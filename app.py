from flask import Flask, request, jsonify
from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
from langchain.retrievers import BM25Retriever
from langchain.retrievers import MultiQueryRetriever
import os
import re
import wikipediaapi
import logging
from functools import lru_cache
import numpy as np
from uuid import uuid4
from datetime import datetime
import pytz

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set OpenAI API key securely
os.environ["OPENAI_API_KEY"] = "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"

# Initialize Wikipedia API
user_agent = "YouTubeVideoChat/1.0 (https://example.com/contact; your-email@example.com)"
wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language='en')

# Global variables
vector_store = None
video_summary = None
bm25_retriever = None
cached_embeddings = {}
conversations = {}  # Store conversations: {conv_id: {"url": str, "key_phrase": str, "messages": list}}

def get_current_time():
    """Return the current time in the format '07:04 PM IST'."""
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime('%I:%M %p IST')
    return current_time

def extract_video_id(url):
    """Extract YouTube video ID from URL."""
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def generate_key_phrase(summary):
    """Generate a key phrase from the video summary."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    key_phrase_prompt = PromptTemplate(
        template="""
        Generate a concise key phrase (3-5 words) that captures the main topic of the following summary.
        Summary: {summary}
        Key Phrase:
        """,
        input_variables=['summary']
    )
    key_phrase_chain = key_phrase_prompt | llm | StrOutputParser()
    try:
        key_phrase = key_phrase_chain.invoke({"summary": summary})
        return key_phrase.strip()
    except Exception as e:
        logger.error(f"Error generating key phrase: {str(e)}")
        return "Video Topic"

def summarize_transcript(transcript):
    """Generate a summary of the transcript using an LLM."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    summary_prompt = PromptTemplate(
        template="""
        You are a helpful assistant. Summarize the following transcript in 3-5 sentences, capturing the main topics discussed.
        Transcript:
        {transcript}
        Summary:
        """,
        input_variables=['transcript']
    )
    summary_chain = summary_prompt | llm | StrOutputParser()
    try:
        summary = summary_chain.invoke({"transcript": transcript})
        logger.info(f"Generated video summary: {summary}")
        return summary
    except Exception as e:
        logger.error(f"Error summarizing transcript: {str(e)}")
        return "Unable to summarize transcript."

@lru_cache(maxsize=100)
def cached_rewrite_query(question):
    """Rewrite query using LLM for better retrieval with caching."""
    if not question or len(question.strip()) < 3:
        logger.warning(f"Invalid question for rewriting: {question}")
        return question
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    rewrite_prompt = PromptTemplate(
        template="""
        You are an expert query rewriter. Rewrite the following question to make it more precise and clear for information retrieval, preserving the original intent. If the question is too vague, rephrase it to focus on key details likely to be in a video transcript.
        Original Question: {question}
        Rewritten Question:
        """,
        input_variables=['question']
    )
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    try:
        rewritten = rewrite_chain.invoke({"question": question})
        return rewritten if rewritten else question
    except Exception as e:
        logger.error(f"Error rewriting query: {str(e)}")
        return question

def determine_domain(question):
    """Determine if question relates to video or general knowledge."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    domain_prompt = PromptTemplate(
        template="""
        Determine if the question is specifically about a video's content or seeking general knowledge.
        Return 'video' for video-specific questions or 'general' for others.
        If unsure, default to 'video'.
        Question: {question}
        Domain:
        """,
        input_variables=['question']
    )
    domain_chain = domain_prompt | llm | StrOutputParser()
    try:
        domain = domain_chain.invoke({"question": question}).strip()
        return domain if domain in ['video', 'general'] else 'video'
    except Exception as e:
        logger.error(f"Error determining domain: {str(e)}")
        return 'video'

@app.route('/process-video', methods=['POST'])
def process_video():
    global vector_store, video_summary, bm25_retriever, conversations
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400
    
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = None
        transcript_language = None
        
        try:
            transcript_obj = transcript_list.find_transcript(['en'])
            transcript = transcript_obj.fetch()
            transcript_language = 'en'
        except NoTranscriptFound:
            for transcript_obj in transcript_list:
                try:
                    transcript = transcript_obj.fetch()
                    transcript_language = transcript_obj.language_code
                    break
                except Exception as e:
                    logger.warning(f"Failed to fetch transcript in {transcript_obj.language_code}: {str(e)}")
                    continue
        
        if not transcript:
            logger.error("No transcripts available")
            return jsonify({"error": "No transcripts available for this video"}), 400
        
        transcript_text = " ".join(chunk.text for chunk in transcript)
        if not transcript_text.strip():
            logger.error("Transcript is empty")
            return jsonify({"error": "Transcript is empty"}), 400
        
        logger.info(f"Transcript fetched for video ID {video_id} in language {transcript_language}")
        
        video_summary = summarize_transcript(transcript_text)
        key_phrase = generate_key_phrase(video_summary)
        
        # Create new conversation
        conv_id = str(uuid4())
        conversations[conv_id] = {
            "url": url,
            "key_phrase": key_phrase,
            "messages": []
        }
        
        # Increase chunk size to reduce number of chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript_text])
        if not chunks:
            logger.error("No chunks created from transcript")
            return jsonify({"error": "Failed to process transcript into chunks"}), 500
        logger.info(f"Transcript split into {len(chunks)} chunks")
        
        # Batch embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=1000)
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Initialize BM25 retriever for hybrid search
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 6
        
        logger.info("Vector store and BM25 retriever created successfully")
        return jsonify({
            "status": "success",
            "language": transcript_language,
            "conv_id": conv_id,
            "key_phrase": key_phrase
        })
    
    except TranscriptsDisabled:
        logger.error("Transcripts disabled for this video")
        return jsonify({"error": "Transcripts are disabled for this video"}), 400
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return jsonify({"error": str(e)}), 500

def hybrid_retrieval(question, retriever, bm25_retriever):
    """Perform hybrid retrieval with MMR and reranking."""
    try:
        vector_results = retriever.get_relevant_documents(question)
    except Exception as e:
        logger.error(f"Error in vector retrieval: {str(e)}")
        vector_results = []
    
    try:
        bm25_results = bm25_retriever.get_relevant_documents(question)
    except Exception as e:
        logger.error(f"Error in BM25 retrieval: {str(e)}")
        bm25_results = []
    
    seen_content = set()
    combined_results = []
    for doc in vector_results + bm25_results:
        if doc.page_content and doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            combined_results.append(doc)
    
    if not combined_results:
        logger.warning("No documents retrieved for hybrid retrieval")
        return []
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    doc_texts = [doc.page_content for doc in combined_results]
    try:
        doc_embeddings = embeddings.embed_documents(doc_texts)
    except Exception as e:
        logger.error(f"Error embedding documents: {str(e)}")
        return []
    
    if question not in cached_embeddings:
        try:
            cached_embeddings[question] = embeddings.embed_query(question)
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            return []
    query_embedding = cached_embeddings[question]
    
    def mmr_score(doc_idx, selected_docs, lambda_param=0.5):
        if doc_idx >= len(doc_embeddings) or not doc_embeddings:
            logger.warning(f"Invalid doc_idx: {doc_idx}, max: {len(doc_embeddings)}")
            return -float('inf')
        doc_embedding = doc_embeddings[doc_idx]
        sim_to_query = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-10
        )
        max_sim_to_selected = max(
            [
                np.dot(doc_embeddings[sel_idx], doc_embedding) / (
                    np.linalg.norm(doc_embeddings[sel_idx]) * np.linalg.norm(doc_embedding) + 1e-10
                )
                for sel_idx in selected_docs if sel_idx < len(doc_embeddings)
            ] if selected_docs else [0]
        )
        return lambda_param * sim_to_query - (1 - lambda_param) * max_sim_to_selected
    
    selected_docs = []
    selected_indices = []
    combined_results_copy = combined_results.copy()
    doc_embeddings_copy = doc_embeddings.copy()
    while combined_results_copy and len(selected_docs) < 6:
        scored_docs = [(i, mmr_score(i, selected_indices)) for i in range(len(combined_results_copy))]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        if not scored_docs or scored_docs[0][1] == -float('inf'):
            logger.warning("No valid documents for MMR scoring")
            break
        selected_idx = scored_docs[0][0]
        selected_docs.append(combined_results_copy[selected_idx])
        selected_indices.append(selected_idx)
        combined_results_copy.pop(selected_idx)
        doc_embeddings_copy.pop(selected_idx)
    
    if not selected_docs:
        logger.warning("No documents selected for reranking")
        return []
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    rerank_prompt = PromptTemplate(
        template="""
        Rank the following documents based on their relevance to the question: {question}
        Return the top 4 most relevant documents in order.
        Documents:
        {docs}
        Ordered Documents:
        """,
        input_variables=['question', 'docs']
    )
    rerank_chain = rerank_prompt | llm | StrOutputParser()
    
    docs_text = "\n".join([f"Doc {i+1}: {doc.page_content}" for i, doc in enumerate(selected_docs)])
    try:
        reranked_text = rerank_chain.invoke({"question": question, "docs": docs_text})
    except Exception as e:
        logger.error(f"Error in reranking: {str(e)}")
        return selected_docs[:4]
    
    reranked_docs = []
    for line in reranked_text.split("\n"):
        if line.startswith("Doc"):
            try:
                doc_idx = int(line.split(":")[0].replace("Doc ", "")) - 1
                if 0 <= doc_idx < len(selected_docs):
                    reranked_docs.append(selected_docs[doc_idx])
            except (ValueError, IndexError):
                logger.warning(f"Invalid doc index in reranking: {line}")
                continue
    
    return reranked_docs[:4] if reranked_docs else selected_docs[:4]

@app.route('/ask-question', methods=['POST'])
def ask_question():
    global vector_store, video_summary, bm25_retriever, conversations
    data = request.get_json()
    question = data.get('question')
    chat_history = data.get('chat_history', [])
    conv_id = data.get('conv_id')
    
    if not conv_id or conv_id not in conversations:
        logger.error("Invalid or missing conversation ID")
        return jsonify({"error": "Invalid or missing conversation ID"}), 400
    
    if vector_store is None or bm25_retriever is None:
        logger.error("No vector store or BM25 retriever available")
        return jsonify({"error": "No video transcript processed yet"}), 400
    
    if not question or len(question.strip()) < 3:
        logger.error("No valid question provided")
        return jsonify({"error": "No valid question provided"}), 400
    
    try:
        current_time = get_current_time()
        history_text = "".join(
            f"User at {msg.get('timestamp', current_time)}: {msg['text']}\n" if msg['type'] == 'user'
            else f"Bot at {msg.get('timestamp', current_time)}: {msg['text']}\n" for msg in chat_history
        )
        
        question_with_time = f"User at {current_time}: {question}"
        logger.info(f"Question with timestamp: {question_with_time}")

        rewritten_question = cached_rewrite_query(question)
        logger.info(f"Rewritten question: {rewritten_question}")

        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=vector_store.as_retriever(),
            llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.4),
            prompt=PromptTemplate(
                template="""
                Generate up to 2 alternative queries for the following question to improve retrieval:
                Question: {question}
                Alternative Queries:
                """,
                input_variables=['question']
            )
        )

        domain = determine_domain(rewritten_question)
        logger.info(f"Determined domain: {domain}")

        if "summarize" in rewritten_question.lower() and "video" in rewritten_question.lower():
            if video_summary:
                logger.info("Returning pre-generated video summary")
                conversations[conv_id]["messages"].append({
                    "type": "user",
                    "text": question,
                    "timestamp": current_time
                })
                conversations[conv_id]["messages"].append({
                    "type": "bot",
                    "text": video_summary,
                    "timestamp": current_time
                })
                return jsonify({
                    "answer": video_summary,
                    "timestamp": current_time,
                    "citations": []
                })

        base_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        compressor = LLMChainExtractor.from_llm(ChatOpenAI(model="gpt-4o-mini", temperature=0.2))
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

        retrieved_docs = hybrid_retrieval(rewritten_question, compression_retriever, bm25_retriever)
        if not retrieved_docs:
            logger.warning("No documents retrieved; falling back to general knowledge")
            domain = 'general'
        
        context = "\n\n".join([f"[Source {i+1}] {doc.page_content}" for i, doc in enumerate(retrieved_docs)])
        if len(context) > 8000:
            context = context[:8000]
        
        answer_prompt = PromptTemplate(
            template="""
            You are a helpful assistant. Answer the question based on the provided context and chat history.
            Provide citations in the format [Source X] where relevant.
            If the context is insufficient, indicate this clearly and use general knowledge if appropriate.
            Avoid speculation and maintain factual accuracy.
            Do not provide sensitive or harmful information.
            
            Chat History:
            {history}
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:
            """,
            input_variables=['history', 'context', 'question']
        )
        
        answer_chain = (
            {
                'history': RunnableLambda(lambda _: history_text),
                'context': RunnableLambda(lambda _: context),
                'question': RunnableLambda(lambda _: rewritten_question)
            }
            | answer_prompt
            | ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
            | StrOutputParser()
        )
        
        answer = answer_chain.invoke({})
        logger.info(f"Generated answer: {answer}")

        citations = []
        for i, doc in enumerate(retrieved_docs, 1):
            if f"[Source {i}]" in answer:
                citations.append({
                    "source": f"Source {i}",
                    "content": doc.page_content[:200]
                })

        guardrail_prompt = PromptTemplate(
            template="""
            Review the following answer for accuracy and appropriateness.
            If the answer contains speculative, harmful, or inappropriate content, return "GUARDRAIL_NONVIOLATION".
            Otherwise, return the original answer.
            Be lenient unless the content is clearly harmful or speculative.
            
            Answer: {answer}
            Result:
            """,
            input_variables=['answer']
        )
        
        guardrail_chain = guardrail_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0.2) | StrOutputParser()
        guardrail_result = guardrail_chain.invoke({"answer": answer})
        
        if guardrail_result == "GUARDRAIL_VIOLATION":
            logger.warning("Answer flagged by guardrail")
            return jsonify({
                "error": "Answer contains inappropriate content",
                "timestamp": current_time,
                "citations": []
            }), 400

        if ("INSUFFICIENT_CONTEXT" in answer.upper() or not citations) and domain == 'video':
            logger.info("Falling back to Wikipedia")
            wiki_title = extract_wiki_title(rewritten_question)
            wiki_page = wiki.page(wiki_title)
            if wiki_page.exists():
                summary = wiki_page.summary[:1000]
                wiki_prompt = PromptTemplate(
                    template="""
                    Answer the question using the Wikipedia summary.
                    Include [Wikipedia] as citation.
                    Chat History: {history}
                    Wikipedia Summary: {summary}
                    Question: {question}
                    Answer:
                    """,
                    input_variables=['history', 'summary', 'question']
                )
                wiki_chain = (
                    {
                        'history': RunnableLambda(lambda _: history_text),
                        'summary': RunnableLambda(lambda _: summary),
                        'question': RunnableLambda(lambda _: rewritten_question)
                    }
                    | wiki_prompt
                    | ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
                    | StrOutputParser()
                )
                answer = wiki_chain.invoke({})
                citations = [{"source": "Wikipedia", "content": summary[:200]}]
            else:
                domain = 'general'

        if domain == 'general':
            logger.info("Using general knowledge")
            general_prompt = PromptTemplate(
                template="""
                Answer the question using general knowledge.
                Indicate that no specific sources were used.
                Chat History: {history}
                Question: {question}
                Answer:
                """,
                input_variables=['history', 'question']
            )
            general_chain = (
                {
                    'history': RunnableLambda(lambda _: history_text),
                    'question': RunnableLambda(lambda _: rewritten_question)
                }
                | general_prompt
                | ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
                | StrOutputParser()
            )
            answer = general_chain.invoke({})
            citations = []

        # Store messages in conversation
        conversations[conv_id]["messages"].append({
            "type": "user",
            "text": question,
            "timestamp": current_time
        })
        conversations[conv_id]["messages"].append({
            "type": "bot",
            "text": answer,
            "timestamp": current_time
        })

        response = {
            "answer": answer,
            "timestamp": current_time,
            "citations": citations
        }
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        return jsonify({
            "error": str(e),
            "timestamp": current_time,
            "citations": []
        }), 500

@app.route('/conversations', methods=['GET'])
def get_conversations():
    """Return a list of all conversations with their IDs, URLs, and key phrases."""
    global conversations
    return jsonify([
        {
            "conv_id": conv_id,
            "url": conv_data["url"],
            "key_phrase": conv_data["key_phrase"],
            "message_count": len(conv_data["messages"])
        }
        for conv_id, conv_data in conversations.items()
    ])

@app.route('/conversation/<conv_id>', methods=['GET'])
def get_conversation(conv_id):
    """Return a specific conversation by ID."""
    global conversations
    if conv_id not in conversations:
        return jsonify({"error": "Conversation not found"}), 404
    
    conv_data = conversations[conv_id]
    return jsonify({
        "conv_id": conv_id,
        "url": conv_data["url"],
        "key_phrase": conv_data["key_phrase"],
        "messages": conv_data["messages"]
    })

def extract_wiki_title(question):
    """Extract a potential Wikipedia page title from the question."""
    question = question.lower().strip("?")
    stop_words = ["can", "you", "please", "what", "is", "the", "this", "about"]
    words = [word for word in question.split() if word not in stop_words]
    if "summarize" in words and "video" in words:
        return "Transformers (machine learning)"
    return " ".join(words).capitalize()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": get_current_time()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

