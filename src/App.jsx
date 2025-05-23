import { useState, useEffect, useRef } from 'react';
import sunIcon from './assets/sun.png';
import moonIcon from './assets/moon.png';
import './index.css';

// Placeholder SVGs for user and bot icons
const UserIcon = () => (
  <svg className="message-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
  </svg>
);

const BotIcon = () => (
  <svg className="message-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
  </svg>
);

function App() {
  const [url, setUrl] = useState('');
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isBotThinking, setIsBotThinking] = useState(false);
  const [error, setError] = useState(null);
  const [isVideoProcessed, setIsVideoProcessed] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');
  const [conversations, setConversations] = useState({});
  const [activeConvId, setActiveConvId] = useState(null);
  const [isSidebarVisible, setIsSidebarVisible] = useState(true);
  const [theme, setTheme] = useState('light'); // Theme state: 'light' or 'dark'
  const chatAreaRef = useRef(null);

  // Load state from localStorage on mount
  useEffect(() => {
    const storedUrl = localStorage.getItem('videoUrl');
    const storedMessages = localStorage.getItem('chatMessages');
    const storedIsVideoProcessed = localStorage.getItem('isVideoProcessed');
    const storedSuccessMessage = localStorage.getItem('successMessage');
    const storedActiveConvId = localStorage.getItem('activeConvId');
    const storedIsSidebarVisible = localStorage.getItem('isSidebarVisible');
    const storedTheme = localStorage.getItem('theme') || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');

    if (storedUrl) setUrl(storedUrl);
    if (storedMessages) setMessages(JSON.parse(storedMessages) || []);
    if (storedIsVideoProcessed) setIsVideoProcessed(JSON.parse(storedIsVideoProcessed) || false);
    if (storedSuccessMessage) setSuccessMessage(storedSuccessMessage);
    if (storedActiveConvId) setActiveConvId(storedActiveConvId);
    if (storedIsSidebarVisible) setIsSidebarVisible(JSON.parse(storedIsSidebarVisible) || true);
    if (storedTheme) setTheme(storedTheme);
  }, []);

  // Save state to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('videoUrl', url);
    localStorage.setItem('chatMessages', JSON.stringify(messages));
    localStorage.setItem('isVideoProcessed', JSON.stringify(isVideoProcessed));
    localStorage.setItem('successMessage', successMessage);
    localStorage.setItem('activeConvId', activeConvId || '');
    localStorage.setItem('isSidebarVisible', JSON.stringify(isSidebarVisible));
    localStorage.setItem('theme', theme);
  }, [url, messages, isVideoProcessed, successMessage, activeConvId, isSidebarVisible, theme]);

  // Apply theme by setting data-theme attribute on the document body
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  // Auto-scroll to the bottom when messages change or bot is thinking
  useEffect(() => {
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }
  }, [messages, isBotThinking]);

  // Fetch conversations for sidebar
  const fetchConversations = async () => {
    try {
      const response = await fetch('http://localhost:8000/conversations', {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
      }
      const data = await response.json();
      console.log('Conversations response:', data); // Debug: Log raw response

      // Ensure data is an array
      if (!Array.isArray(data)) {
        console.error('Invalid response format: Expected an array, got:', data);
        setConversations({});
        setError('Failed to load conversations: Invalid response format');
        return;
      }

      // Group conversations by date
      const grouped = data.reduce((acc, conv) => {
        const messages = Array.isArray(conv.messages) ? conv.messages : [];
        let date = 'Unknown Date';
        if (messages.length > 0 && messages[0].timestamp) {
          try {
            const timestamp = messages[0].timestamp.split(' IST')[0];
            date = new Date(timestamp).toLocaleDateString();
          } catch (e) {
            console.warn(`Failed to parse timestamp for conv ${conv.conv_id}: ${e}`, messages[0].timestamp);
            // Fallback to created_at or current date
            date = conv.created_at ? new Date(conv.created_at).toLocaleDateString() : new Date().toLocaleDateString();
          }
        } else {
          // Use created_at if available, otherwise current date
          date = conv.created_at ? new Date(conv.created_at).toLocaleDateString() : new Date().toLocaleDateString();
        }
        if (!acc[date]) acc[date] = [];
        acc[date].push(conv);
        return acc;
      }, {});
      setConversations(grouped);
      setError(null); // Clear any previous errors
    } catch (err) {
      console.error('Error fetching conversations:', err);
      setError(`Failed to load conversations: ${err.message}`);
      setConversations({}); // Reset to empty object on error
    }
  };

  // Fetch a specific conversation
  const fetchConversation = async (convId) => {
    try {
      const response = await fetch(`http://localhost:8000/conversation/${convId}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}, message: ${await response.text()}`);
      }
      const data = await response.json();
      console.log('Conversation response:', data); // Debug: Log raw response
      if (data.error) {
        setError(data.error);
        return;
      }
      setMessages(data.messages || []);
      setActiveConvId(convId);
      setIsVideoProcessed(true);
      setUrl(data.url || ''); // Set the URL to the conversation's original URL
      setError(null);
    } catch (err) {
      console.error('Error fetching conversation:', err);
      setError(`Failed to load conversation: ${err.message}`);
    }
  };

  // Fetch conversations on mount
  useEffect(() => {
    fetchConversations();
  }, []);

  const processVideo = async () => {
    if (!url.trim()) {
      setError('Please enter a valid YouTube URL');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setSuccessMessage('');

    try {
      const response = await fetch('http://localhost:8000/process-video', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });
      const data = await response.json();
      console.log('Process video response:', data); // Debug: Log raw response
      if (response.ok) {
        setMessages([]); // Clear previous messages
        setIsVideoProcessed(true);
        setSuccessMessage('Successfully fetched');
        setActiveConvId(data.conv_id);
        setUrl(data.url || url); // Ensure URL is set
        await fetchConversations(); // Refresh conversations to include the new one
      } else {
        setError(data.error || 'Failed to process video');
      }
    } catch (err) {
      setError(`Failed to process video: ${err.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const askQuestion = async () => {
    if (!question.trim() || !activeConvId) {
      setError('Please select a conversation and enter a question');
      return;
    }

    const istTime = new Date().toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: true,
      timeZone: 'Asia/Kolkata',
    }) + ' IST';
    const userMessage = { type: 'user', text: question, timestamp: istTime };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);

    setQuestion('');
    setIsBotThinking(true);

    try {
      const response = await fetch('http://localhost:8000/ask-question', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question,
          chat_history: updatedMessages.filter((msg) => msg.type !== 'system'),
          conv_id: activeConvId,
        }),
      });
      const data = await response.json();
      console.log('Ask question response:', data); // Debug: Log raw response
      if (response.ok) {
        setMessages((prev) => [
          ...prev,
          {
            type: 'bot',
            text: data.answer,
            timestamp: data.timestamp || istTime,
            citations: data.citations || [],
          },
        ]);
        await fetchConversations(); // Refresh conversations to update message count
      } else {
        setError(`Error: ${data.error}`);
      }
    } catch (err) {
      setError(`Failed to get answer: ${err.message}`);
    } finally {
      setIsBotThinking(false);
    }
  };

  const toggleSidebar = () => {
    setIsSidebarVisible(!isSidebarVisible);
  };

  const toggleTheme = () => {
    setTheme((prevTheme) => (prevTheme === 'light' ? 'dark' : 'light'));
  };

  return (
    <div className="app-container">
      <div className={`sidebar ${isSidebarVisible ? '' : 'sidebar-hidden'}`}>
        <div className="sidebar-header-container">
          <h2 className="sidebar-header">Conversations</h2>
          <button
            className="sidebar-toggle-button"
            onClick={toggleSidebar}
            data-testid="sidebar-close-button"
          >
            {isSidebarVisible ? '✕' : '☰'}
          </button>
        </div>
        {Object.entries(conversations).length === 0 && (
          <p className="conversation-meta">No conversations available</p>
        )}
        {Object.entries(conversations).map(([date, convList]) => (
          <div key={date} className="conversation-date-group">
            <h3 className="conversation-date">{date}</h3>
            {convList.map((conv) => (
              <div
                key={conv.conv_id}
                className={`conversation-item ${conv.conv_id === activeConvId ? 'active' : ''}`}
                onClick={() => fetchConversation(conv.conv_id)}
              >
                <p className="conversation-title">{conv.key_phrase}</p>
                <p className="conversation-meta">{conv.message_count} messages</p>
                <p className="conversation-url">{conv.url}</p>
              </div>
            ))}
          </div>
        ))}
      </div>
      <div className="main-content" style={{ width: isSidebarVisible ? 'calc(100% - 280px)' : '100%' }}>
        {!isSidebarVisible && (
          <button
            className="sidebar-toggle-button sidebar-open-button"
            onClick={toggleSidebar}
            data-testid="sidebar-open-button"
          >
            ☰
          </button>
        )}
        {/* Fixed Header */}
        <div className="header-section">
          <h1 className="header">Chat With YouTube Video</h1>
          <div className="neumorphic-toggle" onClick={toggleTheme}>
            <div className={`toggle-container ${theme}`}>
              <div className="toggle-thumb">
                <img
                  src={theme === 'light' ? sunIcon : moonIcon}
                  alt={theme === 'light' ? 'Light Mode' : 'Dark Mode'}
                  className="theme-icon"
                />
              </div>
              {theme === 'light' ? (
                <span className="toggle-label left">DARK MODE</span>
              ) : (
                <span className="toggle-label right">LIGHT MODE</span>
              )}
            </div>
          </div>
        </div>

        {/* Fixed URL Input Section */}
        <div className="url-section">
          <label className="label">Enter YouTube Video URL</label>
          <div className="input-group">
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://www.youtube.com/watch?v=..."
              onKeyPress={(e) => e.key === 'Enter' && processVideo()}
              className="input"
            />
            <button
              onClick={processVideo}
              disabled={isProcessing || !url.trim()}
              className={`button ${isProcessing || !url.trim() ? 'button-disabled' : ''}`}
            >
              {isProcessing ? (
                <span className="button-loading">
                  <svg
                    className="spinner"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="spinner-circle"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="spinner-path"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  Processing...
                </span>
              ) : (
                'Process Video'
              )}
            </button>
          </div>
          {successMessage && <p className="success-message">{successMessage}</p>}
          {error && <p className="error-message">{error}</p>}
        </div>

        {/* Scrollable Chat Section */}
        <div className="chat-section">
          <div className="chat-area" ref={chatAreaRef}>
            {messages
              .filter((msg) => msg.type !== 'system')
              .map((msg, index) => (
                <div
                  key={index}
                  className={`message ${
                    msg.type === 'user' ? 'message-user' : 'message-other'
                  }`}
                >
                  {/* <div className="message-icon-wrapper">
                    {msg.type === 'user' ? <UserIcon /> : <BotIcon />}
                  </div> */}
                  <div
                    className={`message-bubble ${
                      msg.type === 'user'
                        ? 'bubble-user'
                        : msg.type === 'bot'
                        ? 'bubble-bot'
                        : 'bubble-system'
                    }`}
                  >
                    {msg.text}
                    <div className="message-timestamp">{msg.timestamp}</div>
                    {msg.citations && msg.citations.length > 0 && (
                      <div>
                        {/* {msg.citations.map((citation, i) => (
                          <p key={i} className="citation">
                            {citation.source}: {citation.content}
                          </p>
                        ))} */}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            {isBotThinking && (
              <div className="message message-other">
                <div className="message-icon-wrapper">
                  <BotIcon />
                </div>
                <div className="message-bubble bubble-bot thinking">
                  <div className="thinking-animation">
                    <span className="thinking-text">Bot is thinking</span>
                    <span className="thinking-particle"></span>
                    <span className="thinking-particle"></span>
                    <span className="thinking-particle"></span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Fixed Chat Input Section */}
        <div className="chat-input-section">
          <div className="chat-input-group">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question about the video..."
              className="input"
              onKeyPress={(e) => e.key === 'Enter' && askQuestion()}
              disabled={!isVideoProcessed}
            />
            <button
              onClick={askQuestion}
              disabled={!question.trim() || !isVideoProcessed || isBotThinking}
              className={`button ${
                !question.trim() || !isVideoProcessed || isBotThinking
                  ? 'button-disabled'
                  : ''
              }`}
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;