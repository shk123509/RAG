import React, { useState, useRef, useEffect } from "react";
import './Chatbot.css';

const CHAT_ENDPOINT = "http://localhost:8000/chat";

const Message = ({ sender, text, sources }) => {
  // Backend se aane wale text ko parts mein divide kar rahe hain
  const parts = text ? text.split("Sources:") : [text];
  const answerContent = parts[0]; 

  return (
    <div className={`msg-wrapper ${sender}`}>
      <div className="msg-avatar">{sender === "bot" ? "AI" : "YOU"}</div>
      <div className="msg-content">
        <div className="msg-bubble">
          {/* Answer text with white-space support for formatting */}
          <div className="formatted-text">{answerContent}</div>
        </div>

        {/* Sources Grid: Jo backend ke "sources" array se aayega */}
        {sources && sources.length > 0 && (
          <div className="sources-wrapper">
            <p className="sources-title">📑 Reference Documents</p>
            <div className="sources-grid">
              {sources.map((url, i) => (
                <a key={i} href={url} target="_blank" rel="noreferrer" className="source-card">
                  <div className="card-top">
                    <span className="file-icon">📄</span>
                    <span className="file-name">{new URL(url).hostname}</span>
                  </div>
                  <div className="card-url">{url}</div>
                </a>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const Chatbot = () => {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "System Ready. Please enter your query.", sources: [] }
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const endRef = useRef(null);

  useEffect(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), [messages, isLoading]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userQuery = input.trim();
    setMessages(prev => [...prev, { sender: "user", text: userQuery }]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch(CHAT_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userQuery }),
      });

      const data = await response.json();

      // Backend response keys: "response" (Text) and "sources" (Array)
      setMessages(prev => [...prev, { 
        sender: "bot", 
        text: data.response, 
        sources: data.sources 
      }]);
    } catch (error) {
      setMessages(prev => [...prev, { sender: "bot", text: "❌ Error: Could not connect to the backend server." }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="status-indicator"></div>
        <h1>RAG Assistant <small>v1.0</small></h1>
      </header>

      <main className="chat-area">
        <div className="chat-limit">
          {messages.map((m, i) => <Message key={i} {...m} />)}
          {isLoading && (
            <div className="msg-wrapper bot">
              <div className="msg-avatar animate-pulse">🤖</div>
              <div className="msg-bubble loading-dots">🧠<span>.</span><span>.</span><span>.</span></div>
            </div>
          )}
          <div ref={endRef} />
        </div>
      </main>

      <footer className="input-area">
        <div className="input-container">
          <input 
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSend()}
            placeholder="Ask from your documents..."
          />
          <button onClick={handleSend} disabled={isLoading || !input.trim()}>
            {isLoading ? "..." : "Send"}
          </button>
        </div>
      </footer>
    </div>
  );
};

export default Chatbot;