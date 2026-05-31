'use client';
import React, { useState, useRef, useEffect, useLayoutEffect, useCallback } from 'react';
import { gsap } from 'gsap';
import { Toaster } from 'sonner';
import ChatInputBox from './ChatBox';
import { sendMessageToBackend } from '../lib/api';
import MarkdownRenderer from '../components/Markdown';
import 'github-markdown-css/github-markdown.css';
import FileUpload from './FileUpload';
import { jwtDecode } from 'jwt-decode';
import { Copy, Check, FileText, Zap, Brain, BookOpen } from 'lucide-react';
import { useChatStore, saveMessageToBackend } from '../components/stores/ChatStore';
import { useUploadStore } from '../components/stores/UploadStore';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

type GoogleUser = { name: string; email: string; picture: string };

interface Message {
  id: number;
  sender: 'user' | 'ai';
  text: string;
  timestamp: number;
}

const suggestions = [
  { icon: <FileText size={16} />, text: 'Summarize this PDF for me' },
  { icon: <Zap size={16} />, text: 'What are the key concepts?' },
  { icon: <Brain size={16} />, text: 'Quiz me on the content' },
  { icon: <BookOpen size={16} />, text: 'Explain the main argument' },
];

const formatTime = (ts: number) => {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

// ── Copy Button ────────────────────────────────────────────────────
const CopyButton: React.FC<{ text: string }> = ({ text }) => {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <button
      onClick={handleCopy}
      className="copy-btn p-1.5 rounded-lg transition-all duration-200 text-white/30 hover:text-white/80 hover:bg-white/10"
      title="Copy response"
    >
      {copied ? <Check size={14} className="text-sky-400" /> : <Copy size={14} />}
    </button>
  );
};

// ── AI Avatar ──────────────────────────────────────────────────────
const AIAvatar = () => (
  <div
    className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium"
    style={{
      background: 'linear-gradient(135deg, #0ea5e9, #2563eb)',
      boxShadow: '0 2px 12px rgba(14,165,233,0.4)',
    }}
  >
    E
  </div>
);

// ── User Avatar ────────────────────────────────────────────────────
const UserAvatar: React.FC<{ name?: string; picture?: string }> = ({ name, picture }) => {
  if (picture) {
    return (
      <img
        src={picture}
        alt={name || 'User'}
        className="flex-shrink-0 w-8 h-8 rounded-full object-cover"
        style={{ boxShadow: '0 2px 12px rgba(0,0,0,0.4)' }}
      />
    );
  }
  const initials = (name || 'U').split(' ').map((n) => n[0]).join('').slice(0, 2).toUpperCase();
  return (
    <div
      className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium text-white"
      style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)' }}
    >
      {initials}
    </div>
  );
};

// ── Thinking Indicator ─────────────────────────────────────────────
const ThinkingIndicator = () => (
  <div className="flex items-start gap-3 justify-start">
    <AIAvatar />
    <div
      className="flex items-center gap-1.5 px-4 py-3 rounded-2xl rounded-tl-sm"
      style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}
    >
      <span className="typing-dot" />
      <span className="typing-dot" />
      <span className="typing-dot" />
      <span className="text-xs text-white/30 ml-1 font-light">Thinking…</span>
    </div>
  </div>
);

// ── Empty State ────────────────────────────────────────────────────
const EmptyState: React.FC<{
  userName?: string;
  onSuggestion: (text: string) => void;
  hasPdf: boolean;
}> = ({ userName, onSuggestion, hasPdf }) => (
  <div className="flex flex-col items-center justify-center min-h-[60vh] px-4 text-center gap-8">
    {/* Greeting */}
    <div className="space-y-3">
      <h1
        className="text-4xl sm:text-5xl md:text-6xl font-light leading-tight
          bg-gradient-to-r from-blue-300 via-sky-400 to-sky-600
          bg-clip-text text-transparent animate-gradient-curl"
        style={{ backgroundSize: '300% 300%' }}
      >
        Hello, {userName?.split(' ')[0] || 'there'} 👋
      </h1>
      <p className="text-white/40 font-light text-lg">
        {hasPdf ? 'Your PDF is ready. Ask me anything about it.' : 'Start by uploading a PDF, then ask me anything.'}
      </p>
    </div>

    {/* File Upload */}
    {!hasPdf && <FileUpload />}

    {/* Suggestion chips */}
    {hasPdf && (
      <div className="flex flex-wrap justify-center gap-3 max-w-lg">
        {suggestions.map((s, i) => (
          <button
            key={i}
            onClick={() => onSuggestion(s.text)}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-light text-white/70 hover:text-white transition-all duration-200"
            style={{
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid rgba(255,255,255,0.1)',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'rgba(56,189,248,0.08)';
              e.currentTarget.style.borderColor = 'rgba(56,189,248,0.3)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'rgba(255,255,255,0.04)';
              e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)';
            }}
          >
            <span className="text-sky-400">{s.icon}</span>
            {s.text}
          </button>
        ))}
      </div>
    )}
  </div>
);

// ── Main Chat Container ────────────────────────────────────────────
const ChatContainer: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isThinking, setIsThinking] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messageRefs = useRef<(HTMLDivElement | null)[]>([]);
  const [user, setUser] = useState<{ name: string; email: string; picture?: string } | null>(null);

  const { uploadedFile } = useUploadStore();
  const { createSession, addMessage, activeSessionId } = useChatStore();

  const sessionIdRef = useRef<string | null>(activeSessionId);

  // Bootstrap or join session
  useEffect(() => {
    if (!sessionIdRef.current) {
      const id = createSession(uploadedFile?.name);
      sessionIdRef.current = id;
    }
  }, []);

  // Update session when PDF changes
  useEffect(() => {
    if (uploadedFile && sessionIdRef.current) {
      // Optionally update pdf name in session
    }
  }, [uploadedFile]);

  // Load user
  useEffect(() => {
    fetch(`${API_URL}/auth/me`, { credentials: 'include' })
      .then((res) => {
        if (!res.ok) throw new Error('Unauthenticated');
        return res.json();
      })
      .then((data) => {
        const userInfo = data.token ? jwtDecode<GoogleUser>(data.token) : data;
        setUser(userInfo);
      })
      .catch(() => {
        window.location.href = '/';
      });
  }, []);

  // Auto scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Animate new messages
  useLayoutEffect(() => {
    if (messages.length > 0) {
      const lastIdx = messages.length - 1;
      const el = messageRefs.current[lastIdx];
      if (el) {
        gsap.fromTo(el, { opacity: 0, y: 16 }, { opacity: 1, y: 0, duration: 0.35, ease: 'power2.out' });
      }
    }
  }, [messages]);

  const handleSendMessage = useCallback(async (text: string) => {
    if (!text.trim()) return;

    const userMsg: Message = { id: Date.now(), sender: 'user', text: text.trim(), timestamp: Date.now() };
    setMessages((prev) => [...prev, userMsg]);

    const sid = sessionIdRef.current;
    if (sid) {
      addMessage(sid, { sender: 'user', text: text.trim() });
      // Persist to MongoDB
      saveMessageToBackend(sid, 'user', text.trim());
    }

    setIsThinking(true);

    try {
      const aiReply = await sendMessageToBackend(text);
      const aiMsg: Message = { id: Date.now() + 1, sender: 'ai', text: aiReply || 'No answer returned.', timestamp: Date.now() };
      setMessages((prev) => [...prev, aiMsg]);

      if (sid) {
        addMessage(sid, { sender: 'ai', text: aiReply || 'No answer returned.' });
        // Persist to MongoDB
        saveMessageToBackend(sid, 'ai', aiReply || 'No answer returned.');
      }
    } catch {
      const errMsg: Message = { id: Date.now() + 1, sender: 'ai', text: 'Something went wrong. Please try again.', timestamp: Date.now() };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setIsThinking(false);
    }
  }, [addMessage]);
  const handleSuggestion = useCallback((text: string) => {
    handleSendMessage(text);
  }, [handleSendMessage]);

  return (
    <div className="flex flex-col h-screen bg-black">
      <Toaster
        position="bottom-center"
        toastOptions={{
          style: {
            marginBottom: '80px',
            borderRadius: '12px',
            fontSize: '14px',
            background: '#1a1a1a',
            border: '1px solid rgba(255,255,255,0.1)',
            color: '#fff',
            boxShadow: '0 6px 20px rgba(0,0,0,0.5)',
          },
        }}
      />

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-4 pb-36 pt-6">
        {messages.length === 0 && !isThinking ? (
          <EmptyState
            userName={user?.name}
            onSuggestion={handleSuggestion}
            hasPdf={!!uploadedFile}
          />
        ) : (
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((message, index) => (
              <div
                key={message.id}
                ref={(el) => { messageRefs.current[index] = el; }}
                className={`msg-group flex gap-3 ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {/* AI Avatar (left) */}
                {message.sender === 'ai' && <AIAvatar />}

                {/* Bubble */}
                <div className={`flex flex-col gap-1 ${message.sender === 'user' ? 'items-end max-w-[78%]' : 'items-start max-w-[85%]'}`}>
                  {message.sender === 'user' ? (
                    <div
                      className="px-4 py-3 rounded-2xl rounded-tr-sm text-sm leading-relaxed text-white"
                      style={{
                        background: 'linear-gradient(135deg, rgba(14,165,233,0.25), rgba(37,99,235,0.25))',
                        border: '1px solid rgba(56,189,248,0.25)',
                      }}
                    >
                      {message.text}
                    </div>
                  ) : (
                    <div
                      className="relative px-4 py-3 rounded-2xl rounded-tl-sm text-sm leading-relaxed"
                      style={{
                        background: 'rgba(255,255,255,0.03)',
                        border: '1px solid rgba(255,255,255,0.08)',
                      }}
                    >
                      <MarkdownRenderer content={message.text} typingSpeed={10} />
                      {/* Copy btn (shown on hover via CSS) */}
                      <div className="flex justify-end mt-1">
                        <CopyButton text={message.text} />
                      </div>
                    </div>
                  )}

                  {/* Timestamp */}
                  <span className="text-[10px] text-white/20 font-light px-1">
                    {formatTime(message.timestamp)}
                  </span>
                </div>

                {/* User Avatar (right) */}
                {message.sender === 'user' && (
                  <UserAvatar name={user?.name} picture={user?.picture} />
                )}
              </div>
            ))}

            {isThinking && <ThinkingIndicator />}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Box */}
      <ChatInputBox onSendMessage={handleSendMessage} />
    </div>
  );
};

export default ChatContainer;
