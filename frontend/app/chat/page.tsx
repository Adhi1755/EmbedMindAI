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
import {
  Copy, Check, FileText, Zap, Brain, BookOpen,
  Cpu, Database, ChevronRight, Circle,
  MessageSquare, Layers,
} from 'lucide-react';
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

const promptCards = [
  { Icon: FileText, title: 'Summarize', description: 'Summarize this document for me', color: '#38bdf8' },
  { Icon: Zap, title: 'Key Concepts', description: 'What are the key concepts?', color: '#818cf8' },
  { Icon: Brain, title: 'Deep Dive', description: 'Explain the main argument in detail', color: '#34d399' },
  { Icon: BookOpen, title: 'Quiz Me', description: 'Create quiz questions from this content', color: '#f59e0b' },
];

const formatTime = (ts: number) =>
  new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

// ── Copy Button ────────────────────────────────────────────────────────────────
const CopyButton: React.FC<{ text: string }> = ({ text }) => {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <button onClick={handleCopy}
      className="copy-btn p-1.5 rounded-lg transition-all duration-200 text-white/25 hover:text-white/70 hover:bg-white/8"
      title="Copy response">
      {copied ? <Check size={13} className="text-sky-400" /> : <Copy size={13} />}
    </button>
  );
};

// ── AI Avatar ──────────────────────────────────────────────────────────────────
const AIAvatar = () => (
  <div className="flex-shrink-0 w-8 h-8 rounded-xl flex items-center justify-center text-xs font-semibold"
    style={{ background: 'linear-gradient(135deg, #0ea5e9, #2563eb)', boxShadow: '0 2px 12px rgba(14,165,233,0.35)' }}>
    E
  </div>
);

// ── User Avatar ────────────────────────────────────────────────────────────────
const UserAvatar: React.FC<{ name?: string; picture?: string }> = ({ name, picture }) => {
  if (picture)
    return <img src={picture} alt={name || 'User'}
      className="flex-shrink-0 w-8 h-8 rounded-xl object-cover"
      style={{ boxShadow: '0 2px 12px rgba(0,0,0,0.5)' }} />;
  const initials = (name || 'U').split(' ').map((n) => n[0]).join('').slice(0, 2).toUpperCase();
  return (
    <div className="flex-shrink-0 w-8 h-8 rounded-xl flex items-center justify-center text-xs font-medium text-white"
      style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)' }}>
      {initials}
    </div>
  );
};

// ── Thinking Indicator ─────────────────────────────────────────────────────────
const ThinkingIndicator = () => (
  <div className="flex items-start gap-3">
    <AIAvatar />
    <div className="flex items-center gap-2 px-4 py-3 rounded-2xl rounded-tl-sm"
      style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)' }}>
      <span className="typing-dot" />
      <span className="typing-dot" />
      <span className="typing-dot" />
      <span className="text-[11px] text-white/30 ml-1 font-light tracking-wide">Analyzing document</span>
    </div>
  </div>
);

// ── Model Info Bar ─────────────────────────────────────────────────────────────
const ModelInfoBar: React.FC<{ pdfName?: string }> = ({ pdfName }) => (
  <div className="flex items-center gap-0 px-4 py-2.5 flex-wrap gap-y-1"
    style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', background: 'rgba(255,255,255,0.015)' }}>
    {/* Live status */}
    <div className="flex items-center gap-1.5 pr-4"
      style={{ borderRight: '1px solid rgba(255,255,255,0.06)' }}>
      <span className="status-dot" />
      <span className="text-[11px] text-green-400/80 font-light">Live</span>
    </div>

    {/* LLM */}
    <div className="flex items-center gap-1.5 px-4"
      style={{ borderRight: '1px solid rgba(255,255,255,0.06)' }}>
      <Zap size={11} className="text-purple-400/70" />
      <span className="text-[11px] text-white/35 font-light font-mono">gemini-2.5-flash</span>
    </div>

    {/* Embeddings */}
    <div className="flex items-center gap-1.5 px-4"
      style={{ borderRight: '1px solid rgba(255,255,255,0.06)' }}>
      <Cpu size={11} className="text-sky-400/70" />
      <span className="text-[11px] text-white/35 font-light font-mono">gemini-embedding-2 · 3072-D</span>
    </div>

    {/* Vector store */}
    <div className="flex items-center gap-1.5 px-4"
      style={{ borderRight: pdfName ? '1px solid rgba(255,255,255,0.06)' : 'none' }}>
      <Database size={11} className="text-emerald-400/70" />
      <span className="text-[11px] text-white/35 font-light">ChromaDB · HNSW</span>
    </div>

    {/* Active document */}
    {pdfName && (
      <div className="flex items-center gap-1.5 px-4">
        <FileText size={11} className="text-amber-400/70" />
        <span className="text-[11px] text-white/50 font-light truncate max-w-[180px]">{pdfName}</span>
      </div>
    )}
  </div>
);

// ── Empty State ────────────────────────────────────────────────────────────────
const EmptyState: React.FC<{
  userName?: string;
  onSuggestion: (text: string) => void;
  hasPdf: boolean;
}> = ({ userName, onSuggestion, hasPdf }) => (
  <div className="flex flex-col items-center justify-center min-h-[60vh] px-6 text-center gap-10">

    {/* Greeting */}
    <div className="space-y-2">
      <div className="w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-4"
        style={{ background: 'linear-gradient(135deg, #0ea5e9, #2563eb)', boxShadow: '0 8px 32px rgba(14,165,233,0.3)' }}>
        <MessageSquare size={22} className="text-white" />
      </div>
      <h2 className="text-3xl sm:text-4xl font-extralight leading-tight"
        style={{ background: 'linear-gradient(135deg, #c0e8ff, #60a5fa, #818cf8)',
          WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}>
        Hello, {userName?.split(' ')[0] || 'there'}
      </h2>
      <p className="text-white/35 font-light text-sm">
        {hasPdf
          ? 'Your document is indexed. What would you like to explore?'
          : 'Upload a PDF to begin. I will embed and index it instantly.'}
      </p>
    </div>

    {/* Upload or prompt cards */}
    {!hasPdf ? (
      <div className="w-full max-w-md">
        <FileUpload />
        {/* Mini stats */}
        <div className="flex justify-center gap-6 mt-8">
          {[
            { Icon: Cpu, label: '3072-D Embeddings', color: '#38bdf8' },
            { Icon: Layers, label: 'MMR Retrieval', color: '#818cf8' },
            { Icon: Database, label: 'ChromaDB', color: '#34d399' },
          ].map(({ Icon, label, color }) => (
            <div key={label} className="flex flex-col items-center gap-1">
              <Icon size={13} style={{ color, opacity: 0.7 }} />
              <span className="text-[10px] text-white/25 font-light">{label}</span>
            </div>
          ))}
        </div>
      </div>
    ) : (
      <div className="grid grid-cols-2 gap-3 max-w-lg w-full">
        {promptCards.map(({ Icon, title, description, color }) => (
          <button key={title} onClick={() => onSuggestion(description)}
            className="flex flex-col items-start gap-2 p-4 rounded-xl text-left group transition-all duration-200"
            style={{ background: 'rgba(255,255,255,0.025)', border: '1px solid rgba(255,255,255,0.07)' }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = `${color}0a`;
              e.currentTarget.style.borderColor = `${color}25`;
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'rgba(255,255,255,0.025)';
              e.currentTarget.style.borderColor = 'rgba(255,255,255,0.07)';
            }}>
            <div className="w-7 h-7 rounded-lg flex items-center justify-center"
              style={{ background: `${color}12`, border: `1px solid ${color}20` }}>
              <Icon size={13} style={{ color }} />
            </div>
            <div>
              <p className="text-xs font-medium text-white/70 mb-0.5">{title}</p>
              <p className="text-[11px] text-white/35 font-light leading-snug">{description}</p>
            </div>
            <ChevronRight size={12} className="text-white/20 group-hover:text-white/50 transition self-end ml-auto" />
          </button>
        ))}
      </div>
    )}
  </div>
);

// ── Main Chat Container ────────────────────────────────────────────────────────
const ChatContainer: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isThinking, setIsThinking] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messageRefs = useRef<(HTMLDivElement | null)[]>([]);
  const [user, setUser] = useState<{ name: string; email: string; picture?: string } | null>(null);

  const { uploadedFile } = useUploadStore();
  const { createSession, addMessage, activeSessionId } = useChatStore();
  const sessionIdRef = useRef<string | null>(activeSessionId);

  useEffect(() => {
    if (!sessionIdRef.current) {
      const id = createSession(uploadedFile?.name);
      sessionIdRef.current = id;
    }
  }, []);

  useEffect(() => {
    fetch(`${API_URL}/auth/me`, { credentials: 'include' })
      .then((res) => { if (!res.ok) throw new Error(); return res.json(); })
      .then((data) => setUser(data.token ? jwtDecode<GoogleUser>(data.token) : data))
      .catch(() => { window.location.href = '/'; });
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useLayoutEffect(() => {
    if (messages.length > 0) {
      const el = messageRefs.current[messages.length - 1];
      if (el) gsap.fromTo(el, { opacity: 0, y: 14 }, { opacity: 1, y: 0, duration: 0.3, ease: 'power2.out' });
    }
  }, [messages]);

  const handleSendMessage = useCallback(async (text: string) => {
    if (!text.trim()) return;

    const userMsg: Message = { id: Date.now(), sender: 'user', text: text.trim(), timestamp: Date.now() };
    setMessages((prev) => [...prev, userMsg]);

    const sid = sessionIdRef.current;
    if (sid) {
      addMessage(sid, { sender: 'user', text: text.trim() });
      saveMessageToBackend(sid, 'user', text.trim());
    }

    setIsThinking(true);
    try {
      const aiReply = await sendMessageToBackend(text);
      const aiMsg: Message = { id: Date.now() + 1, sender: 'ai', text: aiReply || 'No answer returned.', timestamp: Date.now() };
      setMessages((prev) => [...prev, aiMsg]);
      if (sid) {
        addMessage(sid, { sender: 'ai', text: aiReply || 'No answer returned.' });
        saveMessageToBackend(sid, 'ai', aiReply || 'No answer returned.');
      }
    } catch {
      const errMsg: Message = { id: Date.now() + 1, sender: 'ai', text: 'Something went wrong. Please try again.', timestamp: Date.now() };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setIsThinking(false);
    }
  }, [addMessage]);

  const handleSuggestion = useCallback((text: string) => handleSendMessage(text), [handleSendMessage]);

  return (
    <div className="flex flex-col h-screen bg-black">
      <Toaster position="bottom-center"
        toastOptions={{ style: { marginBottom: '80px', borderRadius: '12px', fontSize: '13px',
            background: '#0d1017', border: '1px solid rgba(255,255,255,0.08)',
            color: '#e2e8f0', boxShadow: '0 8px 32px rgba(0,0,0,0.6)' } }} />

      {/* Model Info Bar */}
      <ModelInfoBar pdfName={uploadedFile?.name} />

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 && !isThinking ? (
          <EmptyState userName={user?.name} onSuggestion={handleSuggestion} hasPdf={!!uploadedFile} />
        ) : (
          <div className="max-w-3xl mx-auto px-4 py-8 space-y-7">
            {messages.map((message, index) => (
              <div key={message.id}
                ref={(el) => { messageRefs.current[index] = el; }}
                className={`msg-group flex gap-3 ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}>

                {/* AI avatar left */}
                {message.sender === 'ai' && <AIAvatar />}

                {/* Bubble */}
                <div className={`flex flex-col gap-1 ${message.sender === 'user' ? 'items-end max-w-[76%]' : 'items-start max-w-[86%]'}`}>
                  {message.sender === 'user' ? (
                    /* User bubble */
                    <div className="px-4 py-3 rounded-2xl rounded-tr-sm text-sm leading-relaxed text-white"
                      style={{ background: 'linear-gradient(135deg, rgba(14,165,233,0.18), rgba(37,99,235,0.18))',
                        border: '1px solid rgba(56,189,248,0.2)' }}>
                      {message.text}
                    </div>
                  ) : (
                    /* AI bubble */
                    <div className="relative px-5 py-4 rounded-2xl rounded-tl-sm text-sm leading-relaxed w-full"
                      style={{ background: 'rgba(255,255,255,0.025)', border: '1px solid rgba(255,255,255,0.07)',
                        borderLeft: '2px solid rgba(56,189,248,0.35)' }}>
                      <MarkdownRenderer content={message.text} typingSpeed={10} />
                      <div className="flex items-center justify-between mt-2 pt-2"
                        style={{ borderTop: '1px solid rgba(255,255,255,0.04)' }}>
                        <span className="text-[10px] text-white/20 font-light flex items-center gap-1">
                          <Zap size={9} className="text-purple-400/50" />
                          gemini-2.5-flash
                        </span>
                        <CopyButton text={message.text} />
                      </div>
                    </div>
                  )}

                  {/* Timestamp */}
                  <span className="text-[10px] text-white/18 font-light px-1">{formatTime(message.timestamp)}</span>
                </div>

                {/* User avatar right */}
                {message.sender === 'user' && <UserAvatar name={user?.name} picture={user?.picture} />}
              </div>
            ))}

            {isThinking && <ThinkingIndicator />}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <ChatInputBox onSendMessage={handleSendMessage} />
    </div>
  );
};

export default ChatContainer;
