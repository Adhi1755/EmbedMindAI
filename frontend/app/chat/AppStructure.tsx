'use client';

import React, { useEffect, useState, useLayoutEffect, useRef, useCallback } from 'react';
import { Menu, X, Plus, Search, Trash2, MessageSquare, FileText } from 'lucide-react';
import Image from 'next/image';
import clsx from 'clsx';
import gsap from 'gsap';
import Header from './Header';
import { useChatStore, ChatSession } from '../components/stores/ChatStore';
import { useUploadStore } from '../components/stores/UploadStore';
import { jwtDecode } from 'jwt-decode';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

type GoogleUser = { name: string; email: string; picture: string };

interface AppLayoutProps {
  children: React.ReactNode;
}

interface User {
  name: string;
  email: string;
  picture?: string;
}

const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [hovered, setHovered] = useState(false);
  const [isMobileOpen, setIsMobileOpen] = useState(false);
  const [loginLoading, setLoginLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [searchOpen, setSearchOpen] = useState(false);

  const sidebarRef = useRef<HTMLElement>(null);
  const headerRef = useRef<HTMLDivElement>(null);
  const mainRef = useRef<HTMLElement>(null);

  const { sessions, activeSessionId, setActiveSession, createSession, deleteSession, clearSession, hydrateFromBackend, loadMessagesForSession } = useChatStore();
  const { uploadedFile } = useUploadStore();

  // Load user then hydrate sessions from backend
  useEffect(() => {
    fetch(`${API_URL}/auth/me`, { credentials: 'include' })
      .then((res) => {
        if (!res.ok) throw new Error('Unauthenticated');
        return res.json();
      })
      .then(async (data) => {
        const userInfo: User = data.token ? jwtDecode<GoogleUser>(data.token) : data;
        setUser(userInfo);

        // Hydrate sessions from backend
        try {
          const res2 = await fetch(`${API_URL}/chat/sessions`, { credentials: 'include' });
          if (res2.ok) {
            const backendSessions = await res2.json();
            // Map backend shape → ChatSession shape
            const mapped = backendSessions.map((s: any) => ({
              id:        s.session_id,
              title:     s.title,
              messages:  [],           // loaded lazily on switch
              createdAt: new Date(s.created_at).getTime(),
              updatedAt: new Date(s.updated_at).getTime(),
              pdfName:   s.pdf_name,
              synced:    true,
            }));
            hydrateFromBackend(mapped);
          }
        } catch {
          // If backend is unreachable, local store still works
        }

        setTimeout(() => setLoginLoading(false), 400);
      })
      .catch(() => {
        window.location.href = '/';
      });
  }, []);

  // Ensure there's always an active session
  useEffect(() => {
    if (!loginLoading && sessions.length === 0) {
      createSession(uploadedFile?.name);
    } else if (!loginLoading && !activeSessionId && sessions.length > 0) {
      setActiveSession(sessions[0].id);
    }
  }, [loginLoading, sessions.length]);

  useLayoutEffect(() => {
    if (!loginLoading) {
      const ctx = gsap.context(() => {
        gsap.fromTo(sidebarRef.current, { opacity: 0, x: -30 }, { opacity: 1, x: 0, duration: 0.8, ease: 'power2.out' });
        gsap.fromTo(headerRef.current, { opacity: 0, y: -20 }, { opacity: 1, y: 0, duration: 0.8, delay: 0.15, ease: 'power2.out' });
        gsap.fromTo(mainRef.current, { opacity: 0 }, { opacity: 1, duration: 0.8, delay: 0.3, ease: 'power2.out' });
      });
      return () => ctx.revert();
    }
  }, [loginLoading]);

  const handleNewChat = useCallback(() => {
    createSession(uploadedFile?.name);
    setIsMobileOpen(false);
  }, [createSession, uploadedFile]);

  const handleClearChat = useCallback(() => {
    if (activeSessionId) {
      clearSession(activeSessionId);
    }
  }, [activeSessionId, clearSession]);

  // Load messages from backend when switching to a session with no local messages
  const handleSelectSession = useCallback(async (id: string) => {
    setActiveSession(id);
    const session = sessions.find((s) => s.id === id);
    if (session && session.messages.length === 0) {
      await loadMessagesForSession(id);
    }
  }, [setActiveSession, sessions, loadMessagesForSession]);

  const filteredSessions = sessions.filter((s) =>
    s.title.toLowerCase().includes(search.toLowerCase())
  );

  if (loginLoading) {
    return (
      <div className="flex items-center justify-center h-screen w-screen bg-black">
        <div className="flex flex-col items-center gap-4">
          <div
            className="w-10 h-10 rounded-full flex items-center justify-center text-base font-light"
            style={{ background: 'linear-gradient(135deg, #0ea5e9, #2563eb)', boxShadow: '0 0 30px rgba(14,165,233,0.5)' }}
          >
            E
          </div>
          <div className="text-white/30 text-sm font-light animate-pulse">Loading workspace…</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden bg-black">
      {/* Mobile overlay */}
      {isMobileOpen && (
        <div
          className="lg:hidden fixed inset-0 z-50 bg-black/70 backdrop-blur-sm"
          onClick={() => setIsMobileOpen(false)}
        >
          <aside
            className="w-72 h-full flex flex-col"
            style={{
              background: 'rgba(8,8,12,0.98)',
              borderRight: '1px solid rgba(255,255,255,0.08)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-4 py-4 border-b border-white/6">
              <span className="text-sm font-light bg-gradient-to-r from-sky-400 to-blue-600 bg-clip-text text-transparent">
                EmbedMindAI
              </span>
              <button onClick={() => setIsMobileOpen(false)} className="p-1.5 text-white/40 hover:text-white rounded-lg hover:bg-white/10 transition">
                <X size={16} />
              </button>
            </div>
            <SidebarContent
              user={user}
              hovered
              sessions={filteredSessions}
              activeSessionId={activeSessionId}
              search={search}
              setSearch={setSearch}
              searchOpen={searchOpen}
              setSearchOpen={setSearchOpen}
              onNewChat={handleNewChat}
              onSelectSession={(id) => { handleSelectSession(id); setIsMobileOpen(false); }}
              onDeleteSession={deleteSession}
              uploadedFileName={uploadedFile?.name}
            />
          </aside>
        </div>
      )}

      {/* Desktop sidebar */}
      <aside
        ref={sidebarRef}
        className={clsx(
          'hidden lg:flex flex-col h-full transition-all duration-300 fixed top-0 left-0 z-40 flex-shrink-0',
          hovered ? 'w-64' : 'w-16'
        )}
        style={{
          background: 'rgba(6,6,10,0.98)',
          borderRight: '1px solid rgba(255,255,255,0.07)',
        }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        {/* Logo area */}
        <div className="flex items-center h-14 px-4 border-b border-white/6 overflow-hidden">
          <div
            className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center text-sm"
            style={{ background: 'linear-gradient(135deg, #0ea5e9, #2563eb)' }}
          >
            E
          </div>
          <span
            className={clsx(
              'ml-3 text-sm font-light bg-gradient-to-r from-sky-400 to-blue-600 bg-clip-text text-transparent whitespace-nowrap transition-all duration-300',
              hovered ? 'opacity-100 max-w-[200px]' : 'opacity-0 max-w-0 overflow-hidden'
            )}
          >
            EmbedMindAI
          </span>
        </div>

        <SidebarContent
          user={user}
          hovered={hovered}
          sessions={filteredSessions}
          activeSessionId={activeSessionId}
          search={search}
          setSearch={setSearch}
          searchOpen={searchOpen}
          setSearchOpen={setSearchOpen}
          onNewChat={handleNewChat}
          onSelectSession={handleSelectSession}
          onDeleteSession={deleteSession}
          uploadedFileName={uploadedFile?.name}
        />
      </aside>

      {/* Main content */}
      <div
        className={clsx(
          'flex flex-col flex-1 overflow-hidden transition-all duration-300',
          'lg:ml-16',
          hovered && 'lg:ml-64'
        )}
      >
        {/* Mobile menu toggle */}
        <div className="lg:hidden fixed top-3 left-3 z-30">
          <button
            onClick={() => setIsMobileOpen(true)}
            className="p-2 rounded-xl text-white/60 hover:text-white hover:bg-white/10 transition"
          >
            <Menu size={20} />
          </button>
        </div>

        <div ref={headerRef}>
          <Header user={user} onClearChat={handleClearChat} />
        </div>

        <main ref={mainRef} className="flex-1 overflow-y-auto mt-14">
          {children}
        </main>
      </div>
    </div>
  );
};

export default AppLayout;

// ── Sidebar content ────────────────────────────────────────────────
interface SidebarContentProps {
  user: User | null;
  hovered: boolean;
  sessions: ChatSession[];
  activeSessionId: string | null;
  search: string;
  setSearch: (v: string) => void;
  searchOpen: boolean;
  setSearchOpen: (v: boolean) => void;
  onNewChat: () => void;
  onSelectSession: (id: string) => void;
  onDeleteSession: (id: string) => void;
  uploadedFileName?: string;
}

const SidebarContent: React.FC<SidebarContentProps> = ({
  user,
  hovered,
  sessions,
  activeSessionId,
  search,
  setSearch,
  searchOpen,
  setSearchOpen,
  onNewChat,
  onSelectSession,
  onDeleteSession,
  uploadedFileName,
}) => {
  const labelClass = clsx(
    'text-sm whitespace-nowrap ml-3 transition-all duration-300 font-light',
    hovered ? 'opacity-100 translate-x-0 max-w-[200px]' : 'opacity-0 -translate-x-2 max-w-0 overflow-hidden'
  );

  return (
    <div className="flex flex-col flex-1 overflow-hidden">
      {/* Actions */}
      <div className="p-2 space-y-1">
        {/* New Chat */}
        <button
          onClick={onNewChat}
          className="flex items-center w-full px-3 py-2.5 rounded-xl transition-all duration-200 text-white/70 hover:text-white hover:bg-white/8"
          title="New Chat"
        >
          <Plus size={18} className="flex-shrink-0 text-sky-400" />
          <span className={labelClass}>New Chat</span>
        </button>

        {/* Search */}
        {hovered ? (
          <div className="px-1">
            <div
              className="flex items-center gap-2 px-3 py-2 rounded-xl"
              style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}
            >
              <Search size={14} className="text-white/30 flex-shrink-0" />
              <input
                type="text"
                placeholder="Search chats…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="bg-transparent text-xs text-white/70 outline-none placeholder-white/20 w-full"
              />
            </div>
          </div>
        ) : (
          <button
            className="flex items-center w-full px-3 py-2.5 rounded-xl text-white/40 hover:text-white/70 hover:bg-white/8 transition"
            title="Search"
          >
            <Search size={18} className="flex-shrink-0" />
          </button>
        )}
      </div>

      {/* PDF indicator */}
      {uploadedFileName && hovered && (
        <div className="mx-2 mb-2 px-3 py-2 rounded-xl" style={{ background: 'rgba(56,189,248,0.06)', border: '1px solid rgba(56,189,248,0.15)' }}>
          <div className="flex items-center gap-2">
            <FileText size={13} className="text-sky-400 flex-shrink-0" />
            <span className="text-xs text-sky-300/80 truncate font-light">{uploadedFileName}</span>
          </div>
        </div>
      )}
      {uploadedFileName && !hovered && (
        <div className="flex justify-center my-1">
          <div className="w-8 h-8 rounded-xl flex items-center justify-center" style={{ background: 'rgba(56,189,248,0.08)' }}>
            <FileText size={14} className="text-sky-400" />
          </div>
        </div>
      )}

      {/* Divider */}
      {hovered && sessions.length > 0 && (
        <div className="mx-3 mb-2">
          <p className="text-[10px] text-white/20 uppercase tracking-wider font-medium mb-1">Recent</p>
          <div className="h-px bg-white/6" />
        </div>
      )}

      {/* Chat history */}
      <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-0.5">
        {sessions.map((session) => {
          const isActive = session.id === activeSessionId;
          return (
            <div
              key={session.id}
              className="group relative"
            >
              <button
                onClick={() => onSelectSession(session.id)}
                className={clsx(
                  'flex items-center w-full px-2 py-2 rounded-xl transition-all duration-200 text-left',
                  isActive
                    ? 'bg-white/10 text-white'
                    : 'text-white/40 hover:text-white/70 hover:bg-white/6'
                )}
                title={session.title}
              >
                <MessageSquare size={15} className={clsx('flex-shrink-0', isActive ? 'text-sky-400' : 'text-white/30')} />
                {hovered && (
                  <span className="ml-2.5 text-xs truncate max-w-[140px] font-light">
                    {session.title}
                  </span>
                )}
              </button>

              {/* Delete btn */}
              {hovered && !isActive && (
                <button
                  onClick={(e) => { e.stopPropagation(); onDeleteSession(session.id); }}
                  className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 p-1 rounded-lg text-white/20 hover:text-red-400 hover:bg-red-500/10 transition"
                >
                  <Trash2 size={12} />
                </button>
              )}
            </div>
          );
        })}
      </div>

      {/* User profile */}
      {user && (
        <div className="p-3 border-t border-white/6">
          <div className="flex items-center gap-2 px-2 py-2 rounded-xl">
            {user.picture ? (
              <Image
                src={user.picture}
                alt={user.name}
                width={32}
                height={32}
                className="rounded-full flex-shrink-0"
              />
            ) : (
              <div
                className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium text-white"
                style={{ background: 'linear-gradient(135deg, #0ea5e9, #2563eb)' }}
              >
                {user.name.split(' ').map((n) => n[0]).join('').slice(0, 2).toUpperCase()}
              </div>
            )}
            <div
              className={clsx(
                'transition-all duration-300 overflow-hidden',
                hovered ? 'opacity-100 max-w-[160px]' : 'opacity-0 max-w-0'
              )}
            >
              <p className="text-xs text-white/80 font-medium truncate">{user.name}</p>
              <p className="text-[10px] text-white/30 truncate">{user.email}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
