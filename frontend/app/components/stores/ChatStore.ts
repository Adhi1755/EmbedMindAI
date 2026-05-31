import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface Message {
  id: number;
  sender: 'user' | 'ai';
  text: string;
  timestamp: number;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
  pdfName?: string;
}

interface ChatState {
  sessions: ChatSession[];
  activeSessionId: string | null;

  // Actions
  createSession: (pdfName?: string) => string;
  setActiveSession: (id: string) => void;
  addMessage: (sessionId: string, message: Omit<Message, 'id' | 'timestamp'>) => void;
  clearSession: (sessionId: string) => void;
  updateSessionTitle: (sessionId: string, title: string) => void;
  getActiveSession: () => ChatSession | null;
  deleteSession: (sessionId: string) => void;
}

const generateId = () => Math.random().toString(36).slice(2, 11);

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      sessions: [],
      activeSessionId: null,

      createSession: (pdfName?: string) => {
        const id = generateId();
        const now = Date.now();
        const session: ChatSession = {
          id,
          title: pdfName ? `Chat – ${pdfName}` : `New Chat`,
          messages: [],
          createdAt: now,
          updatedAt: now,
          pdfName,
        };
        set((s) => ({
          sessions: [session, ...s.sessions],
          activeSessionId: id,
        }));
        return id;
      },

      setActiveSession: (id) => set({ activeSessionId: id }),

      addMessage: (sessionId, message) => {
        const newMsg: Message = {
          ...message,
          id: Date.now(),
          timestamp: Date.now(),
        };
        set((s) => ({
          sessions: s.sessions.map((sess) => {
            if (sess.id !== sessionId) return sess;
            const updatedMessages = [...sess.messages, newMsg];
            // Auto-title from first user message
            const title =
              sess.messages.length === 0 && message.sender === 'user'
                ? message.text.slice(0, 50) + (message.text.length > 50 ? '…' : '')
                : sess.title;
            return { ...sess, messages: updatedMessages, title, updatedAt: Date.now() };
          }),
        }));
      },

      clearSession: (sessionId) => {
        set((s) => ({
          sessions: s.sessions.map((sess) =>
            sess.id === sessionId
              ? { ...sess, messages: [], title: 'New Chat', updatedAt: Date.now() }
              : sess
          ),
        }));
      },

      updateSessionTitle: (sessionId, title) => {
        set((s) => ({
          sessions: s.sessions.map((sess) =>
            sess.id === sessionId ? { ...sess, title, updatedAt: Date.now() } : sess
          ),
        }));
      },

      getActiveSession: () => {
        const { sessions, activeSessionId } = get();
        return sessions.find((s) => s.id === activeSessionId) ?? null;
      },

      deleteSession: (sessionId) => {
        set((s) => {
          const filtered = s.sessions.filter((sess) => sess.id !== sessionId);
          const newActive =
            s.activeSessionId === sessionId
              ? filtered[0]?.id ?? null
              : s.activeSessionId;
          return { sessions: filtered, activeSessionId: newActive };
        });
      },
    }),
    {
      name: 'embedmindai-chat-sessions',
      partialize: (s) => ({ sessions: s.sessions, activeSessionId: s.activeSessionId }),
    }
  )
);
