import { create } from 'zustand';
import { persist } from 'zustand/middleware';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

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
  synced?: boolean;   // true once persisted to MongoDB
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
  hydrateFromBackend: (sessions: ChatSession[]) => void;
  loadMessagesForSession: (sessionId: string) => Promise<void>;
}

const generateId = () => Math.random().toString(36).slice(2, 11);

// ── Backend sync helpers ───────────────────────────────────────────────────

async function syncSessionToBackend(
  session_id: string,
  title: string,
  pdf_name?: string
) {
  try {
    await fetch(`${API_URL}/chat/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ session_id, title, pdf_name }),
    });
  } catch {
    // silent — local store always wins
  }
}

async function deleteSessionFromBackend(session_id: string) {
  try {
    await fetch(`${API_URL}/chat/sessions/${session_id}`, {
      method: 'DELETE',
      credentials: 'include',
    });
  } catch {
    // silent
  }
}

async function updateSessionTitleOnBackend(session_id: string, title: string) {
  try {
    await fetch(`${API_URL}/chat/sessions/${session_id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ title }),
    });
  } catch {
    // silent
  }
}

export async function saveMessageToBackend(
  session_id: string,
  sender: 'user' | 'ai',
  text: string
) {
  try {
    await fetch(`${API_URL}/chat/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ session_id, sender, text }),
    });
  } catch {
    // silent
  }
}

// ── Store ──────────────────────────────────────────────────────────────────

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      sessions: [],
      activeSessionId: null,

      createSession: (pdfName?: string) => {
        const id = generateId();
        const now = Date.now();
        const title = pdfName ? `Chat – ${pdfName}` : 'New Chat';
        const session: ChatSession = {
          id,
          title,
          messages: [],
          createdAt: now,
          updatedAt: now,
          pdfName,
          synced: false,
        };
        set((s) => ({
          sessions: [session, ...s.sessions],
          activeSessionId: id,
        }));

        // Persist to backend (fire-and-forget)
        syncSessionToBackend(id, title, pdfName);

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
            // Auto-title: use first user message text
            const isFirst = sess.messages.length === 0 && message.sender === 'user';
            const title = isFirst
              ? message.text.slice(0, 50) + (message.text.length > 50 ? '…' : '')
              : sess.title;

            // Update title on backend if it changed
            if (isFirst) updateSessionTitleOnBackend(sessionId, title);

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
        updateSessionTitleOnBackend(sessionId, title);
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
        deleteSessionFromBackend(sessionId);
      },

      /**
       * Replace the local session list with data fetched from the backend.
       * Called once on mount by AppStructure.
       */
      hydrateFromBackend: (backendSessions: ChatSession[]) => {
        set((s) => {
          // Merge: backend is source of truth for titles/timestamps.
          // Keep local messages for sessions that haven't been loaded yet.
          const merged = backendSessions.map((bs) => {
            const local = s.sessions.find((ls) => ls.id === bs.id);
            return local ? { ...local, title: bs.title, updatedAt: bs.updatedAt } : bs;
          });
          // Keep any local-only sessions that haven't synced yet
          const localOnly = s.sessions.filter(
            (ls) => !backendSessions.find((bs) => bs.id === ls.id)
          );
          return { sessions: [...merged, ...localOnly] };
        });
      },

      /**
       * Fetch and inject messages for a session from the backend.
       * Called when the user switches to a session that has no local messages.
       */
      loadMessagesForSession: async (sessionId: string) => {
        try {
          const res = await fetch(
            `${API_URL}/chat/sessions/${sessionId}/messages`,
            { credentials: 'include' }
          );
          if (!res.ok) return;
          const msgs: Array<{ sender: string; text: string; timestamp: string }> =
            await res.json();

          const mapped: Message[] = msgs.map((m, i) => ({
            id: i,
            sender: m.sender as 'user' | 'ai',
            text: m.text,
            timestamp: new Date(m.timestamp).getTime(),
          }));

          set((s) => ({
            sessions: s.sessions.map((sess) =>
              sess.id === sessionId ? { ...sess, messages: mapped } : sess
            ),
          }));
        } catch {
          // silent
        }
      },
    }),
    {
      name: 'embedmindai-chat-sessions',
      partialize: (s) => ({ sessions: s.sessions, activeSessionId: s.activeSessionId }),
    }
  )
);
