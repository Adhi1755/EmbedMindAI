'use client';
import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { LogOut, Trash2, ChevronDown } from 'lucide-react';
import Image from 'next/image';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface HeaderProps {
  user?: { name: string; email: string; picture?: string } | null;
  onClearChat?: () => void;
}

const Header: React.FC<HeaderProps> = ({ user, onClearChat }) => {
  const [menuOpen, setMenuOpen] = useState(false);
  const router = useRouter();

  const handleLogout = async () => {
    try {
      await fetch(`${API_URL}/auth/logout`, { method: 'GET', credentials: 'include' });
    } catch { /* ignore */ }
    window.location.href = '/';
  };

  const initials = user?.name
    ? user.name.split(' ').map((n) => n[0]).join('').slice(0, 2).toUpperCase()
    : 'U';

  return (
    <header
      className="fixed top-0 right-0 left-0 z-30 flex items-center justify-between px-5 py-3"
      style={{
        background: 'rgba(0,0,0,0.85)',
        backdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(255,255,255,0.07)',
        // On desktop, account for the sidebar (handled by parent AppLayout margin)
      }}
    >
      {/* Logo */}
      <span className="text-lg font-light bg-gradient-to-r from-sky-400 to-blue-600 bg-clip-text text-transparent select-none">
        EmbedMindAI
      </span>

      {/* Right side */}
      <div className="flex items-center gap-2">
        {/* Clear Chat */}
        {onClearChat && (
          <button
            onClick={onClearChat}
            className="hidden md:flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-white/40 hover:text-white/80 hover:bg-white/8 transition-all duration-200"
            style={{ border: '1px solid rgba(255,255,255,0.06)' }}
            title="Clear current chat"
          >
            <Trash2 size={13} />
            Clear Chat
          </button>
        )}

        {/* User dropdown */}
        {user ? (
          <div className="relative">
            <button
              onClick={() => setMenuOpen(!menuOpen)}
              className="flex items-center gap-2 pl-2 pr-3 py-1.5 rounded-full transition-all duration-200 hover:bg-white/8"
              style={{ border: '1px solid rgba(255,255,255,0.1)' }}
            >
              {user.picture ? (
                <Image
                  src={user.picture}
                  alt={user.name}
                  width={28}
                  height={28}
                  className="rounded-full"
                />
              ) : (
                <div
                  className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-medium text-white"
                  style={{ background: 'linear-gradient(135deg, #0ea5e9, #2563eb)' }}
                >
                  {initials}
                </div>
              )}
              <span className="hidden md:block text-sm text-white/80 font-light">
                {user.name?.split(' ')[0]}
              </span>
              <ChevronDown
                size={14}
                className={`text-white/30 transition-transform duration-200 ${menuOpen ? 'rotate-180' : ''}`}
              />
            </button>

            {/* Dropdown */}
            {menuOpen && (
              <div
                className="absolute right-0 top-full mt-2 w-52 rounded-xl overflow-hidden shadow-2xl z-50"
                style={{
                  background: 'rgba(15,15,20,0.98)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  backdropFilter: 'blur(20px)',
                }}
              >
                {/* User info */}
                <div className="px-4 py-3 border-b border-white/8">
                  <p className="text-sm text-white font-medium truncate">{user.name}</p>
                  <p className="text-xs text-white/40 truncate">{user.email}</p>
                </div>

                {/* Actions */}
                {onClearChat && (
                  <button
                    onClick={() => { onClearChat(); setMenuOpen(false); }}
                    className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-white/60 hover:text-white hover:bg-white/6 transition"
                  >
                    <Trash2 size={15} />
                    Clear Chat
                  </button>
                )}

                <button
                  onClick={handleLogout}
                  className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-red-400 hover:text-red-300 hover:bg-red-500/8 transition"
                >
                  <LogOut size={15} />
                  Sign Out
                </button>
              </div>
            )}
          </div>
        ) : (
          <button
            onClick={handleLogout}
            className="flex items-center gap-1.5 px-4 py-2 text-sm font-light text-white/60 hover:text-white rounded-full hover:bg-white/10 transition"
          >
            <LogOut size={14} />
            Logout
          </button>
        )}
      </div>

      {/* Close dropdown when clicking outside */}
      {menuOpen && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setMenuOpen(false)}
        />
      )}
    </header>
  );
};

export default Header;
