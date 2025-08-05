'use client';

import React, { useEffect, useState, useLayoutEffect, useRef } from 'react';
import { Menu, X, Plus, Search, UserCircle2 } from 'lucide-react';
import Image from 'next/image';
import clsx from 'clsx';
import gsap from 'gsap';
import Header from './Header';

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

  const sidebarRef = useRef(null);
  const headerRef = useRef(null);
  const mainRef = useRef(null);

  useEffect(() => {
    fetch('http://localhost:8000/auth/me', { credentials: 'include' })
      .then((res) => res.json())
      .then((data) => {
        if (data.error) {
          window.location.href = '/login';
        } else {
          const userInfo = data.token || data;
          setUser(userInfo);
          setTimeout(() => setLoginLoading(false), 500);
        }
      });
  }, []);

  useLayoutEffect(() => {
    if (!loginLoading) {
      const ctx = gsap.context(() => {
        gsap.fromTo(
          sidebarRef.current,
          { opacity: 0, x: -30 },
          { opacity: 1, x: 0, duration: 1, ease: 'power2.out' }
        );
        gsap.fromTo(
          headerRef.current,
          { opacity: 0, y: -20 },
          { opacity: 1, y: 0, duration: 1, delay: 0.2, ease: 'power2.out' }
        );
        gsap.fromTo(
          mainRef.current,
          { opacity: 0 },
          { opacity: 1, duration: 1, delay: 0.4, ease: 'power2.out' }
        );
      });
      return () => ctx.revert();
    }
  }, [loginLoading]);

  if (loginLoading) {
    return (
      <div className="flex items-center justify-center h-screen w-screen bg-black text-white text-xl">
        <div className="animate-pulse">Loading your workspace...</div>
      </div>
    );
  }

  return (
    <div className="flex h-screen">
      {/* Mobile Toggle */}
      <div className="lg:hidden fixed top-4 right-4 z-50">
        <button
          onClick={() => setIsMobileOpen(true)}
          className="p-2 rounded-md text-white shadow"
        >
          <Menu size={20} />
        </button>
      </div>

      {/* Mobile Sidebar Drawer */}
      {isMobileOpen && (
        <div className="lg:hidden fixed inset-0 z-50 bg-black/60">
          <aside className="w-64 h-full bg-zinc-900 border-r border-zinc-800 p-3 flex flex-col">
            <div className="flex justify-end">
              <button
                onClick={() => setIsMobileOpen(false)}
                className="p-2 text-white"
              >
                <X size={20} />
              </button>
            </div>
            <SidebarContent user={user} hovered />
          </aside>
        </div>
      )}

      {/* Desktop Sidebar */}
      <aside
        ref={sidebarRef}
        className={clsx(
          'hidden lg:flex flex-col h-full bg-neutral-900 text-white border-r border-zinc-800 transition-all duration-500 fixed top-0 left-0 z-40',
          hovered ? 'w-64' : 'w-16'
        )}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        <SidebarContent user={user} hovered={hovered} />
      </aside>

      {/* Main content area */}
      <div className="flex flex-col flex-1 overflow-hidden bg-zinc-950 text-white">
        <div
          className={clsx(
            'flex flex-col transition-all duration-500 ml-0',
            'lg:ml-12',
            hovered && 'lg:ml-60'
          )}
        >
          <Header />
        </div>
        <main ref={mainRef} className="flex-1 overflow-y-auto">{children}</main>
      </div>
    </div>
  );
};

export default AppLayout;

// Sidebar content
const SidebarContent: React.FC<{ user: User | null; hovered: boolean }> = ({ user, hovered }) => (
  <>
    <div className="space-y-2 p-2">
      <SidebarButton icon={<Plus size={20} />} label="New Chat" hovered={hovered} />
      <SidebarButton icon={<Search size={20} />} label="Search" hovered={hovered} />
    </div>

    {user && (
      <div className="p-3 mt-auto">
        <div className="flex items-center gap-3 bg-transparent rounded-full overflow-hidden transition-all duration-300">
          {user.picture ? (
            <Image
              src={user.picture}
              alt="User"
              width={36}
              height={36}
              className="rounded-full"
            />
          ) : (
            <UserCircle2 size={36} />
          )}
          <div
            className={clsx(
              'transition-all duration-300',
              hovered
                ? 'opacity-100 translate-x-0 max-w-[200px]'
                : 'opacity-0 -translate-x-2 max-w-0 overflow-hidden'
            )}
          >
            <div className="flex flex-col">
              <span className="text-sm font-medium truncate">{user.name}</span>
              <span className="text-xs text-zinc-400 truncate">{user.email}</span>
            </div>
          </div>
        </div>
      </div>
    )}
  </>
);

const SidebarButton: React.FC<{ icon: React.ReactNode; label: string; hovered: boolean }> = ({
  icon,
  label,
  hovered,
}) => (
  <button className="flex items-center w-full px-3 py-3 rounded-md bg-transparent hover:bg-zinc-700 transition-all duration-300">
    <span className="text-white">{icon}</span>
    <span
      className={clsx(
        'text-sm whitespace-nowrap ml-3 transition-all duration-300',
        hovered
          ? 'opacity-100 translate-x-0'
          : 'opacity-0 -translate-x-2 pointer-events-none'
      )}
    >
      {label}
    </span>
  </button>
);
