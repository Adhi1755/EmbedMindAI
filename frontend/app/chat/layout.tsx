'use client';
import React from 'react';
import AppLayout from './AppStructure';

export default function ChatLayout({ children }: { children: React.ReactNode }) {
  return <AppLayout>{children}</AppLayout>;
}
