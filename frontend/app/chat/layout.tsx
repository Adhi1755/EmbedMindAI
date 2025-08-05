'use'
import React from 'react';
import AppLayout from './AppStructure';
 // or wherever your layout component is

export const metadata = {
  title: 'EmbedMindAI',
};



export default function ChatLayout({ children }: { children: React.ReactNode }) {
  return <AppLayout>{children}</AppLayout>;
}
