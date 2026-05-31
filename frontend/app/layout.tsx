import type { Metadata } from "next";
import "./globals.css";
import localFont from 'next/font/local'

const outfit = localFont({
  src: [
    { path: './fonts/Outfit/Outfit-Thin.ttf', weight: '100', style: 'normal' },
    { path: './fonts/Outfit/Outfit-ExtraLight.ttf', weight: '200', style: 'normal' },
    { path: './fonts/Outfit/Outfit-Light.ttf', weight: '300', style: 'normal' },
    { path: './fonts/Outfit/Outfit-Regular.ttf', weight: '400', style: 'normal' },
    { path: './fonts/Outfit/Outfit-Medium.ttf', weight: '500', style: 'normal' },
    { path: './fonts/Outfit/Outfit-SemiBold.ttf', weight: '600', style: 'normal' },
    { path: './fonts/Outfit/Outfit-Bold.ttf', weight: '700', style: 'normal' },
    { path: './fonts/Outfit/Outfit-ExtraBold.ttf', weight: '800', style: 'normal' },
    { path: './fonts/Outfit/Outfit-Black.ttf', weight: '900', style: 'normal' },
  ],
  variable: '--font-outfit',
  display: 'swap',
});


export const metadata: Metadata = {
  title: "EmbedMindAI – AI-Powered Document Intelligence",
  description: "Upload any PDF and instantly get AI-powered answers using Retrieval-Augmented Generation. Built with Sentence Transformers, ChromaDB, and Google Gemini.",
  keywords: "PDF AI, RAG, retrieval augmented generation, document intelligence, AI chat",
  openGraph: {
    title: "EmbedMindAI – AI-Powered Document Intelligence",
    description: "Upload PDFs and chat with your documents using advanced RAG technology.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
      className={`${outfit.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
