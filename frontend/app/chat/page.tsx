'use client'
import React, { useState, useRef, useEffect, useLayoutEffect } from 'react';
import { gsap } from 'gsap';
import { Toaster } from 'sonner';
import ChatInputBox from './ChatBox';
import { sendMessageToBackend } from "../lib/api"; 
import MarkdownRenderer from '../components/Markdown';
import 'github-markdown-css/github-markdown.css';
import FileUpload from './FileUpload';
import Header from './Header';
import { jwtDecode } from "jwt-decode";




interface Message {
  id: number;
  sender: 'user' | 'ai';
  text: string;
}

const ChatContainer: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isThinking, setIsThinking] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messageRefs = useRef<(HTMLDivElement | null)[]>([]);
  const [user, setUser] = useState<{ name: string; email: string; picture?: string } | null>(null);
 
  const greeting = `Hello ${user?.name?.split(" ")[0] || "EmbedMindAI"}`;


  type GoogleUser = {
  name: string;
  email: string;
  picture: string;
};


useEffect(() => {
  fetch("http://localhost:8000/auth/me", {
    credentials: "include",
  })
    .then(res => res.json())
    .then(data => {
      if (data.error) {
        window.location.href = "/login";
      } else {
        const userInfo = data.token
          ? jwtDecode<GoogleUser>(data.token)
          : data;
        setUser(userInfo);
      }
    });
}, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useLayoutEffect(() => {
    if (messages.length > 0) {
      const lastMessageIndex = messages.length - 1;
      const lastMessageRef = messageRefs.current[lastMessageIndex];
      if (lastMessageRef) {
        gsap.fromTo(
          lastMessageRef,
          { opacity: 0, y: 20 },
          { opacity: 1, y: 0, duration: 0.4, ease: 'power2.out' }
        );
      }
    }
  }, [messages]);

  const handleSendMessage = async (text: string) => {
    if (text.trim() === '') return;

    const newUserMessage: Message = {
      id: Date.now(),
      sender: 'user',
      text: text.trim(),
    };
    setMessages((prev) => [...prev, newUserMessage]);
    setIsThinking(true);

    const aiReply = await sendMessageToBackend(text); 

    const aiResponse: Message = {
      id: Date.now() + 1,
      sender: 'ai',
      text: aiReply || "no ans"
    };
    setMessages((prev) => [...prev, aiResponse]);
    setIsThinking(false);
  };

  return (
    <div className="flex flex-col justify-center items-center h-screen bg-black">
      
      <Toaster
        position="bottom-center"
        toastOptions={{
          style: {
            marginBottom: "80px", 
            borderRadius: "12px",
            fontSize: "14px",
            boxShadow: "0 6px 20px rgba(0, 0, 0, 0.3)",
          },
        }}
      />

      <div className="relative flex-grow max-h-screen overflow-y-auto w-screen px-4 mb-10">
        {messages.length === 0 && !isThinking && (
          <div className="fixed inset-0 flex justify-center flex-col items-center z-10">
              <h1 className="text-4xl font-light sm:text-5xl md:text-6xl text-center leading-tight 
                bg-gradient-to-r from-blue-300 via-sky-400 to-sky-600 
                bg-clip-text text-transparent animate-gradient-curl">
                {greeting}
              </h1>
            <FileUpload/>
          </div>
        )}

        <div className="max-w-4xl mx-auto  space-y-10 relative z-0 pt-20">
          {messages.map((message, index) => (
            <div
              key={message.id}
              ref={(el) => { messageRefs.current[index] = el; }}
              className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`${
                message.sender === 'user'
                  ? 'bg-[#303030] text-white px-4 py-2 max-w-[80%] rounded-3xl shadow-sm'
                  : ' max-w-[100%]'
              }`}>
                <div className="text-sm sm:text-base md:text-lg lg:text-lg leading-relaxed">
                  {message.sender === 'ai' ? (
                    <MarkdownRenderer content={message.text} typingSpeed={10} />
                  ) : (
                    <span className="text-sm sm:text-base md:text-md">
                      {message.text}
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))}

          {isThinking && (
            <div className="flex justify-start">
              <div className="max-w-[70%] shadow-md bg-transparent animate-pulse">
                Retrieving...
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>
      <ChatInputBox onSendMessage={handleSendMessage} />
    </div>
  );
};

export default ChatContainer;
