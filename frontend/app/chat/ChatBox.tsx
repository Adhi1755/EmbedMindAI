'use client';
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { toast } from 'sonner';
import { ArrowUp, Plus, X, Paperclip } from 'lucide-react';
import { uploadPDF } from '../lib/api';
import { useUploadStore } from '../components/stores/UploadStore';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const MAX_CHARS = 2000;

interface ChatInputProps {
  onSendMessage: (text: string) => void;
}

const ChatInputBox: React.FC<ChatInputProps> = ({ onSendMessage }) => {
  const [inputValue, setInputValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { uploadedFile: pdfFile, setUploadedFile: setPdfFile, isProcessing, setProcessing: setIsProcessing } = useUploadStore();

  const canSend = inputValue.trim().length > 0 && !isProcessing;
  const charCount = inputValue.length;
  const overLimit = charCount > MAX_CHARS;

  // Auto-grow textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
  }, [inputValue]);

  const deletePDF = async (filename: string) => {
    try {
      const res = await fetch(`${API_URL}/delete?filename=${encodeURIComponent(filename)}`, {
        method: 'DELETE',
      });
      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || 'Failed to delete');
      }
      toast.success('File removed', { description: `${filename} was successfully removed.` });
    } catch (err) {
      toast.error('Failed to remove file', { description: (err as Error).message });
    }
  };

  const handleSendClick = useCallback(() => {
    if (!canSend || overLimit) return;
    onSendMessage(inputValue);
    setInputValue('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }, [canSend, overLimit, inputValue, onSendMessage]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendClick();
    }
  };

  const handleButtonClick = async () => {
    if (pdfFile) {
      await deletePDF(pdfFile.name);
      setPdfFile(null);
    } else {
      fileInputRef.current?.click();
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    toast.info('Uploading PDF…', { description: file.name });

    setPdfFile(file);
    setIsProcessing(true);

    try {
      const message = await uploadPDF(file);
      console.log('Upload status:', message);
      toast.success('Upload complete 🎉', {
        description: `${file.name} processed successfully!`,
        duration: 4000,
      });
    } catch {
      toast.error('Upload failed ❌', { description: 'There was an error uploading your file.' });
      setPdfFile(null);
    } finally {
      setIsProcessing(false);
      // Reset so same file can be re-uploaded
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 px-4 pb-5 pt-3"
      style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.95) 70%, transparent)' }}
    >
      <div className="max-w-3xl mx-auto">
        {/* PDF File Pill */}
        {pdfFile && (
          <div className="flex items-center gap-2 mb-2 ml-1">
            <div
              className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs text-sky-300"
              style={{
                background: 'rgba(56,189,248,0.1)',
                border: '1px solid rgba(56,189,248,0.25)',
              }}
            >
              <Paperclip size={12} />
              <span className="truncate max-w-[200px]">{pdfFile.name}</span>
              {isProcessing && (
                <svg className="animate-spin w-3 h-3" viewBox="0 0 50 50">
                  <circle cx="25" cy="25" r="20" fill="none" stroke="#38bdf8" strokeWidth="5"
                    strokeLinecap="round" strokeDasharray="90" strokeDashoffset="60" />
                </svg>
              )}
              {!isProcessing && (
                <button
                  onClick={handleButtonClick}
                  className="text-sky-400/70 hover:text-sky-300 transition ml-0.5"
                >
                  <X size={12} />
                </button>
              )}
            </div>
          </div>
        )}

        {/* Main input container */}
        <div
          className="flex items-end gap-2 rounded-2xl p-2"
          style={{
            background: 'rgba(255,255,255,0.04)',
            border: overLimit
              ? '1px solid rgba(239,68,68,0.5)'
              : '1px solid rgba(255,255,255,0.12)',
            boxShadow: '0 4px 32px rgba(0,0,0,0.4)',
            backdropFilter: 'blur(20px)',
            transition: 'border-color 0.2s ease',
          }}
          onFocusCapture={(e) => {
            if (!overLimit) {
              (e.currentTarget as HTMLElement).style.borderColor = 'rgba(56,189,248,0.4)';
              (e.currentTarget as HTMLElement).style.boxShadow = '0 4px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(56,189,248,0.15)';
            }
          }}
          onBlurCapture={(e) => {
            (e.currentTarget as HTMLElement).style.borderColor = overLimit
              ? 'rgba(239,68,68,0.5)' : 'rgba(255,255,255,0.12)';
            (e.currentTarget as HTMLElement).style.boxShadow = '0 4px 32px rgba(0,0,0,0.4)';
          }}
        >
          {/* Upload Button */}
          <input
            type="file"
            accept="application/pdf"
            ref={fileInputRef}
            onChange={handleFileChange}
            className="hidden"
          />
          <button
            onClick={handleButtonClick}
            disabled={isProcessing}
            title={pdfFile ? 'Remove PDF' : 'Attach PDF'}
            className="flex-shrink-0 w-9 h-9 flex items-center justify-center rounded-xl transition-all duration-200 text-white/40 hover:text-white/80 hover:bg-white/10 disabled:opacity-40 disabled:cursor-wait mb-0.5"
          >
            {isProcessing ? (
              <svg className="animate-spin w-4 h-4" viewBox="0 0 50 50">
                <circle cx="25" cy="25" r="20" fill="none" stroke="#38bdf8" strokeWidth="5"
                  strokeLinecap="round" strokeDasharray="90" strokeDashoffset="60" />
              </svg>
            ) : pdfFile ? (
              <X size={16} className="text-sky-400" />
            ) : (
              <Plus size={18} />
            )}
          </button>

          {/* Textarea */}
          <textarea
            ref={textareaRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything about your PDF… (Shift+Enter for new line)"
            rows={1}
            className="flex-1 bg-transparent outline-none text-white text-sm leading-relaxed placeholder-white/25 resize-none py-2 px-1 min-h-[36px]"
            style={{ maxHeight: '160px', scrollbarWidth: 'none' }}
          />

          {/* Right side: char count + send */}
          <div className="flex flex-col items-end gap-1.5 flex-shrink-0 mb-0.5">
            {/* Char count */}
            {charCount > MAX_CHARS * 0.7 && (
              <span className={`text-[10px] font-light ${overLimit ? 'text-red-400' : 'text-white/25'}`}>
                {charCount}/{MAX_CHARS}
              </span>
            )}

            {/* Send button */}
            <button
              onClick={handleSendClick}
              disabled={!canSend || overLimit}
              id="chat-send-btn"
              className="w-9 h-9 flex items-center justify-center rounded-xl transition-all duration-200 disabled:opacity-30 disabled:cursor-not-allowed"
              style={
                canSend && !overLimit
                  ? {
                      background: 'linear-gradient(135deg, #0ea5e9, #2563eb)',
                      boxShadow: '0 2px 12px rgba(14,165,233,0.5)',
                    }
                  : { background: 'rgba(255,255,255,0.08)' }
              }
              title="Send (Enter)"
            >
              <ArrowUp size={16} className="text-white" />
            </button>
          </div>
        </div>

        {/* Footer hint */}
        <p className="text-center text-[10px] text-white/15 mt-2 font-light">
          EmbedMindAI can make mistakes. Verify important information.
        </p>
      </div>
    </div>
  );
};

export default ChatInputBox;