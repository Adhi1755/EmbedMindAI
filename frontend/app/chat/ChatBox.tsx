'use client'
import React from 'react';
import { toast } from 'sonner';
import { useState , useRef } from 'react';
import { ArrowRight,  Plus, X } from 'lucide-react';
import { uploadPDF } from '../lib/api';
import { useUploadStore } from '../components/stores/UploadStore';

interface ChatInputProps {
  onSendMessage: (text: string) => void;
}
const ChatInputBox:  React.FC<ChatInputProps> = ({ onSendMessage }) => {
    const [inputValue, setInputValue] = useState<string>('');
    const [showUpload, setShowUpload] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const { uploadedFile: pdfFile, setUploadedFile: setPdfFile, isProcessing, setProcessing: setIsProcessing } = useUploadStore();
  
    const deletePDF = async (filename: string): Promise<void> => {
  try {
    const res = await fetch(`http://127.0.0.1:8000/delete?filename=${encodeURIComponent(filename)}`, {
      method: "DELETE",
    });

    if (!res.ok) {
      const error = await res.json();
      throw new Error(error.detail || "Failed to delete");
    }

    console.log("File deleted from backend");

    // ✅ Show success toast
    toast.success("File deleted 🗑️", {
      description: `${filename} was successfully removed.`,
    });

  } catch (err) {
    console.error("Delete error:", err);

    // ❌ Show error toast
    toast.error("Failed to delete file", {
      description: (err as Error).message,
    });
  }
};

    const handleSendClick = () => {
    if (inputValue.trim() !== '') {
      onSendMessage(inputValue);
      setInputValue(''); // Clear input after sending
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && inputValue.trim() !== '') {
      handleSendClick();
    }
  };
  const handleButtonClick = async () => {
    if (pdfFile) {
      await deletePDF(pdfFile.name);
      setPdfFile(null); // Delete
    } else {
      fileInputRef.current?.click(); // Open file picker
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    toast.info("Uploading PDF...");

    setPdfFile(file);
    setIsProcessing(true);

    try {
      const message = await uploadPDF(file);
      console.log("Upload status:", message);
      
      toast.success("Upload complete 🎉", {
        description: `${file.name} processed successfully!`,
        duration: 4000,
      });

    } catch (error) {
      toast.error("Upload failed ❌", {
        description: "There was an error uploading your file.",
      });
    } finally {
      setIsProcessing(false);
    }
  };



 return (
   <div className="fixed bottom-0 left-0 right-0 flex justify-center items-center px-2 sm:px-4 md:px-6 lg:px-8 mb-4 sm:mb-6 backdrop-blur-md z-50  md:backdrop-blur-none">
  
    <div className="relative group">
        {/* Tooltip */}
        <div className="absolute bottom-14 bg-black text-gray-500 text-xs px-3 py-1 rounded-md opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-20">
          {pdfFile ? "Delete" : "Upload"}
        </div>

        {/* Hidden File Input */}
         <input
          type="file"
          accept="application/pdf"
          ref={fileInputRef}
          onChange={handleFileChange}
          className="hidden"
        />

        {/* Upload Button */}
        <div
          className={`mr-2 flex justify-center items-center backdrop-blur-md bg-white/2 border border-white/10 rounded-full w-12 h-12 sm:w-12 sm:h-12 md:w-14 md:h-14 cursor-pointer transition-colors duration-200 ${
            isProcessing ? "animate-pulse cursor-wait" : "hover:bg-white/20"
          }`}
          onClick={handleButtonClick}
        >
          {isProcessing ? (
            <svg
    xmlns="http://www.w3.org/2000/svg"
    className="animate-spin"
    width="40"
    height="40"
    viewBox="0 0 50 50"
  >
    <circle
      cx="25"
      cy="25"
      r="20"
      fill="none"
      stroke="#38BDF8"  // Indigo-600 from Tailwind
      strokeWidth="4"
      strokeLinecap="round"
      strokeDasharray="90"
      strokeDashoffset="60"
    />
  </svg>
          ) : pdfFile ? (
            <X className="text-white w-5 h-5" />
          ) : (
            <Plus className="text-white w-5 h-5" />
          )}
        </div>
      </div>

    <div className="flex items-center  w-full max-w-4xl backdrop-blur-md bg-white/2 border border-white/20 shadow-lg rounded-4xl p-1 sm:p-1 md:p-1">
       {/* Input Field */}
       <input
            type="text"
            placeholder="Type here to query your uploaded content"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            className="flex-grow px-4 sm:px-6 md:px-6 py-1 bg-transparent outline-none text-white/95 text-sm sm:text-base md:text-base placeholder-white/40 transition duration-200"
        />


       {/* Send Button */}
       <div className="border border-white/10  rounded-full cursor-pointer hover:bg-white/20 transition-colors duration-200 flex justify-center items-center w-10 h-10 sm:w-11 sm:h-11 md:w-12 md:h-12">
          <button onClick={handleSendClick}
          disabled={inputValue.trim() === ''}
          className={`p-2 rounded-full transition-colors duration-200
            ${inputValue.trim() === '' ? ' text-gray-500 cursor-not-allowed' : '  hover:bg-white/20 cursor-pointer'}
          `}
          >
            <ArrowRight className='text-white' />
          </button>
           
       </div>
     </div>
   </div>
 );
};


export default ChatInputBox;