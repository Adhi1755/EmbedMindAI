"use client";
import React, { useRef, useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useDropzone } from "react-dropzone";
import { IconUpload, IconCheck } from "@tabler/icons-react";
import { cn } from "@/lib/utils";
import { uploadPDF } from '../lib/api';
import { useUploadStore } from "../components/stores/UploadStore";
import { useAnimationControls } from "framer-motion";

const FileUpload = ({ onChange }: { onChange?: (files: File[]) => void }) => {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [messages, setMessages] = useState<string[]>([]);
  const [isDone, setIsDone] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { getRootProps, getInputProps, isDragActive } = useDropzone();
  const controls = useAnimationControls();

  const { setUploadedFile, setProcessing } = useUploadStore();

  // WebSocket connection for progress messages
  useEffect(() => {
    if (!files.length) return;

    const socket = new WebSocket("ws://localhost:8000/ws/progress");

    socket.onmessage = (event) => {
      const msg = event.data;
      setMessages((prev) => [...prev, msg]);

      if (msg.toLowerCase().includes("setup complete")) {
        setTimeout(() => {
          setIsDone(true);
        }, 1000);
      }
    };

    return () => socket.close();
  }, [files]);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setFiles([file]);
    setMessages([]);
    setIsDone(false);

    setUploadedFile(file);
    setProcessing(true);

    onChange && onChange([file]);

    const formData = new FormData();
    formData.append("file", file);

    try {
      setUploading(true);
      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });

      const result = await res.json();
      if (result.status === "failed") {
        alert(result.error || "Upload failed");
      }
    } catch (err) {
      console.error("Upload failed:", err);
      alert("An error occurred during upload.");
    } finally {
      setUploading(false);
      setProcessing(false);
    }
  };

  const handleClick = () => {
    if (files.length) {
      setFiles([]);
      setUploadedFile(null);
    } else {
      fileInputRef.current?.click();
    }
  };


  const mainVariant = {
  initial: {
    x: 0,
    y: 0,
  },
  animate: {
    x: 20,
    y: -20,
    opacity: 0.9,
  },
};
 
const secondaryVariant = {
  initial: {
    opacity: 0,
  },
  animate: {
    opacity: 1,
  },
};
 

  return (
    <div className="max-w-4xl " {...getRootProps()}>
      <motion.div
        onClick={handleClick}
        onHoverStart={() => controls.start("animate")}
        onHoverEnd={() => controls.start("initial")}
        className="group/file block rounded-lg cursor-pointer w-full relative overflow-hidden"
      >
        <input
          type="file"
          id="file-upload-handle"
          accept="application/pdf"
          ref={fileInputRef}
          onChange={handleFileChange}
          className="hidden"
        />
         

        <div className="absolute inset-0 [mask-image:radial-gradient(ellipse_at_center,white,transparent)]"></div>

        <div className="flex flex-col items-center justify-center">
          {/* Headings (only before file upload) */}
          <AnimatePresence>
            {!files.length && (
              <>
                <motion.p
                  key="subheading"
                  initial={{ opacity: 1 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ duration: 0.3 }}
                  className="relative z-20 font-sans font-normal text-neutral-400 dark:text-neutral-400 text-base"
                >
                  Drag & drop your file here or click to upload
                </motion.p>
              </>
            )}
          </AnimatePresence>

          <div className="relative w-full mt-2 max-w-xl mx-auto">
            {/* Uploaded File Display + Progress */}
            {files.length > 0 && (
  <motion.div
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    className={cn(
      "relative overflow-hidden z-40 bg-transparent flex flex-col items-start justify-start p-4 mt-4 mx-auto rounded-md",
      "w-full max-w-xl h-[250px]" // ✅ Responsive width: full up to max width
    )}
  >
    {isDone ? (
      <motion.div
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        className="w-full text-center flex flex-col items-center justify-center h-full"
      >
        <IconCheck className="text-sky-400 w-8 h-8 mb-2" />
        <p className="text-white font-semibold text-lg">Done</p>
        <p className="text-sky-400 font-extralight">Ask Question from PDF</p>
      </motion.div>
    ) : (
      <>
        {/* File Info */}
        <div className="flex justify-center items-center w-full gap-4 mb-2">
          <p className="text-white text-sm truncate max-w-[70%]">
            {files[0].name}
          </p>
        </div>

        {/* Scrollable Progress Display */}
        <div className="w-full flex-1 overflow-y-auto">
  <ul className="space-y-2 w-full pr-1">
    <AnimatePresence>
      {messages.map((msg, idx) => (
        <motion.li
          key={msg + idx}
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          transition={{
            duration: 0.3,
            delay: idx * 0.2, // ⏱️ Add delay between messages
          }}
          className="text-sm text-sky-400 flex items-center gap-2"
        >
          <IconCheck className="w-4 h-4 text-sky-400 shrink-0" />
          {msg}
        </motion.li>
      ))}
    </AnimatePresence>
  </ul>
</div>
      </>
    )}
  </motion.div>
)}



            {!files.length && (
            <div className="relative h-32 mt-4 w-full max-w-[8rem] mx-auto">
              {/* Secondary (dotted) border */}
              <motion.div
              variants={secondaryVariant}
              initial="initial"
              animate={controls}
              transition={{ duration: 0.4 }}
              className="absolute border border-dashed border-sky-400 inset-0 z-30 bg-transparent rounded-md"
              />

    {/* Upload box with mainVariant animation on hover */}
        <motion.div
          layoutId="file-upload"
          variants={mainVariant}
          initial="initial"
          animate={controls}
          transition={{stiffness: 300, damping: 20 }}
          className={cn(
            "relative z-40 bg-neutral-900 flex flex-col items-center justify-center h-full w-full rounded-md cursor-pointer",
            "shadow-[0px_10px_50px_rgba(0,0,0,0.1)]"
          )}
        >
          <IconUpload className="h-6 w-6 text-neutral-600 dark:text-neutral-300" />
            <p className="text-xs text-neutral-500 mt-2">Upload</p>
        </motion.div>
      </div>
      )}

          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default FileUpload;
