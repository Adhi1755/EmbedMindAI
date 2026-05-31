"use client";
import React, { useRef, useState, useEffect } from "react";
import { motion, AnimatePresence, useAnimationControls } from "motion/react";
import { useDropzone } from "react-dropzone";
import { IconUpload, IconCheck } from "@tabler/icons-react";
import { cn } from "@/lib/utils";
import { uploadPDF } from '../lib/api';
import { useUploadStore } from "../components/stores/UploadStore";
import { FileText, AlertCircle } from "lucide-react";

const FileUpload = ({ onChange }: { onChange?: (files: File[]) => void }) => {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [messages, setMessages] = useState<string[]>([]);
  const [isDone, setIsDone] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ noClick: true });
  const controls = useAnimationControls();

  const { setUploadedFile, setProcessing } = useUploadStore();

  // WebSocket connection for progress messages
  useEffect(() => {
    if (!files.length) return;

    const wsUrl = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace("http", "ws");
    const socket = new WebSocket(`${wsUrl}/ws/progress`);

    socket.onmessage = (event) => {
      const msg = event.data;
      setMessages((prev) => [...prev, msg]);

      if (msg.toLowerCase().includes("setup complete")) {
        setTimeout(() => {
          setIsDone(true);
        }, 800);
      }
    };

    return () => socket.close();
  }, [files]);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Only PDF files are supported.");
      return;
    }

    setFiles([file]);
    setMessages([]);
    setIsDone(false);
    setUploadError("");
    setUploadedFile(file);
    setProcessing(true);

    onChange && onChange([file]);

    const formData = new FormData();
    formData.append("file", file);

    try {
      setUploading(true);
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${apiUrl}/upload`, {
        method: "POST",
        body: formData,
        credentials: "include",
      });

      const result = await res.json();
      if (result.status === "failed") {
        setUploadError(result.error || "Upload failed");
        setFiles([]);
        setUploadedFile(null);
      }
    } catch (err) {
      console.error("Upload failed:", err);
      setUploadError("Network error during upload. Please try again.");
      setFiles([]);
      setUploadedFile(null);
    } finally {
      setUploading(false);
      setProcessing(false);
    }
  };

  const handleClick = () => {
    if (files.length) {
      setFiles([]);
      setMessages([]);
      setIsDone(false);
      setUploadedFile(null);
    } else {
      fileInputRef.current?.click();
    }
  };

  const mainVariant = {
    initial: { x: 0, y: 0 },
    animate: { x: 16, y: -16, opacity: 0.9 },
  };

  const secondaryVariant = {
    initial: { opacity: 0 },
    animate: { opacity: 1 },
  };

  return (
    <div
      className="w-full max-w-sm"
      {...getRootProps()}
    >
      <motion.div
        onClick={handleClick}
        onHoverStart={() => controls.start("animate")}
        onHoverEnd={() => controls.start("initial")}
        className="block rounded-2xl cursor-pointer w-full relative overflow-hidden"
      >
        <input
          type="file"
          id="file-upload-handle"
          accept="application/pdf"
          ref={fileInputRef}
          onChange={handleFileChange}
          className="hidden"
        />

        <div className="flex flex-col items-center justify-center gap-4">
          {/* Error */}
          <AnimatePresence>
            {uploadError && (
              <motion.div
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm text-red-300"
                style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.2)" }}
              >
                <AlertCircle size={14} />
                {uploadError}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Before upload */}
          <AnimatePresence>
            {!files.length && (
              <motion.p
                key="subheading"
                initial={{ opacity: 1 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.2 }}
                className="text-sm text-white/40 font-light text-center"
              >
                Drag & drop your PDF here, or{" "}
                <span className="text-sky-400 hover:text-sky-300 transition">click to browse</span>
              </motion.p>
            )}
          </AnimatePresence>

          <div className={cn("relative w-full mx-auto", files.length ? "max-w-xs" : "max-w-[7rem]")}>
            {/* Upload progress + done state */}
            {files.length > 0 && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="rounded-2xl overflow-hidden"
                style={{
                  background: "rgba(255,255,255,0.03)",
                  border: "1px solid rgba(255,255,255,0.08)",
                  padding: "1.25rem",
                  minHeight: "160px",
                }}
              >
                {isDone ? (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex flex-col items-center justify-center h-full gap-3 py-4"
                  >
                    <div
                      className="w-12 h-12 rounded-full flex items-center justify-center"
                      style={{ background: "rgba(56,189,248,0.12)", border: "1px solid rgba(56,189,248,0.3)" }}
                    >
                      <IconCheck className="text-sky-400 w-6 h-6" />
                    </div>
                    <div className="text-center">
                      <p className="text-white font-medium text-sm">Ready!</p>
                      <p className="text-sky-400 text-xs font-light mt-0.5">Ask a question below</p>
                    </div>
                  </motion.div>
                ) : (
                  <div>
                    {/* File info */}
                    <div className="flex items-center gap-2 mb-3 pb-3" style={{ borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
                      <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                        style={{ background: "rgba(56,189,248,0.1)" }}>
                        <FileText size={14} className="text-sky-400" />
                      </div>
                      <p className="text-white text-xs truncate font-light">{files[0].name}</p>
                    </div>

                    {/* Progress messages */}
                    <div className="max-h-[120px] overflow-y-auto space-y-1.5">
                      <AnimatePresence>
                        {messages.map((msg, idx) => (
                          <motion.div
                            key={msg + idx}
                            initial={{ opacity: 0, x: -8 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0 }}
                            transition={{ duration: 0.25, delay: idx * 0.05 }}
                            className="flex items-start gap-2"
                          >
                            <IconCheck className="w-3.5 h-3.5 text-sky-400 mt-0.5 flex-shrink-0" />
                            <span className="text-xs text-white/60 font-light leading-snug">{msg}</span>
                          </motion.div>
                        ))}
                      </AnimatePresence>

                      {messages.length === 0 && (
                        <div className="flex items-center gap-2">
                          <svg className="animate-spin w-3.5 h-3.5 text-sky-400" viewBox="0 0 50 50">
                            <circle cx="25" cy="25" r="20" fill="none" stroke="#38bdf8" strokeWidth="5"
                              strokeLinecap="round" strokeDasharray="90" strokeDashoffset="60" />
                          </svg>
                          <span className="text-xs text-white/40 font-light">Processing…</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </motion.div>
            )}

            {/* Upload box (before file) */}
            {!files.length && (
              <div className="relative h-28 mt-2 w-full">
                <motion.div
                  variants={secondaryVariant}
                  initial="initial"
                  animate={controls}
                  transition={{ duration: 0.3 }}
                  className="absolute inset-0 z-30 bg-transparent rounded-xl"
                  style={{ border: "1px dashed rgba(56,189,248,0.4)" }}
                />
                <motion.div
                  layoutId="file-upload"
                  variants={mainVariant}
                  initial="initial"
                  animate={controls}
                  transition={{ type: "spring", stiffness: 300, damping: 20 }}
                  className={cn(
                    "relative z-40 flex flex-col items-center justify-center h-full w-full rounded-xl cursor-pointer",
                    isDragActive ? "bg-sky-500/10" : "bg-neutral-900"
                  )}
                  style={{ boxShadow: "0 8px 32px rgba(0,0,0,0.3)" }}
                >
                  <div
                    className="w-10 h-10 rounded-xl flex items-center justify-center mb-2"
                    style={{ background: "rgba(56,189,248,0.08)", border: "1px solid rgba(56,189,248,0.15)" }}
                  >
                    <IconUpload className="h-5 w-5 text-sky-400" />
                  </div>
                  <p className="text-xs text-white/30 font-light">PDF only</p>
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
