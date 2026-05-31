"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import { X, Eye, EyeOff, Loader2, AlertCircle } from "lucide-react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface AuthModalProps {
  open: boolean;
  defaultTab?: "signin" | "signup";
  onClose: () => void;
}

const GoogleIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
    <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
    <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
    <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
  </svg>
);

const PasswordInput = ({
  id,
  placeholder,
  value,
  onChange,
  error,
}: {
  id: string;
  placeholder: string;
  value: string;
  onChange: (v: string) => void;
  error?: string;
}) => {
  const [show, setShow] = useState(false);
  return (
    <div className="space-y-1">
      <div className={`relative flex items-center bg-white/5 border rounded-xl transition-all duration-200 ${error ? "border-red-500/70" : "border-white/10 focus-within:border-sky-400/60"}`}>
        <input
          id={id}
          type={show ? "text" : "password"}
          placeholder={placeholder}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full bg-transparent px-4 py-3 text-sm text-white placeholder-white/30 outline-none"
          autoComplete={id === "signup-password" ? "new-password" : "current-password"}
        />
        <button
          type="button"
          onClick={() => setShow(!show)}
          className="pr-4 text-white/40 hover:text-white/70 transition"
          tabIndex={-1}
        >
          {show ? <EyeOff size={16} /> : <Eye size={16} />}
        </button>
      </div>
      {error && <p className="text-xs text-red-400 flex items-center gap-1"><AlertCircle size={12} />{error}</p>}
    </div>
  );
};

const AuthModal: React.FC<AuthModalProps> = ({ open, defaultTab = "signin", onClose }) => {
  const [tab, setTab] = useState<"signin" | "signup">(defaultTab);
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState("");

  // Sign In state
  const [siEmail, setSiEmail] = useState("");
  const [siPassword, setSiPassword] = useState("");
  const [siErrors, setSiErrors] = useState<{ email?: string; password?: string }>({});

  // Sign Up state
  const [suName, setSuName] = useState("");
  const [suEmail, setSuEmail] = useState("");
  const [suPassword, setSuPassword] = useState("");
  const [suConfirm, setSuConfirm] = useState("");
  const [suErrors, setSuErrors] = useState<{ name?: string; email?: string; password?: string; confirm?: string }>({});

  const backdropRef = useRef<HTMLDivElement>(null);

  // Reset when switching tabs
  useEffect(() => {
    setApiError("");
    setSiErrors({});
    setSuErrors({});
  }, [tab]);

  // Sync tab with prop
  useEffect(() => {
    setTab(defaultTab);
  }, [defaultTab, open]);

  // ESC to close
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  // Prevent body scroll when open
  useEffect(() => {
    document.body.style.overflow = open ? "hidden" : "";
    return () => { document.body.style.overflow = ""; };
  }, [open]);

  const handleGoogleAuth = () => {
    window.location.href = `${API_URL}/auth/login`;
  };

  const validateSignIn = () => {
    const errs: typeof siErrors = {};
    if (!siEmail.trim()) errs.email = "Email is required";
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(siEmail)) errs.email = "Enter a valid email";
    if (!siPassword) errs.password = "Password is required";
    setSiErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const validateSignUp = () => {
    const errs: typeof suErrors = {};
    if (!suName.trim()) errs.name = "Full name is required";
    if (!suEmail.trim()) errs.email = "Email is required";
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(suEmail)) errs.email = "Enter a valid email";
    if (!suPassword) errs.password = "Password is required";
    else if (suPassword.length < 8) errs.password = "Password must be at least 8 characters";
    if (suPassword !== suConfirm) errs.confirm = "Passwords do not match";
    setSuErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateSignIn()) return;
    setLoading(true);
    setApiError("");
    try {
      const res = await fetch(`${API_URL}/auth/email-login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ email: siEmail, password: siPassword }),
      });
      if (!res.ok) {
        const data = await res.json();
        setApiError(data.detail || "Invalid email or password.");
      } else {
        window.location.href = "/chat";
      }
    } catch {
      setApiError("Network error. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateSignUp()) return;
    setLoading(true);
    setApiError("");
    try {
      const res = await fetch(`${API_URL}/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ name: suName, email: suEmail, password: suPassword }),
      });
      if (!res.ok) {
        const data = await res.json();
        setApiError(data.detail || "Could not create account.");
      } else {
        window.location.href = "/chat";
      }
    } catch {
      setApiError("Network error. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  if (!open) return null;

  return (
    <div
      ref={backdropRef}
      className="modal-backdrop fixed inset-0 z-[100] flex items-center justify-center p-4"
      style={{ background: "rgba(0,0,0,0.75)", backdropFilter: "blur(8px)" }}
      onClick={(e) => { if (e.target === backdropRef.current) onClose(); }}
    >
      <div className="modal-panel relative w-full max-w-md rounded-2xl overflow-hidden shadow-2xl"
        style={{
          background: "linear-gradient(145deg, rgba(15,15,20,0.98) 0%, rgba(10,10,15,0.98) 100%)",
          border: "1px solid rgba(56,189,248,0.2)",
          boxShadow: "0 0 60px rgba(56,189,248,0.1), 0 25px 60px rgba(0,0,0,0.6)",
        }}
      >
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 z-10 p-2 rounded-full text-white/40 hover:text-white hover:bg-white/10 transition"
          aria-label="Close"
        >
          <X size={18} />
        </button>

        {/* Top glow accent */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-48 h-px"
          style={{ background: "linear-gradient(90deg, transparent, #38bdf8, transparent)" }}
        />

        <div className="p-8 pt-10">
          {/* Logo */}
          <div className="text-center mb-7">
            <span className="text-2xl font-light bg-gradient-to-r from-sky-400 to-blue-600 bg-clip-text text-transparent">
              EmbedMindAI
            </span>
            <p className="text-sm text-white/40 mt-1">
              {tab === "signin" ? "Welcome back" : "Create your account"}
            </p>
          </div>

          {/* Tabs */}
          <div className="flex rounded-xl bg-white/5 p-1 mb-7 gap-1">
            {(["signin", "signup"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`flex-1 py-2 text-sm font-medium rounded-lg transition-all duration-200 ${
                  tab === t
                    ? "bg-sky-500/20 text-sky-400 shadow-sm border border-sky-500/30"
                    : "text-white/40 hover:text-white/70"
                }`}
              >
                {t === "signin" ? "Sign In" : "Sign Up"}
              </button>
            ))}
          </div>

          {/* Google OAuth */}
          <button
            onClick={handleGoogleAuth}
            className="w-full flex items-center justify-center gap-3 py-3 rounded-xl bg-white text-gray-800 font-medium text-sm hover:bg-gray-100 transition-all duration-200 shadow-md mb-5"
          >
            <GoogleIcon />
            Continue with Google
          </button>

          {/* Divider */}
          <div className="flex items-center gap-3 mb-5">
            <div className="flex-1 h-px bg-white/10" />
            <span className="text-xs text-white/25 font-light">or continue with email</span>
            <div className="flex-1 h-px bg-white/10" />
          </div>

          {/* API Error */}
          {apiError && (
            <div className="mb-4 flex items-start gap-2 rounded-xl bg-red-500/10 border border-red-500/30 px-4 py-3">
              <AlertCircle size={16} className="text-red-400 mt-0.5 shrink-0" />
              <p className="text-sm text-red-300">{apiError}</p>
            </div>
          )}

          {/* Sign In Form */}
          {tab === "signin" && (
            <form onSubmit={handleSignIn} className="space-y-4">
              <div className="space-y-1">
                <div className={`flex items-center bg-white/5 border rounded-xl transition-all duration-200 ${siErrors.email ? "border-red-500/70" : "border-white/10 focus-within:border-sky-400/60"}`}>
                  <input
                    id="signin-email"
                    type="email"
                    placeholder="Email address"
                    value={siEmail}
                    onChange={(e) => setSiEmail(e.target.value)}
                    className="w-full bg-transparent px-4 py-3 text-sm text-white placeholder-white/30 outline-none"
                    autoComplete="email"
                  />
                </div>
                {siErrors.email && <p className="text-xs text-red-400 flex items-center gap-1"><AlertCircle size={12} />{siErrors.email}</p>}
              </div>

              <PasswordInput
                id="signin-password"
                placeholder="Password"
                value={siPassword}
                onChange={setSiPassword}
                error={siErrors.password}
              />

              <div className="flex justify-end">
                <button type="button" className="text-xs text-sky-400 hover:text-sky-300 transition">
                  Forgot password?
                </button>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full py-3 rounded-xl font-medium text-sm text-white transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-60"
                style={{
                  background: "linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%)",
                  boxShadow: loading ? "none" : "0 4px 20px rgba(14,165,233,0.35)",
                }}
              >
                {loading ? <Loader2 size={16} className="animate-spin" /> : null}
                {loading ? "Signing in…" : "Sign In"}
              </button>
            </form>
          )}

          {/* Sign Up Form */}
          {tab === "signup" && (
            <form onSubmit={handleSignUp} className="space-y-4">
              <div className="space-y-1">
                <div className={`flex items-center bg-white/5 border rounded-xl transition-all duration-200 ${suErrors.name ? "border-red-500/70" : "border-white/10 focus-within:border-sky-400/60"}`}>
                  <input
                    id="signup-name"
                    type="text"
                    placeholder="Full name"
                    value={suName}
                    onChange={(e) => setSuName(e.target.value)}
                    className="w-full bg-transparent px-4 py-3 text-sm text-white placeholder-white/30 outline-none"
                    autoComplete="name"
                  />
                </div>
                {suErrors.name && <p className="text-xs text-red-400 flex items-center gap-1"><AlertCircle size={12} />{suErrors.name}</p>}
              </div>

              <div className="space-y-1">
                <div className={`flex items-center bg-white/5 border rounded-xl transition-all duration-200 ${suErrors.email ? "border-red-500/70" : "border-white/10 focus-within:border-sky-400/60"}`}>
                  <input
                    id="signup-email"
                    type="email"
                    placeholder="Email address"
                    value={suEmail}
                    onChange={(e) => setSuEmail(e.target.value)}
                    className="w-full bg-transparent px-4 py-3 text-sm text-white placeholder-white/30 outline-none"
                    autoComplete="email"
                  />
                </div>
                {suErrors.email && <p className="text-xs text-red-400 flex items-center gap-1"><AlertCircle size={12} />{suErrors.email}</p>}
              </div>

              <PasswordInput
                id="signup-password"
                placeholder="Password (min. 8 characters)"
                value={suPassword}
                onChange={setSuPassword}
                error={suErrors.password}
              />

              <PasswordInput
                id="signup-confirm"
                placeholder="Confirm password"
                value={suConfirm}
                onChange={setSuConfirm}
                error={suErrors.confirm}
              />

              <button
                type="submit"
                disabled={loading}
                className="w-full py-3 rounded-xl font-medium text-sm text-white transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-60"
                style={{
                  background: "linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%)",
                  boxShadow: loading ? "none" : "0 4px 20px rgba(14,165,233,0.35)",
                }}
              >
                {loading ? <Loader2 size={16} className="animate-spin" /> : null}
                {loading ? "Creating account…" : "Create Account"}
              </button>

              <p className="text-xs text-white/30 text-center">
                By signing up you agree to our{" "}
                <span className="text-sky-400 cursor-pointer hover:underline">Terms of Service</span>
              </p>
            </form>
          )}
        </div>
      </div>
    </div>
  );
};

export default AuthModal;
