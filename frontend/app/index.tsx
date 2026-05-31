"use client";

import React, { useEffect, useState } from "react";

const API_URL =
  typeof window !== "undefined"
    ? process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
    : "http://localhost:8000";

// ─── Data ────────────────────────────────────────────────────────────────────

const WORDS = ["Research Papers.", "Legal Contracts.", "Study Materials.", "Technical Docs.", "Any PDF."];

const MOCK_MESSAGES = [
  { role: "user",  text: "What are the main findings of Chapter 3?" },
  { role: "ai",    text: "Chapter 3 reveals ocean temperatures have risen by 0.87°C since pre-industrial times, with rates accelerating sharply after 2000..." },
  { role: "user",  text: "What does this mean for coastal cities by 2040?" },
  { role: "ai",    text: "The document identifies three critical risks: increased storm surge intensity, biannual flooding of low-lying districts, and forced migration of 2.4M residents..." },
];

const FEATURES = [
  {
    icon: "◎",
    color: "#7C3AED",
    title: "Semantic Understanding",
    desc: "Goes beyond keyword matching. Understands meaning and context to surface exactly what you need — even when the words don't match.",
  },
  {
    icon: "⚡",
    color: "#EC4899",
    title: "Multi-Stage Retrieval",
    desc: "Advanced RAG pipeline with reranking and MMR diversity selection. Gets the right chunks, not just the closest ones.",
  },
  {
    icon: "✦",
    color: "#F59E0B",
    title: "Powered by Gemini",
    desc: "Google Gemini 2.5 Flash delivers nuanced, human-quality answers grounded strictly in your document's content.",
  },
  {
    icon: "⬡",
    color: "#10B981",
    title: "Your Data, Your Control",
    desc: "Documents are processed in your own environment. Nothing is shared, stored externally, or used for model training.",
  },
];

const STEPS = [
  { num: "01", icon: "↑", title: "Upload", desc: "Drop any PDF — research papers, contracts, textbooks, or reports." },
  { num: "02", icon: "◌", title: "Process", desc: "AI reads, chunks, and indexes your document with high-precision embeddings." },
  { num: "03", icon: "◎", title: "Converse", desc: "Ask natural questions and receive accurate, document-sourced answers." },
];

// ─── Component ────────────────────────────────────────────────────────────────

export default function LandingPage() {
  // Nav scroll state
  const [scrolled, setScrolled] = useState(false);

  // Typewriter
  const [wordIdx, setWordIdx]   = useState(0);
  const [charIdx, setCharIdx]   = useState(0);
  const [typed, setTyped]       = useState("");
  const [deleting, setDeleting] = useState(false);

  // Chat mockup
  const [visibleMsg, setVisibleMsg] = useState(0);

  // ── Scroll listener ──────────────────────────────────────────────────────
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // ── Typewriter ────────────────────────────────────────────────────────────
  useEffect(() => {
    const word = WORDS[wordIdx];
    const speed = deleting ? 38 : 95;

    const t = setTimeout(() => {
      if (!deleting) {
        const next = charIdx + 1;
        setTyped(word.slice(0, next));
        setCharIdx(next);
        if (next === word.length) setTimeout(() => setDeleting(true), 2400);
      } else {
        const next = charIdx - 1;
        setTyped(word.slice(0, next));
        setCharIdx(next);
        if (next === 0) {
          setDeleting(false);
          setWordIdx((w) => (w + 1) % WORDS.length);
        }
      }
    }, speed);

    return () => clearTimeout(t);
  }, [charIdx, deleting, wordIdx]);

  // ── Chat animation ────────────────────────────────────────────────────────
  useEffect(() => {
    const t = setInterval(
      () => setVisibleMsg((v) => (v < MOCK_MESSAGES.length - 1 ? v + 1 : 0)),
      2400
    );
    return () => clearInterval(t);
  }, []);

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="embedmind-root">
      {/* ── Global styles ─────────────────────────────────────────────────── */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        .embedmind-root {
          min-height: 100vh;
          background: #05050F;
          color: #FAFAFA;
          font-family: 'Inter', system-ui, -apple-system, sans-serif;
          overflow-x: hidden;
          -webkit-font-smoothing: antialiased;
        }

        /* Ambient orbs */
        .orb { position: fixed; border-radius: 50%; pointer-events: none; filter: blur(60px); }
        .orb-1 { width: 700px; height: 700px; top: -200px; left: -100px;
                  background: radial-gradient(circle, rgba(124,58,237,0.18) 0%, transparent 65%); }
        .orb-2 { width: 500px; height: 500px; top: 30%; right: -80px;
                  background: radial-gradient(circle, rgba(236,72,153,0.12) 0%, transparent 65%); }
        .orb-3 { width: 400px; height: 400px; bottom: 5%; left: 35%;
                  background: radial-gradient(circle, rgba(245,158,11,0.09) 0%, transparent 65%); }

        /* Gradient text utility */
        .grad {
          background: linear-gradient(135deg, #7C3AED 0%, #EC4899 55%, #F59E0B 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }

        /* Cursor blink */
        .cursor {
          display: inline-block;
          width: 3px;
          height: 0.85em;
          background: #7C3AED;
          margin-left: 2px;
          vertical-align: middle;
          animation: blink 1s step-end infinite;
        }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

        /* Fade-up animation */
        @keyframes fadeUp {
          from { opacity:0; transform: translateY(14px); }
          to   { opacity:1; transform: translateY(0); }
        }
        .fade-up { animation: fadeUp 0.45s cubic-bezier(.22,1,.36,1) forwards; }

        /* Float animation for hero card */
        @keyframes float {
          0%,100%  { transform: translateY(0px) rotate(-1deg); }
          50%      { transform: translateY(-12px) rotate(1deg); }
        }
        .float { animation: float 6s ease-in-out infinite; }

        /* Noise grain overlay */
        .grain::after {
          content: '';
          position: fixed;
          inset: 0;
          background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.035'/%3E%3C/svg%3E");
          pointer-events: none;
          z-index: 1000;
          opacity: 0.6;
        }

        /* Section container */
        .container { max-width: 1180px; margin: 0 auto; padding: 0 24px; }

        /* ─── NAV ─────────────────────────────────── */
        .nav {
          position: fixed; top: 0; left: 0; right: 0; z-index: 200;
          transition: background 0.35s, border-color 0.35s, backdrop-filter 0.35s;
        }
        .nav-scrolled {
          background: rgba(5,5,15,0.75);
          backdrop-filter: blur(24px);
          border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .nav-inner {
          max-width: 1180px; margin: 0 auto; padding: 0 24px;
          display: flex; align-items: center; justify-content: space-between;
          height: 66px;
        }
        .nav-logo { display: flex; align-items: center; gap: 10px; text-decoration: none; color: white; }
        .logo-mark {
          width: 34px; height: 34px; border-radius: 9px;
          background: linear-gradient(135deg, #7C3AED, #EC4899);
          display: flex; align-items: center; justify-content: center;
          font-size: 16px; font-weight: 700; color: white; letter-spacing: -0.5px;
        }
        .nav-links { display: flex; gap: 36px; }
        .nav-link {
          color: rgba(255,255,255,0.5); font-size: 14px; font-weight: 400;
          text-decoration: none; transition: color 0.2s; letter-spacing: 0.01em;
        }
        .nav-link:hover { color: white; }
        .nav-cta {
          background: rgba(124,58,237,0.15);
          border: 1px solid rgba(124,58,237,0.4);
          border-radius: 8px; padding: 9px 22px;
          color: rgba(255,255,255,0.9); font-size: 14px; font-weight: 500;
          cursor: pointer; text-decoration: none; transition: all 0.25s;
          white-space: nowrap;
        }
        .nav-cta:hover { background: rgba(124,58,237,0.3); color: white; }

        /* ─── HERO ───────────────────────────────────── */
        .hero {
          min-height: 100vh;
          display: flex; align-items: center;
          padding: 110px 24px 80px;
        }
        .hero-grid {
          max-width: 1180px; margin: 0 auto;
          display: grid; grid-template-columns: 1fr 1fr;
          gap: 80px; align-items: center;
          width: 100%;
        }
        .hero-badge {
          display: inline-flex; align-items: center; gap: 8px;
          background: rgba(124,58,237,0.10);
          border: 1px solid rgba(124,58,237,0.25);
          border-radius: 100px; padding: 6px 16px;
          margin-bottom: 28px;
        }
        .hero-badge-dot {
          width: 6px; height: 6px; border-radius: 50%;
          background: #7C3AED; flex-shrink: 0;
        }
        .hero-badge-text { font-size: 12.5px; color: rgba(255,255,255,0.65); letter-spacing: 0.01em; }
        .hero-h1 {
          font-size: 68px; font-weight: 800;
          line-height: 1.06; letter-spacing: -0.04em;
          margin-bottom: 22px;
        }
        .hero-sub {
          font-size: 17px; color: rgba(255,255,255,0.45);
          line-height: 1.75; max-width: 460px; margin-bottom: 40px;
          font-weight: 400;
        }
        .hero-actions { display: flex; gap: 14px; align-items: center; margin-bottom: 52px; }
        .btn-primary {
          background: linear-gradient(135deg, #7C3AED, #EC4899);
          border: none; border-radius: 10px; padding: 14px 28px;
          color: white; font-size: 15px; font-weight: 600;
          cursor: pointer; display: flex; align-items: center; gap: 8px;
          box-shadow: 0 0 40px rgba(124,58,237,0.35);
          transition: all 0.25s cubic-bezier(.22,1,.36,1);
          text-decoration: none; white-space: nowrap;
        }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 8px 48px rgba(124,58,237,0.55); }
        .btn-ghost {
          color: rgba(255,255,255,0.45); font-size: 14px; font-weight: 400;
          text-decoration: none; display: flex; align-items: center; gap: 6px;
          transition: color 0.2s;
        }
        .btn-ghost:hover { color: rgba(255,255,255,0.8); }
        .hero-trust {
          display: flex; align-items: center; gap: 16px;
        }
        .avatar-stack { display: flex; }
        .avatar {
          width: 30px; height: 30px; border-radius: 50%;
          border: 2px solid #05050F; margin-left: -8px;
          display: flex; align-items: center; justify-content: center;
          font-size: 11px; font-weight: 600; color: white;
        }
        .avatar:first-child { margin-left: 0; }
        .trust-text { font-size: 13px; color: rgba(255,255,255,0.35); }
        .trust-text strong { color: rgba(255,255,255,0.7); font-weight: 600; }

        /* ─── CHAT MOCKUP ──────────────────────────────── */
        .mockup-wrap { position: relative; }
        .mockup-glow {
          position: absolute; inset: -30px;
          background: radial-gradient(ellipse, rgba(124,58,237,0.22) 0%, transparent 68%);
          border-radius: 28px; filter: blur(24px); pointer-events: none;
        }
        .mockup-card {
          position: relative;
          background: rgba(10,10,20,0.85);
          backdrop-filter: blur(20px);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 20px; overflow: hidden;
          box-shadow: 0 40px 90px rgba(0,0,0,0.6), 0 0 0 0.5px rgba(255,255,255,0.04) inset;
        }
        .mockup-header {
          background: rgba(255,255,255,0.035);
          border-bottom: 1px solid rgba(255,255,255,0.06);
          padding: 14px 18px;
          display: flex; align-items: center; gap: 12px;
        }
        .mockup-pdf-icon {
          width: 38px; height: 38px; border-radius: 10px;
          background: rgba(124,58,237,0.15);
          border: 1px solid rgba(124,58,237,0.25);
          display: flex; align-items: center; justify-content: center;
          font-size: 17px; flex-shrink: 0;
        }
        .mockup-filename { font-size: 13px; font-weight: 500; margin-bottom: 2px; }
        .mockup-meta { font-size: 11px; color: rgba(255,255,255,0.3); }
        .traffic-lights { margin-left: auto; display: flex; gap: 6px; }
        .dot { width: 10px; height: 10px; border-radius: 50%; }
        .mockup-messages {
          padding: 18px; display: flex; flex-direction: column;
          gap: 14px; min-height: 300px;
        }
        .msg { display: flex; }
        .msg-user { justify-content: flex-end; }
        .msg-ai   { justify-content: flex-start; }
        .bubble {
          max-width: 82%; padding: 10px 14px; font-size: 12.5px; line-height: 1.55;
          border-radius: 16px;
        }
        .bubble-user {
          background: linear-gradient(135deg, #7C3AED, #C026D3);
          color: white;
          border-radius: 16px 16px 4px 16px;
        }
        .bubble-ai {
          background: rgba(255,255,255,0.06);
          color: rgba(255,255,255,0.78);
          border-radius: 16px 16px 16px 4px;
          border: 1px solid rgba(255,255,255,0.07);
        }
        .mockup-input {
          padding: 14px 18px; border-top: 1px solid rgba(255,255,255,0.05);
        }
        .input-bar {
          background: rgba(255,255,255,0.05);
          border: 1px solid rgba(255,255,255,0.09);
          border-radius: 10px; padding: 11px 14px;
          display: flex; align-items: center; gap: 12px;
        }
        .input-placeholder {
          flex: 1; font-size: 12.5px; color: rgba(255,255,255,0.25);
        }
        .send-btn {
          width: 28px; height: 28px; border-radius: 8px; flex-shrink: 0;
          background: linear-gradient(135deg, #7C3AED, #EC4899);
          display: flex; align-items: center; justify-content: center;
          font-size: 13px; cursor: pointer;
        }

        /* ─── DIVIDER ──────────────────────────────────── */
        .section-divider {
          max-width: 1180px; margin: 0 auto;
          height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
        }

        /* ─── FEATURES ────────────────────────────────── */
        .features-section { padding: 110px 24px; }
        .section-eyebrow {
          font-size: 12px; font-weight: 600; letter-spacing: 0.12em;
          text-transform: uppercase; margin-bottom: 16px;
        }
        .section-heading {
          font-size: 50px; font-weight: 800;
          letter-spacing: -0.04em; line-height: 1.08;
        }
        .section-sub {
          font-size: 17px; color: rgba(255,255,255,0.4);
          margin-top: 14px; max-width: 520px; line-height: 1.7;
        }
        .features-grid {
          display: grid; grid-template-columns: 1fr 1fr;
          gap: 16px; margin-top: 56px;
        }
        .feature-card {
          background: rgba(10,10,22,0.7);
          border: 1px solid rgba(255,255,255,0.06);
          border-radius: 16px; padding: 32px;
          cursor: default; transition: all 0.3s cubic-bezier(.22,1,.36,1);
          position: relative; overflow: hidden;
        }
        .feature-card:hover {
          border-color: rgba(124,58,237,0.3);
          transform: translateY(-4px);
          box-shadow: 0 24px 60px rgba(0,0,0,0.4);
        }
        .feature-icon-wrap {
          width: 46px; height: 46px; border-radius: 12px; margin-bottom: 20px;
          display: flex; align-items: center; justify-content: center;
          font-size: 20px; font-weight: 300;
        }
        .feature-title { font-size: 18px; font-weight: 600; margin-bottom: 10px; letter-spacing: -0.02em; }
        .feature-desc { font-size: 14px; color: rgba(255,255,255,0.42); line-height: 1.72; }

        /* ─── HOW IT WORKS ────────────────────────────── */
        .how-section {
          padding: 110px 24px;
          background: rgba(255,255,255,0.012);
          border-top: 1px solid rgba(255,255,255,0.04);
          border-bottom: 1px solid rgba(255,255,255,0.04);
        }
        .steps-grid {
          display: grid; grid-template-columns: 1fr 1fr 1fr;
          gap: 48px; margin-top: 60px; position: relative;
        }
        .step { text-align: center; position: relative; }
        .step-connector {
          position: absolute; top: 36px; left: 58%; right: -42%;
          height: 1px; background: linear-gradient(90deg, rgba(124,58,237,0.4), rgba(236,72,153,0.15));
        }
        .step-icon-wrap {
          width: 72px; height: 72px; border-radius: 18px; margin: 0 auto 20px;
          background: rgba(124,58,237,0.08);
          border: 1px solid rgba(124,58,237,0.18);
          display: flex; align-items: center; justify-content: center;
          font-size: 24px; font-weight: 300; color: rgba(124,58,237,0.9);
        }
        .step-num {
          font-size: 11px; font-weight: 700; letter-spacing: 0.1em;
          color: rgba(124,58,237,0.5); display: block; margin-bottom: 8px;
        }
        .step-title { font-size: 21px; font-weight: 700; margin-bottom: 10px; letter-spacing: -0.02em; }
        .step-desc { font-size: 14px; color: rgba(255,255,255,0.4); line-height: 1.7; max-width: 260px; margin: 0 auto; }

        /* ─── CTA SECTION ─────────────────────────────── */
        .cta-section { padding: 130px 24px; text-align: center; position: relative; overflow: hidden; }
        .cta-glow {
          position: absolute; inset: 0; pointer-events: none;
          background: radial-gradient(ellipse 55% 55% at 50% 50%, rgba(124,58,237,0.15) 0%, transparent 70%);
        }
        .cta-heading {
          font-size: 58px; font-weight: 800; letter-spacing: -0.04em;
          line-height: 1.07; margin-bottom: 22px; position: relative;
        }
        .cta-sub {
          font-size: 17px; color: rgba(255,255,255,0.4);
          max-width: 460px; margin: 0 auto 40px; line-height: 1.75;
          position: relative;
        }
        .btn-primary-lg {
          background: linear-gradient(135deg, #7C3AED, #EC4899);
          border: none; border-radius: 12px; padding: 16px 38px;
          color: white; font-size: 17px; font-weight: 700;
          cursor: pointer; position: relative;
          box-shadow: 0 0 64px rgba(124,58,237,0.42);
          transition: all 0.3s cubic-bezier(.22,1,.36,1);
          letter-spacing: -0.01em;
        }
        .btn-primary-lg:hover { transform: scale(1.04); box-shadow: 0 0 80px rgba(124,58,237,0.65); }
        .cta-note { margin-top: 18px; font-size: 13px; color: rgba(255,255,255,0.2); position: relative; }

        /* ─── FOOTER ─────────────────────────────────── */
        .footer {
          border-top: 1px solid rgba(255,255,255,0.05);
          padding: 36px 24px;
        }
        .footer-inner {
          max-width: 1180px; margin: 0 auto;
          display: flex; align-items: center; justify-content: space-between;
        }
        .footer-logo { display: flex; align-items: center; gap: 9px; text-decoration: none; color: white; }
        .footer-logo-mark {
          width: 28px; height: 28px; border-radius: 7px;
          background: linear-gradient(135deg, #7C3AED, #EC4899);
          display: flex; align-items: center; justify-content: center;
          font-size: 13px; font-weight: 700;
        }
        .footer-copy { font-size: 13px; color: rgba(255,255,255,0.22); }
        .footer-copy a { color: rgba(255,255,255,0.4); text-decoration: none; }
        .footer-copy a:hover { color: white; }
        .footer-links { display: flex; gap: 28px; }
        .footer-link {
          font-size: 13px; color: rgba(255,255,255,0.3);
          text-decoration: none; transition: color 0.2s;
        }
        .footer-link:hover { color: rgba(255,255,255,0.7); }
      `}</style>

      {/* Ambient orbs */}
      <div className="orb orb-1" />
      <div className="orb orb-2" />
      <div className="orb orb-3" />

      {/* ── NAV ──────────────────────────────────────────────────────────── */}
      <nav className={`nav${scrolled ? " nav-scrolled" : ""}`}>
        <div className="nav-inner">
          <a href="/" className="nav-logo">
            <div className="logo-mark">E</div>
            <span style={{ fontWeight: 700, fontSize: "17px", letterSpacing: "-0.02em" }}>EmbedMindAI</span>
          </a>

          <div className="nav-links">
            {[
              { label: "Features", href: "#features" },
              { label: "How it works", href: "#how-it-works" },
              { label: "GitHub", href: "https://github.com/Adhi1755/EmbedMindAI" },
            ].map(({ label, href }) => (
              <a key={label} className="nav-link" href={href} target={href.startsWith("http") ? "_blank" : undefined} rel="noreferrer">
                {label}
              </a>
            ))}
          </div>

          <a className="nav-cta" href={`${API_URL}/auth/login`}>
            Sign in with Google
          </a>
        </div>
      </nav>

      {/* ── HERO ─────────────────────────────────────────────────────────── */}
      <section className="hero">
        <div className="hero-grid">

          {/* Left — copy */}
          <div className="fade-up">
            <div className="hero-badge">
              <span className="hero-badge-dot" />
              <span className="hero-badge-text">Powered by Google Gemini 2.5 Flash</span>
            </div>

            <h1 className="hero-h1">
              Chat with your
              <br />
              <span className="grad">
                {typed}
                <span className="cursor" />
              </span>
            </h1>

            <p className="hero-sub">
              Upload any PDF and have an AI-powered conversation with its content.
              Get precise, sourced answers in seconds — not hours of reading.
            </p>

            <div className="hero-actions">
              <a className="btn-primary" href={`${API_URL}/auth/login`}>
                Start for free <span>→</span>
              </a>
              <a className="btn-ghost" href="#how-it-works">
                How it works <span>↓</span>
              </a>
            </div>

            <div className="hero-trust">
              <div className="avatar-stack">
                {["#7C3AED", "#EC4899", "#F59E0B", "#10B981"].map((bg, i) => (
                  <div key={i} className="avatar" style={{ background: bg }}>
                    {["A", "B", "C", "D"][i]}
                  </div>
                ))}
              </div>
              <p className="trust-text">
                Used by <strong>500+</strong> students &amp; researchers
              </p>
            </div>
          </div>

          {/* Right — chat mockup */}
          <div className="mockup-wrap float">
            <div className="mockup-glow" />
            <div className="mockup-card">

              {/* Header */}
              <div className="mockup-header">
                <div className="mockup-pdf-icon">📄</div>
                <div>
                  <p className="mockup-filename">climate_report_2024.pdf</p>
                  <p className="mockup-meta">247 pages · Ready to chat</p>
                </div>
                <div className="traffic-lights">
                  <div className="dot" style={{ background: "#FF5F57" }} />
                  <div className="dot" style={{ background: "#FFBD2E" }} />
                  <div className="dot" style={{ background: "#28C840" }} />
                </div>
              </div>

              {/* Messages */}
              <div className="mockup-messages">
                {MOCK_MESSAGES.slice(0, visibleMsg + 1).map((msg, i) => (
                  <div key={i} className={`msg msg-${msg.role} fade-up`}>
                    <div className={`bubble bubble-${msg.role}`}>{msg.text}</div>
                  </div>
                ))}
              </div>

              {/* Input */}
              <div className="mockup-input">
                <div className="input-bar">
                  <span className="input-placeholder">Ask anything about this document…</span>
                  <div className="send-btn">→</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── SECTION DIVIDER ──────────────────────────────────────────────── */}
      <div className="container"><div className="section-divider" /></div>

      {/* ── FEATURES ─────────────────────────────────────────────────────── */}
      <section id="features" className="features-section">
        <div className="container">
          <p className="section-eyebrow" style={{ color: "#7C3AED" }}>Why EmbedMindAI</p>
          <h2 className="section-heading">
            Not just search.
            <br />
            <span className="grad">True understanding.</span>
          </h2>
          <p className="section-sub">
            Built on a multi-stage RAG pipeline that goes beyond keywords — your PDF becomes a knowledge base you can actually talk to.
          </p>

          <div className="features-grid">
            {FEATURES.map((f, i) => (
              <div key={i} className="feature-card">
                <div className="feature-icon-wrap" style={{ background: `${f.color}15`, border: `1px solid ${f.color}30` }}>
                  <span style={{ color: f.color, fontSize: "22px" }}>{f.icon}</span>
                </div>
                <h3 className="feature-title">{f.title}</h3>
                <p className="feature-desc">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── HOW IT WORKS ─────────────────────────────────────────────────── */}
      <section id="how-it-works" className="how-section">
        <div className="container">
          <div style={{ textAlign: "center" }}>
            <p className="section-eyebrow" style={{ color: "#EC4899" }}>Simple by design</p>
            <h2 className="section-heading">From PDF to insight<br />in three steps.</h2>
          </div>

          <div className="steps-grid">
            {STEPS.map((step, i) => (
              <div key={i} className="step">
                {i < STEPS.length - 1 && <div className="step-connector" />}
                <div className="step-icon-wrap">{step.icon}</div>
                <span className="step-num">STEP {step.num}</span>
                <h3 className="step-title">{step.title}</h3>
                <p className="step-desc">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── FINAL CTA ────────────────────────────────────────────────────── */}
      <section className="cta-section">
        <div className="cta-glow" />
        <div style={{ position: "relative" }}>
          <h2 className="cta-heading">
            Understand every document.
            <br />
            <span className="grad">Instantly.</span>
          </h2>
          <p className="cta-sub">
            Sign in with Google and start chatting with your PDFs in under 60 seconds. No setup. No credit card.
          </p>
          <button className="btn-primary-lg" onClick={() => (window.location.href = `${API_URL}/auth/login`)}>
            Get Started — It&apos;s Free →
          </button>
          <p className="cta-note">Google OAuth · Instant access · Your data stays yours</p>
        </div>
      </section>

      {/* ── FOOTER ───────────────────────────────────────────────────────── */}
      <footer className="footer">
        <div className="footer-inner">
          <a href="/" className="footer-logo">
            <div className="footer-logo-mark">E</div>
            <span style={{ fontWeight: 700, fontSize: "15px", letterSpacing: "-0.02em" }}>EmbedMindAI</span>
          </a>

          <p className="footer-copy">
            © 2025 EmbedMindAI · Built by{" "}
            <a href="https://github.com/Adhi1755" target="_blank" rel="noreferrer">Adithya</a>
          </p>

          <div className="footer-links">
            <a href="https://github.com/Adhi1755/EmbedMindAI" target="_blank" rel="noreferrer" className="footer-link">GitHub</a>
            <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer" className="footer-link">API Docs</a>
          </div>
        </div>
      </footer>
    </div>
  );
}