"use client";

import React, { useEffect, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const CHAT_MSGS = [
  { role: "user", text: "Summarize the key findings from Chapter 3" },
  { role: "ai",   text: "Chapter 3 identifies three critical climate tipping points that could become irreversible after 2047..." },
  { role: "user", text: "What's the projected timeline for coastal cities?" },
  { role: "ai",   text: "According to Figure 3.2, coastal cities face biannual flooding events by 2040 under a 2°C scenario..." },
];

const FEATURES = [
  { tag: "RETRIEVAL",    title: "Multi-Stage RAG",      desc: "Chunks, embeds, and reranks with MMR diversity selection. Gets the right context, every time." },
  { tag: "INTELLIGENCE", title: "Gemini 2.5 Flash",     desc: "Google's frontier model delivers nuanced, document-grounded answers — not hallucinations." },
  { tag: "PRIVACY",      title: "Your Data, Yours",     desc: "Documents never leave your environment. Nothing is shared or used for model training." },
  { tag: "SPEED",        title: "Answers in Seconds",   desc: "From question to insight in under 3 seconds. No more manual page-by-page reading." },
  { tag: "ACCURACY",     title: "Source-Grounded",      desc: "Every answer is backed by specific passages in your document. Fully verifiable." },
  { tag: "FORMATS",      title: "Any PDF",              desc: "Research papers, legal contracts, textbooks, financial reports — all fully supported." },
];

const FOOTER_COLS = [
  { title: "Product",  links: ["Features", "How it works", "Pricing", "Changelog"] },
  { title: "Company",  links: ["About", "Blog", "Careers"] },
  { title: "Build",    links: ["API Docs", "GitHub", "Open Source"] },
  { title: "Support",  links: ["FAQ", "Community", "Contact Us"] },
];

export default function LandingPage() {
  const [scrolled, setScrolled]     = useState(false);
  const [visibleMsg, setVisibleMsg] = useState(0);

  useEffect(() => {
    const fn = () => setScrolled(window.scrollY > 10);
    window.addEventListener("scroll", fn, { passive: true });
    return () => window.removeEventListener("scroll", fn);
  }, []);

  useEffect(() => {
    const t = setInterval(
      () => setVisibleMsg((v) => (v < CHAT_MSGS.length - 1 ? v + 1 : 0)),
      2400
    );
    return () => clearInterval(t);
  }, []);

  return (
    <div id="em-root">

      {/* ─── STYLES ──────────────────────────────────────────────────────── */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        #em-root {
          min-height: 100vh;
          background: #050505;
          color: #fff;
          font-family: 'Inter', system-ui, -apple-system, sans-serif;
          overflow-x: hidden;
          -webkit-font-smoothing: antialiased;
        }

        /* ── Grid texture ─────────────────────────────────────────── */
        .em-grid {
          background-color: #050505;
          background-image:
            linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px);
          background-size: 60px 60px;
        }

        /* ── Nav ──────────────────────────────────────────────────── */
        .em-nav {
          position: fixed; top: 0; left: 0; right: 0; z-index: 200;
          display: flex; align-items: center;
          height: 62px; padding: 0 40px;
          transition: background 0.3s ease, border-color 0.3s ease;
        }
        .em-nav.scrolled {
          background: rgba(5,5,5,0.88);
          backdrop-filter: blur(18px);
          -webkit-backdrop-filter: blur(18px);
          border-bottom: 1px solid rgba(255,255,255,0.07);
        }
        .em-logo {
          display: flex; align-items: center; gap: 9px;
          text-decoration: none; color: #fff;
          font-size: 16px; font-weight: 700; letter-spacing: -0.02em;
          margin-right: 44px; flex-shrink: 0;
        }
        .em-logo-mark {
          width: 30px; height: 30px; border-radius: 7px;
          background: #2563EB;
          display: flex; align-items: center; justify-content: center;
          font-size: 15px; font-weight: 800;
        }
        .em-nav-links { display: flex; gap: 4px; flex: 1; }
        .em-nav-link {
          color: rgba(255,255,255,0.5); font-size: 14px; font-weight: 400;
          text-decoration: none; padding: 6px 12px; border-radius: 7px;
          transition: color 0.15s, background 0.15s;
        }
        .em-nav-link:hover { color: #fff; background: rgba(255,255,255,0.07); }
        .em-nav-actions { display: flex; gap: 8px; align-items: center; }
        .em-signin {
          color: rgba(255,255,255,0.55); font-size: 14px; font-weight: 500;
          text-decoration: none; padding: 8px 16px; border-radius: 8px;
          transition: color 0.15s;
        }
        .em-signin:hover { color: #fff; }
        .em-signup {
          background: #2563EB; color: #fff;
          font-size: 14px; font-weight: 600;
          text-decoration: none; padding: 8px 18px; border-radius: 8px;
          display: flex; align-items: center; gap: 6px;
          transition: background 0.15s;
        }
        .em-signup:hover { background: #1d4ed8; }

        /* ── Hero ─────────────────────────────────────────────────── */
        .em-hero {
          min-height: 100vh;
          display: flex; flex-direction: column;
          align-items: center; justify-content: center;
          text-align: center;
          padding: 130px 24px 80px;
        }
        .em-badge {
          display: inline-flex; align-items: center; gap: 10px;
          border: 1px solid rgba(255,255,255,0.12);
          border-radius: 100px; padding: 5px 16px 5px 6px;
          margin-bottom: 36px; cursor: default;
        }
        .em-badge-pill {
          background: #2563EB; border-radius: 100px;
          padding: 3px 11px; font-size: 10px; font-weight: 700;
          letter-spacing: 0.06em; color: #fff; text-transform: uppercase;
        }
        .em-badge-txt {
          font-size: 13px; color: rgba(255,255,255,0.55);
        }
        .em-h1 {
          font-size: clamp(54px, 8.5vw, 100px);
          font-weight: 800; line-height: 1.03;
          letter-spacing: -0.045em; margin-bottom: 24px;
        }
        .em-sub {
          font-size: 18px; color: rgba(255,255,255,0.42);
          max-width: 540px; line-height: 1.75; margin-bottom: 44px;
        }
        .em-ctas { display: flex; gap: 12px; justify-content: center; margin-bottom: 90px; }
        .em-cta-primary {
          background: #2563EB; color: #fff;
          font-size: 15px; font-weight: 600;
          text-decoration: none; padding: 13px 28px; border-radius: 10px;
          display: flex; align-items: center; gap: 7px;
          transition: background 0.2s;
        }
        .em-cta-primary:hover { background: #1d4ed8; }
        .em-cta-secondary {
          background: rgba(255,255,255,0.07);
          border: 1px solid rgba(255,255,255,0.1);
          color: #fff; font-size: 15px; font-weight: 500;
          text-decoration: none; padding: 13px 28px; border-radius: 10px;
          transition: background 0.2s;
        }
        .em-cta-secondary:hover { background: rgba(255,255,255,0.12); }

        /* ── Hero visual (3-card layout) ─────────────────────────── */
        .em-visual {
          display: grid; grid-template-columns: 1fr 1.6fr 1fr;
          gap: 14px; width: 100%; max-width: 960px;
          align-items: start;
        }
        .em-card {
          background: #0e0e0e;
          border: 1px solid rgba(255,255,255,0.09);
          border-radius: 12px; overflow: hidden;
          box-shadow: 0 32px 80px rgba(0,0,0,0.75);
        }
        .em-card-hdr {
          background: #080808;
          border-bottom: 1px solid rgba(255,255,255,0.07);
          padding: 10px 14px;
          display: flex; align-items: center; gap: 7px;
        }
        .em-dot { width: 9px; height: 9px; border-radius: 50%; }
        .em-card-title { font-size: 11.5px; color: rgba(255,255,255,0.35); margin-left: 4px; }
        .em-card-body { padding: 14px; }
        .em-row {
          display: flex; justify-content: space-between; align-items: center;
          padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .em-row:last-child { border-bottom: none; }
        .em-row-label { font-size: 11px; color: rgba(255,255,255,0.32); }
        .em-row-val {
          font-size: 11px; color: rgba(255,255,255,0.7);
          background: rgba(255,255,255,0.07); padding: 2px 9px; border-radius: 4px;
        }

        /* Center card – chat */
        .em-chat-msgs {
          padding: 14px; display: flex; flex-direction: column;
          gap: 11px; min-height: 220px;
        }
        .em-msg { display: flex; }
        .em-msg-user { justify-content: flex-end; }
        .em-msg-ai   { justify-content: flex-start; }
        .em-bubble {
          max-width: 84%; padding: 9px 13px;
          font-size: 12px; line-height: 1.55; border-radius: 10px;
        }
        .em-bubble-user {
          background: #2563EB; color: #fff;
          border-radius: 10px 10px 2px 10px;
        }
        .em-bubble-ai {
          background: rgba(255,255,255,0.07);
          border: 1px solid rgba(255,255,255,0.08);
          color: rgba(255,255,255,0.78);
          border-radius: 10px 10px 10px 2px;
        }
        .em-chat-input {
          margin: 0 14px 14px;
          background: rgba(255,255,255,0.05);
          border: 1px solid rgba(255,255,255,0.09);
          border-radius: 8px; padding: 9px 12px;
          display: flex; align-items: center; gap: 10px;
        }
        .em-chat-ph { flex: 1; font-size: 11.5px; color: rgba(255,255,255,0.2); }
        .em-send {
          width: 26px; height: 26px; border-radius: 7px;
          background: #2563EB; display: flex; align-items: center;
          justify-content: center; font-size: 12px; flex-shrink: 0;
        }

        /* ── Manifesto ────────────────────────────────────────────── */
        .em-manifesto {
          padding: 180px 24px;
          text-align: center;
        }
        .em-manifesto-tag {
          display: inline-block;
          border: 1px solid rgba(255,255,255,0.1);
          border-radius: 100px; padding: 4px 16px;
          font-size: 11px; font-weight: 500;
          color: rgba(255,255,255,0.35); letter-spacing: 0.06em;
          text-transform: uppercase; margin-bottom: 56px;
        }
        .em-manifesto-text {
          font-size: clamp(30px, 5vw, 62px);
          font-weight: 700; line-height: 1.22;
          letter-spacing: -0.03em; max-width: 880px; margin: 0 auto;
          background: linear-gradient(180deg, #fff 0%, #fff 25%, #444 65%, #111 100%);
          -webkit-background-clip: text; -webkit-text-fill-color: transparent;
          background-clip: text;
        }

        /* ── Features ─────────────────────────────────────────────── */
        .em-features { padding: 100px 24px; }
        .em-features-inner { max-width: 1160px; margin: 0 auto; }
        .em-features-tag {
          font-size: 11px; font-weight: 700; letter-spacing: 0.12em;
          text-transform: uppercase; color: rgba(255,255,255,0.28);
          margin-bottom: 18px;
        }
        .em-features-h2 {
          font-size: clamp(30px, 4vw, 52px); font-weight: 800;
          letter-spacing: -0.04em; line-height: 1.1; margin-bottom: 56px;
        }
        .em-feat-grid {
          display: grid; grid-template-columns: repeat(3, 1fr);
          border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; overflow: hidden;
        }
        .em-feat-item {
          background: #070707; padding: 32px;
          border-right: 1px solid rgba(255,255,255,0.08);
          border-bottom: 1px solid rgba(255,255,255,0.08);
          transition: background 0.2s;
        }
        .em-feat-item:nth-child(3n) { border-right: none; }
        .em-feat-item:nth-child(4),
        .em-feat-item:nth-child(5),
        .em-feat-item:nth-child(6) { border-bottom: none; }
        .em-feat-item:hover { background: #0d0d0d; }
        .em-feat-tag {
          font-size: 10px; font-weight: 700; letter-spacing: 0.12em;
          color: #2563EB; margin-bottom: 14px;
        }
        .em-feat-name {
          font-size: 18px; font-weight: 700; letter-spacing: -0.02em;
          margin-bottom: 10px;
        }
        .em-feat-desc {
          font-size: 13.5px; color: rgba(255,255,255,0.36); line-height: 1.72;
        }

        /* ── Big CTA ──────────────────────────────────────────────── */
        .em-big-cta {
          padding: 150px 24px; text-align: center;
        }
        .em-big-cta-h2 {
          font-size: clamp(40px, 6.5vw, 84px);
          font-weight: 800; letter-spacing: -0.045em; line-height: 1.06;
          margin-bottom: 24px;
        }
        .em-big-cta-sub {
          font-size: 17px; color: rgba(255,255,255,0.38);
          max-width: 460px; margin: 0 auto 44px; line-height: 1.75;
        }

        /* ── Footer ───────────────────────────────────────────────── */
        .em-footer {
          border-top: 1px solid rgba(255,255,255,0.07);
          padding: 64px 40px 0;
          overflow: hidden;
        }
        .em-footer-top {
          max-width: 1160px; margin: 0 auto;
          display: grid; grid-template-columns: 1.4fr 1fr 1fr 1fr 1fr;
          gap: 40px; padding-bottom: 60px;
          border-bottom: 1px solid rgba(255,255,255,0.06);
        }
        .em-footer-brand-desc {
          font-size: 13px; color: rgba(255,255,255,0.32);
          line-height: 1.7; margin-top: 14px; max-width: 240px;
        }
        .em-footer-col-h { font-size: 13px; font-weight: 600; color: #fff; margin-bottom: 20px; }
        .em-footer-link {
          display: block; font-size: 13px; color: rgba(255,255,255,0.32);
          text-decoration: none; margin-bottom: 13px; transition: color 0.15s;
        }
        .em-footer-link:hover { color: rgba(255,255,255,0.65); }
        .em-footer-bottom {
          max-width: 1160px; margin: 0 auto;
          display: flex; justify-content: space-between; align-items: center;
          padding: 22px 0;
        }
        .em-footer-copy { font-size: 11.5px; color: rgba(255,255,255,0.2); letter-spacing: 0.05em; text-transform: uppercase; }
        .em-footer-policy { font-size: 11.5px; color: rgba(255,255,255,0.22); text-decoration: none; }
        .em-footer-policy:hover { color: rgba(255,255,255,0.5); }
        .em-wordmark {
          text-align: center;
          font-size: clamp(72px, 17vw, 230px);
          font-weight: 900; letter-spacing: -0.05em;
          color: rgba(255,255,255,0.032);
          line-height: 0.88; padding: 20px 0 0;
          user-select: none; white-space: nowrap;
          overflow: hidden;
        }

        /* ── Misc ─────────────────────────────────────────────────── */
        @keyframes emFadeUp {
          from { opacity: 0; transform: translateY(10px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .em-anim { animation: emFadeUp 0.4s ease forwards; }

        .em-side-card { opacity: 0.65; }
        .em-side-card-l { transform: perspective(800px) rotateY(10deg) rotateX(4deg) translateY(24px); }
        .em-side-card-r { transform: perspective(800px) rotateY(-10deg) rotateX(4deg) translateY(24px); }
      `}</style>

      {/* ── NAV ──────────────────────────────────────────────────────────── */}
      <nav className={`em-nav${scrolled ? " scrolled" : ""}`}>
        <a href="/" className="em-logo">
          <div className="em-logo-mark">E</div>
          EmbedMindAI
        </a>

        <div className="em-nav-links">
          {["Products", "Developers", "Pricing", "Docs"].map((l) => (
            <a key={l} href="#" className="em-nav-link">{l}</a>
          ))}
        </div>

        <div className="em-nav-actions">
          <a href={`${API_URL}/auth/login`} className="em-signin">Sign in</a>
          <a href={`${API_URL}/auth/login`} className="em-signup">Sign up ↗</a>
        </div>
      </nav>

      {/* ── HERO ─────────────────────────────────────────────────────────── */}
      <section className="em-hero em-grid">
        <div className="em-badge">
          <span className="em-badge-pill">Powered by Gemini</span>
          <span className="em-badge-txt">Upload a PDF. Ask anything. Get answers.</span>
        </div>

        <h1 className="em-h1">
          Understand Any Document.<br />Instantly.
        </h1>

        <p className="em-sub">
          Upload any PDF and have a natural, AI-powered conversation with its content.
          Precise answers, grounded in your documents — not guesses.
        </p>

        <div className="em-ctas">
          <a href={`${API_URL}/auth/login`} className="em-cta-primary">
            Start free account ↗
          </a>
          <a href="#manifesto" className="em-cta-secondary">
            How it works
          </a>
        </div>

        {/* 3-card product visual */}
        <div className="em-visual">

          {/* Left — Document Info */}
          <div className="em-card em-side-card em-side-card-l">
            <div className="em-card-hdr">
              <div className="em-dot" style={{ background: "#FF5F57" }} />
              <div className="em-dot" style={{ background: "#FEBC2E" }} />
              <div className="em-dot" style={{ background: "#28C840" }} />
              <span className="em-card-title">Document Info</span>
            </div>
            <div className="em-card-body">
              <div className="em-row"><span className="em-row-label">File</span><span className="em-row-val">report.pdf</span></div>
              <div className="em-row"><span className="em-row-label">Pages</span><span className="em-row-val">247</span></div>
              <div className="em-row"><span className="em-row-label">Chunks</span><span className="em-row-val">412</span></div>
              <div className="em-row"><span className="em-row-label">Embed model</span><span className="em-row-val">text-emb-004</span></div>
              <div className="em-row">
                <span className="em-row-label">Status</span>
                <span className="em-row-val" style={{ color: "#4ade80" }}>● Ready</span>
              </div>
            </div>
          </div>

          {/* Center — Chat */}
          <div className="em-card">
            <div className="em-card-hdr">
              <div className="em-dot" style={{ background: "#FF5F57" }} />
              <div className="em-dot" style={{ background: "#FEBC2E" }} />
              <div className="em-dot" style={{ background: "#28C840" }} />
              <span className="em-card-title">climate_report_2024.pdf</span>
            </div>
            <div className="em-chat-msgs">
              {CHAT_MSGS.slice(0, visibleMsg + 1).map((m, i) => (
                <div key={i} className={`em-msg em-msg-${m.role} em-anim`}>
                  <div className={`em-bubble em-bubble-${m.role}`}>{m.text}</div>
                </div>
              ))}
            </div>
            <div className="em-chat-input">
              <span className="em-chat-ph">Ask anything about this document…</span>
              <div className="em-send">→</div>
            </div>
          </div>

          {/* Right — Retrieval stats */}
          <div className="em-card em-side-card em-side-card-r">
            <div className="em-card-hdr">
              <div className="em-dot" style={{ background: "#FF5F57" }} />
              <div className="em-dot" style={{ background: "#FEBC2E" }} />
              <div className="em-dot" style={{ background: "#28C840" }} />
              <span className="em-card-title">Retrieval</span>
            </div>
            <div className="em-card-body">
              <div className="em-row"><span className="em-row-label">Stage 1 recall</span><span className="em-row-val">20 chunks</span></div>
              <div className="em-row"><span className="em-row-label">After rerank</span><span className="em-row-val">7 chunks</span></div>
              <div className="em-row"><span className="em-row-label">MMR λ</span><span className="em-row-val">0.7</span></div>
              <div className="em-row"><span className="em-row-label">LLM</span><span className="em-row-val">Gemini 2.5</span></div>
              <div className="em-row"><span className="em-row-label">Latency</span><span className="em-row-val">1.2s</span></div>
            </div>
          </div>

        </div>
      </section>

      {/* ── MANIFESTO ────────────────────────────────────────────────────── */}
      <section id="manifesto" className="em-manifesto em-grid">
        <span className="em-manifesto-tag">More Than Search</span>
        <p className="em-manifesto-text">
          Our multi-stage retrieval system finds exactly what you need, when you need it — with every answer as precise and grounded as the document itself.
        </p>
      </section>

      {/* ── FEATURES ─────────────────────────────────────────────────────── */}
      <section className="em-features">
        <div className="em-features-inner">
          <p className="em-features-tag">Capabilities</p>
          <h2 className="em-features-h2">
            Everything you need to<br />understand any document.
          </h2>
          <div className="em-feat-grid">
            {FEATURES.map((f, i) => (
              <div key={i} className="em-feat-item">
                <p className="em-feat-tag">{f.tag}</p>
                <h3 className="em-feat-name">{f.title}</h3>
                <p className="em-feat-desc">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── BIG CTA ──────────────────────────────────────────────────────── */}
      <section className="em-big-cta em-grid">
        <h2 className="em-big-cta-h2">
          Start understanding your<br />documents today.
        </h2>
        <p className="em-big-cta-sub">
          Sign in with Google and chat with your first PDF in under 60 seconds.
          No setup. No credit card required.
        </p>
        <a href={`${API_URL}/auth/login`} className="em-cta-primary" style={{ display: "inline-flex", justifyContent: "center" }}>
          Start free account ↗
        </a>
      </section>

      {/* ── FOOTER ───────────────────────────────────────────────────────── */}
      <footer className="em-footer">
        <div className="em-footer-top">

          {/* Brand */}
          <div>
            <a href="/" className="em-logo">
              <div className="em-logo-mark">E</div>
              EmbedMindAI
            </a>
            <p className="em-footer-brand-desc">
              AI-powered document intelligence. Upload a PDF, ask anything, get precise answers instantly — grounded in your own content.
            </p>
          </div>

          {/* Columns */}
          {FOOTER_COLS.map((col) => (
            <div key={col.title}>
              <p className="em-footer-col-h">{col.title}</p>
              {col.links.map((l) => (
                <a key={l} href="#" className="em-footer-link">{l}</a>
              ))}
            </div>
          ))}
        </div>

        <div className="em-footer-bottom">
          <span className="em-footer-copy">© Copyright 2025 EmbedMindAI</span>
          <a href="#" className="em-footer-policy">Privacy Policy</a>
        </div>

        {/* Giant wordmark */}
        <div className="em-wordmark">EmbedMindAI</div>
      </footer>

    </div>
  );
}