"use client";

import React, { useEffect, useRef, useCallback, useState } from "react";
import {
  Github,
  FolderUp,
  ScanText,
  Scissors,
  Cpu,
  Database,
  BrainCircuit,
  ShieldCheck,
  Gauge,
  FileText,
  Activity,
  Code2,
  Zap,
  Network,
  Server,
  ChevronRight,
  SearchCode,
  Layers,
  ArrowRight,
  File,
  Settings,
} from "lucide-react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { OrbitingCircles } from "./components/orbiting-circles";
import AuthModal from "./components/AuthModal";

gsap.registerPlugin(ScrollTrigger);

const G = "bg-gradient-to-r from-sky-400 to-blue-600 bg-clip-text text-transparent";

// ── Data ─────────────────────────────────────────────────────────────────────

const pipeline = [
  {
    step: "01",
    Icon: FolderUp,
    title: "PDF Ingestion",
    description:
      "Any PDF — lecture slides, textbooks, research papers — is uploaded through the FastAPI endpoint and saved to the uploads directory with path sanitisation.",
    detail: "pdfplumber  ·  path sanitisation  ·  MIME validation",
    color: "#38bdf8",
  },
  {
    step: "02",
    Icon: ScanText,
    title: "Text Extraction",
    description:
      "pdfplumber parses every page, preserving layout structure. Raw text is then cleaned with regex normalisation to remove noise, repeated whitespace, and non-printable characters.",
    detail: "pdfplumber  ·  regex cleaning  ·  unicode normalisation",
    color: "#818cf8",
  },
  {
    step: "03",
    Icon: Scissors,
    title: "Semantic Chunking",
    description:
      "Text is segmented at natural boundaries — paragraph breaks, sentence endings — into 1 200-character chunks with a 200-character overlap to preserve cross-boundary context.",
    detail: "chunk_size=1200  ·  overlap=200  ·  semantic boundaries",
    color: "#34d399",
  },
  {
    step: "04",
    Icon: Cpu,
    title: "Vector Embedding",
    description:
      "Each chunk is embedded using Google's gemini-embedding-2 model with task_type=RETRIEVAL_DOCUMENT, producing a 3 072-dimensional dense vector that captures deep semantic meaning.",
    detail: "gemini-embedding-2  ·  3072 dims  ·  RETRIEVAL_DOCUMENT",
    color: "#f59e0b",
  },
  {
    step: "05",
    Icon: Database,
    title: "ChromaDB Persistence",
    description:
      "Embeddings and raw chunks are stored in a persistent ChromaDB collection backed by a Docker volume. The HNSW index enables approximate nearest-neighbour lookup in sub-millisecond time.",
    detail: "ChromaDB  ·  HNSW index  ·  cosine similarity  ·  persistent",
    color: "#ef4444",
  },
  {
    step: "06",
    Icon: BrainCircuit,
    title: "Multi-Stage Retrieval",
    description:
      "The query is embedded with RETRIEVAL_QUERY task type. Three retrieval stages run: vector search (Top-K=10) → multi-signal reranking (semantic 40%, lexical 25%, density 20%) → MMR diversity (λ=0.7, final 7 chunks).",
    detail: "Top-K=10  →  Rerank  →  MMR λ=0.7  →  7 chunks",
    color: "#a78bfa",
  },
];

const models = [
  {
    Icon: Cpu,
    label: "Embedding Model",
    name: "gemini-embedding-2",
    badge: "Google AI",
    color: "#38bdf8",
    specs: [
      "3 072 output dimensions",
      "Task-type: RETRIEVAL_DOCUMENT",
      "Task-type: RETRIEVAL_QUERY",
      "Batch processing: 100 chunks / call",
      "Exponential back-off on rate limits",
    ],
  },
  {
    Icon: Zap,
    label: "Language Model",
    name: "gemini-2.5-flash",
    badge: "Google AI",
    color: "#818cf8",
    specs: [
      "Google's latest reasoning model",
      "RAG-structured prompt engineering",
      "Handles long retrieved context",
      "Contextual answer generation",
      "Graceful fallback on API errors",
    ],
  },
  {
    Icon: Database,
    label: "Vector Store",
    name: "ChromaDB",
    badge: "Open Source",
    color: "#34d399",
    specs: [
      "Persistent Docker volume storage",
      "HNSW approximate nearest-neighbour",
      "Cosine similarity metric",
      "Collection reset on document delete",
      "Sub-millisecond query latency",
    ],
  },
];

const retrievalStages = [
  {
    step: "Stage 1",
    title: "Vector Search",
    stat: "TOP-K = 10",
    description:
      "Query embedded with RETRIEVAL_QUERY task type. ChromaDB cosine similarity search returns the 10 most semantically related chunks.",
    barWidth: "40%",
    color: "#38bdf8",
  },
  {
    step: "Stage 2",
    title: "Multi-Signal Reranking",
    stat: "→ TOP 14",
    description:
      "Candidates re-scored using: 40% semantic similarity + 25% term overlap + 20% term density + 10% positional weight + 5% chunk completeness.",
    barWidth: "70%",
    color: "#818cf8",
  },
  {
    step: "Stage 3",
    title: "MMR Diversity Selection",
    stat: "FINAL 7",
    description:
      "Maximal Marginal Relevance (λ=0.7) balances relevance (70%) against redundancy (30%) to select the most informative, non-overlapping 7 chunks.",
    barWidth: "100%",
    color: "#a78bfa",
  },
];

const features = [
  {
    Icon: ShieldCheck,
    title: "Privacy First",
    description:
      "Documents are processed locally. Only your Google API key communicates externally — no document content ever reaches a third-party server.",
  },
  {
    Icon: Gauge,
    title: "Sub-Second Retrieval",
    description:
      "ChromaDB's HNSW index retrieves the top-10 chunks in under a millisecond regardless of PDF size or number of stored vectors.",
  },
  {
    Icon: FileText,
    title: "Format Agnostic",
    description:
      "pdfplumber accurately handles handwritten notes, lecture slides, research papers, and dense multi-column layouts without pre-processing.",
  },
  {
    Icon: BrainCircuit,
    title: "Context-Aware Answers",
    description:
      "Multi-stage retrieval ensures Gemini sees the most relevant, non-redundant chunks — not just the highest cosine-similarity matches.",
  },
  {
    Icon: Activity,
    title: "Real-Time Progress",
    description:
      "WebSocket channel pushes live status messages for every pipeline stage: extraction, chunking, embedding, and ChromaDB storage.",
  },
  {
    Icon: Code2,
    title: "Fully Open Source",
    description:
      "MIT licensed. Every algorithm, every retrieval weight, every prompt template is inspectable. Fork, extend, or self-host freely.",
  },
];

const techStack = [
  { label: "Python 3.11", color: "#3b82f6" },
  { label: "FastAPI", color: "#10b981" },
  { label: "ChromaDB", color: "#f59e0b" },
  { label: "gemini-embedding-2", color: "#38bdf8" },
  { label: "gemini-2.5-flash", color: "#818cf8" },
  { label: "pdfplumber", color: "#f97316" },
  { label: "Next.js 15", color: "#ffffff" },
  { label: "MongoDB", color: "#22c55e" },
  { label: "WebSockets", color: "#06b6d4" },
  { label: "Docker", color: "#2563eb" },
  { label: "GSAP", color: "#ef4444" },
  { label: "Zustand", color: "#a78bfa" },
];

const marqueeItems = [
  { Icon: Cpu, text: "gemini-embedding-2" },
  { Icon: Database, text: "ChromaDB HNSW" },
  { Icon: Zap, text: "gemini-2.5-flash" },
  { Icon: Scissors, text: "Semantic Chunking" },
  { Icon: Network, text: "3072-D Embeddings" },
  { Icon: SearchCode, text: "MMR Diversity" },
  { Icon: Layers, text: "Multi-Stage RAG" },
  { Icon: Server, text: "FastAPI Backend" },
  { Icon: Activity, text: "WebSocket Progress" },
  { Icon: ShieldCheck, text: "Privacy First" },
];

// ── Sub-components ────────────────────────────────────────────────────────────

const SectionBadge = ({ children }: { children: React.ReactNode }) => (
  <span
    className="inline-block px-4 py-1.5 rounded-full text-xs font-medium text-sky-400 mb-4"
    style={{ background: "rgba(56,189,248,0.08)", border: "1px solid rgba(56,189,248,0.2)" }}
  >
    {children}
  </span>
);

// ── Main Component ────────────────────────────────────────────────────────────

const LandingPage = () => {
  const refs = useRef<Record<string, HTMLElement | null>>({
    animatedContent: null,
    heroSection: null,
    heroHeading: null,
    heroBadge: null,
    heroParagraph: null,
    heroStats: null,
    heroButtons: null,
    leftColumn: null,
    orbitingCircles: null,
    header: null,
    footer: null,
    mainContent: null,
    stepsContainer: null,
    featuresSection: null,
    techSection: null,
    ctaSection: null,
    modelsSection: null,
    retrievalSection: null,
    pipelineSection: null,
  });

  const [authModal, setAuthModal] = useState<{ open: boolean; tab: "signin" | "signup" }>({
    open: false,
    tab: "signin",
  });

  const animationControllers = useRef<{
    pageLoadTl: gsap.core.Timeline | null;
    scrollTriggers: ScrollTrigger[];
  }>({ pageLoadTl: null, scrollTriggers: [] });

  const handleLoginClick = useCallback(() => setAuthModal({ open: true, tab: "signin" }), []);
  const handleSignUpClick = useCallback(() => setAuthModal({ open: true, tab: "signup" }), []);

  const initPageLoadAnimations = useCallback(() => {
    const { current: r } = refs;
    const ctx = gsap.context(() => {
      gsap.set(
        [r.header, r.heroBadge, r.heroHeading, r.heroParagraph, r.heroStats, r.heroButtons],
        { opacity: 0, y: 30, force3D: true }
      );
      gsap.set(r.mainContent, { opacity: 0, y: 50, force3D: true });
      gsap.set(r.footer, { opacity: 0, y: 20, force3D: true });

      const tl = gsap.timeline({
        onComplete: () =>
          gsap.set(
            [r.header, r.heroBadge, r.heroHeading, r.heroParagraph, r.heroStats, r.heroButtons, r.mainContent, r.footer],
            { clearProps: "transform" }
          ),
      });
      animationControllers.current.pageLoadTl = tl;

      tl.to(r.header, { opacity: 1, y: 0, duration: 0.8, ease: "back.out(1.2)" }, 0)
        .to(r.heroBadge, { opacity: 1, y: 0, duration: 0.7, ease: "back.out(1.5)" }, 0.03)
        .to(r.heroHeading, { opacity: 1, y: 0, duration: 0.8, ease: "back.out(1.2)" }, 0.08)
        .to(r.heroParagraph, { opacity: 1, y: 0, duration: 0.8, ease: "power2.out" }, 0.13)
        .to(r.heroStats, { opacity: 1, y: 0, duration: 0.7, ease: "power2.out" }, 0.17)
        .to(r.heroButtons, { opacity: 1, y: 0, duration: 0.8, ease: "power2.out" }, 0.21)
        .to(r.mainContent, { opacity: 1, y: 0, duration: 0.8, ease: "power2.out" }, 0.25)
        .to(r.footer, { opacity: 1, y: 0, duration: 0.8, ease: "power2.out" }, 0.3);

      const canvas = document.getElementById("starCanvas");
      if (canvas) gsap.fromTo(canvas, { opacity: 0 }, { opacity: 1, duration: 1.5, ease: "power2.out", delay: 0.3 });
    });
    return ctx;
  }, []);

  const initHeroScrollAnimations = useCallback(() => {
    const { current: r } = refs;
    const ctx = gsap.context(() => {
      const heroElements = [r.heroHeading, r.heroParagraph, r.heroButtons, r.heroStats, r.heroBadge].filter(Boolean);
      if (r.heroSection && heroElements.length) {
        const st = ScrollTrigger.create({
          trigger: r.heroSection,
          start: "bottom 70%",
          end: "bottom 20%",
          scrub: 1,
          onUpdate: (self) => {
            const p = self.progress;
            gsap.to(heroElements, { opacity: Math.max(0.2, 1 - p * 0.8), y: -20 * p, duration: 0.1, overwrite: "auto" });
          },
        });
        animationControllers.current.scrollTriggers.push(st);
      }
    });
    return ctx;
  }, []);

  const initContentScaleAnimation = useCallback(() => {
    const { current: r } = refs;
    const ctx = gsap.context(() => {
      if (r.animatedContent) {
        const st = ScrollTrigger.create({
          trigger: r.animatedContent,
          start: "top bottom",
          end: "top 50%",
          scrub: 1,
          ease: "power2.inOut",
          animation: gsap.fromTo(
            r.animatedContent,
            { scale: 0.7, borderRadius: "5rem", borderWidth: "5px" },
            { scale: 1, borderRadius: "0rem", borderWidth: "0px", transformOrigin: "center center" }
          ),
        });
        animationControllers.current.scrollTriggers.push(st);
      }
    });
    return ctx;
  }, []);

  const initOrbitingAnimation = useCallback(() => {
    const { current: r } = refs;
    const ctx = gsap.context(() => {
      if (r.leftColumn && r.orbitingCircles) {
        const st = ScrollTrigger.create({
          trigger: r.leftColumn,
          start: "top bottom",
          end: "bottom top",
          scrub: 2,
          animation: gsap.fromTo(r.orbitingCircles, { y: -200 }, { y: 0, ease: "none" }),
        });
        animationControllers.current.scrollTriggers.push(st);
      }
    });
    return ctx;
  }, []);

  const initStepsAnimation = useCallback(() => {
    const { current: r } = refs;
    const ctx = gsap.context(() => {
      if (r.stepsContainer) {
        const steps = r.stepsContainer.querySelectorAll(":scope > div");
        steps.forEach((step, index) => {
          if (index === 0) {
            gsap.set(step, { opacity: 1, x: 0 });
          } else {
            gsap.set(step, { opacity: 0, x: -30, force3D: true });
            const st = ScrollTrigger.create({
              trigger: step,
              start: "top 85%",
              end: "bottom 15%",
              toggleActions: "play none none reverse",
              animation: gsap.to(step, { opacity: 1, x: 0, duration: 0.6, ease: "power2.out", clearProps: "transform" }),
            });
            animationControllers.current.scrollTriggers.push(st);
          }
        });
      }
    });
    return ctx;
  }, []);

  const initSectionAnimations = useCallback(() => {
    const { current: r } = refs;
    const ctx = gsap.context(() => {
      const sections = [r.pipelineSection, r.modelsSection, r.retrievalSection, r.featuresSection, r.techSection];
      sections.forEach((section) => {
        if (!section) return;
        const cards = section.querySelectorAll(".feature-card, .model-card, .pipeline-card, .retrieval-card");
        cards.forEach((card, i) => {
          gsap.set(card, { opacity: 0, y: 36 });
          const st = ScrollTrigger.create({
            trigger: card,
            start: "top 92%",
            toggleActions: "play none none none",
            animation: gsap.to(card, { opacity: 1, y: 0, duration: 0.55, delay: (i % 3) * 0.09, ease: "power2.out" }),
          });
          animationControllers.current.scrollTriggers.push(st);
        });
      });
      if (r.techSection) {
        const badges = r.techSection.querySelectorAll(".tech-badge");
        gsap.set(badges, { opacity: 0, scale: 0.82 });
        const st = ScrollTrigger.create({
          trigger: r.techSection,
          start: "top 85%",
          toggleActions: "play none none none",
          animation: gsap.to(badges, { opacity: 1, scale: 1, duration: 0.45, stagger: 0.05, ease: "back.out(1.3)" }),
        });
        animationControllers.current.scrollTriggers.push(st);
      }
    });
    return ctx;
  }, []);

  const initStarryBackground = useCallback(() => {
    const canvas = document.getElementById("starCanvas") as HTMLCanvasElement;
    if (!canvas) return null;
    const ctx = canvas.getContext("2d")!;
    let animId: number;
    const stars: { x: number; y: number; r: number; dx: number; dy: number; opacity: number }[] = [];
    let w = window.innerWidth, h = window.innerHeight;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr; canvas.height = h * dpr;
    canvas.style.width = w + "px"; canvas.style.height = h + "px";
    ctx.scale(dpr, dpr);
    const n = Math.min(140, Math.floor((w * h) / 8000));
    for (let i = 0; i < n; i++)
      stars.push({ x: Math.random() * w, y: Math.random() * h, r: Math.random() * 1.3 + 0.2,
        dx: (Math.random() - 0.5) * 0.35, dy: (Math.random() - 0.5) * 0.35, opacity: Math.random() * 0.5 + 0.4 });
    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      stars.forEach((s) => { ctx.globalAlpha = s.opacity; ctx.fillStyle = "rgba(255,255,255,0.9)";
        ctx.beginPath(); ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2); ctx.fill(); });
      ctx.globalAlpha = 1;
    };
    const update = () => stars.forEach((s) => {
      s.x += s.dx; s.y += s.dy;
      if (s.x > w || s.x < 0) s.dx *= -1;
      if (s.y > h || s.y < 0) s.dy *= -1;
    });
    const animate = () => { draw(); update(); animId = requestAnimationFrame(animate); };
    animate();
    const onResize = () => {
      w = window.innerWidth; h = window.innerHeight;
      canvas.width = w * dpr; canvas.height = h * dpr;
      canvas.style.width = w + "px"; canvas.style.height = h + "px";
      ctx.scale(dpr, dpr);
    };
    window.addEventListener("resize", onResize);
    return () => { cancelAnimationFrame(animId); window.removeEventListener("resize", onResize); };
  }, []);

  useEffect(() => {
    const ctxs = [
      initPageLoadAnimations(), initHeroScrollAnimations(), initContentScaleAnimation(),
      initOrbitingAnimation(), initStepsAnimation(), initSectionAnimations(),
    ];
    const starCleanup = initStarryBackground();
    return () => {
      ctxs.forEach((c) => c?.revert());
      animationControllers.current.scrollTriggers.forEach((st) => st.kill());
      animationControllers.current.scrollTriggers = [];
      animationControllers.current.pageLoadTl?.kill();
      starCleanup?.();
      ScrollTrigger.refresh();
    };
  }, [initPageLoadAnimations, initHeroScrollAnimations, initContentScaleAnimation,
      initOrbitingAnimation, initStepsAnimation, initSectionAnimations, initStarryBackground]);

  const setRef = useCallback((key: string) => (el: HTMLElement | null) => { refs.current[key] = el; }, []);

  return (
    <main className="min-h-screen w-full bg-transparent text-white overflow-x-hidden relative">
      <AuthModal open={authModal.open} defaultTab={authModal.tab}
        onClose={() => setAuthModal((s) => ({ ...s, open: false }))} />

      {/* Starry background */}
      <div className="fixed inset-0 -z-10 pointer-events-none">
        <div className="absolute inset-0 bg-black">
          <canvas id="starCanvas" className="w-full h-full" />
        </div>
        <div className="absolute inset-x-0 bottom-0 h-[600px] pointer-events-none"
          style={{ background: "linear-gradient(to top, rgba(0,191,255,0.28) 0%, rgba(0,191,255,0.04) 100%)",
            borderRadius: "50% 50% 0 0 / 100% 100% 0 0", filter: "blur(130px)" }} />
      </div>

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header ref={setRef("header") as any}
        className="fixed top-0 w-full z-50 px-6 py-4 flex items-center justify-between"
        style={{ background: "rgba(0,0,0,0.75)", backdropFilter: "blur(24px)",
          borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg flex items-center justify-center text-xs font-semibold"
            style={{ background: "linear-gradient(135deg, #0ea5e9, #2563eb)" }}>E</div>
          <span className={`text-lg font-light ${G}`}>EmbedMindAI</span>
        </div>

        <nav className="hidden md:flex items-center gap-7">
          {[["#how-it-works", "Architecture"], ["#pipeline", "Pipeline"], ["#models", "Models"], ["#features", "Features"], ["#tech", "Tech Stack"]].map(([href, label]) => (
            <a key={label} href={href} className="text-xs text-white/50 hover:text-sky-400 transition tracking-wide uppercase">{label}</a>
          ))}
        </nav>

        <div className="flex items-center gap-3">
          <a href="https://github.com/Adhi1755" target="_blank" rel="noopener noreferrer"
            className="hidden md:flex items-center gap-1.5 text-sm text-white/40 hover:text-white transition">
            <Github size={15} />
          </a>
          <button onClick={handleLoginClick} id="header-signin-btn"
            className="text-xs text-white/60 hover:text-white px-4 py-2 rounded-full transition hover:bg-white/8">
            Sign In
          </button>
          <button onClick={handleSignUpClick} id="header-signup-btn"
            className="text-xs px-5 py-2 rounded-full font-medium transition-all duration-200"
            style={{ background: "linear-gradient(135deg, #0ea5e9, #2563eb)", boxShadow: "0 2px 16px rgba(14,165,233,0.35)" }}>
            Get Started
          </button>
        </div>
      </header>

      <div className="pt-24">
        {/* ── Hero ───────────────────────────────────────────────────────── */}
        <div ref={setRef("heroSection") as any}
          className="min-h-[58vh] flex flex-col justify-center items-center gap-7 px-4 text-center">

          {/* Badge */}
          <div ref={setRef("heroBadge") as any}>
            <span className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-medium text-sky-300 pulse-badge"
              style={{ background: "rgba(56,189,248,0.08)", border: "1px solid rgba(56,189,248,0.28)" }}>
              <span className="status-dot" />
              Production RAG System  ·  Gemini 2.5 Flash  ·  ChromaDB
            </span>
          </div>

          {/* Heading */}
          <h1 ref={setRef("heroHeading") as any}
            className={`text-6xl md:text-8xl font-extralight ${G} leading-tight tracking-tight`}>
            EmbedMindAI
          </h1>

          {/* Sub-heading */}
          <p ref={setRef("heroParagraph") as any}
            className="max-w-2xl text-white/50 text-base md:text-lg font-light leading-relaxed">
            A production-grade Retrieval-Augmented Generation system that embeds your PDFs using
            Google&apos;s <span className="text-sky-400 font-normal">gemini-embedding-2</span> into 3 072-dimensional
            vectors and answers questions with <span className="text-sky-400 font-normal">gemini-2.5-flash</span>
            &nbsp;via multi-stage semantic retrieval.
          </p>

          {/* Stats */}
          <div ref={setRef("heroStats") as any} className="flex flex-wrap justify-center gap-10">
            {[
              { Icon: Cpu, value: "3072", label: "Embedding Dimensions", color: "#38bdf8" },
              { Icon: Layers, value: "3-Stage", label: "Retrieval Pipeline", color: "#818cf8" },
              { Icon: Database, value: "< 1 ms", label: "Vector Lookup", color: "#34d399" },
            ].map(({ Icon, value, label, color }) => (
              <div key={label} className="flex flex-col items-center gap-1">
                <div className="flex items-center gap-2">
                  <Icon size={14} style={{ color }} />
                  <span className="text-2xl font-light" style={{ color }}>{value}</span>
                </div>
                <span className="text-[11px] text-white/35 font-light tracking-wide">{label}</span>
              </div>
            ))}
          </div>

          {/* CTAs */}
          <div ref={setRef("heroButtons") as any}
            className="flex flex-col sm:flex-row justify-center items-center gap-4">
            <button id="hero-get-started-btn" onClick={handleSignUpClick}
              className="px-8 py-3 text-sm font-medium rounded-full transition-all duration-200 flex items-center gap-2"
              style={{ background: "linear-gradient(135deg, #0ea5e9, #2563eb)", boxShadow: "0 4px 24px rgba(14,165,233,0.4)" }}
              onMouseEnter={(e) => (e.currentTarget.style.boxShadow = "0 6px 32px rgba(14,165,233,0.6)")}
              onMouseLeave={(e) => (e.currentTarget.style.boxShadow = "0 4px 24px rgba(14,165,233,0.4)")}>
              Get Started <ArrowRight size={15} />
            </button>
            <button id="hero-signin-btn" onClick={handleLoginClick}
              className="px-8 py-3 text-sm font-light rounded-full transition-all duration-200 text-white/60 hover:text-white"
              style={{ border: "1px solid rgba(255,255,255,0.12)", background: "rgba(255,255,255,0.03)" }}
              onMouseEnter={(e) => { e.currentTarget.style.borderColor = "rgba(56,189,248,0.35)"; e.currentTarget.style.background = "rgba(56,189,248,0.06)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.borderColor = "rgba(255,255,255,0.12)"; e.currentTarget.style.background = "rgba(255,255,255,0.03)"; }}>
              Sign In
            </button>
          </div>
          <p className="text-[11px] text-white/20 tracking-wide">No credit card required  ·  MIT licensed  ·  Open source</p>
        </div>

        {/* ── Marquee Strip ──────────────────────────────────────────────── */}
        <div className="relative w-full overflow-hidden py-5 my-4"
          style={{ borderTop: "1px solid rgba(255,255,255,0.05)", borderBottom: "1px solid rgba(255,255,255,0.05)",
            background: "rgba(255,255,255,0.015)" }}>
          {/* Left fade */}
          <div className="absolute left-0 top-0 bottom-0 w-24 z-10 pointer-events-none"
            style={{ background: "linear-gradient(to right, #000, transparent)" }} />
          {/* Right fade */}
          <div className="absolute right-0 top-0 bottom-0 w-24 z-10 pointer-events-none"
            style={{ background: "linear-gradient(to left, #000, transparent)" }} />
          <div className="marquee-track flex gap-10 w-max">
            {[...marqueeItems, ...marqueeItems].map(({ Icon, text }, i) => (
              <div key={i} className="flex items-center gap-2 text-white/35 whitespace-nowrap">
                <Icon size={13} className="text-sky-400/60" />
                <span className="text-xs font-light tracking-wide">{text}</span>
                <span className="text-white/10 ml-4">·</span>
              </div>
            ))}
          </div>
        </div>

        {/* ── How It Works (scale animation section) ─────────────────────── */}
        <div className="relative mt-[-40px] z-10 flex justify-center" id="how-it-works">
          <div ref={setRef("mainContent") as any} className="w-full">
            <div ref={setRef("animatedContent") as any}
              className="border-[5px] border-white/15 rounded-[5rem] overflow-hidden w-full bg-black/70 scale-[0.6]"
              style={{ transformOrigin: "center center", willChange: "transform" }}>
              <section className="px-8 md:px-14 py-14">
                <div className="mb-10">
                  <SectionBadge>Architecture</SectionBadge>
                  <h2 className={`text-3xl md:text-5xl font-extralight ${G}`}>How EmbedMindAI Works</h2>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-16">
                  {/* Steps */}
                  <div ref={setRef("stepsContainer") as any}
                    className="space-y-10 col-span-2 text-white/60 text-sm leading-relaxed">
                    <div ref={setRef("leftColumn") as any}>
                      <p className="text-base md:text-lg font-extralight text-white/50 mb-10 leading-relaxed">
                        RAG — Retrieval-Augmented Generation — pairs a vector retrieval engine with a large language model.
                        EmbedMindAI&apos;s implementation uses Google&apos;s Gemini embedding and generation APIs with a
                        custom three-stage retrieval pipeline for precision and diversity.
                      </p>
                      <div className="flex items-center gap-2 mb-2">
                        <FolderUp size={15} className="text-sky-400" />
                        <h3 className="text-lg font-light text-sky-400">Step 1 — PDF Ingestion</h3>
                      </div>
                      <p className="font-extralight text-white/50">
                        Upload any PDF through the FastAPI endpoint. The file is path-sanitised, saved to the uploads
                        directory, and a WebSocket progress event is emitted to the frontend.
                      </p>
                    </div>

                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <ScanText size={15} className="text-sky-400" />
                        <h3 className="text-lg font-light text-sky-400">Step 2 — Text Extraction</h3>
                      </div>
                      <p className="font-extralight text-white/50">
                        <strong className="text-white/70 font-normal">pdfplumber</strong> extracts raw text from every page.
                        Output is cleaned with regex to remove noise, repeated whitespace, and non-printable characters.
                      </p>
                    </div>

                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <Scissors size={15} className="text-sky-400" />
                        <h3 className="text-lg font-light text-sky-400">Step 3 — Semantic Chunking</h3>
                      </div>
                      <p className="font-extralight text-white/50">
                        Text is split at semantic boundaries into <code>chunk_size=1200</code> character segments with
                        <code>overlap=200</code> to preserve cross-boundary context.
                      </p>
                    </div>

                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <Cpu size={15} className="text-sky-400" />
                        <h3 className="text-lg font-light text-sky-400">Step 4 — Gemini Embedding</h3>
                      </div>
                      <p className="font-extralight text-white/50">
                        Each chunk is embedded with <code>gemini-embedding-2</code> using
                        <code>task_type=RETRIEVAL_DOCUMENT</code>, producing a <strong className="text-white/70 font-normal">3 072-dimensional</strong> dense
                        vector capturing deep semantic meaning.
                      </p>
                    </div>

                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <Database size={15} className="text-sky-400" />
                        <h3 className="text-lg font-light text-sky-400">Step 5 — ChromaDB Storage</h3>
                      </div>
                      <p className="font-extralight text-white/50">
                        Vectors and raw chunks are persisted in <strong className="text-white/70 font-normal">ChromaDB</strong> with HNSW approximate
                        nearest-neighbour indexing, backed by a Docker volume for cross-restart persistence.
                      </p>
                    </div>

                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <BrainCircuit size={15} className="text-sky-400" />
                        <h3 className="text-lg font-light text-sky-400">Step 6 — Multi-Stage Retrieval + Generation</h3>
                      </div>
                      <p className="font-extralight text-white/50">
                        Queries run through three stages: vector search (Top-K=10) → multi-signal reranking → MMR diversity selection (λ=0.7).
                        The final 7 chunks are injected into a structured prompt for <code>gemini-2.5-flash</code>.
                      </p>
                    </div>
                  </div>

                  {/* Orbiting circles */}
                  <div className="relative col-span-2 flex items-center justify-center md:scale-150">
                    <div ref={setRef("orbitingCircles") as any}
                      className="flex h-[400px] w-full flex-col items-center justify-center overflow-hidden">
                      <OrbitingCircles iconSize={100}>
                        <File /><Settings /><File /><File /><Settings />
                      </OrbitingCircles>
                      <OrbitingCircles iconSize={20} radius={100} reverse speed={1}>
                        <File /><Settings /><File /><File /><Settings />
                      </OrbitingCircles>
                    </div>
                  </div>
                </div>
              </section>
            </div>
          </div>
        </div>

        {/* ── Pipeline Deep Dive ─────────────────────────────────────────── */}
        <section id="pipeline" ref={setRef("pipelineSection") as any}
          className="px-6 md:px-16 py-28 max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <SectionBadge>Pipeline</SectionBadge>
            <h2 className={`text-4xl md:text-5xl font-extralight ${G} mb-4`}>The RAG Pipeline</h2>
            <p className="text-white/35 font-light text-base max-w-xl mx-auto">
              Six precisely engineered stages from raw PDF bytes to a grounded, contextual answer.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {pipeline.map(({ step, Icon, title, description, detail, color }) => (
              <div key={step} className="pipeline-card glass-card rounded-2xl p-6 relative overflow-hidden">
                {/* Step number watermark */}
                <span className="absolute top-4 right-5 text-5xl font-bold text-white/3 select-none">{step}</span>
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-9 h-9 rounded-xl flex items-center justify-center"
                    style={{ background: `${color}12`, border: `1px solid ${color}25` }}>
                    <Icon size={16} style={{ color }} />
                  </div>
                  <span className="text-[11px] font-medium tracking-widest uppercase" style={{ color }}>Step {step}</span>
                </div>
                <h3 className="text-base font-medium text-white mb-2">{title}</h3>
                <p className="text-sm text-white/45 font-light leading-relaxed mb-4">{description}</p>
                <div className="mt-auto">
                  <code className="text-[11px] leading-loose">{detail}</code>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ── Models & Infrastructure ───────────────────────────────────── */}
        <section id="models" ref={setRef("modelsSection") as any}
          className="px-6 md:px-16 py-24 max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <SectionBadge>Infrastructure</SectionBadge>
            <h2 className={`text-4xl md:text-5xl font-extralight ${G} mb-4`}>Models & Storage</h2>
            <p className="text-white/35 font-light text-base max-w-xl mx-auto">
              Every component chosen for accuracy, speed, and production reliability.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {models.map(({ Icon, label, name, badge, color, specs }) => (
              <div key={name} className="model-card glass-card rounded-2xl p-7 flex flex-col gap-5">
                <div className="flex items-start justify-between">
                  <div className="w-11 h-11 rounded-xl flex items-center justify-center"
                    style={{ background: `${color}10`, border: `1px solid ${color}22` }}>
                    <Icon size={20} style={{ color }} />
                  </div>
                  <span className="text-[10px] font-medium px-2.5 py-1 rounded-full tracking-wider"
                    style={{ background: `${color}12`, border: `1px solid ${color}25`, color }}>
                    {badge}
                  </span>
                </div>
                <div>
                  <p className="text-[11px] text-white/35 tracking-widest uppercase mb-1">{label}</p>
                  <h3 className="text-lg font-medium text-white font-mono">{name}</h3>
                </div>
                <ul className="flex flex-col gap-2 mt-auto">
                  {specs.map((s) => (
                    <li key={s} className="flex items-center gap-2 text-xs text-white/45 font-light">
                      <ChevronRight size={11} style={{ color, flexShrink: 0 }} />
                      {s}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>

        {/* ── Retrieval Intelligence ────────────────────────────────────── */}
        <section ref={setRef("retrievalSection") as any}
          className="px-6 md:px-16 py-24 max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <SectionBadge>Retrieval</SectionBadge>
            <h2 className={`text-4xl md:text-5xl font-extralight ${G} mb-4`}>Retrieval Intelligence</h2>
            <p className="text-white/35 font-light text-base max-w-xl mx-auto">
              Three cascaded stages that balance precision, recall, and contextual diversity.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {retrievalStages.map(({ step, title, stat, description, barWidth, color }) => (
              <div key={step} className="retrieval-card glass-card rounded-2xl p-7 flex flex-col gap-4">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[11px] tracking-widest uppercase font-medium" style={{ color }}>{step}</span>
                  <span className="text-xs font-mono px-2.5 py-1 rounded-md"
                    style={{ background: `${color}12`, border: `1px solid ${color}25`, color }}>
                    {stat}
                  </span>
                </div>
                <h3 className="text-base font-medium text-white">{title}</h3>

                {/* Bar */}
                <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
                  <div className="h-full rounded-full rerank-bar"
                    style={{ width: barWidth, background: `linear-gradient(to right, ${color}60, ${color})` }} />
                </div>

                <p className="text-sm text-white/40 font-light leading-relaxed">{description}</p>
              </div>
            ))}
          </div>

          {/* Score breakdown */}
          <div className="mt-8 glass-card rounded-2xl p-6">
            <p className="text-xs text-white/30 tracking-widest uppercase mb-4 font-medium">Stage 2 — Reranking Signal Weights</p>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {[
                { label: "Semantic Similarity", weight: "40%", color: "#38bdf8" },
                { label: "Term Overlap", weight: "25%", color: "#818cf8" },
                { label: "Term Density", weight: "20%", color: "#34d399" },
                { label: "Position Weight", weight: "10%", color: "#f59e0b" },
                { label: "Chunk Completeness", weight: "5%", color: "#ef4444" },
              ].map(({ label, weight, color }) => (
                <div key={label} className="text-center">
                  <div className="text-xl font-light mb-1" style={{ color }}>{weight}</div>
                  <div className="text-[11px] text-white/35 font-light">{label}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── Features ──────────────────────────────────────────────────── */}
        <section id="features" ref={setRef("featuresSection") as any}
          className="px-6 md:px-16 py-24 max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <SectionBadge>Why EmbedMindAI</SectionBadge>
            <h2 className={`text-4xl md:text-5xl font-extralight ${G} mb-4`}>Built for Intelligence</h2>
            <p className="text-white/35 font-light text-base max-w-xl mx-auto">
              Every feature engineered for accuracy, privacy, and speed.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {features.map(({ Icon, title, description }) => (
              <div key={title} className="feature-card glass-card rounded-2xl p-6">
                <div className="w-10 h-10 rounded-xl flex items-center justify-center mb-5"
                  style={{ background: "rgba(56,189,248,0.08)", border: "1px solid rgba(56,189,248,0.14)" }}>
                  <Icon size={17} className="text-sky-400" />
                </div>
                <h3 className="text-sm font-medium text-white mb-2">{title}</h3>
                <p className="text-sm text-white/40 leading-relaxed font-light">{description}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ── Tech Stack ─────────────────────────────────────────────────── */}
        <section id="tech" ref={setRef("techSection") as any}
          className="px-6 md:px-16 py-20 max-w-7xl mx-auto">
          <div className="text-center mb-12">
            <SectionBadge>Stack</SectionBadge>
            <h2 className={`text-3xl md:text-4xl font-extralight ${G} mb-3`}>Powered By</h2>
            <p className="text-white/35 font-light text-sm">Production-grade tools, thoughtfully assembled.</p>
          </div>
          <div className="flex flex-wrap justify-center gap-3">
            {techStack.map(({ label, color }) => (
              <span key={label} className="tech-badge px-5 py-2.5 rounded-full text-xs font-light cursor-default"
                style={{ background: "rgba(255,255,255,0.03)", border: `1px solid ${color}28`,
                  color: color === "#ffffff" ? "rgba(255,255,255,0.7)" : color }}
                onMouseEnter={(e) => { e.currentTarget.style.background = `${color}12`; e.currentTarget.style.borderColor = `${color}55`; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = "rgba(255,255,255,0.03)"; e.currentTarget.style.borderColor = `${color}28`; }}>
                {label}
              </span>
            ))}
          </div>
        </section>

        {/* ── CTA ───────────────────────────────────────────────────────── */}
        <section ref={setRef("ctaSection") as any} className="px-6 md:px-16 py-28 flex justify-center">
          <div className="relative max-w-3xl w-full rounded-3xl overflow-hidden px-10 py-16 text-center"
            style={{ background: "linear-gradient(135deg, rgba(14,165,233,0.1), rgba(37,99,235,0.1))",
              border: "1px solid rgba(56,189,248,0.18)", boxShadow: "0 0 80px rgba(56,189,248,0.06)" }}>
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-64 h-px"
              style={{ background: "linear-gradient(90deg, transparent, #38bdf8, transparent)" }} />

            <div className="w-12 h-12 rounded-2xl flex items-center justify-center mx-auto mb-6"
              style={{ background: "linear-gradient(135deg, #0ea5e9, #2563eb)", boxShadow: "0 4px 24px rgba(14,165,233,0.4)" }}>
              <BrainCircuit size={22} className="text-white" />
            </div>

            <h2 className={`text-4xl md:text-5xl font-extralight ${G} mb-4`}>Ready to Begin?</h2>
            <p className="text-white/40 text-base font-light mb-8 max-w-md mx-auto">
              Upload your first PDF and experience precision AI document intelligence powered by Gemini and ChromaDB.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button id="cta-get-started-btn" onClick={handleSignUpClick}
                className="px-10 py-3.5 rounded-full font-medium text-sm transition-all duration-200 flex items-center gap-2 justify-center"
                style={{ background: "linear-gradient(135deg, #0ea5e9, #2563eb)", boxShadow: "0 4px 24px rgba(14,165,233,0.45)" }}
                onMouseEnter={(e) => (e.currentTarget.style.boxShadow = "0 6px 36px rgba(14,165,233,0.65)")}
                onMouseLeave={(e) => (e.currentTarget.style.boxShadow = "0 4px 24px rgba(14,165,233,0.45)")}>
                Start for Free <ArrowRight size={15} />
              </button>
              <a href="https://github.com/Adhi1755" target="_blank" rel="noopener noreferrer"
                className="inline-flex items-center justify-center gap-2 px-10 py-3.5 rounded-full font-light text-sm text-white/60 hover:text-white transition-all duration-200"
                style={{ border: "1px solid rgba(255,255,255,0.12)", background: "rgba(255,255,255,0.03)" }}>
                <Github size={16} />View on GitHub
              </a>
            </div>
          </div>
        </section>

        {/* ── Footer ────────────────────────────────────────────────────── */}
        <footer ref={setRef("footer") as any} className="border-t"
          style={{ background: "rgba(0,0,0,0.85)", borderColor: "rgba(255,255,255,0.06)" }}>
          <div className="max-w-7xl mx-auto px-6 py-8">
            <div className="flex flex-col md:flex-row items-center justify-between gap-6">
              <div className="flex items-center gap-2.5">
                <div className="w-6 h-6 rounded-md flex items-center justify-center text-xs font-semibold"
                  style={{ background: "linear-gradient(135deg, #0ea5e9, #2563eb)" }}>E</div>
                <div>
                  <span className={`text-sm font-light ${G}`}>EmbedMindAI</span>
                  <p className="text-[11px] text-white/25">AI-Powered Document Intelligence  ·  2025</p>
                </div>
              </div>

              <div className="flex items-center gap-6 text-xs text-white/35">
                {[["#how-it-works", "Architecture"], ["#pipeline", "Pipeline"], ["#models", "Models"], ["#features", "Features"], ["#tech", "Stack"]].map(([href, label]) => (
                  <a key={label} href={href} className="hover:text-sky-400 transition">{label}</a>
                ))}
              </div>

              <div className="flex items-center gap-2">
                {[
                  { href: "https://github.com/Adhi1755", label: "GitHub", svg: <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" /> },
                  { href: "https://www.linkedin.com/in/adithyanagamuneendran/", label: "LinkedIn", svg: <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" /> },
                ].map(({ href, label, svg }) => (
                  <a key={label} href={href} target="_blank" rel="noopener noreferrer" aria-label={label}
                    className="p-2 rounded-full text-white/35 hover:text-white hover:bg-white/8 transition-all">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">{svg}</svg>
                  </a>
                ))}
              </div>
            </div>
            <div className="mt-6 pt-6 border-t border-white/5 text-center">
              <p className="text-[11px] text-white/18 font-light">
                Designed & Built by{" "}
                <a href="https://www.linkedin.com/in/adithyanagamuneendran/" className="text-sky-400/50 hover:text-sky-400 transition" target="_blank" rel="noopener noreferrer">Adithya</a>
                {" "}· Open Source on{" "}
                <a href="https://github.com/Adhi1755" className="text-sky-400/50 hover:text-sky-400 transition" target="_blank" rel="noopener noreferrer">GitHub</a>
              </p>
            </div>
          </div>
        </footer>
      </div>
    </main>
  );
};

export default LandingPage;