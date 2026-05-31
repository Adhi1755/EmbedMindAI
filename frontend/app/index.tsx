"use client";

import React, { useEffect, useRef, useCallback, useState } from "react";
import { Github } from "lucide-react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { OrbitingCircles } from "./components/orbiting-circles";
import { File, Settings, Search } from "lucide-react";
import AuthModal from "./components/AuthModal";

gsap.registerPlugin(ScrollTrigger);

const gradientText =
  "bg-gradient-to-r from-sky-400 to-blue-600 bg-clip-text text-transparent";

// ── Feature card data ───────────────────────────────────────────────
const features = [
  {
    icon: "🔒",
    title: "Privacy First",
    description:
      "Your documents never leave your machine. Everything runs locally with zero data sent to third-party servers.",
  },
  {
    icon: "⚡",
    title: "Instant Answers",
    description:
      "Semantic vector search retrieves the most relevant chunks in milliseconds, no matter the PDF size.",
  },
  {
    icon: "📄",
    title: "Any PDF Format",
    description:
      "Handwritten notes, lecture slides, textbooks — pdfplumber accurately extracts text from all layouts.",
  },
  {
    icon: "🧠",
    title: "Context-Aware AI",
    description:
      "Cosine similarity + Gemini LLM understand nuance, not just keywords, for truly insightful answers.",
  },
  {
    icon: "🔄",
    title: "Real-Time Progress",
    description:
      "WebSocket-powered live updates show chunking, embedding, and indexing progress as it happens.",
  },
  {
    icon: "🌐",
    title: "Open Source",
    description:
      "Fully transparent, MIT-licensed codebase. Inspect, fork, extend, or self-host with zero restrictions.",
  },
];

// ── Tech stack badges ───────────────────────────────────────────────
const techStack = [
  { label: "Python", color: "#3b82f6" },
  { label: "FastAPI", color: "#10b981" },
  { label: "ChromaDB", color: "#f59e0b" },
  { label: "Sentence Transformers", color: "#8b5cf6" },
  { label: "Google Gemini", color: "#ef4444" },
  { label: "Next.js", color: "#ffffff" },
  { label: "MongoDB", color: "#22c55e" },
  { label: "WebSockets", color: "#06b6d4" },
];

const LandingPage = () => {
  const refs = useRef({
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
  });

  const [authModal, setAuthModal] = useState<{
    open: boolean;
    tab: "signin" | "signup";
  }>({ open: false, tab: "signin" });

  const animationControllers = useRef({
    pageLoadTl: null,
    scrollTriggers: [],
  });

  const handleLoginClick = useCallback(() => {
    setAuthModal({ open: true, tab: "signin" });
  }, []);

  const handleSignUpClick = useCallback(() => {
    setAuthModal({ open: true, tab: "signup" });
  }, []);

  const initPageLoadAnimations = useCallback(() => {
    const { current: r } = refs;
    const ctx = gsap.context(() => {
      gsap.set(
        [
          r.header,
          r.heroBadge,
          r.heroHeading,
          r.heroParagraph,
          r.heroStats,
          r.heroButtons,
        ],
        { opacity: 0, y: 30, force3D: true }
      );
      gsap.set(r.mainContent, { opacity: 0, y: 50, force3D: true });
      gsap.set(r.footer, { opacity: 0, y: 20, force3D: true });

      const tl = gsap.timeline({
        onComplete: () => {
          gsap.set(
            [
              r.header,
              r.heroBadge,
              r.heroHeading,
              r.heroParagraph,
              r.heroStats,
              r.heroButtons,
              r.mainContent,
              r.footer,
            ],
            { clearProps: "transform" }
          );
        },
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
      if (canvas) {
        gsap.fromTo(canvas, { opacity: 0 }, { opacity: 1, duration: 1.5, ease: "power2.out", delay: 0.3 });
      }
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
            const progress = self.progress;
            gsap.to(heroElements, {
              opacity: Math.max(0.2, 1 - progress * 0.8),
              y: -20 * progress,
              duration: 0.1,
              overwrite: "auto",
            });
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
              animation: gsap.to(step, {
                opacity: 1,
                x: 0,
                duration: 0.6,
                ease: "power2.out",
                clearProps: "transform",
              }),
            });
            animationControllers.current.scrollTriggers.push(st);
          }
        });
      }
    });
    return ctx;
  }, []);

  const initFeaturesAnimation = useCallback(() => {
    const { current: r } = refs;
    const ctx = gsap.context(() => {
      if (r.featuresSection) {
        const cards = r.featuresSection.querySelectorAll(".feature-card");
        cards.forEach((card, i) => {
          gsap.set(card, { opacity: 0, y: 40 });
          const st = ScrollTrigger.create({
            trigger: card,
            start: "top 90%",
            toggleActions: "play none none none",
            animation: gsap.to(card, {
              opacity: 1,
              y: 0,
              duration: 0.6,
              delay: (i % 3) * 0.1,
              ease: "power2.out",
            }),
          });
          animationControllers.current.scrollTriggers.push(st);
        });
      }
    });
    return ctx;
  }, []);

  const initTechAnimation = useCallback(() => {
    const { current: r } = refs;
    const ctx = gsap.context(() => {
      if (r.techSection) {
        const badges = r.techSection.querySelectorAll(".tech-badge");
        gsap.set(badges, { opacity: 0, scale: 0.8 });
        const st = ScrollTrigger.create({
          trigger: r.techSection,
          start: "top 85%",
          toggleActions: "play none none none",
          animation: gsap.to(badges, {
            opacity: 1,
            scale: 1,
            duration: 0.5,
            stagger: 0.06,
            ease: "back.out(1.3)",
          }),
        });
        animationControllers.current.scrollTriggers.push(st);
      }
    });
    return ctx;
  }, []);

  const initStarryBackground = useCallback(() => {
    const canvas = document.getElementById("starCanvas") as HTMLCanvasElement;
    if (!canvas) return null;
    const ctx = canvas.getContext("2d");
    let animationId: number;
    const stars: { x: number; y: number; r: number; dx: number; dy: number; opacity: number }[] = [];
    let w = window.innerWidth;
    let h = window.innerHeight;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    ctx.scale(dpr, dpr);

    const numStars = Math.min(120, Math.floor((w * h) / 10000));
    for (let i = 0; i < numStars; i++) {
      stars.push({
        x: Math.random() * w,
        y: Math.random() * h,
        r: Math.random() * 1.2 + 0.3,
        dx: (Math.random() - 0.5) * 0.4,
        dy: (Math.random() - 0.5) * 0.4,
        opacity: Math.random() * 0.5 + 0.5,
      });
    }

    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      stars.forEach((star) => {
        ctx.globalAlpha = star.opacity;
        ctx.fillStyle = "rgba(255,255,255,0.8)";
        ctx.beginPath();
        ctx.arc(star.x, star.y, star.r, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.globalAlpha = 1;
    };

    const update = () => {
      stars.forEach((star) => {
        star.x += star.dx;
        star.y += star.dy;
        if (star.x > w || star.x < 0) star.dx *= -1;
        if (star.y > h || star.y < 0) star.dy *= -1;
      });
    };

    const animate = () => {
      draw();
      update();
      animationId = requestAnimationFrame(animate);
    };
    animate();

    const handleResize = () => {
      w = window.innerWidth;
      h = window.innerHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      canvas.style.width = w + "px";
      canvas.style.height = h + "px";
      ctx.scale(dpr, dpr);
    };
    window.addEventListener("resize", handleResize);
    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  useEffect(() => {
    const contexts = [
      initPageLoadAnimations(),
      initHeroScrollAnimations(),
      initContentScaleAnimation(),
      initOrbitingAnimation(),
      initStepsAnimation(),
      initFeaturesAnimation(),
      initTechAnimation(),
    ];
    const starCleanup = initStarryBackground();

    return () => {
      contexts.forEach((ctx) => ctx?.revert());
      animationControllers.current.scrollTriggers.forEach((st) => st.kill());
      animationControllers.current.scrollTriggers = [];
      animationControllers.current.pageLoadTl?.kill();
      starCleanup?.();
      ScrollTrigger.refresh();
    };
  }, [
    initPageLoadAnimations,
    initHeroScrollAnimations,
    initContentScaleAnimation,
    initOrbitingAnimation,
    initStepsAnimation,
    initFeaturesAnimation,
    initTechAnimation,
    initStarryBackground,
  ]);

  const setRef = useCallback(
    (key: string) => (el: HTMLElement | null) => {
      refs.current[key] = el;
    },
    []
  );

  return (
    <main className="min-h-screen w-full bg-transparent text-white overflow-x-hidden relative">
      {/* Auth Modal */}
      <AuthModal
        open={authModal.open}
        defaultTab={authModal.tab}
        onClose={() => setAuthModal((s) => ({ ...s, open: false }))}
      />

      {/* Starry background and center glow */}
      <div className="fixed inset-0 -z-10 pointer-events-none">
        <div className="absolute inset-0 bg-black">
          <canvas id="starCanvas" className="w-full h-full" />
        </div>
        <div
          className="absolute inset-x-0 bottom-0 h-[600px] pointer-events-none z-0"
          style={{
            background:
              "linear-gradient(to top, rgba(0,191,255,0.3) 0%, rgba(0,191,255,0.05) 100%)",
            borderRadius: "50% 50% 0 0 / 100% 100% 0 0",
            filter: "blur(120px)",
          }}
        />
      </div>

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header
        ref={setRef("header")}
        className="fixed top-0 w-full z-50 px-6 py-4 flex items-center justify-between"
        style={{
          background: "rgba(0,0,0,0.7)",
          backdropFilter: "blur(20px)",
          borderBottom: "1px solid rgba(255,255,255,0.06)",
        }}
      >
        <h1 className={`text-xl font-light ${gradientText}`}>EmbedMindAI</h1>

        <nav className="hidden md:flex items-center gap-8">
          <a href="#" className="text-sm text-white/60 hover:text-sky-400 transition">Home</a>
          <a href="#features" className="text-sm text-white/60 hover:text-sky-400 transition">Features</a>
          <a href="#how-it-works" className="text-sm text-white/60 hover:text-sky-400 transition">How It Works</a>
          <a href="#tech" className="text-sm text-white/60 hover:text-sky-400 transition">Tech Stack</a>
        </nav>

        <div className="flex items-center gap-3">
          <a
            href="https://github.com/Adhi1755"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden md:flex items-center gap-1.5 text-sm text-white/50 hover:text-white transition"
          >
            <Github size={16} />
          </a>
          <button
            onClick={handleLoginClick}
            id="header-signin-btn"
            className="text-sm text-white/70 hover:text-white px-4 py-2 rounded-full transition hover:bg-white/10"
          >
            Sign In
          </button>
          <button
            onClick={handleSignUpClick}
            id="header-signup-btn"
            className="text-sm px-5 py-2 rounded-full font-medium transition-all duration-200"
            style={{
              background: "linear-gradient(135deg, #0ea5e9, #2563eb)",
              boxShadow: "0 2px 16px rgba(14,165,233,0.35)",
            }}
          >
            Get Started
          </button>
        </div>
      </header>

      {/* ── Page content wrapper ────────────────────────────────────────── */}
      <div className="pt-24">
        {/* ── Hero Section ────────────────────────────────────────────── */}
        <div
          ref={setRef("heroSection")}
          className="min-h-[55vh] flex flex-col justify-center items-center gap-6 px-4 text-center"
        >
          {/* Badge */}
          <div ref={setRef("heroBadge")}>
            <span
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-medium text-sky-300 pulse-badge"
              style={{
                background: "rgba(56,189,248,0.1)",
                border: "1px solid rgba(56,189,248,0.3)",
              }}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-sky-400 animate-pulse" />
              ✨ AI-Powered Document Intelligence · RAG Technology
            </span>
          </div>

          {/* Main heading */}
          <h2
            ref={setRef("heroHeading")}
            className={`text-6xl md:text-8xl font-light ${gradientText} leading-tight`}
          >
            EmbedMindAI
          </h2>

          {/* Sub-heading */}
          <p
            ref={setRef("heroParagraph")}
            className="max-w-xl sm:max-w-2xl text-gray-300 sm:text-base md:text-lg lg:text-xl font-light leading-relaxed"
          >
            Upload any PDF and instantly get AI-powered answers using Retrieval-Augmented Generation.
            Ask questions, extract insights, and learn from your documents like never before.
          </p>

          {/* Stats */}
          <div
            ref={setRef("heroStats")}
            className="flex flex-wrap justify-center gap-8 text-center"
          >
            {[
              { value: "6", label: "RAG Pipeline Steps" },
              { value: "100%", label: "Local Processing" },
              { value: "∞", label: "PDF Size Support" },
            ].map((stat) => (
              <div key={stat.label} className="flex flex-col items-center gap-0.5">
                <span className={`text-3xl font-light ${gradientText}`}>{stat.value}</span>
                <span className="text-xs text-white/40 font-light">{stat.label}</span>
              </div>
            ))}
          </div>

          {/* CTA Buttons */}
          <div
            ref={setRef("heroButtons")}
            className="flex flex-col sm:flex-row justify-center items-center gap-4 w-full sm:w-auto"
          >
            <button
              id="hero-get-started-btn"
              onClick={handleSignUpClick}
              className="w-full sm:w-auto px-8 py-3 text-sm sm:text-base font-medium rounded-full transition-all duration-200"
              style={{
                background: "linear-gradient(135deg, #0ea5e9, #2563eb)",
                boxShadow: "0 4px 24px rgba(14,165,233,0.4)",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.boxShadow = "0 6px 32px rgba(14,165,233,0.6)")}
              onMouseLeave={(e) => (e.currentTarget.style.boxShadow = "0 4px 24px rgba(14,165,233,0.4)")}
            >
              Get Started Free →
            </button>
            <button
              id="hero-signin-btn"
              onClick={handleLoginClick}
              className="w-full sm:w-auto px-8 py-3 text-sm sm:text-base font-light rounded-full transition-all duration-200 text-white/70 hover:text-white"
              style={{
                border: "1px solid rgba(255,255,255,0.15)",
                background: "rgba(255,255,255,0.04)",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = "rgba(56,189,248,0.4)";
                e.currentTarget.style.background = "rgba(56,189,248,0.06)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "rgba(255,255,255,0.15)";
                e.currentTarget.style.background = "rgba(255,255,255,0.04)";
              }}
            >
              Sign In with Google
            </button>
          </div>

          {/* Trust badges */}
          <p className="text-xs text-white/25 font-light">
            No credit card required · Open source · MIT licensed
          </p>
        </div>

        {/* ── Animated "How it Works" section ─────────────────────────── */}
        <div className="relative mt-[-80px] z-10 flex justify-center" id="how-it-works">
          <div ref={setRef("mainContent")} className="w-full">
            <div
              ref={setRef("animatedContent")}
              className="border-[5px] border-white/20 rounded-[5rem] overflow-hidden h-full w-full bg-black/65 scale-[0.6] px-10 py-10"
              style={{ transformOrigin: "center center", willChange: "transform" }}
            >
              <div className="w-full h-full">
                <section className="md:px-10 md:py-10">
                  <h2 className={`text-3xl md:text-5xl font-light mb-8 ${gradientText} opacity-100`}>
                    How it Works
                  </h2>

                  <div className="grid grid-cols-1 md:grid-cols-4 gap-20">
                    {/* Left: Scrolling Steps */}
                    <div
                      ref={setRef("stepsContainer")}
                      className="space-y-10 col-span-2 text-gray-300 text-base md:text-lg leading-relaxed"
                    >
                      <div ref={setRef("leftColumn")}>
                        <p className="text-xl md:text-2xl font-extralight text-gray-300 mb-10">
                          RAG – Retrieval-Augmented Generation – combines document retrieval with
                          AI-generated answers, allowing the system to find relevant information
                          from your uploaded PDF and respond to your questions in a clear,
                          contextual way.
                        </p>
                        <h3 className="text-xl md:text-2xl font-extralight text-sky-400 mb-2">
                          Step 1: Upload PDFs
                        </h3>
                        <p className="font-extralight">
                          Users begin by uploading their study materials in PDF format. These
                          files typically contain handwritten notes, lecture slides, or textbook
                          content that the system will analyze.
                        </p>
                      </div>

                      <div>
                        <h3 className="text-xl md:text-2xl font-extralight text-sky-400 mb-2">
                          Step 2: Text Extraction with PDFPlumber
                        </h3>
                        <p className="font-extralight">
                          Once the PDF is uploaded, a Python library called{" "}
                          <strong>pdfplumber</strong> is used to extract raw text from each
                          page. This library accurately pulls out text content, preserving layout
                          where possible.
                        </p>
                      </div>

                      <div>
                        <h3 className="text-xl md:text-2xl font-extralight text-sky-400 mb-2">
                          Step 3: Chunking the Extracted Text
                        </h3>
                        <p className="font-extralight">
                          The extracted text is broken into smaller segments called chunks. This
                          is done using specialized logic from <strong>LangChain</strong> or{" "}
                          <strong>NLTK</strong>, helping efficiently retrieve relevant
                          information later.
                        </p>
                      </div>

                      <div>
                        <h3 className="text-xl md:text-2xl font-extralight text-sky-400 mb-2">
                          Step 4: Embedding with Sentence Transformers
                        </h3>
                        <p className="font-extralight">
                          Each chunk is passed through a <strong>Sentence Transformer</strong>{" "}
                          model that converts natural language into high-dimensional vectors.
                          These embeddings capture semantic meaning and are stored in ChromaDB.
                        </p>
                      </div>

                      <div>
                        <h3 className="text-xl md:text-2xl font-extralight text-sky-400 mb-2">
                          Step 5: Query Processing and Similarity Search
                        </h3>
                        <p className="font-extralight">
                          When a user enters a question, it is also converted into an embedding.
                          A similarity search using <strong>cosine similarity</strong> retrieves
                          the most relevant chunks based on semantic closeness.
                        </p>
                      </div>

                      <div>
                        <h3 className="text-xl md:text-2xl font-extralight text-sky-400 mb-2">
                          Step 6: Answer Generation with LLM
                        </h3>
                        <p className="font-extralight">
                          The retrieved chunks and the user&apos;s query are sent to{" "}
                          <strong>Google Gemini</strong>, which generates a precise, contextually
                          accurate answer tailored to the uploaded material.
                        </p>
                      </div>
                    </div>

                    {/* Right: Orbiting Circles */}
                    <div className="relative col-span-2 flex items-center justify-center md:scale-150">
                      <div
                        ref={setRef("orbitingCircles")}
                        className="flex h-[400px] w-full flex-col items-center justify-center overflow-hidden"
                      >
                        <OrbitingCircles iconSize={100}>
                          <File />
                          <Settings />
                          <File />
                          <File />
                          <Settings />
                        </OrbitingCircles>
                        <OrbitingCircles iconSize={20} radius={100} reverse speed={1}>
                          <File />
                          <Settings />
                          <File />
                          <File />
                          <Settings />
                        </OrbitingCircles>
                      </div>
                    </div>
                  </div>
                </section>
              </div>
            </div>
          </div>
        </div>

        {/* ── Features Section ─────────────────────────────────────────── */}
        <section
          id="features"
          ref={setRef("featuresSection")}
          className="relative px-6 md:px-16 py-24 max-w-7xl mx-auto"
        >
          {/* Section heading */}
          <div className="text-center mb-16">
            <span
              className="inline-block px-4 py-1 rounded-full text-xs text-sky-400 mb-4"
              style={{ background: "rgba(56,189,248,0.1)", border: "1px solid rgba(56,189,248,0.25)" }}
            >
              Why EmbedMindAI
            </span>
            <h2 className={`text-4xl md:text-5xl font-light ${gradientText} mb-4`}>
              Built for Intelligence
            </h2>
            <p className="text-white/40 font-light text-lg max-w-xl mx-auto">
              Every feature designed to make document understanding fast, private, and insightful.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feat, i) => (
              <div
                key={i}
                className="feature-card glass-card rounded-2xl p-6 transition-all duration-300"
              >
                <div
                  className="w-12 h-12 rounded-xl flex items-center justify-center text-2xl mb-5"
                  style={{ background: "rgba(56,189,248,0.08)", border: "1px solid rgba(56,189,248,0.15)" }}
                >
                  {feat.icon}
                </div>
                <h3 className="text-lg font-medium text-white mb-2">{feat.title}</h3>
                <p className="text-sm text-white/50 leading-relaxed font-light">{feat.description}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ── Tech Stack Section ───────────────────────────────────────── */}
        <section
          id="tech"
          ref={setRef("techSection")}
          className="px-6 md:px-16 py-20 max-w-7xl mx-auto"
        >
          <div className="text-center mb-12">
            <h2 className={`text-3xl md:text-4xl font-light ${gradientText} mb-3`}>
              Powered By
            </h2>
            <p className="text-white/40 font-light">
              Production-grade tools, thoughtfully assembled.
            </p>
          </div>

          <div className="flex flex-wrap justify-center gap-3">
            {techStack.map((tech) => (
              <span
                key={tech.label}
                className="tech-badge px-5 py-2.5 rounded-full text-sm font-light transition-all duration-200 cursor-default"
                style={{
                  background: "rgba(255,255,255,0.04)",
                  border: `1px solid ${tech.color}30`,
                  color: tech.color === "#ffffff" ? "rgba(255,255,255,0.8)" : tech.color,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = `${tech.color}15`;
                  e.currentTarget.style.borderColor = `${tech.color}60`;
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "rgba(255,255,255,0.04)";
                  e.currentTarget.style.borderColor = `${tech.color}30`;
                }}
              >
                {tech.label}
              </span>
            ))}
          </div>
        </section>

        {/* ── CTA Section ──────────────────────────────────────────────── */}
        <section
          ref={setRef("ctaSection")}
          className="px-6 md:px-16 py-24 flex justify-center"
        >
          <div
            className="relative max-w-3xl w-full rounded-3xl overflow-hidden px-10 py-16 text-center"
            style={{
              background: "linear-gradient(135deg, rgba(14,165,233,0.12) 0%, rgba(37,99,235,0.12) 100%)",
              border: "1px solid rgba(56,189,248,0.2)",
              boxShadow: "0 0 80px rgba(56,189,248,0.08)",
            }}
          >
            {/* Decorative glow */}
            <div
              className="absolute top-0 left-1/2 -translate-x-1/2 w-64 h-px"
              style={{ background: "linear-gradient(90deg, transparent, #38bdf8, transparent)" }}
            />
            <span className="text-5xl mb-6 block">🚀</span>
            <h2 className={`text-4xl md:text-5xl font-light ${gradientText} mb-4`}>
              Ready to Get Started?
            </h2>
            <p className="text-white/50 text-lg font-light mb-8 max-w-md mx-auto">
              Upload your first PDF and experience the power of AI-driven document intelligence in seconds.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button
                id="cta-get-started-btn"
                onClick={handleSignUpClick}
                className="px-10 py-3.5 rounded-full font-medium text-base transition-all duration-200"
                style={{
                  background: "linear-gradient(135deg, #0ea5e9, #2563eb)",
                  boxShadow: "0 4px 24px rgba(14,165,233,0.45)",
                }}
                onMouseEnter={(e) => (e.currentTarget.style.boxShadow = "0 6px 36px rgba(14,165,233,0.65)")}
                onMouseLeave={(e) => (e.currentTarget.style.boxShadow = "0 4px 24px rgba(14,165,233,0.45)")}
              >
                Start for Free →
              </button>
              <a
                href="https://github.com/Adhi1755"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center gap-2 px-10 py-3.5 rounded-full font-light text-base text-white/70 hover:text-white transition-all duration-200"
                style={{ border: "1px solid rgba(255,255,255,0.15)", background: "rgba(255,255,255,0.04)" }}
              >
                <Github size={18} />
                View on GitHub
              </a>
            </div>
          </div>
        </section>

        {/* ── Footer ───────────────────────────────────────────────────── */}
        <footer
          ref={setRef("footer")}
          className="border-t mt-auto"
          style={{ background: "rgba(0,0,0,0.8)", borderColor: "rgba(255,255,255,0.08)" }}
        >
          <div className="max-w-7xl mx-auto px-6 py-8">
            <div className="flex flex-col md:flex-row items-center justify-between gap-6">
              {/* Brand */}
              <div className="flex flex-col items-center md:items-start gap-1">
                <span className={`text-lg font-light ${gradientText}`}>EmbedMindAI</span>
                <p className="text-xs text-white/30">
                  AI-Powered Document Intelligence · © 2025
                </p>
              </div>

              {/* Links */}
              <div className="flex items-center gap-6 text-sm text-white/40">
                <a href="#how-it-works" className="hover:text-sky-400 transition">How It Works</a>
                <a href="#features" className="hover:text-sky-400 transition">Features</a>
                <a href="#tech" className="hover:text-sky-400 transition">Tech Stack</a>
              </div>

              {/* Social Icons */}
              <div className="flex items-center gap-3">
                <a
                  href="https://github.com/Adhi1755"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="GitHub"
                  className="p-2 rounded-full transition-all duration-300 text-white/50 hover:text-white hover:bg-white/10"
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                  </svg>
                </a>
                <a
                  href="https://www.linkedin.com/in/adithyanagamuneendran/"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="LinkedIn"
                  className="p-2 rounded-full transition-all duration-300 text-white/50 hover:text-white hover:bg-white/10"
                >
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                  </svg>
                </a>
                <a
                  href="https://mail.google.com/mail/?view=cm&fs=1&to=adithya1755@gmail.com"
                  aria-label="Email"
                  className="p-2 rounded-full transition-all duration-300 text-white/50 hover:text-white hover:bg-white/10"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                </a>
              </div>
            </div>

            <div className="mt-6 pt-6 border-t border-white/5 text-center">
              <p className="text-xs text-white/20 font-light">
                Designed & Built by{" "}
                <a
                  href="https://www.linkedin.com/in/adithyanagamuneendran/"
                  className="text-sky-400/60 hover:text-sky-400 transition"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Adithya
                </a>{" "}
                · Open Source on{" "}
                <a
                  href="https://github.com/Adhi1755"
                  className="text-sky-400/60 hover:text-sky-400 transition"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  GitHub
                </a>
              </p>
            </div>
          </div>
        </footer>
      </div>
    </main>
  );
};

export default LandingPage;