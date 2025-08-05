"use client";

import React, { useEffect, useRef, useCallback, useMemo } from "react";
import { Github, Link } from "lucide-react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { OrbitingCircles } from "./components/orbiting-circles";
import { File, Settings, Search } from "lucide-react";
import { useRouter } from 'next/navigation';
import FlipCard from "./components/Filp_card";

gsap.registerPlugin(ScrollTrigger);

const gradientText = "bg-gradient-to-r from-sky-400 to-blue-600 bg-clip-text text-transparent";

const LandingPage = () => {
  // Refs consolidated for better organization
  const refs = useRef({
    animatedContent: null,
    heroSection: null,
    heroHeading: null,
    heroParagraph: null,
    heroButtons: null,
    leftColumn: null,
    orbitingCircles: null,
    header: null,
    footer: null,
    mainContent: null,
    stepsContainer: null
  });

  const router = useRouter();
  
  // Animation controllers for cleanup
  const animationControllers = useRef({
    pageLoadTl: null,
    scrollTriggers: []
  });

  const handleLoginClick = useCallback(() => {
    window.location.href = "http://localhost:8000/auth/login";
  }, []);

  // Optimized page load animations with proper sequencing
  const initPageLoadAnimations = useCallback(() => {
    const { current: r } = refs;
    const ctx = gsap.context(() => {
      
      // Set initial states with transform3d for GPU acceleration
      gsap.set([r.header, r.heroHeading, r.heroParagraph, r.heroButtons], {
        opacity: 0,
        y: 30,
        force3D: true
      });

      gsap.set(r.mainContent, {
        opacity: 0,
        y: 50,
        force3D: true
      });

      gsap.set(r.footer, {
        opacity: 0,
        y: 20,
        force3D: true
      });

      // Main timeline with proper easing and performance optimization
      const tl = gsap.timeline({ 
        onComplete: () => {
          // Clear transforms after animation for better performance
          gsap.set([r.header, r.heroHeading, r.heroParagraph, r.heroButtons, r.mainContent, r.footer], {
            clearProps: "transform"
          });
        }
      });

      animationControllers.current.pageLoadTl = tl;

      tl.to(r.header, {
  opacity: 1,
  y: 0,
  duration: 0.8,
  ease: "back.out(1.2)"
}, 0)

.to(r.heroHeading, {
  opacity: 1,
  y: 0,
  duration: 0.8,
  ease: "back.out(1.2)"
}, 0.05)

.to(r.heroParagraph, {
  opacity: 1,
  y: 0,
  duration: 0.8,
  ease: "power2.out"
}, 0.1)

.to(r.mainContent, {
  opacity: 1,
  y: 0,
  duration: 0.8,
  ease: "power2.out"
}, 0.15)

.to(r.heroButtons, {
  opacity: 1,
  y: 0,
  duration: 0.8,
  ease: "power2.out"
}, 0.2)

.to(r.footer, {
  opacity: 1,
  y: 0,
  duration: 0.8,
  ease: "power2.out"
}, 0.25);
      // Background elements animation
      const canvas = document.getElementById("starCanvas");
      if (canvas) {
        gsap.fromTo(canvas, {
          opacity: 0
        }, {
          opacity: 1,
          duration: 1.5,
          ease: "power2.out",
          delay: 0.3
        });
      }

    });

    return ctx;
  }, []);

  // Optimized hero scroll animations
  const initHeroScrollAnimations = useCallback(() => {
    const { current: r } = refs;
    
    const ctx = gsap.context(() => {
      const heroElements = [r.heroHeading, r.heroParagraph, r.heroButtons].filter(Boolean);
      
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
              overwrite: "auto"
            });
          }
        });
        
        animationControllers.current.scrollTriggers.push(st);
      }
    });

    return ctx;
  }, []);

  // Optimized content scale animation
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
          animation: gsap.fromTo(r.animatedContent, {
            scale: 0.7,
            borderRadius: "5rem",
            borderWidth: "5px"
          }, {
            scale: 1,
            borderRadius: "0rem",
            borderWidth: "0px",
            transformOrigin: "center center"
          })
        });
        
        animationControllers.current.scrollTriggers.push(st);
      }
    });

    return ctx;
  }, []);

  // Optimized orbiting circles animation
  const initOrbitingAnimation = useCallback(() => {
    const { current: r } = refs;
    
    const ctx = gsap.context(() => {
      if (r.leftColumn && r.orbitingCircles) {
        const st = ScrollTrigger.create({
          trigger: r.leftColumn,
          start: "top bottom",
          end: "bottom top",
          scrub: 2,
          animation: gsap.fromTo(r.orbitingCircles, {
            y: -200
          }, {
            y: 0,
            ease: "none"
          })
        });
        
        animationControllers.current.scrollTriggers.push(st);
      }
    });

    return ctx;
  }, []);

  // Optimized steps animation with intersection observer pattern
  const initStepsAnimation = useCallback(() => {
    const { current: r } = refs;
    
    const ctx = gsap.context(() => {
      if (r.stepsContainer) {
        const steps = r.stepsContainer.querySelectorAll(':scope > div');
        
        steps.forEach((step, index) => {
          if (index === 0) {
            // First step always visible
            gsap.set(step, { opacity: 1, x: 0 });
          } else {
            // Set initial state
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
                clearProps: "transform"
              })
            });
            
            animationControllers.current.scrollTriggers.push(st);
          }
        });
      }
    });

    return ctx;
  }, []);

  // Canvas animation with optimized performance
  const initStarryBackground = useCallback(() => {
    const canvas = document.getElementById("starCanvas");
    if (!canvas) return null;

    const ctx = canvas.getContext("2d");
    let animationId;
    let stars = [];
    let w = window.innerWidth;
    let h = window.innerHeight;
    
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr);

    const numStars = Math.min(120, Math.floor((w * h) / 10000)); // Responsive star count

    // Initialize stars
    for (let i = 0; i < numStars; i++) {
      stars.push({
        x: Math.random() * w,
        y: Math.random() * h,
        r: Math.random() * 1.2 + 0.3,
        dx: (Math.random() - 0.5) * 0.4,
        dy: (Math.random() - 0.5) * 0.4,
        opacity: Math.random() * 0.5 + 0.5
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
      canvas.style.width = w + 'px';
      canvas.style.height = h + 'px';
      ctx.scale(dpr, dpr);
    };

    window.addEventListener("resize", handleResize);
    
    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  // Main effect for initializing all animations
  useEffect(() => {
    const contexts = [
      initPageLoadAnimations(),
      initHeroScrollAnimations(),
      initContentScaleAnimation(),
      initOrbitingAnimation(),
      initStepsAnimation()
    ];

    const starCleanup = initStarryBackground();

    // Cleanup function
    return () => {
      contexts.forEach(ctx => ctx?.revert());
      
      // Kill all scroll triggers
      animationControllers.current.scrollTriggers.forEach(st => st.kill());
      animationControllers.current.scrollTriggers = [];
      
      // Kill main timeline
      animationControllers.current.pageLoadTl?.kill();
      
      // Canvas cleanup
      starCleanup?.();
      
      // Final GSAP cleanup
      ScrollTrigger.refresh();
    };
  }, [initPageLoadAnimations, initHeroScrollAnimations, initContentScaleAnimation, initOrbitingAnimation, initStepsAnimation, initStarryBackground]);

  // Memoized ref assignment function
  const setRef = useCallback((key) => (el) => {
    refs.current[key] = el;
  }, []);

  return (
    <main className="min-h-screen w-full bg-transparent text-white overflow-x-hidden relative">
      {/* Starry background and center glow */}
      <div className="fixed inset-0 -z-10 pointer-events-none">
        <div className="absolute inset-0 bg-black">
          <canvas id="starCanvas" className="w-full h-full" />
        </div>
        <div
          className="absolute inset-x-0 bottom-0 h-[600px] pointer-events-none z-0"
          style={{
            background: "linear-gradient(to top, rgba(0,191,255,0.3) 0%, rgba(0,191,255,0.05) 100%)",
            borderRadius: "50% 50% 0 0 / 100% 100% 0 0",
            filter: "blur(120px)",
          }}
        />
      </div>

      {/* Header */}
      <header 
        ref={setRef('header')}
        className="fixed top-0 w-full z-50 bg-black px-6 py-4 flex items-center justify-between"
      >
        <h1 className={`text-xl font-light ${gradientText}`}>EmbedMindAI</h1>
        <nav className="hidden md:flex gap-10">
          <a href="#" className="hover:text-sky-400 transition">Home</a>
          <a href="#tech" className="hover:text-sky-400 transition">Features</a>
          <a href="#rag" className="hover:text-sky-400 transition">About</a>
        </nav>
        <a
          href="https://github.com/your-repo"
          target="_blank"
          rel="noopener noreferrer"
          className="hover:text-sky-400 transition flex items-center gap-1"
        >
          <Github size={20} /> GitHub
        </a>
      </header>

      {/* Page content wrapper */}
      <div className="pt-30">
        {/* Hero Section */}
        <div
          ref={setRef('heroSection')}
          className="min-h-[30vh] md:min-h-[50vh] flex flex-col justify-center items-center gap-6 px-4 text-center"
        >
          <h2
            ref={setRef('heroHeading')}
            className={`text-6xl md:text-8xl font-light ${gradientText}`}
          >
            EmbedMindAI
          </h2>

          <p
            ref={setRef('heroParagraph')}
            className="max-w-xl sm:max-w-2xl text-gray-300 sm:text-base md:text-lg lg:text-xl font-light"
          >
            An AI-powered tool that lets you upload a PDF, ask questions from it, and learn through interactive answers.
          </p>

          <div
            ref={setRef('heroButtons')}
            className="flex flex-row sm:flex-row justify-center items-center gap-4 w-full sm:w-auto"
          >
            <button
              onClick={handleLoginClick}
              className="w-full sm:w-auto px-6 py-2 text-sm sm:text-base bg-sky-400 hover:bg-sky-600 text-white rounded-full transition"
            >
              Login with Google
            </button>
          </div>
        </div>

        {/* Animated div content */}
        <div className="relative mt-[-100px] z-10 flex justify-center">
          <div ref={setRef('mainContent')} className="w-full">
            <div
              ref={setRef('animatedContent')}
              className="border-[5px] border-white/20 rounded-[5rem] overflow-hidden h-full w-full bg-black/65 scale-[0.6] px-10 py-10"
              style={{
                transformOrigin: "center center",
                willChange: "transform",
              }}
            >
              <div className="w-full h-full">
                <section className="md:px-10 md:py-10">
                  <h2 className={`text-3xl md:text-5xl font-light mb-8 ${gradientText} opacity-100`}>
                    How it Works
                  </h2>

                  <div className="grid grid-cols-1 md:grid-cols-4 gap-20">
                    {/* Left: Scrolling Steps */}
                    <div 
                      ref={setRef('stepsContainer')}
                      className="space-y-10 col-span-2 text-gray-300 text-base md:text-lg leading-relaxed"
                    >
                      <div ref={setRef('leftColumn')}>
                        <p className="text-xl md:text-2xl font-extralight text-gray-300 mb-10">
                          RAG – Retrieval-Augmented Generation – combines document retrieval with AI-generated answers, allowing the system to find relevant information from your uploaded PDF and respond to your questions in a clear, contextual way.
                        </p>
                        <h3 className="text-xl md:text-2xl font-extralight text-sky-400 mb-2">Step 1: Upload PDFs</h3>
                        <p className="font-extralight">
                          Users begin by uploading their study materials in PDF format. These files typically contain handwritten notes, lecture slides, or textbook content that the system will analyze.
                        </p>
                      </div>

                      <div>
                        <h3 className="text-xl md:text-2xl font-extralight text-sky-400 mb-2">Step 2: Text Extraction with PDFPlumber</h3>
                        <p className="font-extralight">
                          Once the PDF is uploaded, a Python library called <strong>pdfplumber</strong> is used to extract raw text from each page. This library is capable of accurately pulling out text content, preserving the layout where possible, and ignoring irrelevant formatting or artifacts.
                        </p>
                      </div>

                      <div>
                        <h3 className="text-xl md:text-2xl font-extralight text-sky-400 mb-2">Step 3: Chunking the Extracted Text</h3>
                        <p className="font-extralight">
                          The extracted text is broken down into smaller, manageable segments called chunks. This is done using simple Python logic or specialized libraries like <strong>LangChain</strong> or <strong>NLTK</strong>, depending on the desired granularity. Chunking helps in efficiently retrieving relevant information later.
                        </p>
                      </div>

                      <div>
                        <h3 className="text-xl md:text-2xl font-extralight text-sky-400 mb-2">Step 4: Embedding with Sentence Transformers</h3>
                        <p className="font-extralight">
                          Each chunk is passed through a <strong>Sentence Transformer</strong>, a deep learning model that converts natural language into high-dimensional vectors (embeddings). These embeddings capture the semantic meaning of the text and are stored in a vector database for fast retrieval.
                        </p>
                      </div>

                      <div>
                        <h3 className="text-xl md:text-2xl font-extralight text-sky-400 mb-2">Step 5: Query Processing and Similarity Search</h3>
                        <p className="font-extralight">
                          When a user enters a question, it is also converted into an embedding using the same transformer model. A similarity search is performed between the query embedding and all stored chunk embeddings using a mathematical operation called <strong>cosine similarity</strong>. This retrieves the most relevant chunks based on semantic closeness.
                        </p>
                      </div>

                      <div>
                        <h3 className="text-xl md:text-2xl font-extralight text-sky-400 mb-2">Step 6: Answer Generation with LLM</h3>
                        <p className="font-extralight">
                          The retrieved chunks and the user's query are sent to a large language model (LLM) — such as Google Gemini — which analyzes the context and generates a precise, informative answer. The system ensures that the responses are contextually accurate and tailored to the uploaded material.
                        </p>
                      </div>
                    </div>

                    {/* Right: Animated Orbiting Circles */}
                    <div className="relative col-span-2 flex items-center justify-center md:scale-150">
                      <div 
                        ref={setRef('orbitingCircles')}
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
      </div>

      {/* Footer */}
      <footer 
        ref={setRef('footer')}
        className="bg-black/65 border-t border-white/20 mt-auto"
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex flex-col sm:flex-row items-center gap-2 sm:gap-4 text-center sm:text-left">
              <p className="text-sm text-black dark:text-white">
                © 2025 EmbedMindAI. All rights reserved.
              </p>
              <span className="hidden sm:inline text-black/50 dark:text-white/50">•</span>
              <p className="text-xs text-black/70 dark:text-white/70">
                Designed & Built by Adithya
              </p>
            </div>

            <div className="flex items-center gap-3">
              <a
                href="https://github.com/Adhi1755"
                target="_blank"
                rel="noopener noreferrer"
                className="group relative p-2 rounded-full backdrop-blur-sm bg-white/5 border border-black/10 dark:border-white/10 shadow-sm hover:shadow-md transition-all duration-300 hover:scale-110 hover:bg-black/10 dark:hover:bg-white/10"
                aria-label="GitHub"
              >
                <svg className="w-5 h-5 text-black/80 dark:text-white/80 group-hover:text-black dark:group-hover:text-white transition-colors duration-300" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
              </a>

              <a
                href="https://www.linkedin.com/in/adithyanagamuneendran/"
                target="_blank"
                rel="noopener noreferrer"
                className="group relative p-2 rounded-full backdrop-blur-sm bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10 shadow-sm hover:shadow-md transition-all duration-300 hover:scale-110 hover:bg-black/10 dark:hover:bg-white/10"
                aria-label="LinkedIn"
              >
                <svg className="w-5 h-5 text-black/80 dark:text-white/80 group-hover:text-black dark:group-hover:text-white transition-colors duration-300" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                </svg>
              </a>

              <a
                href="https://mail.google.com/mail/?view=cm&fs=1&to=adithya1755@gmail.com"
                className="group relative p-2 rounded-full backdrop-blur-sm bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10 shadow-sm hover:shadow-md transition-all duration-300 hover:scale-110 hover:bg-black/10 dark:hover:bg-white/10"
                aria-label="Email"
              >
                <svg className="w-5 h-5 text-black/80 dark:text-white/80 group-hover:text-black dark:group-hover:text-white transition-colors duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
              </a>
            </div>
          </div>
        </div>
      </footer>
    </main>
  );
};

export default LandingPage;