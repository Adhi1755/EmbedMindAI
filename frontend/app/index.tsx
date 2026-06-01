"use client";
import React, { useEffect, useState, useRef } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const LOGOS = ["Stripe","Notion","Vercel","Shopify","Linear","Figma","Loom","Retool"];

const STEPS = [
  { num:"01", emoji:"📁", title:"Embed Your Sources",          body:"Upload PDFs, paste URLs, connect APIs or databases. We chunk, vectorize, and index everything automatically." },
  { num:"02", emoji:"🧠", title:"Build Semantic Memory",       body:"Our engine stores embeddings in a high-performance vector store with metadata filtering for pinpoint retrieval." },
  { num:"03", emoji:"💬", title:"Query With Natural Language", body:"Ask anything. Get answers grounded in your exact data — with sources cited, hallucinations minimized." },
];

const DEEP = [
  { tag:"RETRIEVAL",    title:"Blazing-Fast Semantic Search",        desc:"Sub-100ms similarity search across millions of vectors. Hybrid search: dense + sparse retrieval for maximum relevance.",             bullets:["< 80ms average query latency","Hybrid dense + sparse retrieval","Metadata filtering & re-ranking"],    rev:false },
  { tag:"INTELLIGENCE", title:"Context-Aware Answer Generation",     desc:"Not just retrieval — full RAG pipeline with LLM reasoning grounded in your knowledge base. Cite sources automatically.",              bullets:["Full RAG pipeline with source citations","Hallucination minimization","Configurable LLM backends"],        rev:true  },
  { tag:"INTEGRATIONS", title:"Drop-In Integrations",                desc:"REST API, Python SDK, JS/TS client. Plug into Slack, Notion, Intercom, or any custom stack in minutes.",                              bullets:["REST API + Python & JS SDKs","Webhook & streaming support","Pre-built connectors for 20+ apps"],            rev:false },
];

const STATS = [
  { val:80,   pre:"<", suf:"ms", lbl:"Average query latency", dec:false },
  { val:99.9, pre:"",  suf:"%",  lbl:"Uptime SLA",            dec:true  },
  { val:10,   pre:"",  suf:"M+", lbl:"Vectors indexed daily", dec:false },
  { val:4200, pre:"",  suf:"+",  lbl:"Active developers",     dec:false },
];

const TESTI = [
  { name:"Sarah K.",  role:"Head of DevRel @ Syntax.io",  text:"EmbedMind cut our support ticket volume by 60% in the first week. Our docs finally talk back.",                                                          color:"#6366F1" },
  { name:"Marcus T.", role:"CTO @ Loopback Labs",          text:"The retrieval accuracy is unreal. We tried 4 other tools — this is the only one that doesn't hallucinate our pricing.",                                 color:"#06B6D4" },
  { name:"Priya M.",  role:"Senior Engineer @ Meridian",   text:"5-minute setup, production-ready API, and the best DX I've used in years. Genuinely impressive.",                                                       color:"#F59E0B" },
  { name:"Alex R.",   role:"Founder @ Quickpath AI",       text:"We replaced our entire internal search infrastructure with EmbedMind in a weekend. The latency improvements alone paid for the subscription.",          color:"#10B981" },
  { name:"Jordan L.", role:"ML Engineer @ Tensora",        text:"Hybrid retrieval — dense plus sparse — is a game changer for our legal doc search. Accuracy went from 71% to 94%.",                                     color:"#8B5CF6" },
  { name:"Mei C.",    role:"Product Lead @ Stackward",     text:"Our customers asked for AI search for months. EmbedMind let us ship it in 3 days. They love it.",                                                       color:"#EC4899" },
];

const PLANS = [
  { name:"Starter",    pm:0,    py:0,    desc:"Perfect for side projects and exploration", feats:["500K vectors","1 project","Community support","REST API access","Basic analytics"],                                   cta:"Get Started Free",  hot:false, badge:"" },
  { name:"Pro",        pm:49,   py:39,   desc:"For teams building production-grade AI",    feats:["10M vectors","5 projects","Email support","Custom prompts","Advanced analytics","Priority retrieval"],                 cta:"Start Pro Trial",   hot:true,  badge:"Most Popular" },
  { name:"Enterprise", pm:null, py:null, desc:"For organizations at scale",                feats:["Unlimited vectors","Unlimited projects","SLA guarantee","Dedicated infra","SSO & SAML","24/7 support"],                cta:"Contact Sales",     hot:false, badge:"" },
];

const FAQS = [
  { q:"Is my data private and secure?",      a:"Absolutely. Your documents are processed in an isolated environment and never used for training. All data is encrypted at rest and in transit. You can delete your data at any time." },
  { q:"What file types do you support?",     a:"We support PDF, DOCX, TXT, MD, HTML, CSV, and JSON out of the box. URLs and APIs can be connected via our crawler or webhook ingestion pipeline." },
  { q:"Which LLMs can I use?",               a:"EmbedMind supports GPT-4o, Claude 3.5, Gemini 2.5 Flash, Mistral, and any OpenAI-compatible endpoint. You can also bring your own API key." },
  { q:"How does pricing scale with usage?",  a:"The free tier gives you 500K vectors. Pro starts at $49/mo for 10M vectors. Beyond that, vectors are $0.10 per million. No surprise bills — you set a hard cap." },
  { q:"Are there API rate limits?",          a:"Free: 60 requests/min. Pro: 500 requests/min. Enterprise: unlimited with dedicated infrastructure. All tiers support burst capacity." },
  { q:"Can I self-host EmbedMind?",          a:"Yes! Our Docker image and Kubernetes helm chart are available on GitHub. The self-hosted version is fully open-source under the MIT license." },
  { q:"How accurate is the retrieval?",      a:"Our hybrid retrieval achieves >90% recall@5 on standard RAG benchmarks. Accuracy depends on your data quality — we provide eval tools to measure and improve." },
  { q:"How long does setup take?",           a:"Most developers have their first query running in under 5 minutes using our quickstart guide. Full production deployment typically takes an afternoon." },
];

const FCOLS = [
  { title:"Product",   links:["Features","Pricing","Changelog","Roadmap","Status"] },
  { title:"Resources", links:["Documentation","API Reference","Blog","Tutorials","Community"] },
  { title:"Company",   links:["About","Careers","Press Kit","Partners","Contact"] },
  { title:"Legal",     links:["Privacy Policy","Terms of Service","Cookie Policy","Security","DPA"] },
];

const PARTS = [
  {x:12,y:22,s:4,d:0,   dur:3.5},{x:83,y:17,s:3,d:1.2,dur:4.1},
  {x:88,y:65,s:5,d:0.4, dur:3.2},{x:7, y:73,s:3,d:2.1,dur:5.0},
  {x:45,y:84,s:4,d:0.7, dur:4.4},{x:93,y:40,s:3,d:1.6,dur:3.9},
  {x:22,y:50,s:5,d:0.2, dur:4.7},{x:65,y:20,s:3,d:2.3,dur:3.6},
  {x:35,y:78,s:4,d:1.0, dur:4.0},{x:72,y:57,s:3,d:0.6,dur:4.8},
];

// ─── Component ─────────────────────────────────────────────────────────────

export default function LandingPage() {
  const [scrolled,   setScrolled]   = useState(false);
  const [mobOpen,    setMobOpen]    = useState(false);
  const [yearly,     setYearly]     = useState(false);
  const [openFaq,    setOpenFaq]    = useState<number|null>(null);
  const statsRef   = useRef<HTMLDivElement>(null);
  const statsRan   = useRef(false);

  useEffect(() => {
    const fn = () => setScrolled(window.scrollY > 80);
    window.addEventListener("scroll", fn, { passive:true });
    return () => window.removeEventListener("scroll", fn);
  }, []);

  useEffect(() => {
    const obs = new IntersectionObserver(
      (es) => es.forEach(e => e.isIntersecting && e.target.classList.add("vis")),
      { threshold:0.12, rootMargin:"0px 0px -32px 0px" }
    );
    document.querySelectorAll(".rev").forEach(el => obs.observe(el));
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    const el = statsRef.current; if (!el) return;
    const obs = new IntersectionObserver((es) => {
      if (!es[0].isIntersecting || statsRan.current) return;
      statsRan.current = true;
      document.querySelectorAll("[data-cnt]").forEach(node => {
        const target = parseFloat(node.getAttribute("data-cnt")!);
        const isDec  = node.getAttribute("data-dec") === "1";
        const pre    = node.getAttribute("data-pre") || "";
        const suf    = node.getAttribute("data-suf") || "";
        const t0 = performance.now();
        const tick = (now:number) => {
          const p = Math.min((now-t0)/2000,1), e = 1-Math.pow(1-p,3);
          (node as HTMLElement).textContent = pre+(isDec?(e*target).toFixed(1):Math.round(e*target).toLocaleString())+suf;
          if (p<1) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      });
    }, { threshold:0.4 });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <div id="eml">
    <style>{`
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,700;12..96,800&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@400;500&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
:root{
  --bg:#050508;--indigo:#6366F1;--cyan:#06B6D4;--amber:#F59E0B;
  --txt:#FFF;--body:#94A3B8;--muted:#64748B;
  --surf:rgba(255,255,255,0.04);--bdr:rgba(255,255,255,0.08);
  --grad:linear-gradient(135deg,#6366F1 0%,#06B6D4 100%);
  --glow:0 0 40px rgba(99,102,241,0.38);--glow2:0 0 64px rgba(99,102,241,0.6);
  --fd:'Bricolage Grotesque',system-ui,sans-serif;
  --fb:'DM Sans',system-ui,sans-serif;
  --fm:'JetBrains Mono',monospace;
  --ease:cubic-bezier(0.16,1,0.3,1);
}
#eml{background:var(--bg);color:var(--txt);font-family:var(--fb);overflow-x:hidden;-webkit-font-smoothing:antialiased}
.gt{background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.rev{opacity:0;transform:translateY(32px);transition:opacity 600ms var(--ease),transform 600ms var(--ease)}
.rev.vis{opacity:1;transform:translateY(0)}
.d1{transition-delay:80ms}.d2{transition-delay:160ms}.d3{transition-delay:240ms}.d4{transition-delay:320ms}
.wrap{max-width:1200px;margin:0 auto;padding:0 48px}
.sec{padding:120px 0}
.pill{display:inline-block;background:var(--surf);border:1px solid var(--bdr);border-radius:999px;padding:6px 18px;font-size:13px;font-weight:500;color:var(--body);margin-bottom:24px}
.sh2{font-family:var(--fd);font-size:clamp(34px,4vw,58px);font-weight:800;line-height:1.08;letter-spacing:-0.04em;margin-bottom:48px}
/* NAV */
#nav{position:fixed;top:0;left:0;right:0;z-index:100;padding:0 48px;transition:background 0.3s,border-color 0.3s}
#nav.sc{background:rgba(5,5,8,0.86);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);border-bottom:1px solid rgba(255,255,255,0.06)}
.ni{max-width:1200px;margin:0 auto;display:flex;align-items:center;height:72px}
.lw{display:flex;align-items:baseline;text-decoration:none;font-family:var(--fd);font-weight:800;font-size:20px;letter-spacing:-0.04em;flex-shrink:0;margin-right:48px}
.le{color:#fff}.lm{background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.lai{font-size:10px;font-weight:700;background:var(--amber);color:#000;border-radius:4px;padding:1px 5px;margin-left:3px;vertical-align:super;-webkit-text-fill-color:#000}
.nls{display:flex;gap:4px;flex:1}
.nl{color:var(--body);font-size:14px;text-decoration:none;padding:7px 13px;border-radius:8px;transition:color 0.15s,background 0.15s}
.nl:hover{color:#fff;background:var(--surf)}
.nas{display:flex;gap:10px;align-items:center}
.bsi{color:var(--body);font-size:14px;font-weight:500;text-decoration:none;padding:9px 18px;border-radius:10px;transition:color 0.15s}
.bsi:hover{color:#fff}
.bgs{background:var(--grad);color:#fff;font-size:14px;font-weight:600;text-decoration:none;padding:9px 22px;border-radius:10px;box-shadow:var(--glow);transition:box-shadow 0.2s,transform 0.15s;white-space:nowrap}
.bgs:hover{box-shadow:var(--glow2);transform:scale(1.03)}
.bgs:active{transform:scale(0.97)}
.hbtn{display:none;flex-direction:column;gap:5px;background:none;border:none;cursor:pointer;margin-left:auto;padding:8px}
.hl{width:22px;height:2px;background:#fff;border-radius:2px;transition:all 0.25s}
.hl.o1{transform:rotate(45deg) translate(4px,4px)}.hl.o2{opacity:0;transform:scaleX(0)}.hl.o3{transform:rotate(-45deg) translate(4px,-4px)}
.mm{background:rgba(5,5,8,0.97);backdrop-filter:blur(20px);padding:16px 48px 28px;display:flex;flex-direction:column;gap:4px;border-bottom:1px solid var(--bdr)}
.mml{color:var(--body);font-size:16px;text-decoration:none;padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.05);transition:color 0.15s}
.mml:hover{color:#fff}
/* HERO */
#hero{min-height:100vh;display:flex;align-items:center;justify-content:center;position:relative;overflow:hidden;padding:110px 24px 80px}
.orb1{position:absolute;width:900px;height:900px;border-radius:50%;background:radial-gradient(circle,rgba(99,102,241,0.18) 0%,rgba(6,182,212,0.07) 40%,transparent 70%);top:50%;left:35%;transform:translate(-50%,-50%);animation:orbD 8s ease-in-out infinite;pointer-events:none;filter:blur(2px)}
@keyframes orbD{0%,100%{transform:translate(-50%,-50%) scale(1) translate(0,0)}33%{transform:translate(-50%,-50%) scale(1.07) translate(-28px,22px)}66%{transform:translate(-50%,-50%) scale(0.94) translate(22px,-28px)}}
.dgrid{position:absolute;inset:0;pointer-events:none;background-image:radial-gradient(circle,rgba(255,255,255,0.04) 1px,transparent 1px);background-size:24px 24px}
.hin{position:relative;z-index:2;display:flex;flex-direction:column;align-items:center;text-align:center;max-width:900px;width:100%;margin:0 auto}
.hbadge{position:relative;overflow:hidden;display:inline-flex;align-items:center;gap:8px;background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.35);border-radius:999px;padding:8px 22px;font-size:14px;color:var(--body);font-weight:500;margin-bottom:36px;animation:fD 0.6s 0.2s both}
.shim{position:absolute;top:0;left:-100%;width:100%;height:100%;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.12),transparent);animation:sh 2.5s 1s infinite}
@keyframes sh{from{left:-100%}to{left:200%}}
@keyframes fD{from{opacity:0;transform:translateY(-16px)}to{opacity:1;transform:translateY(0)}}
.hh1{font-family:var(--fd);font-size:clamp(40px,7.5vw,92px);font-weight:800;line-height:1.03;letter-spacing:-0.045em;margin-bottom:24px;display:flex;flex-direction:column;gap:2px}
.hl1{animation:fU 0.7s 0.3s both}.hl2{animation:fU 0.7s 0.42s both}.hl3{animation:fU 0.7s 0.54s both}
@keyframes fU{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
.hsub{font-size:18px;color:var(--body);line-height:1.75;max-width:560px;margin-bottom:40px;animation:fU 0.7s 0.7s both}
.hctas{display:flex;gap:14px;justify-content:center;flex-wrap:wrap;margin-bottom:24px;animation:fU 0.7s 0.9s both}
.bgl{background:var(--grad);color:#fff;font-family:var(--fb);font-size:16px;font-weight:600;text-decoration:none;padding:16px 32px;border-radius:12px;display:inline-flex;align-items:center;gap:8px;box-shadow:var(--glow);transition:box-shadow 0.2s,transform 0.15s}
.bgl:hover{box-shadow:var(--glow2);transform:scale(1.03)}.bgl:active{transform:scale(0.97)}
.bghl{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);color:#fff;font-family:var(--fb);font-size:16px;font-weight:500;text-decoration:none;padding:16px 32px;border-radius:12px;transition:background 0.2s,border-color 0.2s,transform 0.15s}
.bghl:hover{background:rgba(255,255,255,0.1);border-color:rgba(255,255,255,0.22);transform:scale(1.02)}
.proof{font-size:13px;color:var(--muted);margin-bottom:64px;animation:fU 0.7s 1.1s both}
.proof strong{color:var(--body)}
/* Hero visual */
.hvw{position:relative;width:100%;max-width:860px;animation:hcI 0.9s 1.1s both}
@keyframes hcI{from{opacity:0;transform:translateY(40px) scale(0.97)}to{opacity:1;transform:translateY(0) scale(1)}}
.fb_{position:absolute;z-index:10;background:rgba(8,8,16,0.92);border:1px solid var(--bdr);backdrop-filter:blur(12px);border-radius:10px;padding:8px 14px;font-size:12px;font-weight:600;white-space:nowrap}
.fb1{top:-20px;right:10%;animation:bf1 4s ease-in-out infinite}
.fb2{top:28%;left:-44px;animation:bf2 5s 0.5s ease-in-out infinite}
.fb3{bottom:-16px;right:18%;animation:bf3 4.5s 1s ease-in-out infinite}
@keyframes bf1{0%,100%{transform:translateY(0) rotate(-1deg)}50%{transform:translateY(-10px) rotate(1deg)}}
@keyframes bf2{0%,100%{transform:translateY(0) rotate(1deg)}50%{transform:translateY(-8px) rotate(-1deg)}}
@keyframes bf3{0%,100%{transform:translateY(0) rotate(-2deg)}50%{transform:translateY(-12px) rotate(2deg)}}
.hcard{background:rgba(8,8,18,0.88);backdrop-filter:blur(20px);border:1px solid rgba(99,102,241,0.25);border-radius:20px;overflow:hidden;box-shadow:0 0 80px rgba(99,102,241,0.18),0 40px 80px rgba(0,0,0,0.65);animation:fY 3s ease-in-out infinite}
@keyframes fY{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
.chard{background:rgba(255,255,255,0.03);border-bottom:1px solid var(--bdr);padding:12px 18px;display:flex;align-items:center;gap:12px}
.cdots{display:flex;gap:6px}
.dot_{width:10px;height:10px;border-radius:50%}
.dr{background:#FF5F57}.dy{background:#FEBC2E}.dg{background:#28C840}
.ctabs{display:flex;gap:4px}
.ctab{font-family:var(--fm);font-size:11px;color:var(--muted);padding:4px 12px;border-radius:6px;cursor:pointer}
.ctab.on{background:rgba(99,102,241,0.15);color:rgba(99,102,241,0.9)}
.cbod{display:grid;grid-template-columns:1fr 1fr;min-height:180px}
.cpan{padding:20px;border-right:1px solid var(--bdr)}
.ccode{font-family:var(--fm);font-size:12px;color:var(--body);line-height:1.7;text-align:left;white-space:pre}
.rpan{padding:20px}
.rhd{display:flex;align-items:center;gap:8px;font-family:var(--fm);font-size:11px;color:var(--muted);margin-bottom:14px}
.gdot{width:7px;height:7px;border-radius:50%;background:#28C840;display:inline-block}
.lat{color:#28C840;font-weight:600}
.rtxt{font-size:13px;color:var(--body);line-height:1.65;margin-bottom:14px;text-align:left}
.rsrc{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
.rsl{font-family:var(--fm);font-size:10px;color:var(--muted);text-transform:uppercase}
.rsrc_{font-family:var(--fm);font-size:10px;background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.25);color:rgba(99,102,241,0.9);padding:2px 8px;border-radius:4px}
.rcon{display:flex;align-items:center;gap:10px}
.rbw{flex:1;height:4px;border-radius:2px;background:rgba(255,255,255,0.06);overflow:hidden}
.rbi{height:100%;border-radius:2px;background:var(--grad)}
.rcv{font-family:var(--fm);font-size:10px;color:var(--muted);white-space:nowrap}
/* LOGOS */
#logos{padding:60px 0;border-top:1px solid var(--bdr);border-bottom:1px solid var(--bdr);overflow:hidden}
.llbl{text-align:center;font-size:11px;font-weight:600;letter-spacing:0.12em;color:var(--muted);margin-bottom:28px}
.mqw{overflow:hidden}
.mqt{display:flex;animation:mq 30s linear infinite;width:max-content}
.mqw:hover .mqt{animation-play-state:paused}
@keyframes mq{from{transform:translateX(0)}to{transform:translateX(-50%)}}
.li_{padding:0 52px;color:rgba(255,255,255,0.2);font-family:var(--fd);font-size:18px;font-weight:700;letter-spacing:-0.02em;white-space:nowrap;transition:color 0.2s;cursor:default}
.li_:hover{color:rgba(255,255,255,0.55)}
/* HOW */
#how{padding:120px 0}
.stp{display:grid;grid-template-columns:repeat(3,1fr);gap:28px;position:relative;margin-top:48px}
.sc_{background:var(--surf);border:1px solid var(--bdr);border-radius:16px;padding:32px;position:relative;transition:border-color 0.2s,transform 0.2s,box-shadow 0.2s}
.sc_:hover{border-color:rgba(99,102,241,0.4);transform:translateY(-4px);box-shadow:0 20px 60px rgba(0,0,0,0.4)}
.scon{position:absolute;top:42%;right:-14px;width:28px;height:1px;z-index:2;background:linear-gradient(90deg,rgba(99,102,241,0.5),rgba(6,182,212,0.2))}
.snum{font-family:var(--fd);font-size:44px;font-weight:800;letter-spacing:-0.05em;line-height:1;background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:16px}
.sico{font-size:32px;margin-bottom:16px}
.stit{font-family:var(--fd);font-size:20px;font-weight:700;margin-bottom:12px}
.sbod{font-size:14px;color:var(--body);line-height:1.72}
/* FEATURES */
#feats{padding:40px 0 120px}
.fr_{display:grid;grid-template-columns:1fr 1fr;gap:80px;align-items:center;padding:80px 0;border-top:1px solid var(--bdr)}
.fr_:first-child{border-top:none;padding-top:0}
.rr_ .ft_{order:2}.rr_ .fv_{order:1}
.ftag_{display:inline-block;font-size:11px;font-weight:700;letter-spacing:0.12em;border:1px solid rgba(99,102,241,0.4);border-radius:999px;padding:4px 14px;background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:16px}
.ftt_{font-family:var(--fd);font-size:clamp(26px,2.8vw,38px);font-weight:800;letter-spacing:-0.03em;line-height:1.1;margin-bottom:16px}
.fd_{font-size:16px;color:var(--body);line-height:1.75;margin-bottom:28px}
.fbl{list-style:none;display:flex;flex-direction:column;gap:12px}
.fbl li{display:flex;align-items:center;gap:10px;font-size:15px;color:var(--body)}
.ck{color:#06B6D4;font-weight:700;font-size:16px}
.fgc_{background:var(--surf);border:1px solid var(--bdr);border-radius:16px;overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,0.4)}
.fch_{background:rgba(255,255,255,0.03);border-bottom:1px solid var(--bdr);padding:12px 16px;display:flex;align-items:center}
.fvs_{display:flex;align-items:center;gap:12px}
.fvsl{font-size:12px;color:var(--muted);width:140px;flex-shrink:0}
.fvsb{flex:1;height:6px;border-radius:3px;background:rgba(255,255,255,0.06);overflow:hidden}
.fvsi{height:100%;border-radius:3px;background:var(--grad)}
.fvsv{font-family:var(--fm);font-size:12px;color:var(--body);width:38px;text-align:right}
.fpul{margin-top:20px;display:flex;align-items:center;gap:8px}
.pd_{width:8px;height:8px;border-radius:50%;background:#06B6D4;animation:pd_ 2s ease-in-out infinite}
@keyframes pd_{0%,100%{box-shadow:0 0 0 0 rgba(6,182,212,0.4)}50%{box-shadow:0 0 0 8px rgba(6,182,212,0)}}
.pt_{font-family:var(--fm);font-size:12px;color:var(--muted)}
/* STATS */
#stats{padding:100px 0;position:relative;overflow:hidden;background:rgba(99,102,241,0.03)}
.sg_{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:600px;height:300px;background:radial-gradient(ellipse,rgba(99,102,241,0.15) 0%,transparent 70%);pointer-events:none;filter:blur(30px)}
.sth{font-family:var(--fd);font-size:clamp(28px,3.5vw,50px);font-weight:800;letter-spacing:-0.04em;text-align:center;margin-bottom:60px}
.stg{display:grid;grid-template-columns:repeat(4,1fr)}
.sti{text-align:center;padding:0 40px;position:relative}
.stv{font-family:var(--fd);font-size:clamp(42px,5vw,72px);font-weight:800;letter-spacing:-0.04em;line-height:1;display:block;background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:12px}
.stl{font-family:var(--fm);font-size:12px;color:var(--muted);letter-spacing:0.02em}
.sdiv{position:absolute;right:0;top:10%;bottom:10%;width:1px;background:var(--bdr)}
/* TESTIMONIALS */
#testi{padding:120px 0}
.tg_{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}
.tc_{background:var(--surf);border:1px solid var(--bdr);border-radius:16px;padding:28px;transition:transform 0.2s,box-shadow 0.2s,border-color 0.2s}
.tc_:hover{transform:translateY(-4px);box-shadow:0 24px 60px rgba(0,0,0,0.4);border-color:rgba(255,255,255,0.14)}
.tst{color:var(--amber);font-size:14px;margin-bottom:16px;letter-spacing:2px}
.ttx{font-size:15px;color:var(--body);line-height:1.7;margin-bottom:24px;font-style:italic}
.tau{display:flex;align-items:center;gap:12px}
.tav{width:40px;height:40px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:16px;color:#fff;flex-shrink:0}
.tnm{font-size:14px;font-weight:600;margin-bottom:2px}
.tro{font-size:12px;color:var(--muted)}
/* PRICING */
#price{padding:120px 0}
.ptog{display:flex;align-items:center;gap:14px;justify-content:center;margin-bottom:56px;font-size:15px;flex-wrap:wrap}
.ton{color:#fff;font-weight:600}.toff{color:var(--muted)}
.togb{width:44px;height:24px;border-radius:12px;background:rgba(99,102,241,0.25);border:none;cursor:pointer;position:relative;flex-shrink:0}
.togp{position:absolute;top:2px;left:2px;width:20px;height:20px;border-radius:50%;background:var(--grad);transition:transform 0.25s cubic-bezier(0.34,1.56,0.64,1);display:block}
.sbdg{background:rgba(6,182,212,0.15);border:1px solid rgba(6,182,212,0.3);color:#06B6D4;font-size:11px;font-weight:600;padding:2px 9px;border-radius:999px;margin-left:6px}
.pg2{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;align-items:center}
.pc_{background:var(--surf);border:1px solid var(--bdr);border-radius:20px;padding:36px;position:relative;transition:transform 0.2s}
.pc_.hot{border-color:rgba(99,102,241,0.45);box-shadow:0 0 60px rgba(99,102,241,0.2);transform:scale(1.04);background:rgba(99,102,241,0.05)}
.pc_.hot:hover{transform:scale(1.06)}
.hbdg{position:absolute;top:-14px;left:50%;transform:translateX(-50%);background:var(--grad);color:#fff;font-size:12px;font-weight:700;padding:4px 18px;border-radius:999px;white-space:nowrap}
.pnm{font-family:var(--fd);font-size:22px;font-weight:800;letter-spacing:-0.02em;margin-bottom:16px}
.ppr{display:flex;align-items:baseline;gap:4px;margin-bottom:12px}
.ppv{font-family:var(--fd);font-size:52px;font-weight:800;letter-spacing:-0.04em;line-height:1;background:var(--grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.ppm{font-size:16px;color:var(--muted)}
.pds{font-size:14px;color:var(--muted);margin-bottom:28px;line-height:1.5}
.pfl{list-style:none;display:flex;flex-direction:column;gap:12px;margin-bottom:32px}
.pfl li{display:flex;align-items:center;gap:10px;font-size:14px;color:var(--body)}
.bng{display:block;text-align:center;text-decoration:none;background:var(--grad);color:#fff;font-size:15px;font-weight:600;padding:14px;border-radius:12px;box-shadow:0 0 30px rgba(99,102,241,0.3);transition:box-shadow 0.2s,transform 0.15s}
.bng:hover{box-shadow:0 0 48px rgba(99,102,241,0.55);transform:scale(1.02)}
.bgh_{display:block;text-align:center;text-decoration:none;background:transparent;color:#fff;border:1px solid rgba(255,255,255,0.12);font-size:15px;font-weight:500;padding:14px;border-radius:12px;transition:background 0.2s,border-color 0.2s}
.bgh_:hover{background:rgba(255,255,255,0.07);border-color:rgba(255,255,255,0.25)}
/* FAQ */
#faq{padding:120px 0}
.flist{display:flex;flex-direction:column;gap:4px}
.fi_{border:1px solid var(--bdr);border-radius:12px;overflow:hidden;transition:border-color 0.2s}
.fi_.op{border-color:rgba(99,102,241,0.45);border-left-width:3px;border-left-color:var(--indigo)}
.fq_{width:100%;display:flex;align-items:center;justify-content:space-between;padding:20px 24px;background:none;border:none;color:#fff;font-size:16px;font-weight:500;cursor:pointer;text-align:left;font-family:var(--fb);transition:background 0.15s}
.fq_:hover{background:rgba(255,255,255,0.03)}
.fchv{font-size:18px;color:var(--muted);transition:transform 0.3s ease;flex-shrink:0;margin-left:16px}
.fans{overflow:hidden;transition:max-height 0.35s var(--ease)}
.fans p{padding:0 24px 20px;font-size:15px;color:var(--body);line-height:1.75}
/* FINAL CTA */
#fcta{min-height:520px;display:flex;align-items:center;justify-content:center;text-align:center;position:relative;overflow:hidden;padding:120px 24px}
.corb{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:700px;height:400px;border-radius:50%;background:radial-gradient(ellipse,rgba(99,102,241,0.2) 0%,rgba(6,182,212,0.1) 45%,transparent 70%);animation:corb_ 5s ease-in-out infinite;filter:blur(20px);pointer-events:none}
@keyframes corb_{0%,100%{transform:translate(-50%,-50%) scale(1)}50%{transform:translate(-50%,-50%) scale(1.15)}}
.part_{position:absolute;border-radius:50%;background:radial-gradient(circle,rgba(99,102,241,0.8),rgba(6,182,212,0.4));animation:pf_ linear infinite;pointer-events:none}
@keyframes pf_{0%,100%{transform:translateY(0) scale(1);opacity:0.6}50%{transform:translateY(-22px) scale(1.2);opacity:1}}
.ctain{position:relative;z-index:2}
.ctah{font-family:var(--fd);font-size:clamp(40px,6vw,76px);font-weight:800;letter-spacing:-0.04em;line-height:1.07;margin-bottom:20px}
.ctas{font-size:18px;color:var(--body);margin:0 auto 44px;max-width:480px;line-height:1.7}
.ctaa{display:flex;gap:16px;justify-content:center;flex-wrap:wrap}
/* FOOTER */
#foot{border-top:1px solid var(--bdr);padding:72px 48px 40px}
.fin{max-width:1200px;margin:0 auto}
.ftop{display:grid;grid-template-columns:1fr 2.5fr;gap:64px;margin-bottom:56px}
.ftag_{font-size:14px;color:var(--muted);margin-top:12px;max-width:220px;line-height:1.6}
.fcols{display:grid;grid-template-columns:repeat(4,1fr);gap:32px}
.fch_{font-size:13px;font-weight:600;color:#fff;margin-bottom:20px}
.fcl{display:block;font-size:13px;color:var(--muted);text-decoration:none;margin-bottom:12px;transition:color 0.15s}
.fcl:hover{color:rgba(255,255,255,0.7)}
.fbot{display:flex;justify-content:space-between;align-items:center;padding-top:28px;border-top:1px solid var(--bdr);font-size:13px;color:var(--muted)}
.fsoc{display:flex;gap:24px}
.fsl{color:var(--muted);text-decoration:none;font-size:14px;font-weight:500;transition:color 0.15s}
.fsl:hover{color:#fff}
/* RESPONSIVE */
@media(max-width:1024px){
  .fr_{grid-template-columns:1fr;gap:40px}
  .rr_ .ft_,.rr_ .fv_{order:unset}
  .pc_.hot{transform:none}
  .pg2{grid-template-columns:1fr;max-width:440px;margin:0 auto}
}
@media(max-width:768px){
  .wrap{padding:0 24px}.sec{padding:72px 0}
  #nav,#foot{padding-left:24px;padding-right:24px}
  .nls,.nas{display:none}
  .hbtn{display:flex}.mm{padding-left:24px;padding-right:24px}
  .hh1{font-size:clamp(36px,9vw,56px)}.hsub{font-size:16px}
  .hctas{flex-direction:column;align-items:center}
  .cbod{grid-template-columns:1fr}.cpan{border-right:none;border-bottom:1px solid var(--bdr)}
  .fb2,.fb3{display:none}
  .stp{grid-template-columns:1fr}.scon{display:none}
  .stg{grid-template-columns:1fr 1fr;gap:48px 20px}.sdiv{display:none}
  .tg_{grid-template-columns:1fr}
  .ftop{grid-template-columns:1fr;gap:32px}
  .fcols{grid-template-columns:repeat(2,1fr);gap:24px}
  .fbot{flex-direction:column;gap:16px;text-align:center}
  .lw{margin-right:0}
}
@media(max-width:480px){
  .stg{grid-template-columns:1fr}.ctaa{flex-direction:column;align-items:center}
}
    `}</style>

      {/* NAV */}
      <nav id="nav" className={scrolled ? "sc" : ""}>
        <div className="ni">
          <a href="/" className="lw">
            <span className="le">Embed</span><span className="lm">Mind</span><span className="lai">AI</span>
          </a>
          <div className="nls">
            {["Features","Pricing","Docs","Blog","Enterprise"].map(l=>(
              <a key={l} href="#" className="nl">{l}</a>
            ))}
          </div>
          <div className="nas">
            <a href={`${API}/auth/login`} className="bsi">Sign in</a>
            <a href={`${API}/auth/login`} className="bgs">Get Started Free →</a>
          </div>
          <button className="hbtn" onClick={()=>setMobOpen(!mobOpen)} aria-label="Toggle menu">
            <span className={`hl${mobOpen?" o1":""}`}/><span className={`hl${mobOpen?" o2":""}`}/><span className={`hl${mobOpen?" o3":""}`}/>
          </button>
        </div>
        {mobOpen&&(
          <div className="mm">
            {["Features","Pricing","Docs","Blog","Enterprise"].map(l=>(
              <a key={l} href="#" className="mml">{l}</a>
            ))}
            <a href={`${API}/auth/login`} className="bgs" style={{display:"block",textAlign:"center",marginTop:"16px"}}>Get Started Free →</a>
          </div>
        )}
      </nav>

      {/* HERO */}
      <section id="hero">
        <div className="orb1"/>
        <div className="dgrid"/>
        <div className="hin">
          <div className="hbadge">
            <span className="shim"/>
            <span>✦ Now with GPT-4o Embeddings</span>
          </div>
          <h1 className="hh1">
            <span className="hl1">Turn Any Knowledge Into</span>
            <span className="hl2"><span className="gt">Intelligent Answers</span> —</span>
            <span className="hl3">Instantly.</span>
          </h1>
          <p className="hsub">EmbedMind AI lets you embed your docs, PDFs, websites, and databases into a semantic knowledge engine that answers questions with razor-sharp precision.</p>
          <div className="hctas">
            <a href={`${API}/auth/login`} className="bgl">Start Building Free →</a>
            <a href="#how" className="bghl">Watch Demo ▶</a>
          </div>
          <p className="proof">★★★★★ &nbsp;Loved by <strong>4,200+</strong> developers · No credit card required · Setup in 5 minutes</p>
          <div className="hvw">
            <div className="fb_ fb1">⚡ 99ms response</div>
            <div className="fb_ fb2">🤖 GPT-4o</div>
            <div className="fb_ fb3">📐 512-dim vectors</div>
            <div className="hcard">
              <div className="chard">
                <div className="cdots"><span className="dot_ dr"/><span className="dot_ dy"/><span className="dot_ dg"/></div>
                <div className="ctabs"><span className="ctab on">query.ts</span><span className="ctab">response.json</span></div>
              </div>
              <div className="cbod">
                <div className="cpan">
                  <pre className="ccode">{`const result = await embedmind.query({
  query: "What's our refund policy?",
  topK: 5,
  filter: { source: "docs" }
});

console.log(result.answer);`}</pre>
                </div>
                <div className="rpan">
                  <div className="rhd"><span className="gdot"/> Response · <span className="lat">73ms</span></div>
                  <p className="rtxt">Based on your documentation, customers may request a full refund within 30 days of purchase. Digital products are subject to evaluation before approval...</p>
                  <div className="rsrc">
                    <span className="rsl">Sources</span>
                    <span className="rsrc_">docs/refunds.pdf</span>
                    <span className="rsrc_">help/billing.md</span>
                  </div>
                  <div className="rcon">
                    <div className="rbw"><div className="rbi" style={{width:"94%"}}/></div>
                    <span className="rcv">94% confidence</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* LOGOS */}
      <section id="logos">
        <p className="llbl">TRUSTED BY TEAMS AT</p>
        <div className="mqw">
          <div className="mqt">
            {[...LOGOS,...LOGOS].map((l,i)=><span key={i} className="li_">{l}</span>)}
          </div>
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section id="how" className="sec">
        <div className="wrap" style={{textAlign:"center"}}>
          <div className="pill rev">✦ Simple Setup</div>
          <h2 className="sh2 rev d1" style={{textAlign:"center"}}>From raw data to<br/><span className="gt">smart answers</span> in 3 steps.</h2>
          <div className="stp">
            {STEPS.map((s,i)=>(
              <div key={i} className={`sc_ rev d${i+1}`}>
                {i<2&&<div className="scon"/>}
                <div className="snum">{s.num}</div>
                <div className="sico">{s.emoji}</div>
                <h3 className="stit">{s.title}</h3>
                <p className="sbod">{s.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* FEATURES DEEP */}
      <section id="feats" className="sec">
        <div className="wrap">
          {DEEP.map((f,i)=>(
            <div key={i} className={`fr_${f.rev?" rr_":""}`}>
              <div className={`ft_ rev${f.rev?" d2":""}`}>
                <span className="ftag_">{f.tag}</span>
                <h3 className="ftt_">{f.title}</h3>
                <p className="fd_">{f.desc}</p>
                <ul className="fbl">
                  {f.bullets.map((b,j)=><li key={j}><span className="ck">✓</span>{b}</li>)}
                </ul>
              </div>
              <div className={`fv_ rev${f.rev?"":" d2"}`}>
                <div className="fgc_">
                  <div className="fch_">
                    <div className="cdots"><span className="dot_ dr"/><span className="dot_ dy"/><span className="dot_ dg"/></div>
                    <span style={{fontFamily:"var(--fm)",fontSize:"11px",color:"var(--muted)",marginLeft:"10px"}}>{f.tag.toLowerCase()}.ts</span>
                  </div>
                  <div style={{padding:"24px"}}>
                    <div className="fvs_"><span className="fvsl">Query vectors</span><div className="fvsb"><div className="fvsi" style={{width:`${72+i*9}%`}}/></div><span className="fvsv">{[78,88,96][i]}%</span></div>
                    <div className="fvs_" style={{marginTop:"14px"}}><span className="fvsl">Retrieval accuracy</span><div className="fvsb"><div className="fvsi" style={{width:`${64+i*11}%`}}/></div><span className="fvsv">{[71,84,94][i]}%</span></div>
                    <div className="fvs_" style={{marginTop:"14px"}}><span className="fvsl">Confidence score</span><div className="fvsb"><div className="fvsi" style={{width:`${82+i*4}%`}}/></div><span className="fvsv">{[89,92,96][i]}%</span></div>
                    <div className="fpul"><span className="pd_"/><span className="pt_">Live · {[73,61,88][i]}ms avg latency</span></div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* STATS */}
      <section id="stats" ref={statsRef}>
        <div className="sg_"/>
        <div className="wrap">
          <h2 className="sth rev">Built for performance at scale.</h2>
          <div className="stg">
            {STATS.map((s,i)=>(
              <div key={i} className={`sti rev d${i+1}`}>
                <span className="stv" data-cnt={s.val} data-pre={s.pre} data-suf={s.suf} data-dec={s.dec?"1":"0"}>
                  {s.pre}{s.val}{s.suf}
                </span>
                <div className="stl">{s.lbl}</div>
                {i<3&&<div className="sdiv"/>}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* TESTIMONIALS */}
      <section id="testi" className="sec">
        <div className="wrap">
          <h2 className="sh2 rev" style={{textAlign:"center",marginBottom:"56px"}}>What developers are saying.</h2>
          <div className="tg_">
            {TESTI.map((t,i)=>(
              <div key={i} className={`tc_ rev d${(i%3)+1}`}>
                <div className="tst">★★★★★</div>
                <p className="ttx">&ldquo;{t.text}&rdquo;</p>
                <div className="tau">
                  <div className="tav" style={{background:t.color}}>{t.name[0]}</div>
                  <div><p className="tnm">{t.name}</p><p className="tro">{t.role}</p></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* PRICING */}
      <section id="price" className="sec">
        <div className="wrap">
          <div style={{textAlign:"center"}}>
            <div className="pill rev">Pricing</div>
            <h2 className="sh2 rev d1" style={{textAlign:"center"}}>Simple, transparent pricing.</h2>
          </div>
          <div className="ptog rev d2">
            <span className={!yearly?"ton":"toff"}>Monthly</span>
            <button className="togb" onClick={()=>setYearly(!yearly)} aria-label="Toggle yearly">
              <span className="togp" style={{transform:yearly?"translateX(20px)":"translateX(0)"}}/>
            </button>
            <span className={yearly?"ton":"toff"}>Yearly <span className="sbdg">Save 20%</span></span>
          </div>
          <div className="pg2">
            {PLANS.map((p,i)=>(
              <div key={i} className={`pc_ rev d${i+1}${p.hot?" hot":""}`}>
                {p.hot&&<div className="hbdg">{p.badge}</div>}
                <h3 className="pnm">{p.name}</h3>
                <div className="ppr">
                  {p.pm===null?<span className="ppv">Custom</span>:p.pm===0?<span className="ppv">Free</span>:<><span className="ppv">${yearly?p.py:p.pm}</span><span className="ppm">/mo</span></>}
                </div>
                <p className="pds">{p.desc}</p>
                <ul className="pfl">{p.feats.map((f,j)=><li key={j}><span className="ck">✓</span>{f}</li>)}</ul>
                <a href={`${API}/auth/login`} className={p.hot?"bng":"bgh_"}>{p.cta}</a>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* FAQ */}
      <section id="faq" className="sec">
        <div className="wrap" style={{maxWidth:"800px"}}>
          <h2 className="sh2 rev" style={{textAlign:"center",marginBottom:"48px"}}>Frequently asked questions.</h2>
          <div className="flist">
            {FAQS.map((f,i)=>(
              <div key={i} className={`fi_ rev d${(i%4)+1}${openFaq===i?" op":""}`}>
                <button className="fq_" onClick={()=>setOpenFaq(openFaq===i?null:i)} aria-expanded={openFaq===i}>
                  <span>{f.q}</span>
                  <span className="fchv" style={{transform:openFaq===i?"rotate(180deg)":"rotate(0deg)"}}>▾</span>
                </button>
                <div className="fans" style={{maxHeight:openFaq===i?"300px":"0"}}>
                  <p>{f.a}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* FINAL CTA */}
      <section id="fcta">
        <div className="corb"/>
        {PARTS.map((p,i)=>(
          <div key={i} className="part_" style={{left:`${p.x}%`,top:`${p.y}%`,width:`${p.s}px`,height:`${p.s}px`,animationDelay:`${p.d}s`,animationDuration:`${p.dur}s`}}/>
        ))}
        <div className="ctain">
          <h2 className="ctah rev">Your knowledge base,<br/><span className="gt">finally intelligent.</span></h2>
          <p className="ctas rev d1">Join 4,200+ developers building smarter products with EmbedMind AI.</p>
          <div className="ctaa rev d2">
            <a href={`${API}/auth/login`} className="bgl">Start Building Free →</a>
            <a href="#" className="bghl">Talk to an Expert</a>
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer id="foot">
        <div className="fin">
          <div className="ftop">
            <div>
              <a href="/" className="lw"><span className="le">Embed</span><span className="lm">Mind</span><span className="lai">AI</span></a>
              <p className="ftag_">AI-powered semantic knowledge. Upload, embed, and query your data in minutes.</p>
            </div>
            <div className="fcols">
              {FCOLS.map(col=>(
                <div key={col.title}>
                  <h4 className="fch_">{col.title}</h4>
                  {col.links.map(l=><a key={l} href="#" className="fcl">{l}</a>)}
                </div>
              ))}
            </div>
          </div>
          <div className="fbot">
            <p>© 2026 EmbedMind AI · All rights reserved</p>
            <div className="fsoc">
              <a href="#" className="fsl">𝕏</a>
              <a href="https://github.com/Adhi1755/EmbedMindAI" target="_blank" rel="noreferrer" className="fsl">GitHub</a>
              <a href="#" className="fsl">Discord</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}