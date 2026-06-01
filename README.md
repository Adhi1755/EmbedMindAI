<div align="center">

# EmbedMindAI

### AI-Powered PDF Intelligence Platform

Upload any PDF. Ask anything. Get precise, document-grounded answers powered by a multi-stage RAG pipeline and Google Gemini.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-15-000000?style=flat-square&logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6B35?style=flat-square)](https://trychroma.com)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?style=flat-square&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)

[Report Bug](https://github.com/Adhi1755/EmbedMindAI/issues) · [Request Feature](https://github.com/Adhi1755/EmbedMindAI/issues)

</div>

---

## Overview

**EmbedMindAI** is a full-stack Retrieval-Augmented Generation (RAG) application. Upload a PDF, ask questions in natural language, and receive accurate answers grounded in the document — not hallucinated.

The retrieval engine uses a three-stage pipeline: semantic vector search, multi-signal reranking, and Maximal Marginal Relevance (MMR) diversity selection, before passing context to Gemini 2.5 Flash for answer generation.

```
PDF Upload  →  Extract  →  Chunk  →  Embed  →  ChromaDB
                                                    ↓
Question  →  Embed Query  →  Search  →  Rerank  →  MMR  →  Gemini  →  Answer
```

---

## Features

- **Multi-stage RAG** — semantic search + multi-signal reranking + MMR diversity selection
- **Google OAuth** — secure sign-in with sessions persisted in MongoDB
- **Real-time progress** — WebSocket updates during PDF processing
- **Formatted responses** — Markdown rendering with syntax-highlighted code blocks
- **Animated landing page** — GSAP scroll animations and particle canvas
- **Docker-ready** — full stack runs with a single `docker compose up`
- **CPU-optimized** — no GPU required; PyTorch CPU build, E5-small embeddings

---

## Tech Stack

### Backend
| | |
|---|---|
| **FastAPI** | Async REST API + WebSocket |
| **ChromaDB** | Persistent local vector database |
| **intfloat/e5-small-v2** | SentenceTransformer embedding model |
| **PDFPlumber** | Layout-aware PDF text extraction |
| **Google Gemini 2.5 Flash** | LLM answer generation |
| **MongoDB + PyMongo** | User session persistence |
| **Python 3.11 / uvicorn** | Runtime + ASGI server |

### Frontend
| | |
|---|---|
| **Next.js 15 + React 19** | Framework with App Router |
| **TypeScript** | Type safety |
| **Tailwind CSS v4** | Styling |
| **GSAP** | Animations and scroll effects |
| **react-markdown** | Render AI responses |
| **react-syntax-highlighter** | Code block highlighting |
| **Zustand** | Upload state management |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Docker Network                     │
│                                                     │
│   ┌──────────────┐   HTTP    ┌──────────────────┐   │
│   │  Next.js 15  │ ◄───────► │    FastAPI       │   │
│   │  Port 3000   │  WS       │    Port 8000     │   │
│   └──────────────┘           └────────┬─────────┘   │
│                                       │             │
│                          ┌────────────┼──────────┐  │
│                          │            │          │  │
│                    ┌─────▼───┐  ┌─────▼──┐  ┌───▼─┐ │
│                    │ChromaDB │  │MongoDB │  │ GCP │ │
│                    │(volume) │  │Sessions│  │APIs │ │
│                    └─────────┘  └────────┘  └─────┘ │
└─────────────────────────────────────────────────────┘
```

---

## Project Structure

```
EmbedMindAI/
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   ├── main.py              # FastAPI app, routes, WebSocket
│   ├── rag_logic.py         # Full RAG pipeline
│   ├── auth.py              # Google OAuth + MongoDB sessions
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── Dockerfile
    ├── package.json
    └── app/
        ├── page.tsx         # Root — redirects to landing
        ├── index.tsx        # Landing page (GSAP animations)
        ├── chat/
        │   ├── page.tsx     # Chat interface
        │   ├── ChatBox.tsx  # Message input + upload button
        │   ├── FileUpload.tsx
        │   └── Header.tsx
        ├── components/
        │   ├── Markdown.tsx         # Markdown + syntax highlighting
        │   ├── ProgressCheck.tsx    # WebSocket progress indicator
        │   └── stores/UploadStore.ts
        └── lib/
            └── api.ts       # API client
```

---

## Getting Started

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) — recommended
- Or: Python 3.11+, Node.js 20+, MongoDB running locally

### 1. Clone and configure

```bash
git clone https://github.com/Adhi1755/EmbedMindAI.git
cd EmbedMindAI
cp backend/.env.example backend/.env
```

Edit `backend/.env`:

```env
# Google OAuth — https://console.cloud.google.com/
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-your-secret

# Gemini API — https://aistudio.google.com/apikey
GOOGLE_API_KEY=AIzaSy...

# MongoDB (optional — defaults to localhost:27017)
MONGODB_URI=mongodb://localhost:27017/embedmindai

# CORS origins (comma-separated)
ALLOWED_ORIGINS=http://localhost:3000

# OAuth redirect URLs
BACKEND_URL=http://localhost:8000
FRONTEND_URL=http://localhost:3000
```

### 2. Google OAuth setup

1. Open [Google Cloud Console](https://console.cloud.google.com/) → **APIs & Services** → **Credentials**
2. Create an **OAuth 2.0 Client ID** (Web application)
3. Add authorized redirect URI: `http://localhost:8000/auth/callback`
4. Paste the Client ID and Secret into `.env`

### 3. Run with Docker

```bash
docker compose up --build
```

> First build takes ~5–8 minutes (downloads PyTorch CPU and the embedding model). Subsequent builds are fast.

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Swagger docs | http://localhost:8000/docs |

### 4. Run locally (development)

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

---

## API Reference

### Auth
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/auth/login` | Redirect to Google OAuth |
| `GET` | `/auth/callback` | OAuth callback — sets session cookie |
| `GET` | `/auth/me` | Return current user |
| `GET` | `/auth/logout` | Clear session |

### Documents
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload and process a PDF |
| `DELETE` | `/delete?filename={name}` | Remove PDF and its vectors |

### Query
| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ask` | Ask a question (`query` form field) |
| `GET` | `/api/test` | Health check |

### WebSocket
| Endpoint | Description |
|---|---|
| `WS /ws/progress` | Real-time PDF processing progress |

---

## RAG Pipeline

### Chunking

Documents are split into overlapping chunks (~1,200 chars, 200-char overlap). Boundary detection respects semantic structure — preferred split points in order:

1. Double newlines (paragraph breaks)
2. Sentence-ending punctuation followed by a capital letter
3. Colon + newline (definitions, lists)
4. Semicolons
5. Word boundaries (last resort)

Chunks under 100 characters are discarded.

### Retrieval Scoring

After semantic search returns 20 candidates, each is re-scored:

```
score = 0.40 × semantic_similarity   # cosine distance from ChromaDB
      + 0.25 × term_overlap          # query term coverage in chunk
      + 0.20 × term_density          # query term frequency
      + 0.10 × position_score        # earlier appearances weighted higher
      + 0.05 × completeness_score    # longer chunks preferred
```

### MMR Selection

Top candidates are filtered through Maximal Marginal Relevance to balance relevance with diversity:

```
MMR(cᵢ) = λ · relevance(cᵢ) − (1 − λ) · max_similarity(cᵢ, selected_chunks)

λ = 0.7   →   70% relevance, 30% diversity
```

Final context window: 7 diverse, high-scoring chunks passed to Gemini.

---

## Roadmap

- [x] Core RAG pipeline
- [x] Google OAuth + MongoDB sessions
- [x] ChromaDB vector storage
- [x] Docker Compose deployment
- [x] Real-time WebSocket progress
- [ ] Streaming LLM responses (SSE)
- [ ] Per-user PDF library
- [ ] Persistent chat history
- [ ] Conversation memory (multi-turn)
- [ ] Document summary on upload
- [ ] Suggested follow-up questions
- [ ] Rate limiting

---

## Author

**Adithya Nagamuneendran**

[![GitHub](https://img.shields.io/badge/GitHub-Adhi1755-181717?style=flat-square&logo=github)](https://github.com/Adhi1755)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-adithyanagamuneendran-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/adithyanagamuneendran/)

---

## License

MIT
