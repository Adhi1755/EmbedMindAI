<div align="center">

# 🧠 EmbedMindAI

### AI-Powered PDF Intelligence Platform

*Upload any PDF. Ask anything. Get precise, contextual answers powered by RAG + Google Gemini.*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-15.4-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6B35?style=for-the-badge)](https://trychroma.com)
[![Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini)

[Live Demo](#) · [Report Bug](https://github.com/Adhi1755/EmbedMindAI/issues) · [Request Feature](https://github.com/Adhi1755/EmbedMindAI/issues)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Environment Variables](#environment-variables)
  - [Run with Docker (Recommended)](#run-with-docker-recommended)
  - [Run Locally (Dev)](#run-locally-dev)
- [API Reference](#-api-reference)
- [RAG Pipeline Deep Dive](#-rag-pipeline-deep-dive)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

**EmbedMindAI** is a full-stack Retrieval-Augmented Generation (RAG) application that lets you upload any PDF document and have an intelligent conversation with it. It combines state-of-the-art embedding models, a vector database, and Google's Gemini LLM to give you accurate, document-grounded answers.

Whether you're studying research papers, analyzing legal documents, or reviewing technical manuals — EmbedMindAI turns your static PDFs into interactive knowledge bases.

<div align="center">

```
Upload PDF → Extract Text → Chunk → Embed → Store in ChromaDB
                                                     ↓
Ask Question → Embed Query → Similarity Search → Rerank → Gemini → Answer
```

</div>

---

## 🔬 How It Works

EmbedMindAI implements a **multi-stage RAG pipeline** with advanced retrieval:

### 1. 📄 Document Ingestion
- PDF text extracted using **PDFPlumber** (layout-aware extraction)
- Text cleaned and normalized (header/footer removal, whitespace normalization)

### 2. ✂️ Semantic Chunking
- Documents split into overlapping chunks (1200 chars, 200 char overlap)
- Boundary-aware splitting: prefers paragraph breaks → sentence ends → clause boundaries
- Minimum chunk quality filter (< 100 chars discarded)

### 3. 🔢 Embedding Generation
- Each chunk embedded using **`intfloat/e5-small-v2`** SentenceTransformer
- E5 model uses `passage: {text}` prefix for document embeddings
- Embeddings L2-normalized for stable cosine similarity

### 4. 💾 Vector Storage
- Embeddings + chunks stored in **ChromaDB** (persistent local storage)
- Per-document metadata (chunk index, length) stored alongside vectors

### 5. 🔍 Multi-Stage Retrieval
When you ask a question, EmbedMindAI runs a 3-stage retrieval pipeline:

| Stage | What Happens |
|-------|-------------|
| **Stage 1: Semantic Search** | Query embedded with `query: {text}` prefix, top-20 candidates retrieved from ChromaDB |
| **Stage 2: Multi-Signal Reranking** | Candidates re-scored using semantic similarity (40%) + term overlap (25%) + term density (20%) + position (10%) + completeness (5%) |
| **Stage 3: MMR Selection** | Maximal Marginal Relevance applied to pick top-7 diverse, relevant chunks |

### 6. 🤖 Answer Generation
- Retrieved chunks + user query sent to **Gemini 2.5 Flash**
- Structured prompt instructs Gemini to be comprehensive, accurate, and document-grounded
- Response streamed back to the user

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose |
|-----------|---------|
| **FastAPI** | Async REST API framework |
| **Python 3.11** | Backend runtime |
| **ChromaDB** | Persistent vector database |
| **SentenceTransformers** | `intfloat/e5-small-v2` embedding model |
| **PDFPlumber** | PDF text extraction |
| **Google Gemini 2.5 Flash** | LLM for answer generation (`google-genai`) |
| **PyMongo** | MongoDB driver for session persistence |
| **uvicorn** | ASGI server with hot-reload |

### Frontend
| Technology | Purpose |
|-----------|---------|
| **Next.js 15** | React framework with App Router |
| **React 19** | UI library |
| **TypeScript** | Type safety |
| **Tailwind CSS v4** | Utility-first styling |
| **GSAP** | Smooth animations & scroll effects |
| **react-markdown** | Render AI responses as formatted Markdown |
| **react-syntax-highlighter** | Code block syntax highlighting |
| **Zustand** | Lightweight state management |
| **Sonner** | Toast notifications |
| **Lucide React** | Icon library |

### Infrastructure
| Technology | Purpose |
|-----------|---------|
| **Docker + Docker Compose** | Containerization & orchestration |
| **MongoDB** | Persistent user session storage |
| **Google OAuth 2.0** | User authentication |

---

## ✨ Features

- 🔐 **Google OAuth Authentication** — Secure sign-in, persistent sessions via MongoDB
- 📤 **PDF Upload & Processing** — Real-time progress via WebSocket
- 🧠 **Advanced RAG Pipeline** — Multi-stage retrieval with MMR-based diversity selection
- 💬 **Interactive Chat** — Conversational interface with animated message rendering
- 📝 **Markdown Rendering** — AI responses with syntax-highlighted code blocks
- 🎨 **Animated Landing Page** — GSAP-powered scroll animations and particle canvas
- 🐳 **Docker Ready** — One command to run the full stack
- ⚡ **Fast Inference** — CPU-optimized PyTorch, E5-small model baked into image

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Docker Network                        │
│                                                             │
│  ┌─────────────────┐         ┌──────────────────────────┐  │
│  │   Frontend       │         │       Backend             │  │
│  │   Next.js 15     │◄───────►│       FastAPI             │  │
│  │   Port: 3000     │  HTTP   │       Port: 8000          │  │
│  └─────────────────┘         └──────────┬───────────────┘  │
│                                          │                   │
│                               ┌──────────┼──────────┐       │
│                               │          │          │       │
│                        ┌──────▼──┐ ┌─────▼──┐ ┌────▼────┐  │
│                        │ChromaDB │ │MongoDB │ │Google   │  │
│                        │Vectors  │ │Sessions│ │APIs     │  │
│                        │(volume) │ │        │ │OAuth+AI │  │
│                        └─────────┘ └────────┘ └─────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
EmbedMindAI/
├── 📄 docker-compose.yml          # Orchestrates frontend + backend
├── 📄 .gitignore                  # Root gitignore
├── 📄 .dockerignore               # Root Docker ignore
│
├── 🐍 backend/
│   ├── Dockerfile                 # Python 3.11-slim, CPU PyTorch
│   ├── main.py                    # FastAPI app, routes, WebSocket
│   ├── rag_logic.py               # Full RAG pipeline (embed, chunk, retrieve)
│   ├── auth.py                    # Google OAuth + MongoDB sessions
│   ├── requirements.txt           # Python dependencies
│   ├── .env.example               # Environment variable template
│   ├── .gitignore                 # Backend-specific ignores
│   └── .dockerignore
│
└── ⚛️  frontend/
    ├── Dockerfile                 # Node 20-alpine, Next.js standalone
    ├── next.config.ts             # Next.js config (standalone output)
    ├── package.json
    ├── .gitignore
    ├── .dockerignore
    └── app/
        ├── page.tsx               # Root → redirects to landing
        ├── layout.tsx             # Root layout
        ├── globals.css            # Global styles
        ├── index.tsx              # Landing page (GSAP animations)
        ├── lib/
        │   └── api.ts             # API client (env-based URLs)
        ├── chat/
        │   ├── page.tsx           # Chat interface
        │   ├── ChatBox.tsx        # Message input + PDF upload button
        │   ├── FileUpload.tsx     # Drag-and-drop PDF uploader
        │   └── Header.tsx         # Chat header with user info
        └── components/
            ├── Markdown.tsx       # Markdown + syntax highlighting renderer
            ├── ProgressCheck.tsx  # WebSocket progress indicator
            ├── orbiting-circles.tsx # Animated landing page component
            ├── sonner.tsx         # Toast provider
            └── stores/
                └── UploadStore.ts # Zustand upload state
```

---

## 🚀 Getting Started

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (recommended)
- OR: Python 3.11+, Node.js 20+, MongoDB

### Environment Variables

Copy the example file and fill in your credentials:

```bash
cp backend/.env.example backend/.env
```

```env
# backend/.env

# Google OAuth (https://console.cloud.google.com/)
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-your-secret

# Google Gemini API (https://aistudio.google.com/apikey)
GOOGLE_API_KEY=AIzaSy...

# MongoDB (optional — defaults to localhost)
MONGODB_URI=mongodb://localhost:27017/embedmindai

# CORS (comma-separated allowed origins)
ALLOWED_ORIGINS=http://localhost:3000

# URLs (for OAuth redirects)
BACKEND_URL=http://localhost:8000
FRONTEND_URL=http://localhost:3000
```

### Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/) → **APIs & Services** → **Credentials**
2. Create an **OAuth 2.0 Client ID** (Web application)
3. Add Authorized redirect URIs:
   - `http://localhost:8000/auth/callback`
4. Copy the Client ID and Secret to your `.env`

---

### Run with Docker (Recommended)

```bash
# Clone the repo
git clone https://github.com/Adhi1755/EmbedMindAI.git
cd EmbedMindAI

# Set up environment
cp backend/.env.example backend/.env
# Edit backend/.env with your API keys

# Build and start everything
docker compose up --build
```

> ⏱️ **First build takes ~5-8 minutes** — it downloads PyTorch (CPU) and the embedding model. Subsequent builds are instant due to Docker layer caching.

| Service | URL |
|---------|-----|
| 🌐 Frontend | http://localhost:3000 |
| ⚙️ Backend API | http://localhost:8000 |
| 📖 API Docs (Swagger) | http://localhost:8000/docs |

---

### Run Locally (Dev)

**Backend:**
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies (CPU-only PyTorch)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend

npm install
npm run dev
```

---

## 📡 API Reference

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/auth/login` | Redirect to Google OAuth |
| `GET` | `/auth/callback` | OAuth callback — sets session cookie |
| `GET` | `/auth/me` | Get current logged-in user |
| `GET` | `/auth/logout` | Clear session and cookie |

### Document Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload and process a PDF file |
| `DELETE` | `/delete?filename={name}` | Delete a PDF and clear its vectors |

### Query
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ask` | Ask a question (form field: `query`) |
| `GET` | `/api/test` | Health check |

### WebSocket
| Endpoint | Description |
|----------|-------------|
| `WS /ws/progress` | Real-time processing progress updates |

---

## 🔬 RAG Pipeline Deep Dive

### Chunking Strategy

```
Raw PDF Text
     │
     ▼
Semantic Split (regex: paragraph/section boundaries)
     │
     ▼ (chunks > 1200 chars)
Advanced Semantic Chunking
  - Preferred split points (priority order):
    1. Double newlines (paragraph breaks)
    2. Sentence-ending punctuation + capital
    3. Colon + newline (definition/list)
    4. Semicolons
    5. Word boundaries
  - 200-char overlap between consecutive chunks
     │
     ▼
Quality Filter (< 100 chars → discard)
     │
     ▼
Final Chunks (avg ~800-1200 chars each)
```

### Retrieval Scoring

```python
final_score = (
    0.40 * semantic_similarity  +  # ChromaDB cosine distance
    0.25 * term_overlap         +  # Query term coverage
    0.20 * term_density         +  # Query term frequency in chunk
    0.10 * position_score       +  # Earlier mentions weighted higher
    0.05 * completeness_score      # Longer chunks preferred
)
```

### MMR Diversity Formula

```
MMR(cᵢ) = λ × relevance(cᵢ) - (1-λ) × max_similarity(cᵢ, selected)

λ = 0.7  (70% relevance, 30% diversity)
```

---

## 🗺️ Roadmap

- [x] Core RAG pipeline (chunk → embed → retrieve → generate)
- [x] Google OAuth authentication
- [x] ChromaDB vector storage
- [x] Docker containerization
- [x] Persistent MongoDB sessions
- [ ] **Streaming LLM responses** (SSE)
- [ ] **Persistent chat history** per user
- [ ] **Multi-PDF management** (library per user)
- [ ] **User-isolated ChromaDB collections**
- [ ] **Conversation memory** (multi-turn context)
- [ ] **Document summary** on upload
- [ ] **Suggested follow-up questions**
- [ ] Rate limiting
- [ ] Background PDF processing (async)
- [ ] Export chat as Markdown / PDF

---

## 🤝 Contributing

Contributions are welcome!

```bash
# Fork the repo, then:
git checkout -b feature/amazing-feature
git commit -m 'feat: add amazing feature'
git push origin feature/amazing-feature
# Open a Pull Request
```

---

## 👨‍💻 Author

**Adithya Nagamuneendran**

[![GitHub](https://img.shields.io/badge/GitHub-Adhi1755-181717?style=flat&logo=github)](https://github.com/Adhi1755)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-adithyanagamuneendran-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/adithyanagamuneendran/)

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

Made with ❤️ and a lot of ☕

⭐ Star this repo if you found it useful!

</div>
