# rag_logic.py

import os
import re
import time
import numpy as np
import pdfplumber
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import List, Tuple
import logging
from fastapi import APIRouter
import chromadb

router = APIRouter()

@router.get("/test")
async def test():
    return {"message": "RAG router working!"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(".env")

client = genai.Client()

# Directories
UPLOAD_DIR = "uploads"
CHROMA_DB_PATH = "chromadb_storage"
COLLECTION_NAME = "rag_documents"

# Google Embedding model — 3072-dim, task-type aware
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 3072
EMBED_BATCH_SIZE = 100  # Google API max per request

# Retrieval configuration
TOP_K = 10
SIMILARITY_THRESHOLD = 0.25
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
RERANK_TOP_K = 7
MMR_DIVERSITY_LAMBDA = 0.7

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

_chroma_client = None
_collection = None
_current_chunks = []


# ------------------ Google Embedding API ------------------

def _get_google_embeddings(texts: List[str], task_type: str) -> np.ndarray:
    """Embed a list of texts using Google text-embedding-004 with batching.

    task_type should be "RETRIEVAL_DOCUMENT" for chunks and "RETRIEVAL_QUERY"
    for user queries. The model is optimised differently for each role.
    """
    if not texts:
        return np.array([], dtype=np.float32).reshape(0, EMBEDDING_DIMENSIONS)

    all_embeddings: List[List[float]] = []

    for batch_start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[batch_start : batch_start + EMBED_BATCH_SIZE]
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            all_embeddings.extend([e.values for e in result.embeddings])

            # Small back-off between batches to respect rate limits
            if batch_start + EMBED_BATCH_SIZE < len(texts):
                time.sleep(0.05)

        except Exception as e:
            logger.error(f"Embedding batch starting at {batch_start} failed: {e}")
            raise

    embeddings = np.array(all_embeddings, dtype=np.float32)
    logger.info(
        f"Generated Google embeddings: {embeddings.shape} "
        f"[task={task_type}]"
    )
    return embeddings


# ------------------ ChromaDB helpers ------------------

def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        logger.info(f"ChromaDB client initialised at: {CHROMA_DB_PATH}")
    return _chroma_client


def _reset_collection():
    global _collection, _current_chunks
    try:
        db = _get_chroma_client()
        try:
            db.delete_collection(name=COLLECTION_NAME)
            logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
        except Exception:
            pass
        _collection = db.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "RAG document chunks"},
            embedding_function=None,
        )
        _current_chunks = []
        logger.info(f"Created fresh collection: {COLLECTION_NAME}")
    except Exception as e:
        logger.error(f"Failed to reset collection: {e}")
        raise


def _get_or_create_collection():
    global _collection
    if _collection is None:
        db = _get_chroma_client()
        try:
            _collection = db.get_collection(name=COLLECTION_NAME)
            logger.info(f"Retrieved existing collection: {COLLECTION_NAME}")
        except Exception:
            _collection = db.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "RAG document chunks"},
                embedding_function=None,
            )
            logger.info(f"Created new collection: {COLLECTION_NAME}")
    return _collection


# ------------------ PDF Extraction ------------------

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    cleaned = _clean_extracted_text(page_text)
                    if cleaned:
                        text += f"\n--- Page {page_num} ---\n{cleaned}\n"
    except Exception as e:
        logger.error(f"PDF extraction failed for {file_path}: {e}")
        raise
    return text.strip()


def _clean_extracted_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'\/\@\#\$\%\&\*\+\=]", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ------------------ Chunking ------------------

def chunk_by_method_blocks(text: str) -> List[str]:
    """Semantic chunking with larger windows for better embedding quality."""
    if not text or not text.strip():
        return []

    semantic_splits = re.split(
        r"\n(?=\s*(?:\w+\(.*\)\s*:?|[A-Z][^.]*:|---|\*\*|##))", text
    )

    chunks: List[str] = []
    for segment in semantic_splits:
        segment = segment.strip()
        if not segment:
            continue
        if len(segment) > CHUNK_SIZE:
            chunks.extend(_advanced_semantic_chunking(segment, CHUNK_SIZE, CHUNK_OVERLAP))
        else:
            chunks.append(segment)

    final = [c for c in chunks if len(c.strip()) > 100]
    avg = sum(len(c) for c in final) // len(final) if final else 0
    logger.info(f"Created {len(final)} chunks (avg {avg} chars)")
    return final


def _advanced_semantic_chunking(text: str, chunk_size: int, overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break
        split_point = _find_optimal_split_point(text, start, end)
        chunk = text[start:split_point].strip()
        if chunk:
            chunks.append(chunk)
        start = max(split_point - overlap, start + chunk_size // 4)
    return chunks


def _find_optimal_split_point(text: str, start: int, end: int) -> int:
    region = text[max(0, end - 200) : end + 100]
    patterns = [
        (r"\n\n+", 2),
        (r"[.!?]\s+[A-Z]", 1),
        (r"[.!?]\n", 1),
        (r":\s*\n", 1),
        (r";\s+", 1),
        (r"\n", 1),
        (r"\s+", 1),
    ]
    for pattern, _ in patterns:
        matches = list(re.finditer(pattern, region))
        if matches:
            target = len(region) // 2
            best = min(matches, key=lambda m: abs(m.end() - target))
            return max(0, end - 200) + best.end()
    return end


# ------------------ Embedding (Google API) ------------------

def embed_chunks(chunks: List[str]) -> np.ndarray:
    """Embed document chunks with RETRIEVAL_DOCUMENT task type."""
    if not chunks:
        return np.array([], dtype=np.float32).reshape(0, EMBEDDING_DIMENSIONS)
    return _get_google_embeddings(chunks, task_type="RETRIEVAL_DOCUMENT")


def embed_query(query: str) -> np.ndarray:
    """Embed a user query with RETRIEVAL_QUERY task type."""
    return _get_google_embeddings([query], task_type="RETRIEVAL_QUERY")[0]


# ------------------ ChromaDB Storage ------------------

def save_embeddings_faiss(embeddings: np.ndarray, path: str = None) -> None:
    """Persist embeddings + chunks into ChromaDB."""
    global _current_chunks

    if embeddings.size == 0:
        logger.warning("No embeddings to save")
        return
    if not _current_chunks:
        logger.error("No chunks available — cannot save embeddings")
        return
    if len(_current_chunks) != len(embeddings):
        logger.error(
            f"Chunk/embedding count mismatch: {len(_current_chunks)} vs {len(embeddings)}"
        )
        return

    try:
        _reset_collection()
        collection = _get_or_create_collection()

        collection.add(
            embeddings=embeddings.tolist(),
            documents=_current_chunks.copy(),
            ids=[f"chunk_{i}" for i in range(len(_current_chunks))],
            metadatas=[
                {"chunk_index": i, "chunk_length": len(c)}
                for i, c in enumerate(_current_chunks)
            ],
        )
        logger.info(f"Saved {len(embeddings)} embeddings to ChromaDB")
    except Exception as e:
        logger.error(f"ChromaDB save failed: {e}")
        raise


def load_index(path: str = None):
    collection = _get_or_create_collection()
    logger.info(f"ChromaDB collection has {collection.count()} vectors")
    return collection


# ------------------ Retrieval (multi-stage) ------------------

def retrieve_relevant_chunks(
    query: str, chunks: List[str], embeddings: np.ndarray, top_k: int = TOP_K
) -> List[str]:
    """Multi-stage retrieval: vector search → rerank → MMR diversity."""
    global _current_chunks

    if not chunks or embeddings.size == 0:
        logger.warning("No chunks or embeddings available")
        return []

    _current_chunks = chunks.copy()

    # Query embedding uses RETRIEVAL_QUERY task type — key RAG improvement
    try:
        query_embedding = embed_query(query).tolist()
    except Exception as e:
        logger.error(f"Query embedding failed: {e}")
        return []

    try:
        collection = _get_or_create_collection()

        if collection.count() != len(chunks):
            logger.info("Syncing ChromaDB with current chunks...")
            save_embeddings_faiss(embeddings)

        initial_k = min(TOP_K * 2, len(chunks))
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=initial_k,
            include=["documents", "distances", "metadatas"],
        )

        documents = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []

        candidates = []
        for doc, dist, meta in zip(documents, distances, metadatas):
            similarity = max(0.0, 1.0 - dist)
            candidates.append(
                {"document": doc, "similarity": similarity, "distance": dist, "metadata": meta}
            )

        logger.info(f"Stage 1 — retrieved {len(candidates)} candidates")

        reranked = _rerank_candidates(query, candidates, RERANK_TOP_K)
        logger.info(f"Stage 2 — reranked to {len(reranked)} candidates")

        final = _apply_mmr_selection(query_embedding, reranked, RERANK_TOP_K)
        logger.info(f"Stage 3 — MMR selected {len(final)} chunks")

        return final

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return _fallback_text_search(query, chunks, top_k)


def _rerank_candidates(
    query: str, candidates: List[dict], target_k: int
) -> List[dict]:
    """Multi-signal reranking: semantic + lexical + density + position."""
    query_terms = set(query.lower().split())

    for c in candidates:
        doc = c["document"].lower()
        doc_terms = set(doc.split())

        semantic = c["similarity"]
        term_overlap = (
            len(query_terms & doc_terms) / len(query_terms) if query_terms else 0
        )
        total_hits = sum(doc.count(t) for t in query_terms)
        density = total_hits / len(doc_terms) if doc_terms else 0
        position = 0.0
        for t in query_terms:
            pos = doc.find(t)
            if pos != -1:
                position += max(0, 1 - pos / len(doc))
        position /= len(query_terms) if query_terms else 1
        completeness = min(1.0, len(c["document"]) / 800)

        c["rerank_score"] = (
            0.40 * semantic
            + 0.25 * term_overlap
            + 0.20 * density
            + 0.10 * position
            + 0.05 * completeness
        )

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[: target_k * 2]


def _apply_mmr_selection(
    query_embedding: List[float], candidates: List[dict], final_k: int
) -> List[str]:
    """Maximal Marginal Relevance for a diverse, relevant final set."""
    if not candidates:
        return []

    selected = [candidates[0]]
    remaining = candidates[1:]

    while len(selected) < final_k and remaining:
        best_score = -1.0
        best_idx = -1

        for i, cand in enumerate(remaining):
            relevance = cand["rerank_score"]
            cand_terms = set(cand["document"].lower().split())
            max_sim = max(
                len(cand_terms & set(s["document"].lower().split()))
                / len(cand_terms | set(s["document"].lower().split()))
                for s in selected
            ) if selected else 0.0
            score = MMR_DIVERSITY_LAMBDA * relevance - (1 - MMR_DIVERSITY_LAMBDA) * max_sim
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx >= 0:
            selected.append(remaining.pop(best_idx))
        else:
            break

    return [c["document"] for c in selected]


def _fallback_text_search(query: str, chunks: List[str], top_k: int) -> List[str]:
    query_words = query.lower().split()
    scored = []
    for chunk in chunks:
        cl = chunk.lower()
        exact = sum(cl.count(w) for w in query_words)
        coverage = sum(1 for w in query_words if w in cl) / len(query_words)
        score = exact * 2 + coverage * 10
        if score > 0:
            scored.append((chunk, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:top_k]]


# ------------------ LLM (Gemini) ------------------

def ask_llm(query: str, context_chunks: List[str]) -> str:
    """Generate an answer from Gemini given the retrieved context chunks."""
    if not context_chunks:
        return (
            "I don't have relevant information in the document to answer your question. "
            "Try rephrasing or check if the document covers this topic."
        )

    context_parts = [
        f"[Context {i} — {len(chunk)} chars]\n{chunk.strip()}"
        for i, chunk in enumerate(context_chunks, 1)
    ]
    context = "\n\n".join(context_parts)

    prompt = f"""You are an expert document analyst. Answer using ONLY the provided context.

INSTRUCTIONS:
1. Provide comprehensive, structured answers using all relevant context sections.
2. For lists, benefits, or features — compile complete information from every context section.
3. Use bullet points or numbered lists where appropriate.
4. Only include information explicitly stated or clearly implied in the context.
5. If the context is insufficient, state what is available and note any gaps.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        answer = response.text.strip()
        if not answer or len(answer) < 10:
            return "Could not generate a response. Please rephrase your question."
        return answer
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        return "An error occurred while processing your question. Please try again."


# ------------------ Full Upload Pipeline ------------------

def process_uploaded_pdf(file_path: str):
    """Extract → chunk → embed (Google API) → store in ChromaDB."""
    global _current_chunks

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    logger.info(f"Processing PDF: {file_path}")
    text = extract_text_from_pdf(file_path)

    if len(text.strip()) < 100:
        logger.warning("Extracted text is very short")

    logger.info(f"Extracted {len(text)} chars from PDF")

    chunks = chunk_by_method_blocks(text)
    if not chunks:
        raise ValueError("No valid chunks created from PDF")

    _current_chunks = chunks.copy()
    avg_size = sum(len(c) for c in chunks) // len(chunks)
    logger.info(f"Created {len(chunks)} chunks (avg {avg_size} chars)")

    logger.info("Generating embeddings via Google text-embedding-004...")
    embeddings = embed_chunks(chunks)

    if embeddings.size == 0:
        raise ValueError("Embedding generation failed")

    logger.info(f"Embeddings shape: {embeddings.shape}")

    save_embeddings_faiss(embeddings)
    logger.info("PDF processing complete")

    return chunks, embeddings
