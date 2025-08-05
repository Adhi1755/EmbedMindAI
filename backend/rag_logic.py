#rag_logic.py

import os
import re
import numpy as np
import pdfplumber
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
from typing import List, Tuple
import logging
from fastapi import APIRouter
import chromadb
from chromadb.config import Settings
import uuid
import json


router = APIRouter()

@router.get("/test")
async def test():
    return {"message": "RAG router working!"}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(".env")

# Initialize Gemini client
client = genai.Client()

# Enhanced Configurations - IMPROVED FOR BETTER RETRIEVAL
UPLOAD_DIR = "uploads"
CHROMA_DB_PATH = "chromadb_storage"
COLLECTION_NAME = "rag_documents"
MODEL_NAME = "intfloat/e5-small-v2"
BACKUP_MODEL = "multi-qa-mpnet-base-dot-v1"

# 📈 IMPROVEMENT 1: Increased chunk size and top-k for better semantic coverage
TOP_K = 10  # Increased from 5 to 10 for better recall
SIMILARITY_THRESHOLD = 0.25  # Lowered slightly for better recall
CHUNK_SIZE = 1200  # Increased from 500 to 1200 tokens equivalent
CHUNK_OVERLAP = 200  # Increased from 100 to 200 for better context preservation

# 📈 IMPROVEMENT 2: Added reranking parameters
RERANK_TOP_K = 7  # Final number of chunks after reranking
MMR_DIVERSITY_LAMBDA = 0.7  # Balance between relevance and diversity (0.7 = more relevance)

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# Global variables for consistency
_embedding_model = None
_chroma_client = None
_collection = None
_current_chunks = []

def _get_embedding_model():
    """Get embedding model with fallback strategy"""
    global _embedding_model
    if _embedding_model is None:
        try:
            logger.info(f"Loading primary embedding model: {MODEL_NAME}")
            _embedding_model = SentenceTransformer(MODEL_NAME)
        except Exception as e:
            logger.warning(f"Failed to load {MODEL_NAME}, trying backup: {e}")
            try:
                _embedding_model = SentenceTransformer(BACKUP_MODEL)
                logger.info(f"Loaded backup model: {BACKUP_MODEL}")
            except Exception as e2:
                logger.error(f"Failed to load backup model: {e2}")
                _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Loaded fallback model: all-MiniLM-L6-v2")
    return _embedding_model

def _get_chroma_client():
    """Initialize ChromaDB client with persistent storage"""
    global _chroma_client
    if _chroma_client is None:
        try:
            # Initialize ChromaDB with persistent storage
            _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            logger.info(f"Initialized ChromaDB client with persistent storage at: {CHROMA_DB_PATH}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
    return _chroma_client

def _reset_collection():
    """Reset and create a fresh collection"""
    global _collection, _current_chunks
    try:
        client = _get_chroma_client()
        
        # Delete existing collection if it exists
        try:
            client.delete_collection(name=COLLECTION_NAME)
            logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
        except Exception:
            pass
        
        # Create fresh collection
        _collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "RAG document chunks collection"},
            embedding_function=None
        )
        _current_chunks = []
        logger.info(f"Created fresh collection: {COLLECTION_NAME}")
        
    except Exception as e:
        logger.error(f"Failed to reset collection: {e}")
        raise

def _get_or_create_collection():
    """Get or create ChromaDB collection"""
    global _collection
    if _collection is None:
        try:
            client = _get_chroma_client()
            try:
                _collection = client.get_collection(name=COLLECTION_NAME)
                logger.info(f"Retrieved existing collection: {COLLECTION_NAME}")
            except Exception:
                _collection = client.create_collection(
                    name=COLLECTION_NAME,
                    metadata={"description": "RAG document chunks collection"},
                    embedding_function=None
                )
                logger.info(f"Created new collection: {COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Failed to get/create collection: {e}")
            raise
    return _collection

# ------------------ PDF Extraction ------------------
def extract_text_from_pdf(file_path):
    """Enhanced PDF text extraction with better cleaning"""
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    cleaned_text = _clean_extracted_text(page_text)
                    if cleaned_text:
                        text += f"\n--- Page {page_num} ---\n{cleaned_text}\n"
    except Exception as e:
        logger.error(f"Error extracting PDF {file_path}: {e}")
        raise
    
    return text.strip()

def _clean_extracted_text(text: str) -> str:
    """Clean extracted text for better processing"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'\/\@\#\$\%\&\*\+\=]', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

# ------------------ 📈 IMPROVED CHUNKING STRATEGY ------------------
def chunk_by_method_blocks(text: str) -> list[str]:
    """📈 ENHANCED: Larger semantic chunks with better overlap for improved retrieval"""
    if not text or not text.strip():
        return []
    
    # Split by semantic boundaries (paragraphs, sections, methods)
    semantic_splits = re.split(r'\n(?=\s*(?:\w+\(.*\)\s*:?|[A-Z][^.]*:|---|\*\*|##))', text)
    
    enhanced_chunks = []
    
    for chunk in semantic_splits:
        chunk = chunk.strip()
        if not chunk:
            continue
            
        # 📈 IMPROVEMENT: Use larger chunk size for better semantic coverage
        if len(chunk) > CHUNK_SIZE:
            sub_chunks = _advanced_semantic_chunking(chunk, CHUNK_SIZE, CHUNK_OVERLAP)
            enhanced_chunks.extend(sub_chunks)
        else:
            enhanced_chunks.append(chunk)
    
    # Filter out very small chunks but keep more content
    final_chunks = [chunk for chunk in enhanced_chunks if len(chunk.strip()) > 100]
    
    logger.info(f"📈 Created {len(final_chunks)} enhanced semantic chunks (avg size: {sum(len(c) for c in final_chunks) // len(final_chunks) if final_chunks else 0} chars)")
    return final_chunks

def _advanced_semantic_chunking(text: str, chunk_size: int, overlap: int) -> List[str]:
    """📈 ENHANCED: Advanced semantic-aware chunking with better context preservation"""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break
        
        # 📈 IMPROVEMENT: Better semantic boundary detection
        split_point = _find_optimal_split_point(text, start, end)
        
        chunk = text[start:split_point].strip()
        if chunk:
            chunks.append(chunk)
        
        # 📈 IMPROVEMENT: Smarter overlap calculation
        start = max(split_point - overlap, start + chunk_size // 4)
    
    return chunks

def _find_optimal_split_point(text: str, start: int, end: int) -> int:
    """📈 ENHANCED: More sophisticated semantic boundary detection"""
    search_region = text[max(0, end-200):end+100]  # Larger search region
    
    # Priority order for semantic boundaries
    boundary_patterns = [
        (r'\n\n+', 2),           # Paragraph breaks (highest priority)
        (r'[.!?]\s+[A-Z]', 1),   # Sentence boundaries with capital
        (r'[.!?]\n', 1),         # Sentence boundaries with newline
        (r':\s*\n', 1),          # Colon with newline (lists, definitions)
        (r';\s+', 1),            # Semicolon boundaries
        (r',\s+(?=\w+\s+(?:is|are|was|were|has|have))', 1),  # Clause boundaries
        (r'\n', 1),              # Line breaks
        (r'\s+', 1),             # Word boundaries
    ]
    
    for pattern, offset in boundary_patterns:
        matches = list(re.finditer(pattern, search_region))
        if matches:
            # Find the match closest to our target split point
            target_pos = len(search_region) // 2
            best_match = min(matches, key=lambda m: abs(m.end() - target_pos))
            return max(0, end-200) + best_match.end()
    
    return end

# ------------------ Enhanced Embedding & Normalization ------------------
def embed_chunks(chunks: list[str], model_name=MODEL_NAME) -> np.ndarray:
    """Enhanced embedding with query optimization"""
    if not chunks:
        return np.array([]).reshape(0, 384)
    
    model = _get_embedding_model()
    
    # 📈 IMPROVEMENT: Better preprocessing for different model types
    processed_chunks = []
    for chunk in chunks:
        if "e5" in model_name.lower():
            processed_chunk = f"passage: {chunk}"
        elif "instructor" in model_name.lower():
            processed_chunk = f"Represent this document for retrieval: {chunk}"
        else:
            processed_chunk = chunk
        processed_chunks.append(processed_chunk)
    
    try:
        embeddings = model.encode(
            processed_chunks,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32  # 📈 IMPROVEMENT: Optimized batch size
        )
        
        embeddings = normalize_embeddings(embeddings)
        logger.info(f"📈 Generated optimized embeddings for {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise
    
    return embeddings

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Enhanced normalization with numerical stability"""
    if embeddings.size == 0:
        return embeddings
    
    epsilon = 1e-8
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, epsilon)
    
    normalized = embeddings / norms
    
    verification_norms = np.linalg.norm(normalized, axis=1)
    if not np.allclose(verification_norms, 1.0, atol=1e-6):
        logger.warning("Embedding normalization may be imperfect")
    
    return normalized.astype(np.float32)

# ------------------ ChromaDB Storage ------------------
def save_embeddings_faiss(embeddings: np.ndarray, path: str = None) -> None:
    """Save embeddings to ChromaDB with proper chunk synchronization"""
    global _current_chunks
    
    if embeddings.size == 0:
        logger.warning("No embeddings to save")
        return
    
    if not _current_chunks:
        logger.error("No chunks available to save with embeddings")
        return
    
    if len(_current_chunks) != len(embeddings):
        logger.error(f"Chunk count ({len(_current_chunks)}) doesn't match embedding count ({len(embeddings)})")
        return
    
    try:
        _reset_collection()
        collection = _get_or_create_collection()
        
        embeddings_list = embeddings.tolist()
        ids = [f"chunk_{i}" for i in range(len(_current_chunks))]
        documents = _current_chunks.copy()
        metadatas = [{"chunk_index": i, "chunk_length": len(chunk)} for i, chunk in enumerate(_current_chunks)]
        
        collection.add(
            embeddings=embeddings_list,
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        
        logger.info(f"📈 Successfully saved {len(embeddings_list)} embeddings to ChromaDB")
        
    except Exception as e:
        logger.error(f"Error saving embeddings to ChromaDB: {e}")
        raise

def load_index(path: str = None):
    """Load ChromaDB collection"""
    try:
        collection = _get_or_create_collection()
        count = collection.count()
        logger.info(f"📈 Loaded ChromaDB collection with {count} vectors")
        return collection
        
    except Exception as e:
        logger.error(f"Error loading ChromaDB collection: {e}")
        raise

# ------------------ 📈 GREATLY IMPROVED SIMILARITY SEARCH WITH RERANKING ------------------
def retrieve_relevant_chunks(query: str, chunks: list[str], embeddings: np.ndarray, top_k: int = TOP_K) -> list[str]:
    """📈 ENHANCED: Multi-stage retrieval with semantic search + lightweight reranking"""
    global _current_chunks
    
    if not chunks or embeddings.size == 0:
        logger.warning("No chunks or embeddings available for retrieval")
        return []
    
    _current_chunks = chunks.copy()
    model = _get_embedding_model()
    
    # 📈 IMPROVEMENT: Better query preprocessing
    if "e5" in MODEL_NAME.lower():
        optimized_query = f"query: {query}"
    elif "instructor" in MODEL_NAME.lower():
        optimized_query = f"Represent this query for searching relevant passages: {query}"
    else:
        optimized_query = query
    
    # Generate query embedding
    try:
        query_vector = model.encode([optimized_query], convert_to_numpy=True, normalize_embeddings=True)
        query_vector = normalize_embeddings(query_vector).astype(np.float32)
        query_embedding = query_vector[0].tolist()
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        return []
    
    # 📈 STAGE 1: Initial retrieval with higher recall
    try:
        collection = _get_or_create_collection()
        
        current_count = collection.count()
        if current_count != len(chunks):
            logger.info("📈 Updating ChromaDB collection with current chunks...")
            save_embeddings_faiss(embeddings)
        
        # 📈 IMPROVEMENT: Retrieve more candidates for reranking
        initial_k = min(TOP_K * 2, len(chunks))  # Get 2x candidates
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=initial_k,
            include=['documents', 'distances', 'metadatas']
        )
        
        # Process initial results
        candidate_results = []
        documents = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        
        print(f"\n🔍 Stage 1 - Initial Retrieval ({len(documents)} candidates):")
        
        for i, (doc, distance, metadata) in enumerate(zip(documents, distances, metadatas)):
            similarity = max(0, 1 - distance)
            candidate_results.append({
                'document': doc,
                'similarity': similarity,
                'distance': distance,
                'metadata': metadata,
                'index': i
            })
            
            if i < 5:  # Show top 5 candidates
                print(f"Candidate {i+1}: Similarity={similarity:.4f}, Length={len(doc)}")
        
        # 📈 STAGE 2: Advanced reranking with multiple signals
        print(f"\n🎯 Stage 2 - Advanced Reranking:")
        reranked_results = _advanced_rerank_candidates(query, candidate_results, RERANK_TOP_K)
        
        # 📈 STAGE 3: Diversity-aware final selection (MMR-like)
        print(f"\n🌟 Stage 3 - Diversity-Aware Selection:")
        final_chunks = _apply_mmr_selection(query_embedding, reranked_results, RERANK_TOP_K)
        
        logger.info(f"📈 Multi-stage retrieval completed: {len(final_chunks)} high-quality chunks selected")
        
        # Debug output
        print(f"\n✅ Final Selected Chunks ({len(final_chunks)}):")
        for i, chunk in enumerate(final_chunks):
            print(f"\n=== Final Chunk {i+1} ===")
            print(f"Length: {len(chunk)} chars")
            print(f"Preview: {chunk[:200]}{'...' if len(chunk) > 200 else ''}")
        
        return final_chunks
        
    except Exception as e:
        logger.error(f"Error in enhanced similarity search: {e}")
        return _fallback_text_search(query, chunks, top_k)

def _advanced_rerank_candidates(query: str, candidates: List[dict], target_k: int) -> List[dict]:
    """📈 NEW: Advanced reranking using multiple relevance signals"""
    query_terms = set(query.lower().split())
    
    for candidate in candidates:
        doc = candidate['document'].lower()
        
        # Signal 1: Semantic similarity (already have)
        semantic_score = candidate['similarity']
        
        # Signal 2: Term overlap score
        doc_terms = set(doc.split())
        term_overlap = len(query_terms.intersection(doc_terms)) / len(query_terms) if query_terms else 0
        
        # Signal 3: Query term density
        total_matches = sum(doc.count(term) for term in query_terms)
        density_score = total_matches / len(doc.split()) if doc.split() else 0
        
        # Signal 4: Position-based scoring (earlier mentions weighted higher)
        position_score = 0
        for term in query_terms:
            pos = doc.find(term)
            if pos != -1:
                position_score += max(0, 1 - (pos / len(doc)))
        position_score /= len(query_terms) if query_terms else 1
        
        # Signal 5: Chunk completeness (prefer longer, more complete chunks)
        completeness_score = min(1.0, len(candidate['document']) / 800)  # Normalize around 800 chars
        
        # 📈 IMPROVEMENT: Weighted combination of all signals
        final_score = (
            0.40 * semantic_score +      # Primary: semantic similarity
            0.25 * term_overlap +        # Secondary: term coverage
            0.20 * density_score +       # Tertiary: term density
            0.10 * position_score +      # Quaternary: early mentions
            0.05 * completeness_score    # Quinary: chunk completeness
        )
        
        candidate['rerank_score'] = final_score
        candidate['signals'] = {
            'semantic': semantic_score,
            'term_overlap': term_overlap,
            'density': density_score,
            'position': position_score,
            'completeness': completeness_score
        }
    
    # Sort by reranked score
    reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
    
    # Debug output
    print("Top reranked candidates:")
    for i, candidate in enumerate(reranked[:5]):
        signals = candidate['signals']
        print(f"  {i+1}. Final Score: {candidate['rerank_score']:.3f}")
        print(f"     Signals - Semantic: {signals['semantic']:.3f}, Terms: {signals['term_overlap']:.3f}, "
              f"Density: {signals['density']:.3f}, Position: {signals['position']:.3f}")
    
    return reranked[:target_k * 2]  # Return more for diversity selection

def _apply_mmr_selection(query_embedding: List[float], candidates: List[dict], final_k: int) -> List[str]:
    """📈 NEW: Maximal Marginal Relevance for diversity-aware selection"""
    if not candidates:
        return []
    
    selected = []
    remaining = candidates.copy()
    query_vector = np.array(query_embedding)
    
    # Select the highest scoring candidate first
    best_candidate = remaining.pop(0)
    selected.append(best_candidate)
    
    # MMR selection for remaining slots
    while len(selected) < final_k and remaining:
        best_score = -1
        best_idx = -1
        
        for i, candidate in enumerate(remaining):
            # Relevance score (to query)
            relevance = candidate['rerank_score']
            
            # Diversity score (maximum similarity to already selected)
            max_sim_to_selected = 0
            candidate_text = candidate['document'].lower()
            
            for selected_candidate in selected:
                selected_text = selected_candidate['document'].lower()
                # Simple text-based similarity for diversity
                text_sim = len(set(candidate_text.split()) & set(selected_text.split())) / \
                          len(set(candidate_text.split()) | set(selected_text.split()))
                max_sim_to_selected = max(max_sim_to_selected, text_sim)
            
            # MMR formula: λ * relevance - (1-λ) * max_similarity_to_selected
            mmr_score = MMR_DIVERSITY_LAMBDA * relevance - (1 - MMR_DIVERSITY_LAMBDA) * max_sim_to_selected
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        
        if best_idx >= 0:
            selected.append(remaining.pop(best_idx))
        else:
            break
    
    print(f"MMR Selection: Selected {len(selected)} diverse, relevant chunks")
    return [candidate['document'] for candidate in selected]

def _fallback_text_search(query: str, chunks: list[str], top_k: int) -> list[str]:
    """Enhanced fallback search with better scoring"""
    query_lower = query.lower()
    query_words = query_lower.split()
    scored_chunks = []
    
    for chunk in chunks:
        chunk_lower = chunk.lower()
        
        # Multiple scoring signals
        exact_matches = sum(chunk_lower.count(word) for word in query_words)
        term_coverage = sum(1 for word in query_words if word in chunk_lower) / len(query_words)
        
        # Combined score
        score = exact_matches * 2 + term_coverage * 10
        
        if score > 0:
            scored_chunks.append((chunk, score))
    
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in scored_chunks[:top_k]]

# ------------------ Enhanced Gemini LLM ------------------
def ask_llm(query: str, context_chunks: list[str]) -> str:
    """📈 ENHANCED: Improved LLM prompting with better context organization"""
    if not context_chunks:
        return "I don't have any relevant information in the document to answer your question. Please try rephrasing your question or check if the document contains the information you're looking for."
    
    # 📈 IMPROVEMENT: Better context organization and ranking
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        # Add context indicators for better LLM understanding
        chunk_preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
        context_parts.append(f"[Context {i} - {len(chunk)} chars]\n{chunk.strip()}")
    
    context = "\n\n".join(context_parts)
    
    # 📈 IMPROVEMENT: Enhanced prompt with better instructions
    full_prompt = f"""You are an expert document analyst. Your task is to provide accurate, comprehensive answers using ONLY the provided context.

CONTEXT ANALYSIS INSTRUCTIONS:
1. **Comprehensive Coverage**: If the answer requires multiple aspects or examples, draw from ALL relevant context sections
2. **Prioritize Completeness**: For questions asking for lists, benefits, features, or explanations, provide complete information from across all contexts
3. **Structured Responses**: Organize your answer logically (use bullet points, numbered lists, or clear paragraphs as appropriate)
4. **Accuracy First**: Only include information that is explicitly stated or clearly implied in the context
5. **Contextual Attribution**: You may reference "the document" or "according to the provided information" when appropriate

RESPONSE STRATEGY:
- For conceptual questions: Provide comprehensive explanations using all relevant context
- For list-type questions: Compile complete lists from all context sections
- For specific facts: Provide precise, accurate information
- If information is incomplete: State what's available and note any limitations

CONTEXT SECTIONS:
{context}

USER QUESTION: {query}

COMPREHENSIVE ANSWER:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt
        )
        
        answer = response.text.strip()
        
        if not answer or len(answer) < 10:
            return "I couldn't generate a proper response. Please try rephrasing your question."
        
        # 📈 IMPROVEMENT: Better post-processing
        if any(phrase in answer.lower() for phrase in ["not in document", "not mentioned", "no information", "cannot determine"]):
            # Try to extract any partial information that might be useful
            if len(answer) > 50:  # If there's substantial content despite the disclaimer
                return f"Based on the available context: {answer}"
            else:
                return "The document doesn't contain specific information to fully answer your question. You might want to try a more specific query or check if the document covers this topic."
        
        return answer
        
    except Exception as e:
        logger.error(f"Error querying LLM: {e}")
        return "I encountered an error while processing your question. Please try again."

# ------------------ Enhanced Upload + Process Flow ------------------
def process_uploaded_pdf(file_path: str):
    """📈 ENHANCED: Improved PDF processing with better chunking and validation"""
    global _current_chunks
    
    try:
        logger.info(f"📄 Processing PDF with enhanced pipeline: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        # Extract text
        text = extract_text_from_pdf(file_path)
        
        if not text or len(text.strip()) < 100:
            logger.warning("⚠️ Extracted text is very short, this may affect quality")
        
        logger.info(f"📄 Extracted {len(text)} characters from PDF")
        
        # 📈 IMPROVEMENT: Enhanced chunking with better semantic preservation
        print("🔪 Applying advanced semantic chunking (larger chunks, better overlap)...")
        chunks = chunk_by_method_blocks(text)
        
        if not chunks:
            raise ValueError("No valid chunks were created from the PDF")
        
        _current_chunks = chunks.copy()
        avg_chunk_size = sum(len(chunk) for chunk in chunks) // len(chunks)
        print(f"📈 Created {len(chunks)} optimized chunks (average size: {avg_chunk_size} chars)")
        
        # Generate embeddings
        print("🔗 Generating enhanced embeddings with optimized preprocessing...")
        embeddings = embed_chunks(chunks)
        
        if embeddings.size == 0:
            raise ValueError("Failed to generate embeddings")
        
        print(f"✅ Generated embeddings: {embeddings.shape}")
        
        # Save to ChromaDB
        print("💾 Saving to ChromaDB with enhanced indexing...")
        save_embeddings_faiss(embeddings)
        
        print("🎉 PDF processing completed with enhanced retrieval pipeline!")
        print(f"📊 Pipeline Stats: {len(chunks)} chunks, avg {avg_chunk_size} chars/chunk, {CHUNK_OVERLAP} char overlap")
        
        return chunks, embeddings
        
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {e}")
        raise