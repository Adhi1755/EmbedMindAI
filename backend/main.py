# main.py
import os
import time
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from rag_logic import *                          # brings in np, re, logger, etc.
from rag_logic import router as rag_router
from auth import router as auth_router
from chat_routes import router as chat_router
from database import get_db, get_current_user, init_indexes

app = FastAPI(title="EmbedMindAI API", version="2.0.0")

# ── CORS ──────────────────────────────────────────────────────────────────────
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(rag_router, prefix="/api")
app.include_router(auth_router)
app.include_router(chat_router)

# ── Startup / shutdown ────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    init_indexes()
    logger.info("EmbedMindAI API started")

@app.on_event("shutdown")
async def on_shutdown():
    from database import close_db
    close_db()

# ── WebSocket progress broadcaster ───────────────────────────────────────────
connected_clients: list[WebSocket] = []

@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()   # keep-alive
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)

async def notify_clients(message: str):
    dead = []
    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception:
            dead.append(client)
    for d in dead:
        if d in connected_clients:
            connected_clients.remove(d)


# ── In-memory PDF state (per-process; replaced by per-user later) ─────────────
pdf_chunks: list[str] = []
pdf_embeddings = np.array([], dtype=np.float32)


# ── Upload endpoint ───────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    """Upload and process a PDF; save document metadata to MongoDB."""

    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are supported", "status": "failed"}

    safe_filename = re.sub(r"[^\w\-_\.]", "_", file.filename)
    file_path = f"uploads/{safe_filename}"
    os.makedirs("uploads", exist_ok=True)

    logger.info(f"Saving uploaded file: {safe_filename}")
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        await notify_clients("Starting PDF processing…")
    except Exception:
        pass

    text = extract_text_from_pdf(file_path)
    try:
        await notify_clients("Extracting text from PDF…")
    except Exception:
        pass

    if not text or len(text.strip()) < 50:
        os.remove(file_path)
        return {"error": "PDF appears to be empty or contains insufficient text", "status": "failed"}

    global pdf_chunks, pdf_embeddings

    try:
        await notify_clients("Applying enhanced chunking strategy…")
    except Exception:
        pass
    pdf_chunks = chunk_by_method_blocks(text)

    if not pdf_chunks:
        os.remove(file_path)
        return {"error": "Failed to create meaningful chunks from PDF", "status": "failed"}

    try:
        await notify_clients(f"Created {len(pdf_chunks)} semantic chunks")
    except Exception:
        pass

    try:
        await notify_clients("Generating embeddings…")
    except Exception:
        pass
    pdf_embeddings = embed_chunks(pdf_chunks)

    if pdf_embeddings.size == 0:
        os.remove(file_path)
        return {"error": "Failed to generate embeddings", "status": "failed"}

    try:
        await notify_clients("Saving to ChromaDB — Setup Complete")
    except Exception:
        pass
    save_embeddings_faiss(pdf_embeddings)

    # ── Save document record to MongoDB ──────────────────────────────────────
    doc_id = None
    try:
        user = get_current_user(request)
        user_email = user["email"] if user else "anonymous"
        result = get_db().documents.insert_one({
            "user_email":       user_email,
            "filename":         file.filename,
            "safe_filename":    safe_filename,
            "file_path":        file_path,
            "file_size_bytes":  len(content),
            "chunk_count":      len(pdf_chunks),
            "text_length":      len(text),
            "embedding_dims":   int(pdf_embeddings.shape[1]) if pdf_embeddings.size > 0 else 0,
            "uploaded_at":      datetime.now(timezone.utc),
            "status":           "ready",
        })
        doc_id = str(result.inserted_id)
        logger.info(f"Document record saved: {doc_id}")
    except Exception as e:
        logger.warning(f"Could not save document record: {e}")

    return {
        "message": "✅ PDF uploaded and processed successfully.",
        "details": {
            "filename":           safe_filename,
            "text_length":        len(text),
            "chunks_created":     len(pdf_chunks),
            "embedding_dimensions": pdf_embeddings.shape[1] if pdf_embeddings.size > 0 else 0,
            "vector_store":       "ChromaDB",
            "document_id":        doc_id,
        },
        "status": "success",
    }


# ── Delete endpoint ───────────────────────────────────────────────────────────
@app.delete("/delete")
async def delete_pdf(filename: str, request: Request):
    safe_filename = re.sub(r"[^\w\-_\.]", "_", filename)
    file_path = f"uploads/{safe_filename}"

    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        raise HTTPException(status_code=404, detail="PDF not found")

    # Clear ChromaDB
    try:
        from rag_logic import _reset_collection
        _reset_collection()
    except Exception as e:
        logger.warning(f"Could not clear ChromaDB: {e}")

    # Mark document as deleted in MongoDB
    try:
        user = get_current_user(request)
        user_email = user["email"] if user else "anonymous"
        get_db().documents.update_one(
            {"safe_filename": safe_filename, "user_email": user_email},
            {"$set": {"status": "deleted", "deleted_at": datetime.now(timezone.utc)}},
        )
    except Exception as e:
        logger.warning(f"Could not update document status: {e}")

    global pdf_chunks, pdf_embeddings
    pdf_chunks = []
    pdf_embeddings = np.array([], dtype=np.float32)

    return {"message": "PDF and associated data deleted successfully"}


# ── Documents listing endpoint ────────────────────────────────────────────────
@app.get("/documents")
async def list_documents(request: Request):
    """Return all documents uploaded by the current user."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    db = get_db()
    docs = list(
        db.documents.find(
            {"user_email": user["email"], "status": {"$ne": "deleted"}},
            {"_id": 0, "password_hash": 0},
        ).sort("uploaded_at", -1)
    )
    for d in docs:
        if "uploaded_at" in d and d["uploaded_at"]:
            d["uploaded_at"] = d["uploaded_at"].isoformat()
    return docs


# ── Ask endpoint ──────────────────────────────────────────────────────────────
@app.post("/ask")
async def ask_question(
    request: Request,
    query: str = Form(...),
    session_id: str = Form(None),
):
    global pdf_chunks, pdf_embeddings

    if not pdf_chunks or pdf_embeddings.size == 0:
        return {"answer": "Please upload and process a PDF first."}

    logger.info(f"Query received: {query[:80]}")
    start = time.time()

    relevant = retrieve_relevant_chunks(query, pdf_chunks, pdf_embeddings)
    response = ask_llm(query, relevant)

    elapsed_ms = int((time.time() - start) * 1000)

    # ── Log query to MongoDB ─────────────────────────────────────────────────
    try:
        user = get_current_user(request)
        get_db().query_logs.insert_one({
            "user_email":       user["email"] if user else "anonymous",
            "session_id":       session_id,
            "query":            query,
            "answer_length":    len(response),
            "chunks_retrieved": len(relevant),
            "response_time_ms": elapsed_ms,
            "created_at":       datetime.now(timezone.utc),
        })
    except Exception as e:
        logger.warning(f"Could not save query log: {e}")

    return {"answer": response}


# ── Analytics endpoints ───────────────────────────────────────────────────────
@app.get("/stats/me")
async def my_stats(request: Request):
    """Return basic usage statistics for the current user."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    db = get_db()
    email = user["email"]

    total_docs    = db.documents.count_documents({"user_email": email, "status": {"$ne": "deleted"}})
    total_sessions = db.chat_sessions.count_documents({"user_email": email})
    total_messages = db.messages.count_documents({"user_email": email})
    total_queries  = db.query_logs.count_documents({"user_email": email})

    avg_response_ms = 0
    pipeline = [
        {"$match": {"user_email": email}},
        {"$group": {"_id": None, "avg": {"$avg": "$response_time_ms"}}},
    ]
    result = list(db.query_logs.aggregate(pipeline))
    if result:
        avg_response_ms = int(result[0]["avg"])

    return {
        "total_documents":   total_docs,
        "total_sessions":    total_sessions,
        "total_messages":    total_messages,
        "total_queries":     total_queries,
        "avg_response_ms":   avg_response_ms,
    }