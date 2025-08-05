# main.py - Only minimal changes needed for ChromaDB compatibility

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from rag_logic import *
from fastapi import HTTPException
from fastapi import WebSocket , WebSocketDisconnect
from fastapi.responses import HTMLResponse
from rag_logic import router as rag_router 
from auth import router as auth_router
from pydantic import BaseModel
import asyncio


app = FastAPI()
connected_clients = []
app.include_router(rag_router, prefix="/api")
app.include_router(auth_router)



@app.websocket("/ws/progress")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)

    try:
        while True:
            await websocket.receive_text()  # keep connection alive
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

async def notify_clients(message: str):
    disconnected_clients = []

    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception as e:
            print(f"Client disconnected: {e}")
            disconnected_clients.append(client)

    for dc in disconnected_clients:
        connected_clients.remove(dc)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pdf_chunks = []
pdf_embeddings = np.array([], dtype=np.float32)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Enhanced PDF upload endpoint with ChromaDB processing"""
    
    try:
        if not file.filename.lower().endswith('.pdf'):
            return {"error": "Only PDF files are supported", "status": "failed"}
        
        safe_filename = re.sub(r'[^\w\-_\.]', '_', file.filename)
        file_path = f"uploads/{safe_filename}"
        
        os.makedirs("uploads", exist_ok=True)
        
        logger.info(f"Saving uploaded file: {safe_filename}")
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        logger.info("Starting enhanced PDF processing with ChromaDB...")
        try:
            await notify_clients("Starting enhanced PDF processing with ChromaDB...")
        except:
            pass

        text = extract_text_from_pdf(file_path)
        try:
            await notify_clients("Extracting text from pdf...")
        except:
            pass

        if not text or len(text.strip()) < 50:
            return {
                "error": "PDF appears to be empty or contains insufficient text content",
                "status": "failed"
            }

        global pdf_chunks, pdf_embeddings
        
        logger.info("Applying enhanced chunking strategy...")
        try:
            await notify_clients("Applying enhanced chunking strategy...")
        except:
            pass
        
        pdf_chunks = chunk_by_method_blocks(text)

        if not pdf_chunks:
            return {
                "error": "Failed to create meaningful chunks from PDF content",
                "status": "failed"
            }
        
        logger.info(f"Created {len(pdf_chunks)} semantic chunks")
        try:
            await notify_clients(f"Created {len(pdf_chunks)} semantic chunks")
        except:
            pass

        logger.info("Generating enhanced embeddings...")
        try:
            await notify_clients("Generating enhanced embeddings...")
        except:
            pass

        pdf_embeddings = embed_chunks(pdf_chunks)

        if pdf_embeddings.size == 0:
            return {
                "error": "Failed to generate embeddings from PDF content",
                "status": "failed"
            }

        logger.info(f"Generated embeddings with shape: {pdf_embeddings.shape}")
        try:
            await notify_clients(f"Generated embeddings with shape: {pdf_embeddings.shape}")
        except:
            pass

        logger.info("Saving to ChromaDB collection...")
        try:
            await notify_clients("Saving to ChromaDB - Setup Complete")
        except:
            pass

        save_embeddings_faiss(pdf_embeddings)  # Same function name, now uses ChromaDB internally
        return {
            "message": "✅ PDF uploaded and processed successfully with enhanced ChromaDB RAG system.",
            "details": {
                "filename": safe_filename,
                "text_length": len(text),
                "chunks_created": len(pdf_chunks),
                "embedding_dimensions": pdf_embeddings.shape[1] if pdf_embeddings.size > 0 else 0,
                "processing_model": MODEL_NAME,
                "vector_store": "ChromaDB",  # Updated to reflect ChromaDB usage
                "collection_saved": True
            },
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error processing PDF upload: {str(e)}", exc_info=True)
        
        if 'file_path' in locals() and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        
        return {
            "error": f"Failed to process PDF: {str(e)}",
            "status": "failed"
        }


@app.delete("/delete")
async def delete_pdf(filename: str):
    safe_filename = re.sub(r'[^\w\-_\.]', '_', filename)
    file_path = f"uploads/{safe_filename}"

    # 1. Delete the uploaded PDF
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        raise HTTPException(status_code=404, detail="PDF not found")

    # 2. Clear ChromaDB collection
    try:
        from rag_logic import _reset_collection
        _reset_collection()  # This properly clears the collection
        logger.info("Cleared ChromaDB collection")
    except Exception as e:
        logger.warning(f"Could not clear ChromaDB collection: {e}")

    # 3. Clear in-memory variables
    global pdf_chunks, pdf_embeddings
    pdf_chunks = []
    pdf_embeddings = np.array([], dtype=np.float32)

    return {"message": "PDF and associated ChromaDB data deleted successfully"}

@app.post("/ask")
async def ask_question(query: str = Form(...)):
    global pdf_chunks, pdf_embeddings

    if not pdf_chunks or pdf_embeddings.size == 0:
        return {"answer": "Please upload and process a PDF first."}

    print("📥 Query:", query)

    # This function now uses ChromaDB internally but maintains the same interface
    relevant = retrieve_relevant_chunks(query, pdf_chunks, pdf_embeddings)
    print("🔍 Top Relevant Chunks (via ChromaDB):")
    for i, c in enumerate(relevant):
        print(f"\n--- Chunk {i+1} ---\n{c}")

    response = ask_llm(query, "\n".join(relevant))
    return {"answer": response}