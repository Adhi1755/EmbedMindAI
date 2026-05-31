FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (build tools + OpenMP for numpy/torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install all Python dependencies
# Install CPU-only PyTorch FIRST to prevent pip from pulling CUDA packages (~2GB)
COPY requirements.txt .
RUN pip install --upgrade pip --root-user-action=ignore && \
    pip install --no-cache-dir --root-user-action=ignore \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Pre-download the sentence-transformers model so first startup is instant
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-small-v2')"

# Copy application source
COPY . .

# Create persistent storage directories
RUN mkdir -p uploads chromadb_storage

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/test')" || exit 1

# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
