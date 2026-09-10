# ==============================================================================
# FedTrust-Credit — Cloud Production Dockerfile
# Course: CLOUD COMPUTING – BITE412L, Fall Semester 2026-27
# Lightweight container (<400MB) for Aggregator & FastAPI Credit Risk Service
# ==============================================================================

FROM python:3.11-slim AS base

# Install system dependencies (libgomp1 for LightGBM OpenMP support, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code, web dashboard, docs, results, and pre-trained models
COPY src/ ./src/
COPY results/ ./results/
COPY docs/ ./docs/
COPY models/ ./models/
COPY run_service.py .
COPY README.md .
COPY LICENSE .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0

# Expose HTTP API & Dashboard port
EXPOSE 8000

# Health check using FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production Uvicorn server on all interfaces
CMD ["python", "run_service.py", "--host", "0.0.0.0", "--port", "8000"]
