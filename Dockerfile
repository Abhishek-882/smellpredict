FROM python:3.11-slim

LABEL maintainer="SmellPredict Research Team"
LABEL description="SmellPredict — Empirical Code Smell & Bug-Fix Defect Intelligence Platform"
LABEL version="2.0.0"

# System dependencies: git, libgomp1 (for LightGBM), curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
COPY requirements-lock.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-lock.txt

# Copy source code and configurations
COPY src/ src/
COPY config/ config/
COPY platform_ui/ platform_ui/

# Copy trained champion models into the container
COPY models/ models/

# Create runtime directories
RUN mkdir -p data/raw data/processed data/labels data/external logs reports

# Environment variables
ENV PYTHONPATH=src
ENV SMELLPREDICT_MODEL_PATH=models/best_model_final.pkl
ENV SMELLPREDICT_DATA_DIR=data
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Expose default HTTP port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI server on dynamic $PORT (supports Render, Railway, Fly.io, Cloud Run, Docker)
CMD ["sh", "-c", "uvicorn smellpredict.platform.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
