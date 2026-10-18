# ==============================================================================
# FairPriceLK Consolidated Production Dockerfile
# Optimized for low-memory environments (Oracle Cloud Free Tier 1GB RAM)
# ==============================================================================

FROM python:3.10-slim

WORKDIR /app

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:/app/api-gateway:/app/gpu-price-model/src" \
    CONSOLIDATED_MODE="true" \
    PORT=8000

# Install runtime and build dependencies for ML C-extensions (LightGBM, XGBoost, CatBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade packaging tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Leverage Docker layer caching for requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and trained model artifacts
COPY . /app/

EXPOSE 8000

# Healthcheck to verify the consolidated gateway and downstream models are healthy
HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Launch the consolidated in-memory ASGI gateway (consumes ~240MB RAM vs ~700MB across 5 processes)
CMD ["python", "api-gateway/gateway.py"]
