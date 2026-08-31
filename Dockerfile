# Use Python 3.10 slim as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for C extensions, LightGBM, XGBoost, CatBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy root requirements first for layer caching
COPY requirements.txt /app/

# Install all Python dependencies together to avoid version conflicts
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . /app/

# Set Python path environment variable
ENV PYTHONPATH="/app:/app/api-gateway:/app/gpu-price-model/src"
ENV PORT=10000

EXPOSE 10000

# Start command: launches all model microservices and API gateway together
CMD ["python", "api-gateway/start_all.py"]
