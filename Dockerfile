# Use Python 3.10 slim as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for C extensions, LightGBM, XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app/

# Install Python requirements across all microservices
RUN pip install --no-cache-dir -r gpu-price-model/requirements.txt
RUN pip install --no-cache-dir -r mobile-price-model/requirements.txt
RUN pip install --no-cache-dir -r vehicle-price-model/requirements.txt
RUN pip install --no-cache-dir -r electronics-price-model/requirements.txt
RUN pip install --no-cache-dir -r api-gateway/requirements.txt

# Set Python path environment variable
ENV PYTHONPATH="/app/gpu-price-model/src"
ENV PORT=7860

# Hugging Face Spaces default port
EXPOSE 7860

# Start command: launches all services on ports 8001-8004 and API gateway on Hugging Face port 7860
CMD ["python", "-m", "uvicorn", "api-gateway.gateway:app", "--host", "0.0.0.0", "--port", "7860"]
