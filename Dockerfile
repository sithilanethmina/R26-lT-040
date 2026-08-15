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
ENV PYTHONPATH="/app:/app/api-gateway:/app/gpu-price-model/src"
ENV PORT=10000

EXPOSE 10000

# Start command: launches all model microservices and API gateway together
CMD ["python", "api-gateway/start_all.py"]
