# Use Python 3.11 slim as base image for better performance and smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Clone the repository
RUN git clone https://github.com/pradeepodela/conductor-python-ML-Workers.git .

# Install Python dependencies if requirements.txt exists
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Install Conductor SDK and common ML libraries

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash conductor && \
    chown -R conductor:conductor /app
USER conductor

# Expose port (adjust if your workers use a different port)
EXPOSE 8080

# Set default environment variables for Conductor
ENV CONDUCTOR_SERVER_URL="" \
    CONDUCTOR_AUTH_KEY="" \
    CONDUCTOR_AUTH_SECRET="" \
    WORKER_DOMAIN="" \
    POLLING_INTERVAL=1000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=5)" || exit 1

# Default command - adjust based on your main worker file
CMD ["python", "runs.py"]