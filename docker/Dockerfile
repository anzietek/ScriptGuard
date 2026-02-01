# ============================================
# Stage 1: Base Builder
# ============================================
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS base-builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    ca-certificates \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Install uv for faster package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build


# ============================================
# Stage 2: Dependencies Builder
# ============================================
FROM base-builder AS dependencies

# Copy dependency files
COPY pyproject.toml README.md ./

# Install Python dependencies
RUN uv pip install --system --no-deps \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN uv pip install --system .

# Clean up
RUN apt-get clean && \
    rm -rf /root/.cache/pip /root/.cache/uv


# ============================================
# Stage 3: Training Image
# ============================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS training

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-distutils \
    libpq5 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ /app/src/
COPY config.yaml /app/config.yaml

# Set environment variables
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Create directories for models and logs
RUN mkdir -p /app/models /app/logs /app/data

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "src/main.py"]


# ============================================
# Stage 4: Inference API Image
# ============================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS inference

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-distutils \
    libpq5 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy only necessary application code for inference
COPY src/scriptguard/api/ /app/src/scriptguard/api/
COPY src/scriptguard/models/ /app/src/scriptguard/models/
COPY src/scriptguard/rag/ /app/src/scriptguard/rag/
COPY src/scriptguard/database/ /app/src/scriptguard/database/
COPY src/scriptguard/__init__.py /app/src/scriptguard/__init__.py

# Set environment variables
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Create directories
RUN mkdir -p /app/models /app/logs

# Expose port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API server
CMD ["uvicorn", "scriptguard.api.inference:app", "--host", "0.0.0.0", "--port", "8000"]


# ============================================
# Stage 5: Development Image
# ============================================
FROM base-builder AS development

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    nano \
    tmux \
    htop \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages
COPY --from=dependencies /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Install development dependencies
RUN pip install --no-cache-dir \
    ipython \
    jupyter \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy

WORKDIR /app

# Copy all source code
COPY . /app/

ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Expose Jupyter port
EXPOSE 8888

CMD ["/bin/bash"]
