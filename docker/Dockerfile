# Use a CUDA-enabled base image for GPU support
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install Python and essential tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for high-performance package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy dependency files
COPY pyproject.toml .
# COPY README.md .

# Install dependencies using uv
RUN uv pip install --system .

# Copy source code
COPY src /app/src

# Set environment variables
ENV PYTHONPATH="/app/src:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "scriptguard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
