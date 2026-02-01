# ScriptGuard: Production-Ready Malware Detection for Scripts

ScriptGuard is an advanced AI-powered system designed to detect malicious and dangerous scripts using state-of-the-art Python techniques, ZenML pipelines, and RAG architecture.

## Architecture

- **Base Model:** `bigcode/starcoder2-3b` (Optimized for code analysis).
- **Fine-tuning:** Parameter-efficient fine-tuning using **QLoRA** (4-bit quantization).
- **Orchestration:** **ZenML** manages the end-to-end ML lifecycle (Ingestion -> Preprocessing -> Training -> Evaluation).
- **RAG:** **Qdrant** stores embeddings of known CVEs and exploits to provide context during inference.
- **Tracking:** **Comet.ml** monitors experiments and metrics.
- **Inference:** **FastAPI** provides a high-performance REST API for script analysis.
- **Containerization:** **Docker Compose** orchestrates the API and Vector DB services.

## Tech Stack

- **Package Manager:** `uv`
- **Orchestration:** `ZenML`
- **Vector DB:** `Qdrant`
- **Fine-tuning:** `PEFT (LoRA/QLoRA)`
- **Experiment Tracking:** `Comet.ml`
- **Serving:** `FastAPI` & `Uvicorn`

## Project Structure

```text
├── docker/                 # Containerization configs
├── src/
│   ├── scriptguard/
│   │   ├── api/            # FastAPI inference service
│   │   ├── models/         # QLoRA fine-tuning logic
│   │   ├── pipelines/      # ZenML pipeline definitions
│   │   ├── rag/            # Qdrant RAG store implementation
│   │   └── steps/          # ZenML atomic steps
│   └── main.py             # Pipeline entry point
├── pyproject.toml          # uv dependency management
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker & Docker Compose (with NVIDIA Container Toolkit for GPU support)

### Setup

1. Clone the repository.
2. Install dependencies:
   ```bash
   uv pip install -e .
   ```
3. Set up environment variables in a `.env` file (see `.env.example`).

### Running the Training Pipeline

```bash
python src/main.py
```

### Deployment

To start the inference API and Qdrant:

```bash
docker-compose -f docker/docker-compose.yml up --build
```

## API Usage

Analyze a script:

```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"script_content": "import os; os.system(\"rm -rf /\")"}'
```
