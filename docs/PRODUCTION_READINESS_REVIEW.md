# ScriptGuard Production Readiness Review (API & Inference)

This document reviews the current repository state from a production usage perspective.
It is strictly code-focused: API behavior, inference correctness, security controls implemented in code, and operability implemented in code.

## Scope

**Reviewed:**
- `src/scriptguard/api/main.py`
- `src/main.py` (config loading pattern)
- `config.yaml` (inference-related settings)
- `src/scriptguard/schemas/config_schema.py`
- `src/scriptguard/utils/logger.py`
- `src/scriptguard/rag/qdrant_store.py`

## Review Goal

Provide an actionable list of code changes needed to make the API/inference layer production-ready.

## Out of Scope

- Docker, Kubernetes, reverse proxies, TLS termination
- Runtime infrastructure planning and operational runbooks
- Training pipeline improvements

## Current API/Inference Implementation (Observed)

- API module: `src/scriptguard/api/main.py`.
- Startup loads tokenizer and a causal LM (`AutoModelForCausalLM`) and optionally a LoRA adapter.
- Endpoints:
  - `GET /health`
  - `GET /ready`
  - `POST /analyze` (protected by optional API key)
- Request correlation:
  - middleware generates `request.state.request_id` and returns it as `X-Request-ID`.
- Input validation:
  - request size is limited using `config.validation.max_length` (fallback exists).
- RAG:
  - Qdrant is initialized via `AppState._load_rag()`.
  - bootstrapping is gated by env var `BOOTSTRAP_QDRANT` and/or `config.qdrant.bootstrap_on_startup`.
  - when retrieval returns results, the service uses few-shot prompt formatting.
- Classification:
  - prompt enforces one-word output (`BENIGN` or `MALICIOUS`), parsing uses `parse_classification_output()`.

## Code-Only Production Readiness Gaps (Prioritized)

### P0 — Blocking issues

#### 1) Constrained decoding correctness (tokenization edge cases)

**Current**
- API enforces constrained decoding via a custom `LogitsProcessor` to allow only `BENIGN` or `MALICIOUS` at the first generated token.
- The implementation attempts to resolve each label to a *single token id* (preferring a leading-space variant).

**Residual risk**
- Tokenization may split labels into multiple tokens for some tokenizers/models.
- If a label is not representable as a single token, constrained decoding must be disabled or replaced with a multi-token constraint or a classifier head.

**Code changes still recommended**
- Keep (and log) the single-token resolution check.
- If single-token resolution fails:
  - preferred: use a sequence-classification head (`AutoModelForSequenceClassification`), or
  - alternative: implement a multi-token constraint strategy.

#### 2) Confidence definition and calibration

**Current**
- Confidence is derived from model scores at the *first decoding step*.
- When constrained decoding is enabled, confidence is computed as a calibrated probability:
  - `confidence = softmax([logit(BENIGN), logit(MALICIOUS)])[chosen]`
- When constrained decoding is disabled (labels not single-token), confidence falls back to the chosen-token probability in the full vocabulary (less interpretable).

**Residual risk**
- In the unconstrained fallback path, “confidence” is not comparable to the constrained path.
- There is no dataset-level calibration (e.g., temperature scaling, ROC-based thresholding).

**Code changes still recommended**
- Consider returning an explicit confidence type/semantics (e.g., `confidence_mode: constrained|unconstrained`).
- Optionally add offline calibration for the constrained probability.

## Code Review Findings (Re-check)

### Fixed / Implemented (confirmed in code)

- API key header alias (`X-API-Key`).
- Request ID propagation via `request.state.request_id` and `X-Request-ID` response header.
- Health/readiness endpoints: `GET /health` and `GET /ready`.
- Config-driven request size limit (`config.validation.max_length`) with safe fallback.
- Few-shot prompt usage when RAG retrieval returns results.
- Qdrant bootstrap is gated (disabled by default).
- Optional DB logging via optional `asyncpg` import.

### Remaining issues / improvements (still relevant)

1) **Constrained decoding tokenization edge cases**
- Labels may be multi-token depending on tokenizer.
- Single-token resolution must be validated per tokenizer/model.

2) **Confidence semantics in unconstrained fallback path**
- When constrained decoding is disabled, the confidence fallback is probability of the chosen token in the full vocabulary.
- This is a different quantity than P(label|{BENIGN,MALICIOUS}).

3) **API schema versioning**
- There is no explicit compatibility/versioning mechanism for clients.

## Concrete Code Change Checklist

### P0
- [x] Minimal auth hook (API key) with explicit header alias (`X-API-Key`)
- [x] `/health` and `/ready`
- [x] Input validation + size limits (config-driven)
- [x] Confidence derived from measurable model signal (logits/probabilities) — *implemented for constrained decoding path*
- [ ] Deterministic classification for tokenizers where labels are multi-token (fallback strategy)

### P1
- [ ] Response schema versioning strategy (e.g., `/v1/*` or response `version` field)
- [ ] (Optional) Calibrate the constrained probability (dataset-level)

### P2
- [ ] (Optional) Pytest suite for API behaviors (intentionally deferred for now)
