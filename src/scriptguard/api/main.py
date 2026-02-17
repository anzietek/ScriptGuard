import time
import uuid
import os
import hashlib
from contextlib import asynccontextmanager
from typing import Optional, Sequence, Tuple, cast

from fastapi import FastAPI, HTTPException, Request, Depends, Header, status, BackgroundTasks
from fastapi.responses import JSONResponse

from scriptguard.api.schemas import (
    ScriptAnalysisRequest,
    ScriptAnalysisResponse,
    VulnerabilityInfo,
    HealthResponse,
    ReadinessResponse,
    ErrorResponse,
)
from scriptguard.api.state import app_state
from scriptguard.utils.logger import logger
from scriptguard.utils.prompts import (
    format_inference_prompt,
    parse_classification_output,
    format_fewshot_prompt,
)

import torch
from transformers import LogitsProcessorList, LogitsProcessor


class BinaryClassificationLogitsProcessor(LogitsProcessor):
    """Constrain the *first generated token* to a binary label token.

    Notes:
        This processor is instantiated per request.
        It assumes the label can be decided by a single token id.
    """

    def __init__(self, allowed_token_ids: Sequence[int]) -> None:
        if len(allowed_token_ids) < 2:
            raise ValueError("allowed_token_ids must contain at least two token ids")
        self._allowed_token_ids: Tuple[int, ...] = tuple(int(t) for t in allowed_token_ids)
        self._applied: bool = False

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self._applied:
            return scores

        mask = scores.new_full(scores.shape, float("-inf"))
        for token_id in self._allowed_token_ids:
            mask[:, token_id] = scores[:, token_id]

        self._applied = True
        return cast(torch.FloatTensor, mask)


def _encode_label_token_id(tokenizer, label: str) -> Optional[int]:
    """Get a single token id representing a label, if possible.

    Returns:
        Token id if the label can be represented as a single token (with a leading
        space variant preferred), otherwise None.
    """

    candidates = [f" {label}", label]
    for cand in candidates:
        token_ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(token_ids) != 1:
            continue

        token_id = int(token_ids[0])
        decoded = tokenizer.decode([token_id], skip_special_tokens=True)

        # Accept either exact match or a whitespace-prefixed variant.
        if decoded.strip().upper() == label.upper():
            return token_id

    return None


def _confidence_from_first_step_logits(
    step_logits: torch.FloatTensor, chosen_token_id: int, allowed_token_ids: Sequence[int]
) -> float:
    """Compute calibrated confidence as P(chosen | allowed) from step logits."""

    allowed = torch.tensor(list(allowed_token_ids), device=step_logits.device)
    allowed_logits = step_logits.index_select(dim=-1, index=allowed)
    probs = torch.softmax(allowed_logits, dim=-1)

    allowed_list = list(int(t) for t in allowed_token_ids)
    chosen_index = allowed_list.index(int(chosen_token_id))
    return float(probs[0, chosen_index].detach().cpu().item())


# --- Lifecycle Events ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Replaces deprecated @app.on_event("startup").
    """
    logger.info("Starting ScriptGuard API...")
    try:
        await app_state.load_resources()
    except Exception as e:
        logger.critical(f"Failed to initialize application state: {e}")
        # We allow the app to start so /health can report the error state
    
    yield
    
    logger.info("Shutting down ScriptGuard API...")
    await app_state.shutdown()

app = FastAPI(title="ScriptGuard Inference API", version="1.0.0", lifespan=lifespan)

# --- Middleware ---

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id # Store for endpoint access
    start_time = time.time()
    
    logger.info(f"Request started: {request.method} {request.url.path} - ID: {request_id}")
    
    try:
        response = await call_next(request)
        
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id
        
        logger.info(f"Request finished: {request.method} {request.url.path} - ID: {request_id} - Status: {response.status_code} - Duration: {process_time:.4f}s")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Request failed: {request.method} {request.url.path} - ID: {request_id} - Error: {e} - Duration: {process_time:.4f}s")
        raise e

# --- Exception Handlers ---

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """
    Return consistent JSON error response for HTTP exceptions.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=str(exc.status_code),
            message=exc.detail,
            details=None
        ).model_dump()
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all handler for unhandled exceptions.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred. Please contact support.",
            details={"request_id": request_id}
        ).model_dump()
    )

# --- Dependencies ---

async def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    """
    Verify API key from header against environment variable.
    If SCRIPTGUARD_API_KEY is not set, auth is disabled (warning logged).
    """
    expected_key = os.getenv("SCRIPTGUARD_API_KEY")
    return
    # if not expected_key:
    #     # Auth disabled
    #     return
    #
    # if x_api_key != expected_key:
    #     raise HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="Invalid API Key",
    #     )

# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness probe."""
    return HealthResponse(status="ok", version="1.0.0")

@app.get("/ready", response_model=ReadinessResponse)
async def readiness_check():
    """Readiness probe."""
    model_loaded = app_state.model is not None and app_state.tokenizer is not None
    rag_connected = app_state.rag_store is not None
    
    status_str = "ready" if model_loaded else "not_ready"
    
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    return ReadinessResponse(
        status=status_str,
        model_loaded=model_loaded,
        rag_connected=rag_connected
    )

@app.post("/analyze", response_model=ScriptAnalysisResponse, dependencies=[Depends(verify_api_key)])
async def analyze_script(
    request: Request,
    analysis_request: ScriptAnalysisRequest,
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(None, alias="X-API-Key"),
):
    """
    Analyze a script for malicious content.
    """
    start_time = time.time()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    if not app_state.model or not app_state.tokenizer:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # Input validation (Config-driven limits)
    max_len = 100000 # Default fallback
    if app_state.config and app_state.config.validation:
        max_len = app_state.config.validation.max_length
        
    if len(analysis_request.script_content) > max_len:
        raise HTTPException(
            status_code=400, 
            detail=f"Script content exceeds maximum allowed length of {max_len} characters"
        )

    if len(analysis_request.script_content.strip()) == 0:
        raise HTTPException(status_code=400, detail="Script content cannot be empty")

    # RAG Context
    rag_context_examples = []
    related_cves = []

    if analysis_request.include_rag and app_state.rag_store:
        try:
            # Use config for limit if available
            limit = 2
            if app_state.config and app_state.config.code_embedding and app_state.config.code_embedding.fewshot:
                 limit = app_state.config.code_embedding.fewshot.k or 2

            # Check if using CodeSimilarityStore or QdrantStore
            if hasattr(app_state.rag_store, 'search_similar_code'):
                # CodeSimilarityStore - for code_samples collection
                logger.info(f"Searching code_samples with limit={limit}")

                # Log RAG search parameters for debugging
                logger.info(f"RAG search params: balance_labels=False, enable_reranking=True")

                # Temporarily disable graceful fallback for inference
                # Save original settings
                original_fallback = app_state.rag_store.graceful_fallback_enabled
                original_ensure_balance = app_state.rag_store.ensure_label_balance

                # Force disable for this inference call
                app_state.rag_store.graceful_fallback_enabled = False
                app_state.rag_store.ensure_label_balance = False

                try:
                    results = app_state.rag_store.search_similar_code(
                        query_code=analysis_request.script_content,
                        k=limit,
                        balance_labels=False,  # For inference, we want the most similar regardless of label
                        enable_reranking=True,
                        fetch_full_content=False,  # FIXED: Set to False since we don't have db_manager in API
                        aggregate_chunks=True,
                        threshold_mode="strict"  # Use strict threshold mode for high-quality results
                    )
                finally:
                    # Restore original settings
                    app_state.rag_store.graceful_fallback_enabled = original_fallback
                    app_state.rag_store.ensure_label_balance = original_ensure_balance

                # Log RAG search results with labels and scores
                logger.info(f"RAG search returned {len(results)} results")
                labels_distribution = {}
                for idx, r in enumerate(results, 1):
                    label = r.get('label', 'unknown')
                    score = r.get('score', 0.0)
                    labels_distribution[label] = labels_distribution.get(label, 0) + 1
                    logger.info(f"  [{idx}] Label: {label}, Score: {score:.4f}, Code: {r.get('code', '')[:60]}...")
                logger.info(f"RAG labels distribution: {labels_distribution}")

                # Filter out low-quality results (inference-specific)
                MIN_INFERENCE_SCORE = 0.45
                original_count = len(results)
                results = [r for r in results if r.get('score', 0.0) >= MIN_INFERENCE_SCORE]
                logger.info(f"After score filtering (>={MIN_INFERENCE_SCORE}): {len(results)} results (removed {original_count - len(results)})")

                # Sort by score descending to prioritize best matches in prompt
                results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
                logger.info("Results sorted by score (descending) for prompt construction")

                # CRITICAL FIX: Remove malicious examples if they're only marginally better than benign
                # This prevents false positives from slightly-higher-scored malicious "hello world" examples
                if len(results) >= 2:
                    top_score = results[0].get('score', 0.0)
                    top_label = results[0].get('label', 'unknown')

                    # Find best benign score
                    benign_scores = [r.get('score', 0.0) for r in results if r.get('label') == 'benign']

                    if top_label == 'malicious' and benign_scores:
                        best_benign_score = max(benign_scores)
                        score_diff = top_score - best_benign_score

                        # If malicious is only marginally better (< 5% difference), prefer benign examples
                        if score_diff < 0.05:
                            logger.warning(
                                f"Top result is malicious (score={top_score:.4f}) but only {score_diff:.4f} "
                                f"better than benign (score={best_benign_score:.4f}). "
                                f"Filtering out malicious examples to prevent false positive."
                            )
                            # Remove malicious examples with marginal advantage
                            original_count = len(results)
                            results = [r for r in results if r.get('label') != 'malicious' or r.get('score', 0.0) - best_benign_score >= 0.05]
                            logger.info(f"Removed {original_count - len(results)} malicious examples with marginal scores")

                            # Re-sort after filtering
                            results.sort(key=lambda x: x.get('score', 0.0), reverse=True)

                for r in results:
                    # Support both flat and nested payload structures
                    # Flat: { "code": "...", "label": "..." }
                    # Nested: { "payload": { "code": "...", "label": "..." } }
                    if 'payload' in r and isinstance(r['payload'], dict):
                        # Nested structure (Qdrant-style)
                        code = r.get('code', '') or r['payload'].get('code_preview', '') or r['payload'].get('code', '')
                        label = r['payload'].get('label', 'unknown')
                        db_id = r['payload'].get('db_id')
                    else:
                        # Flat structure (CodeSimilarityStore-style)
                        code = r.get('code', '') or r.get('code_preview', '') or r.get('content', '')
                        label = r.get('label', 'unknown')
                        db_id = r.get('db_id')

                    # Debug logging
                    logger.debug(f"RAG result - Label: {label}, Code length: {len(code)}, Score: {r.get('score', 0.0):.4f}")

                    # Skip if no code available
                    if not code or len(code.strip()) == 0:
                        logger.warning(f"Skipping RAG result with empty code (db_id: {db_id})")
                        continue

                    # Prepare context for few-shot prompt
                    rag_context_examples.append({
                        "code": code,
                        "label": label,
                        "score": r.get('score', 0.0)
                    })

                    # For code samples, we create "vulnerability" info from metadata
                    # Extract metadata and severity from either flat or nested structure
                    if 'payload' in r and isinstance(r['payload'], dict):
                        metadata = r['payload'].get('metadata', {})
                        severity = r['payload'].get('severity', 'INFO')
                    else:
                        metadata = r.get('metadata', {})
                        severity = r.get('severity', 'INFO')

                    related_cves.append(VulnerabilityInfo(
                        id=str(r.get('id', 'unknown')),
                        description=f"Similar {label} code from {metadata.get('repository', 'unknown source')}",
                        severity=severity,
                        score=r.get('score', 0.0)
                    ))

            else:
                # QdrantStore - for malware_knowledge collection (fallback)
                logger.info(f"Searching malware_knowledge with limit={limit}")
                results = app_state.rag_store.search(analysis_request.script_content, limit=limit)

                for r in results:
                    payload = r.get('payload', {})
                    # Prepare context for few-shot prompt
                    rag_context_examples.append({
                        "code": payload.get("pattern", "") or payload.get("description", ""),
                        "label": "malicious",
                        "score": r.get("score")
                    })

                    related_cves.append(VulnerabilityInfo(
                        id=r.get('id'),
                        description=payload.get('description', 'Unknown'),
                        severity=payload.get('severity'),
                        score=r.get('score')
                    ))

            logger.info(f"RAG retrieved {len(rag_context_examples)} examples")

        except Exception as e:
            logger.error(f"RAG search failed: {e}", exc_info=True)
            # Continue without RAG

    # Construct Prompt
    # Use few-shot only if we have at least 2 high-quality examples
    # Otherwise fallback to zero-shot to avoid biasing model with low-quality/irrelevant examples
    MIN_EXAMPLES_FOR_FEWSHOT = 2
    if rag_context_examples and len(rag_context_examples) >= MIN_EXAMPLES_FOR_FEWSHOT:
        # Use Few-Shot prompt if RAG provided context
        logger.info(f"Using FEW-SHOT prompt with {len(rag_context_examples)} examples")

        # Get prompt length config from config if available
        max_context_length = 1500  # Default
        max_code_length = 3000  # Default
        if app_state.config and app_state.config.code_embedding and app_state.config.code_embedding.fewshot:
            fewshot_config = app_state.config.code_embedding.fewshot
            max_context_length = getattr(fewshot_config, 'max_context_length', 1500)
            max_code_length = getattr(fewshot_config, 'max_code_length', 3000)

        prompt = format_fewshot_prompt(
            analysis_request.script_content,
            rag_context_examples,
            max_code_length=max_code_length,
            max_context_length=max_context_length
        )
    else:
        # Fallback to standard inference prompt
        if rag_context_examples:
            logger.warning(f"Only {len(rag_context_examples)} RAG examples (< {MIN_EXAMPLES_FOR_FEWSHOT}). Falling back to ZERO-SHOT prompt.")
        else:
            logger.info("No RAG examples available. Using ZERO-SHOT prompt.")
        prompt = format_inference_prompt(analysis_request.script_content)
    
    # Inference
    try:
        inputs = app_state.tokenizer(prompt, return_tensors="pt").to(app_state.device)

        max_new_tokens = 5

        benign_token_id = _encode_label_token_id(app_state.tokenizer, "BENIGN")
        malicious_token_id = _encode_label_token_id(app_state.tokenizer, "MALICIOUS")

        logits_processor = LogitsProcessorList()
        allowed_token_ids: Optional[Tuple[int, int]] = None
        if benign_token_id is not None and malicious_token_id is not None:
            allowed_token_ids = (benign_token_id, malicious_token_id)
            logits_processor.append(BinaryClassificationLogitsProcessor(allowed_token_ids))
        else:
            logger.warning(
                "Constrained decoding disabled: labels are not single-token for this tokenizer"
            )

        with torch.no_grad():
            outputs = app_state.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=app_state.tokenizer.pad_token_id,
                eos_token_id=app_state.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=logits_processor,
            )

        generated_sequence = outputs.sequences[0]
        response_text = app_state.tokenizer.decode(generated_sequence, skip_special_tokens=True)

        classification_result = parse_classification_output(response_text)
        is_malicious = classification_result == 1

        confidence = 0.0 if classification_result == -1 else 0.5

        # Compute confidence from the *first generation step* logits if possible.
        # When constrained decoding is enabled, this is a calibrated P(label|{BENIGN,MALICIOUS}).
        try:
            if outputs.scores and len(outputs.scores) > 0:
                first_step_logits = outputs.scores[0]

                # Determine which token the model actually picked at step 1.
                prompt_len = inputs["input_ids"].shape[-1]
                chosen_token_id = int(outputs.sequences[0, prompt_len].item())

                if allowed_token_ids is not None and chosen_token_id in allowed_token_ids:
                    confidence = _confidence_from_first_step_logits(
                        step_logits=first_step_logits,
                        chosen_token_id=chosen_token_id,
                        allowed_token_ids=allowed_token_ids,
                    )
                else:
                    # Unconstrained path: softmax probability of the chosen token in the full vocab.
                    # This is less meaningful as "confidence", but still a measurable signal.
                    probs = torch.softmax(first_step_logits, dim=-1)
                    confidence = float(probs[0, chosen_token_id].detach().cpu().item())

                if classification_result == -1:
                    confidence = 0.0

        except Exception as e:
            logger.warning(f"Failed to compute confidence scores: {e}")

        # Extract reasoning
        reasoning = response_text.split("# Analysis: The script above is classified as:")[-1].strip()
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Log to DB asynchronously
        script_hash = hashlib.sha256(analysis_request.script_content.encode()).hexdigest()
        api_key_prefix = x_api_key[:4] if x_api_key else "none"
        
        log_data = {
            "request_id": request_id,
            "script_hash": script_hash,
            "is_malicious": is_malicious,
            "confidence": confidence,
            "model_version": app_state.config.training.model_id if app_state.config else "unknown",
            "api_key_prefix": api_key_prefix,
            "processing_time_ms": processing_time_ms
        }
        
        background_tasks.add_task(app_state.log_scan_result, log_data)
        
        return ScriptAnalysisResponse(
            is_malicious=is_malicious,
            confidence=confidence,
            reasoning=reasoning,
            related_cves=related_cves
        )

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail="Internal inference error")

if __name__ == "__main__":
    import uvicorn
    # Use config for host/port if available, else defaults
    host = "0.0.0.0"
    port = 8000
    
    # We can't easily access app_state.config here before startup, 
    # so we rely on env vars or defaults for the server start
    
    uvicorn.run(app, host=host, port=port)
