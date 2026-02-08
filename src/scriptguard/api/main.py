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
    
    if not expected_key:
        # Auth disabled
        return
        
    if x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )

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

            results = app_state.rag_store.search(analysis_request.script_content, limit=limit)
            
            for r in results:
                payload = r.get('payload', {})
                # Prepare context for few-shot prompt
                rag_context_examples.append({
                    "code": payload.get("pattern", "") or payload.get("description", ""), # Use pattern if available as code example
                    "label": "malicious", # Assuming RAG returns malicious patterns/CVEs
                    "score": r.get("score")
                })
                
                related_cves.append(VulnerabilityInfo(
                    id=r.get('id'),
                    description=payload.get('description', 'Unknown'),
                    severity=payload.get('severity'),
                    score=r.get('score')
                ))
            
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")
            # Continue without RAG

    # Construct Prompt
    if rag_context_examples:
        # Use Few-Shot prompt if RAG provided context
        prompt = format_fewshot_prompt(analysis_request.script_content, rag_context_examples)
    else:
        # Fallback to standard inference prompt
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
