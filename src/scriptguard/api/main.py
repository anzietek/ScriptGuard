import time
import uuid
import os
import hashlib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Depends, Header, status, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
from scriptguard.api.schemas import (
    ScriptAnalysisRequest, 
    ScriptAnalysisResponse, 
    VulnerabilityInfo,
    HealthResponse,
    ReadinessResponse,
    ErrorResponse
)
from scriptguard.api.state import app_state
from scriptguard.utils.logger import logger
from scriptguard.utils.prompts import format_inference_prompt, parse_classification_output
import torch

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
    start_time = time.time()
    
    # Add request ID to logger context if possible, or just log it
    # For simplicity here, we'll just log the start
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
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred. Please contact support.",
            details={"request_id": request.headers.get("X-Request-ID")}
        ).model_dump()
    )

# --- Dependencies ---

async def verify_api_key(x_api_key: str = Header(None)):
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
    request: ScriptAnalysisRequest, 
    background_tasks: BackgroundTasks,
    x_request_id: str = Header(None),
    x_api_key: str = Header(None)
):
    """
    Analyze a script for malicious content.
    """
    start_time = time.time()
    
    if not app_state.model or not app_state.tokenizer:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # Input validation (basic length check handled by Pydantic, but we can add more)
    if len(request.script_content.strip()) == 0:
        raise HTTPException(status_code=400, detail="Script content cannot be empty")

    # RAG Context
    rag_context = ""
    related_cves = []
    
    if request.include_rag and app_state.rag_store:
        try:
            # Use config for limit if available
            limit = 2
            if app_state.config and app_state.config.code_embedding and app_state.config.code_embedding.fewshot:
                 limit = app_state.config.code_embedding.fewshot.k or 2

            results = app_state.rag_store.search(request.script_content, limit=limit)
            
            for r in results:
                payload = r.get('payload', {})
                related_cves.append(VulnerabilityInfo(
                    id=r.get('id'),
                    description=payload.get('description', 'Unknown'),
                    severity=payload.get('severity'),
                    score=r.get('score')
                ))
            
            rag_context = "\n".join([f"Known Vulnerability: {c.description}" for c in related_cves])
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")
            # Continue without RAG

    # Construct Prompt using centralized utility
    # Note: We are currently not using the RAG context in the prompt format provided by prompts.py
    # If we want to use RAG context, we should update prompts.py or use format_fewshot_prompt if applicable
    # For now, we stick to the standard inference prompt to ensure consistency with training
    
    prompt = format_inference_prompt(request.script_content)
    
    # Inference
    try:
        inputs = app_state.tokenizer(prompt, return_tensors="pt").to(app_state.device)
        
        # Get generation config from app_state.config if available
        max_new_tokens = 20 # Reduced since we only expect BENIGN/MALICIOUS
        temperature = 0.1
        if app_state.config:
            # Override max_length for classification task
            temperature = app_state.config.inference.temperature

        with torch.no_grad():
            outputs = app_state.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False, # Deterministic
                temperature=temperature if temperature > 0 else None, # Only if sampling
                pad_token_id=app_state.tokenizer.pad_token_id,
                eos_token_id=app_state.tokenizer.eos_token_id
            )

        response_text = app_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-processing / Parsing using centralized utility
        classification_result = parse_classification_output(response_text)
        
        is_malicious = classification_result == 1
        confidence = 0.95 if classification_result != -1 else 0.5
        
        # Extract reasoning (everything after the classification)
        # This is a heuristic since the prompt asks for one word, but the model might generate more
        reasoning = response_text.split("# Analysis: The script above is classified as:")[-1].strip()
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Log to DB asynchronously
        script_hash = hashlib.sha256(request.script_content.encode()).hexdigest()
        api_key_prefix = x_api_key[:4] if x_api_key else "none"
        
        log_data = {
            "request_id": x_request_id or str(uuid.uuid4()),
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
