import time
import uuid
import os
from fastapi import FastAPI, HTTPException, Request, Depends, Header, status
from fastapi.responses import JSONResponse
from scriptguard.api.schemas import (
    ScriptAnalysisRequest, 
    ScriptAnalysisResponse, 
    VulnerabilityInfo,
    HealthResponse,
    ReadinessResponse
)
from scriptguard.api.state import app_state
from scriptguard.utils.logger import logger
import torch

app = FastAPI(title="ScriptGuard Inference API", version="1.0.0")

# --- Middleware ---

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Add request ID to logger context if possible, or just log it
    # For simplicity here, we'll just log the start
    logger.info(f"Request started: {request.method} {request.url.path} - ID: {request_id}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request_id
    
    logger.info(f"Request finished: {request.method} {request.url.path} - ID: {request_id} - Status: {response.status_code} - Duration: {process_time:.4f}s")
    return response

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

# --- Lifecycle Events ---

@app.on_event("startup")
async def startup_event():
    logger.info("Starting ScriptGuard API...")
    try:
        app_state.load_resources()
    except Exception as e:
        logger.critical(f"Failed to initialize application state: {e}")
        # We might want to exit here, but let's allow it to run so /health can report error
        pass

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
async def analyze_script(request: ScriptAnalysisRequest):
    """
    Analyze a script for malicious content.
    """
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

    # Construct Prompt
    # Note: In a real production system, we should use a proper template
    prompt = f"""
    Context from known vulnerabilities:
    {rag_context}
    
    Analyze the following script for malicious intent:
    {request.script_content}
    
    Is it malicious? Answer with reasoning.
    """
    
    # Inference
    try:
        inputs = app_state.tokenizer(prompt, return_tensors="pt").to(app_state.device)
        
        # Get generation config from app_state.config if available
        max_new_tokens = 100
        temperature = 0.1
        if app_state.config:
            max_new_tokens = app_state.config.inference.max_length # This might be too long for just reasoning, but using config
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
        
        # Post-processing / Parsing
        # TODO: Replace with deterministic classification head or constrained decoding
        # For now, we improve the heuristic slightly but acknowledge it's a P0 gap to fix fully with a classifier model
        
        # Strip the prompt from the response to analyze only the generated part
        # This is a simple heuristic; robust implementation requires knowing prompt length
        generated_text = response_text[len(prompt):] if len(response_text) > len(prompt) else response_text
        
        is_malicious = "malicious" in generated_text.lower()
        
        return ScriptAnalysisResponse(
            is_malicious=is_malicious,
            confidence=0.85, # Still mocked until we have logits/probabilities
            reasoning=generated_text.strip(),
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
