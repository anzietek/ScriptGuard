from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from scriptguard.rag.qdrant_store import QdrantStore
from scriptguard.utils.logger import logger
import os

app = FastAPI(title="ScriptGuard Inference API")

class ScriptAnalysisRequest(BaseModel):
    script_content: str
    include_rag: bool = True

class ScriptAnalysisResponse(BaseModel):
    is_malicious: bool
    confidence: float
    reasoning: str
    related_cves: List[dict] = []

# Global variables for model and tokenizer (loaded on startup)
model = None
tokenizer = None
rag_store = None

@app.on_event("startup")
def load_resources():
    global model, tokenizer, rag_store
    
    model_id = os.getenv("BASE_MODEL_ID", "bigcode/starcoder2-3b")
    adapter_path = os.getenv("ADAPTER_PATH", "./model_checkpoints/final_adapter")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    if os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print(f"Loaded adapter from {adapter_path}")
    else:
        model = base_model
        print(f"Adapter not found at {adapter_path}, using base model.")
        
    rag_store = QdrantStore(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333))
    )

@app.post("/analyze", response_model=ScriptAnalysisResponse)
async def analyze_script(request: ScriptAnalysisRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # RAG Context
    rag_context = ""
    related_cves = []
    if request.include_rag:
        results = rag_store.search(request.script_content, limit=2)
        related_cves = results
        rag_context = "\n".join([f"Known Vulnerability: {r['description']}" for r in results])
    
    prompt = f"""
    Context from known vulnerabilities:
    {rag_context}
    
    Analyze the following script for malicious intent:
    {request.script_content}
    
    Is it malicious? Answer with reasoning.
    """
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Simplified logic for demonstration
    is_malicious = "malicious" in response_text.lower()
    
    return ScriptAnalysisResponse(
        is_malicious=is_malicious,
        confidence=0.85, # Mocked
        reasoning=response_text,
        related_cves=related_cves
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
