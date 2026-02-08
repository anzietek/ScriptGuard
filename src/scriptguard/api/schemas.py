"""
API Request and Response Schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ScriptAnalysisRequest(BaseModel):
    """Request model for script analysis."""
    script_content: str = Field(..., description="The source code content to analyze", min_length=1, max_length=100000)
    include_rag: bool = Field(True, description="Whether to include RAG context in analysis")

class VulnerabilityInfo(BaseModel):
    """Information about a detected vulnerability or related CVE."""
    id: Optional[str] = None
    description: str
    severity: Optional[str] = None
    score: Optional[float] = None

class ScriptAnalysisResponse(BaseModel):
    """Response model for script analysis."""
    is_malicious: bool = Field(..., description="Whether the script is classified as malicious")
    confidence: float = Field(..., description="Confidence score of the classification (0.0 to 1.0)")
    reasoning: str = Field(..., description="Explanation for the classification")
    related_cves: List[VulnerabilityInfo] = Field(default_factory=list, description="List of related CVEs or vulnerabilities found via RAG")
    
class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str

class ReadinessResponse(BaseModel):
    """Response model for readiness check."""
    status: str
    model_loaded: bool
    rag_connected: bool
