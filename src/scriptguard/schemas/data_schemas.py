"""
Pydantic schemas for data validation between pipeline steps.
Ensures consistency and type safety across the data pipeline.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum


class LabelType(str, Enum):
    """Classification labels"""
    BENIGN = "benign"
    MALICIOUS = "malicious"


class CodeSample(BaseModel):
    """Schema for raw code sample from data sources"""
    content: str = Field(..., min_length=1, description="Source code content")
    label: LabelType = Field(..., description="Classification label")
    source: str = Field(..., description="Data source (github, malwarebazaar, etc)")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata")

    @validator("content")
    def content_not_empty(cls, v):
        """Ensure content is not just whitespace"""
        if not v.strip():
            raise ValueError("Content cannot be empty or whitespace")
        return v


class ValidatedCodeSample(CodeSample):
    """Schema for validated code sample (after validation step)"""
    syntax_valid: bool = Field(..., description="Whether syntax is valid")
    length: int = Field(..., gt=0, description="Content length in characters")
    num_lines: int = Field(..., gt=0, description="Number of code lines")


class FeatureExtractedSample(ValidatedCodeSample):
    """Schema for sample with extracted features"""
    features: dict = Field(..., description="Extracted features (AST, entropy, etc)")
    feature_vector: Optional[List[float]] = Field(None, description="Numerical feature vector")


class ProcessedSample(BaseModel):
    """Schema for preprocessed sample ready for training"""
    text: str = Field(..., min_length=1, description="Formatted training text")
    original_label: LabelType = Field(..., description="Original classification label")

    @validator("text")
    def text_contains_classification(cls, v):
        """Ensure formatted text follows expected structure"""
        if "Classification:" not in v:
            raise ValueError("Formatted text must contain 'Classification:' marker")
        return v


def validate_data_batch(data: List[dict], schema: type[BaseModel]) -> List[BaseModel]:
    """
    Validate a batch of data against a Pydantic schema.

    Args:
        data: List of dictionaries to validate
        schema: Pydantic model class to validate against

    Returns:
        List of validated Pydantic model instances

    Raises:
        ValidationError: If any item fails validation
    """
    validated = []
    errors = []

    for idx, item in enumerate(data):
        try:
            validated.append(schema(**item))
        except Exception as e:
            errors.append(f"Item {idx}: {str(e)}")

    if errors:
        raise ValueError(f"Validation errors:\n" + "\n".join(errors[:10]))  # Show first 10 errors

    return validated
