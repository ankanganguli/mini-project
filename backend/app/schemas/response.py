from pydantic import BaseModel, Field
from typing import Optional

class ForensicBreakdown(BaseModel):
    """Detailed forensic analysis breakdown from textual and visual pipelines."""
    text_fake_probability: Optional[float] = Field(None, description="Probability of text being fake (0.0 to 1.0)")
    visual_entropy: Optional[float] = Field(None, description="Visual edge or Laplacian variance/entropy")
    compression_discrepancy: Optional[float] = Field(None, description="Error level analysis compression difference metric")
    cross_modal_conflict_detected: bool = Field(False, description="True if text and visual forensics severely contradict each other")

class VerifyResponse(BaseModel):
    """Response returned upon verification of text and/or media."""
    verdict: str = Field(..., description="Categorical outcome: 'Likely Authentic', 'Manipulated Media', or 'High Risk Contradiction'")
    confidence_score: float = Field(..., description="Overall confidence score from 0.0 to 100.0")
    breakdown: ForensicBreakdown
    explanation: str = Field(..., description="Human-readable explanation of the verdict")
    timestamp: str = Field(..., description="ISO 8601 formatted timestamp of the verification")
    record_id: str = Field(..., description="Unique identifier for the audit record stored in the database")

class TrainJobResponse(BaseModel):
    """Response returned when a background training job is initiated or queried."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Current job status: 'queued', 'processing', 'completed', or 'failed'")
    started_at: str = Field(..., description="ISO 8601 formatted timestamp when the job started")

class MetricsResponse(BaseModel):
    """Overall system metrics and model performance stats."""
    model_accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_verifications: int
    dataset_samples: int
