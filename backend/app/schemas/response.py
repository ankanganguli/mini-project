from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ModalitySummary(BaseModel):
    """Individual forensic report for a single analyzed modality."""
    modality: str = Field(..., description="Modality name: 'Text', 'Image', 'Video', or 'URL'")
    verdict: str = Field(..., description="Individual verdict for this modality")
    authenticity_score: float = Field(..., description="Authenticity score from 0.0 to 100.0")
    manipulation_probability: float = Field(..., description="Probability of manipulation/fakeness from 0.0 to 1.0")
    confidence_score: float = Field(..., description="Confidence in this modality's verdict (0.0 to 100.0)")
    summary: str = Field(..., description="Short summary of findings for this modality")
    details: Dict[str, Any] = Field(default_factory=dict, description="Modality-specific forensic metrics")


class CombinedReport(BaseModel):
    """Comprehensive cross-modal forensic synthesis report when analyzing one or multiple modalities."""
    is_multimodal: bool = Field(False, description="True if 2 or more modalities were analyzed together")
    modalities_analyzed: List[str] = Field(default_factory=list, description="List of analyzed modalities")
    overall_verdict: str = Field(..., description="Unified verdict across all modalities")
    overall_confidence: float = Field(..., description="Unified confidence score (0.0 to 100.0)")
    overall_authenticity_score: float = Field(..., description="Overall authenticity score (0.0 to 100.0)")
    overall_manipulation_risk: float = Field(..., description="Combined probability of manipulation (0.0 to 100.0)")
    cross_modal_agreement: str = Field(
        ...,
        description="'Strong Agreement', 'Moderate Agreement', 'Cross-Modal Contradiction', or 'Single Modality'"
    )
    modality_reports: List[ModalitySummary] = Field(
        default_factory=list,
        description="Individual breakdown for each provided modality"
    )
    executive_summary: str = Field(..., description="Detailed narrative synthesis of all modalities")
    key_findings: List[str] = Field(
        default_factory=list,
        description="Bullet-point forensic findings across text, image, video, and URL"
    )
    recommendation: str = Field(..., description="Actionable verification recommendation")


class ForensicBreakdown(BaseModel):
    """Detailed forensic analysis breakdown from textual, visual, and URL pipelines."""
    text_fake_probability: Optional[float] = Field(None, description="Probability of text being fake (0.0 to 1.0)")
    visual_entropy: Optional[float] = Field(None, description="Visual edge/Laplacian variance or video temporal variance")
    compression_discrepancy: Optional[float] = Field(None, description="Error level analysis compression difference metric")
    cross_modal_conflict_detected: bool = Field(False, description="True if modalities severely contradict each other")
    image_ela_score: Optional[float] = Field(None, description="Image ELA & ViT score (0.0 to 1.0)")
    image_laplacian_score: Optional[float] = Field(None, description="Image Laplacian sharpness inconsistency (0.0 to 1.0)")
    video_mean_ela: Optional[float] = Field(None, description="Video frame-averaged ELA score (0.0 to 1.0)")
    video_temporal_variance: Optional[float] = Field(None, description="Video temporal consistency variance (0.0 to 1.0)")
    url_source: Optional[str] = Field(None, description="Normalized URL if a URL was analyzed")
    url_domain: Optional[str] = Field(None, description="Source domain of the analyzed URL")
    url_title: Optional[str] = Field(None, description="Extracted headline or title from the URL")
    url_type: Optional[str] = Field(None, description="Detected URL content type ('news_article', 'video_url', 'image_url', 'video_platform')")
    url_extracted_text_preview: Optional[str] = Field(None, description="Excerpt of text extracted from the URL")


class VerifyResponse(BaseModel):
    """Response returned upon verification of text, image, video, URL, or combined multimodal input."""
    verdict: str = Field(..., description="Categorical outcome: 'Likely Authentic', 'Manipulated Media', or 'High Risk Contradiction'")
    confidence_score: float = Field(..., description="Overall confidence score from 0.0 to 100.0")
    breakdown: ForensicBreakdown
    explanation: str = Field(..., description="Human-readable explanation of the verdict")
    timestamp: str = Field(..., description="ISO 8601 formatted timestamp of the verification")
    record_id: str = Field(..., description="Unique identifier for the audit record stored in the database")
    combined_report: Optional[CombinedReport] = Field(None, description="Detailed combined report across all submitted modalities")


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
