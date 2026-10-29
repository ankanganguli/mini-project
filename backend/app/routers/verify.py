import asyncio
import uuid
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Form, UploadFile, File, HTTPException, Request
from typing import Optional

from backend.app.schemas.response import VerifyResponse, ForensicBreakdown
from backend.app.services.text_service import text_service
from backend.app.services.vision_service import vision_service
from backend.app.services.fusion_service import fusion_service
from backend.app.database import db_client

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/verify", response_model=VerifyResponse)
async def verify_content(
    request: Request,
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Analyzes submitted text and/or media files to determine authenticity.
    Uses multimodal parallel execution and late-fusion inference.
    """
    if not text and not file:
        raise HTTPException(status_code=400, detail="Must provide either text or a file for verification.")
        
    start_time = datetime.now(timezone.utc)
    
    # 1. Parallel Execution
    text_task = None
    vision_task = None
    file_bytes = None
    filename = None
    content_type = None

    if text:
        text_task = asyncio.to_thread(text_service.predict, text)
        
    if file:
        filename = file.filename
        content_type = file.content_type
        file_bytes = await file.read()
        
        # Decide which vision analysis to run based on content type
        if content_type and content_type.startswith("image/"):
            # Run both ELA and Laplacian in parallel threads
            vision_task = asyncio.gather(
                asyncio.to_thread(vision_service.analyze_image_ela, file_bytes),
                asyncio.to_thread(vision_service.analyze_laplacian_sharpness, file_bytes)
            )
        elif content_type and content_type.startswith("video/"):
            vision_task = asyncio.to_thread(vision_service.analyze_video, file_bytes)
        else:
            # Fallback for unknown types - skip vision analysis
            logger.warning(f"Unsupported file type: {content_type}")

    # Await parallel results
    try:
        text_result = await text_task if text_task else None
        vision_result = await vision_task if vision_task else None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Extract metrics from results
    text_fake_prob = None
    if text_result:
        text_fake_prob = text_result.get("p_fake")
        
    visual_entropy = None
    compression_discrepancy = None
    
    if vision_result:
        if isinstance(vision_result, (list, tuple)):
            # Image analysis result (ELA, Laplacian)
            compression_discrepancy, visual_entropy = vision_result
        elif isinstance(vision_result, dict):
            # Video analysis result
            compression_discrepancy = vision_result.get("mean_ela")
            visual_entropy = vision_result.get("temporal_variance")

    # For late fusion, we need a single visual probability score. We'll use a combination of ELA and entropy if available.
    visual_prob = None
    if compression_discrepancy is not None and visual_entropy is not None:
        visual_prob = min((compression_discrepancy + visual_entropy) / 2.0, 1.0)
    elif compression_discrepancy is not None:
        visual_prob = compression_discrepancy
    elif visual_entropy is not None:
        visual_prob = visual_entropy
        
    # 3. Decision Fusion
    verdict, confidence_score, explanation, cross_modal_conflict = fusion_service.fuse(
        text_prob=text_fake_prob,
        visual_prob=visual_prob
    )
    
    breakdown = ForensicBreakdown(
        text_fake_probability=text_fake_prob,
        visual_entropy=visual_entropy,
        compression_discrepancy=compression_discrepancy,
        cross_modal_conflict_detected=cross_modal_conflict
    )
    
    record_id = str(uuid.uuid4())
    timestamp_str = start_time.isoformat()
    
    # 4. Construct Audit Log
    audit_record = {
        "record_id": record_id,
        "timestamp": timestamp_str,
        "client_ip": request.client.host if request.client else "unknown",
        "input_summary": {
            "text_length": len(text) if text else 0,
            "has_file": file is not None,
            "filename": filename,
            "content_type": content_type
        },
        "breakdown": breakdown.model_dump(),
        "verdict": verdict,
        "confidence_score": confidence_score,
        "explanation": explanation
    }
    
    # Write to MongoDB asynchronously
    try:
        await db_client.verifications.insert_one(audit_record)
    except Exception as e:
        logger.error(f"Failed to persist audit log: {e}", exc_info=True)
        # We don't fail the request if audit logging fails
        
    return VerifyResponse(
        verdict=verdict,
        confidence_score=confidence_score,
        breakdown=breakdown,
        explanation=explanation,
        timestamp=timestamp_str,
        record_id=record_id
    )
