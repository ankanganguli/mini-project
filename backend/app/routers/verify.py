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
from backend.app.services.url_service import url_service
from backend.app.database import db_client

logger = logging.getLogger(__name__)

router = APIRouter()


def _is_image_file(content_type: Optional[str], filename: Optional[str]) -> bool:
    if content_type and content_type.lower().startswith("image/"):
        return True
    if filename and filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")):
        return True
    return False


def _is_video_file(content_type: Optional[str], filename: Optional[str]) -> bool:
    if content_type and content_type.lower().startswith("video/"):
        return True
    if filename and filename.lower().endswith((".mp4", ".mov", ".webm", ".mkv", ".avi")):
        return True
    return False


@router.post("/verify", response_model=VerifyResponse)
async def verify_content(
    request: Request,
    text: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None),
    video_file: Optional[UploadFile] = File(None),
):
    """
    Analyzes submitted text, URL (news or video/media URL), image, and/or video files
    to determine authenticity. Supports single-modality or combined multimodal
    verification (e.g. Text + Image + Video + URL all together) and returns a
    unified Combined Report.
    """
    clean_text = text.strip() if text and text.strip() else None
    clean_url = url.strip() if url and url.strip() else None

    has_legacy_file = file is not None and bool(file.filename)
    has_image_file = image_file is not None and bool(image_file.filename)
    has_video_file = video_file is not None and bool(video_file.filename)

    if not any([clean_text, clean_url, has_legacy_file, has_image_file, has_video_file]):
        raise HTTPException(
            status_code=400,
            detail="At least one input modality (text, url, image, or video) must be provided for verification.",
        )

    start_time = datetime.now(timezone.utc)

    # 1. URL Ingestion (if URL provided)
    url_info = None
    if clean_url:
        try:
            url_info = await url_service.extract_from_url(clean_url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected URL extraction error for {clean_url}: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Failed to process URL: {str(e)}")

    # 2. Resolve Image and Video bytes from uploaded files or URL media
    img_bytes: Optional[bytes] = None
    img_filename: Optional[str] = None
    vid_bytes: Optional[bytes] = None
    vid_filename: Optional[str] = None
    primary_content_type: Optional[str] = None

    if has_legacy_file and file:
        raw_bytes = await file.read()
        if _is_image_file(file.content_type, file.filename):
            img_bytes = raw_bytes
            img_filename = file.filename
            primary_content_type = file.content_type or "image/jpeg"
        elif _is_video_file(file.content_type, file.filename):
            vid_bytes = raw_bytes
            vid_filename = file.filename
            primary_content_type = file.content_type or "video/mp4"
        else:
            logger.warning(f"Unsupported legacy file type: {file.content_type} ({file.filename})")

    if has_image_file and image_file:
        img_bytes = await image_file.read()
        img_filename = image_file.filename
        primary_content_type = image_file.content_type or "image/jpeg"

    if has_video_file and video_file:
        vid_bytes = await video_file.read()
        vid_filename = video_file.filename
        primary_content_type = video_file.content_type or "video/mp4"

    # If URL extracted image or video and user didn't upload an overriding file for that modality
    is_editorial_thumb = False
    if url_info:
        if img_bytes is None and url_info.get("image_bytes"):
            img_bytes = url_info["image_bytes"]
            img_filename = f"URL Media ({url_info.get('publisher') or url_info.get('domain', 'remote')})"
            is_editorial_thumb = bool(url_info.get("thumbnail_is_editorial", False))
            if not primary_content_type:
                primary_content_type = "image/jpeg"
        if vid_bytes is None and url_info.get("video_bytes"):
            vid_bytes = url_info["video_bytes"]
            vid_filename = f"URL Video ({url_info.get('publisher') or url_info.get('domain', 'remote')})"
            if not primary_content_type:
                primary_content_type = "video/mp4"

    # 3. Resolve Effective Text (user text + URL extracted text if present)
    effective_text: Optional[str] = None
    text_source_label = "Direct Input"
    is_verified_src = bool(url_info and url_info.get("is_verified_source", False))
    if clean_text and url_info and url_info.get("extracted_text"):
        effective_text = f"{clean_text}\n\n{url_info['extracted_text']}"
        text_source_label = f"Direct Input + URL ({url_info.get('publisher') or url_info.get('domain')})"
    elif clean_text:
        effective_text = clean_text
        text_source_label = "Direct Input"
    elif url_info and url_info.get("extracted_text"):
        effective_text = url_info["extracted_text"]
        pub_label = url_info.get("publisher") or url_info.get("domain")
        text_source_label = f"Extracted from {pub_label}" + (" (Verified Publisher)" if is_verified_src else "")

    # 4. Launch Parallel Inference Tasks across all active modalities
    text_task = (
        asyncio.to_thread(text_service.predict, effective_text, is_verified_src)
        if effective_text
        else None
    )
    image_task = (
        asyncio.gather(
            asyncio.to_thread(vision_service.analyze_image_ela, img_bytes, is_editorial_thumb),
            asyncio.to_thread(vision_service.analyze_laplacian_sharpness, img_bytes),
        )
        if img_bytes
        else None
    )
    video_task = asyncio.to_thread(vision_service.analyze_video, vid_bytes) if vid_bytes else None

    try:
        text_result = await text_task if text_task else None
        image_result = await image_task if image_task else None
        video_result = await video_task if video_task else None
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 5. Extract Modality Metrics
    text_fake_prob: Optional[float] = None
    text_meta = None
    if text_result:
        text_fake_prob = text_result.get("p_fake")
        text_meta = {
            "word_count": len(effective_text.split()) if effective_text else 0,
            "source": text_source_label,
        }

    image_scores = None
    image_meta = None
    image_ela_score: Optional[float] = None
    image_lap_score: Optional[float] = None
    if image_result:
        image_ela_score, image_lap_score = image_result
        image_scores = {"ela": image_ela_score, "laplacian": image_lap_score}
        image_meta = {"filename": img_filename or "Uploaded Image"}

    video_scores = None
    video_meta = None
    video_mean_ela: Optional[float] = None
    video_temp_var: Optional[float] = None
    if video_result and isinstance(video_result, dict):
        video_mean_ela = video_result.get("mean_ela", 0.0)
        video_temp_var = video_result.get("temporal_variance", 0.0)
        video_scores = {"mean_ela": video_mean_ela, "temporal_variance": video_temp_var}
        video_meta = {"filename": vid_filename or "Uploaded Video"}

    # Backward-compatible aggregate visual metrics for ForensicBreakdown
    comp_candidates = [x for x in (image_ela_score, video_mean_ela) if x is not None]
    ent_candidates = [x for x in (image_lap_score, video_temp_var) if x is not None]
    compression_discrepancy = round(sum(comp_candidates) / len(comp_candidates), 4) if comp_candidates else None
    visual_entropy = round(sum(ent_candidates) / len(ent_candidates), 4) if ent_candidates else None

    # 6. Multimodal Fusion & Combined Report Generation
    verdict, confidence_score, explanation, cross_modal_conflict, combined_report = (
        fusion_service.build_combined_report(
            text_prob=text_fake_prob,
            text_meta=text_meta,
            image_scores=image_scores,
            image_meta=image_meta,
            video_scores=video_scores,
            video_meta=video_meta,
            url_info=url_info,
        )
    )

    url_preview = None
    if url_info and url_info.get("extracted_text"):
        raw_ext = url_info["extracted_text"]
        url_preview = raw_ext[:400] + ("..." if len(raw_ext) > 400 else "")

    breakdown = ForensicBreakdown(
        text_fake_probability=text_fake_prob,
        visual_entropy=visual_entropy,
        compression_discrepancy=compression_discrepancy,
        cross_modal_conflict_detected=cross_modal_conflict,
        image_ela_score=image_ela_score,
        image_laplacian_score=image_lap_score,
        video_mean_ela=video_mean_ela,
        video_temporal_variance=video_temp_var,
        url_source=url_info.get("url") if url_info else None,
        url_domain=url_info.get("domain") if url_info else None,
        url_title=url_info.get("title") if url_info else None,
        url_type=url_info.get("url_type") if url_info else None,
        url_extracted_text_preview=url_preview,
    )

    record_id = str(uuid.uuid4())
    timestamp_str = start_time.isoformat()

    # Determine human-readable submission title for history
    submission_label = (
        (url_info.get("title") if url_info else None)
        or (clean_text[:60] if clean_text else None)
        or img_filename
        or vid_filename
        or "Multimodal Scan"
    )

    # 7. Construct & Persist Audit Log
    audit_record = {
        "record_id": record_id,
        "timestamp": timestamp_str,
        "client_ip": request.client.host if request.client else "unknown",
        "input_summary": {
            "title": submission_label,
            "modalities": combined_report.modalities_analyzed,
            "is_multimodal": combined_report.is_multimodal,
            "text_length": len(effective_text) if effective_text else 0,
            "url": url_info.get("url") if url_info else None,
            "has_file": bool(img_bytes or vid_bytes),
            "filename": ", ".join([f for f in (img_filename, vid_filename) if f]) or None,
            "content_type": primary_content_type,
        },
        "breakdown": breakdown.model_dump(),
        "verdict": verdict,
        "confidence_score": confidence_score,
        "explanation": explanation,
        "combined_report": combined_report.model_dump(),
    }

    try:
        await db_client.verifications.insert_one(audit_record)
    except Exception as e:
        logger.error(f"Failed to persist audit log: {e}", exc_info=True)

    return VerifyResponse(
        verdict=verdict,
        confidence_score=confidence_score,
        breakdown=breakdown,
        explanation=explanation,
        timestamp=timestamp_str,
        record_id=record_id,
        combined_report=combined_report,
    )
