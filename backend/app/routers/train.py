import os
import aiofiles
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.app.schemas.response import TrainJobResponse
from backend.app.services.training_service import training_orchestrator
from backend.app.config import settings

router = APIRouter()

@router.post("/train", response_model=TrainJobResponse)
async def upload_training_dataset(file: UploadFile = File(...)):
    """
    Accepts a CSV dataset for retraining the text classification model.
    Validates the format, saves it to disk, and triggers a background training job.
    """
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported for training.")
        
    # Ensure upload directory exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    safe_filename = f"{timestamp}_{file.filename.replace(' ', '_')}"
    file_path = os.path.join(settings.UPLOAD_DIR, safe_filename)
    
    try:
        # Write uploaded file to disk asynchronously
        async with aiofiles.open(file_path, 'wb') as out_file:
            while content := await file.read(1024 * 1024):  # 1MB chunks
                await out_file.write(content)
                
        # Start background training job
        job_id = await training_orchestrator.start_training_job(file_path)
        
        return TrainJobResponse(
            job_id=job_id,
            status="queued",
            started_at=datetime.now(timezone.utc).isoformat()
        )
    except Exception as e:
        # Clean up file on failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to initiate training: {str(e)}")
