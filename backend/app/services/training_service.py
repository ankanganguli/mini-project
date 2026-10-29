import asyncio
import uuid
import sys
import logging
from datetime import datetime, timezone
from backend.app.database import db_client
from backend.app.services.text_service import text_service
from backend.app.config import settings
import os

logger = logging.getLogger(__name__)

class TrainingOrchestrator:
    """
    Manages background subprocess execution of model retraining,
    tracks status in MongoDB, and triggers hot-reloading.
    """
    
    async def start_training_job(self, dataset_path: str) -> str:
        """
        Initializes a training job, stores the initial state in DB, and
        dispatches the background task.
        Returns the job_id.
        """
        job_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc).isoformat()
        
        job_doc = {
            "job_id": job_id,
            "status": "queued",
            "dataset_path": dataset_path,
            "started_at": started_at,
            "updated_at": started_at,
            "logs": []
        }
        
        await db_client.training_jobs.insert_one(job_doc)
        
        # Dispatch background task without blocking
        asyncio.create_task(self._run_training_subprocess(job_id, dataset_path))
        
        return job_id

    async def _update_status(self, job_id: str, status: str, log_message: str):
        """Updates the job status and appends a log message in the database."""
        await db_client.training_jobs.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": status,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                },
                "$push": {"logs": log_message}
            }
        )
        logger.info(f"Training Job [{job_id}] {status}: {log_message}")

    async def _run_training_subprocess(self, job_id: str, dataset_path: str):
        """Executes train_engine.py in a subprocess and monitors completion."""
        await self._update_status(job_id, "processing", "Started ingestion and training.")
        
        try:
            # Assumes train_engine.py is located at the project root or a known path.
            # We use sys.executable to run the python script safely.
            script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../train_engine.py"))
            
            # Start the subprocess with correct --data-path flag
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                "--data-path", dataset_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                await self._update_status(job_id, "completed", "Training completed successfully.")
                # Trigger hot-reload of the newly trained model
                text_service.reload_model()
                await self._update_status(job_id, "completed", "Hot-reload triggered in memory.")
            else:
                error_msg = stderr.decode().strip() or "Unknown error"
                await self._update_status(job_id, "failed", f"Training failed: {error_msg}")
        except Exception as e:
            logger.error(f"Error executing background training {job_id}: {e}", exc_info=True)
            await self._update_status(job_id, "failed", f"Exception during execution: {str(e)}")

# Global instance
training_orchestrator = TrainingOrchestrator()
