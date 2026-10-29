import logging
from fastapi import APIRouter, Query
from typing import List, Optional
from backend.app.database import db_client

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/history")
async def get_verification_history(
    limit: int = Query(50, ge=1, le=200, description="Max number of records to return"),
    content_type: Optional[str] = Query(None, alias="type", description="Filter by content type: text, image, video")
):
    """
    Returns the most recent verification audit records from MongoDB,
    sorted by timestamp descending. Used by the frontend history page.
    """
    try:
        query_filter = {}
        if content_type and content_type.lower() != "all":
            # Match against the content_type stored in input_summary
            query_filter["input_summary.content_type"] = {
                "$regex": f"^{content_type.lower()}/",
                "$options": "i"
            }
            # For text entries, content_type in input_summary is None / not set
            if content_type.lower() == "text":
                query_filter = {
                    "$or": [
                        {"input_summary.has_file": False},
                        {"input_summary.content_type": None},
                    ]
                }

        cursor = db_client.verifications.find(
            query_filter,
            {
                "_id": 0,
                "record_id": 1,
                "timestamp": 1,
                "input_summary": 1,
                "verdict": 1,
                "confidence_score": 1,
                "breakdown": 1,
                "explanation": 1,
            }
        ).sort("timestamp", -1).limit(limit)

        records = await cursor.to_list(length=limit)
        return {"records": records, "count": len(records)}

    except Exception as e:
        logger.error(f"Error fetching history: {e}", exc_info=True)
        return {"records": [], "count": 0}
