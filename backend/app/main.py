import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.database import db_client
from backend.app.services.text_service import text_service
from backend.app.routers import verify, train, metrics, history

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    Handles startup (DB connection, model loading) and shutdown events.
    """
    logger.info("Starting up FastAPI application...")
    
    # Connect to MongoDB
    await db_client.connect_db()
    
    # Note: text_service model is loaded on instantiation, but we could reload here if needed.
    # text_service.load_model()
    
    yield
    
    logger.info("Shutting down FastAPI application...")
    # Close MongoDB connection
    await db_client.close_db()

# Initialize FastAPI app
app = FastAPI(
    title="Smart Fake News Detection Engine API",
    description="Asynchronous Backend for Multimodal Forensics and NLP Analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, this should be restricted to specific domains (e.g., frontend host)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(verify.router, prefix="/api/v1", tags=["Verification"])
app.include_router(train.router, prefix="/api/v1", tags=["Training"])
app.include_router(metrics.router, prefix="/api/v1", tags=["Metrics"])
app.include_router(history.router, prefix="/api/v1", tags=["History"])

@app.get("/health", tags=["Health"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}
