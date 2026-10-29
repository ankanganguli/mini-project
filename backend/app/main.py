import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

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

@app.middleware("http")
async def disable_static_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Include API Routers
app.include_router(verify.router, prefix="/api/v1", tags=["Verification"])
app.include_router(train.router, prefix="/api/v1", tags=["Training"])
app.include_router(metrics.router, prefix="/api/v1", tags=["Metrics"])
app.include_router(history.router, prefix="/api/v1", tags=["History"])

@app.get("/health", tags=["Health"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}

# Mount frontend static files and redirect root to login
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
if os.path.exists(frontend_dir):
    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/login.html")

    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

