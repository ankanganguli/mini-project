from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application settings, loaded from environment variables using pydantic-settings.
    Provides default values suitable for local development or basic deployment.
    """
    MONGO_URI: str = "mongodb://localhost:27017"
    DB_NAME: str = "fake_news_detector"
    MODEL_PATH: str = "artifacts/model_pipeline.pkl"
    METRICS_PATH: str = "artifacts/metrics.json"
    UPLOAD_DIR: str = "data/uploads"

    class Config:
        env_file = ".env"

# Global settings instance
settings = Settings()
