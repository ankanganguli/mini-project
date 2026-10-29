import logging
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
from backend.app.config import settings

logger = logging.getLogger(__name__)

class DatabaseClient:
    """
    Manages the asynchronous MongoDB connection lifecycle and provides access
    to collections for verifications and training jobs.
    """
    client: Optional[AsyncIOMotorClient] = None
    db = None

    async def connect_db(self):
        """Initializes the database connection and verifies it with a ping."""
        logger.info(f"Connecting to MongoDB at {settings.MONGO_URI}")
        try:
            self.client = AsyncIOMotorClient(
                settings.MONGO_URI,
                serverSelectionTimeoutMS=3000,  # fail fast → fall back to mongomock
            )
            self.db = self.client[settings.DB_NAME]
            # Perform a heartbeat check
            await self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB and verified heartbeat.")
        except Exception as e:
            logger.warning(f"Failed to connect to real MongoDB: {e}. Falling back to mongomock-motor for local execution.")
            from mongomock_motor import AsyncMongoMockClient
            self.client = AsyncMongoMockClient()
            self.db = self.client[settings.DB_NAME]
            logger.info("Successfully connected to mongomock (in-memory MongoDB).")

    async def close_db(self):
        """Closes the MongoDB connection gracefully."""
        if self.client:
            logger.info("Closing MongoDB connection.")
            self.client.close()
            logger.info("MongoDB connection closed.")

    @property
    def verifications(self):
        """Returns the verifications collection for audit trails."""
        if self.db is None:
            raise RuntimeError("Database connection not initialized")
        return self.db["verifications"]

    @property
    def training_jobs(self):
        """Returns the training_jobs collection for tracking background tasks."""
        if self.db is None:
            raise RuntimeError("Database connection not initialized")
        return self.db["training_jobs"]

# Global database client instance
db_client = DatabaseClient()
