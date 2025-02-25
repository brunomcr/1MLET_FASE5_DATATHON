import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.containers import Container
from app.routers import predictions_v1_router


# Logging configuration
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize resources on startup and cleanup on shutdown
    """
    # Initialize container
    container = Container()
    app.container = container

    # Eagerly initialize Spark and model resources
    logger.info("Initializing Spark session and loading model...")
    predictor = container.predictor()
    logger.info("Initialization complete!")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down Spark session...")
    predictor.spark.stop()


# Initialize the FastAPI application
app = FastAPI(lifespan=lifespan)

# Add middlewares
# app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.ALLOWED_HOSTS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predictions_v1_router)