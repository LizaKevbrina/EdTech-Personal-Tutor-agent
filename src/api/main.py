"""
FastAPI application entry point.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from src.agent.llm.llm_provider import llm_provider
from src.agent.retrieval.embeddings import embeddings_manager
from src.agent.retrieval.qdrant_client import qdrant_manager
from src.api.middleware.error_handler import error_handler_middleware
from src.api.middleware.logging_middleware import logging_middleware
from src.api.routes import chat, health, progress, quiz
from src.core.config import settings
from src.core.logging import get_logger, setup_logging
from src.core.rate_limiter import rate_limiter

# Setup logging first
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for startup and shutdown events.
    
    Args:
        app: FastAPI application
    """
    # Startup
    logger.info(
        "application_starting",
        environment=settings.environment,
        version=settings.api_version,
    )

    try:
        # Initialize components
        logger.info("initializing_components")

        # Initialize Qdrant
        await qdrant_manager.initialize()

        # Initialize embeddings
        embeddings_manager.initialize()

        # Initialize LLM provider
        llm_provider.initialize()

        # Initialize rate limiter
        rate_limiter.initialize()

        logger.info("components_initialized_successfully")

        yield

    finally:
        # Shutdown
        logger.info("application_shutting_down")

        # Close connections
        await qdrant_manager.close()
        rate_limiter.close()

        logger.info("application_stopped")


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Production-ready EdTech Personal Tutor Agent",
    lifespan=lifespan,
    docs_url=f"{settings.api_prefix}/docs",
    redoc_url=f"{settings.api_prefix}/redoc",
    openapi_url=f"{settings.api_prefix}/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
app.middleware("http")(logging_middleware)
app.middleware("http")(error_handler_middleware)

# Include routers
app.include_router(
    health.router,
    prefix=settings.api_prefix,
    tags=["Health"],
)

app.include_router(
    chat.router,
    prefix=settings.api_prefix,
    tags=["Chat"],
)

app.include_router(
    quiz.router,
    prefix=settings.api_prefix,
    tags=["Quiz"],
)

app.include_router(
    progress.router,
    prefix=settings.api_prefix,
    tags=["Progress"],
)

# Mount Prometheus metrics
if settings.prometheus_enabled:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)


@app.get("/")
async def root() -> JSONResponse:
    """Root endpoint."""
    return JSONResponse(
        content={
            "name": settings.api_title,
            "version": settings.api_version,
            "environment": settings.environment,
            "status": "running",
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
        log_config=None,  # Use our custom logging
    )
