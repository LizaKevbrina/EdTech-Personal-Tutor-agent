"""
Health check endpoints for monitoring and readiness.
"""
from typing import Any

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from src.agent.retrieval.qdrant_client import qdrant_manager
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check() -> JSONResponse:
    """
    Basic health check endpoint.
    
    Returns:
        Health status
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "healthy",
            "service": settings.api_title,
            "version": settings.api_version,
            "environment": settings.environment,
        },
    )


@router.get("/health/ready")
async def readiness_check() -> JSONResponse:
    """
    Readiness check - verifies all dependencies are available.
    
    Returns:
        Readiness status with component health
    """
    components: dict[str, Any] = {}
    all_healthy = True

    # Check Qdrant
    try:
        qdrant_healthy = await qdrant_manager.health_check()
        components["qdrant"] = {
            "status": "healthy" if qdrant_healthy else "unhealthy",
            "url": settings.qdrant_url,
        }
        if not qdrant_healthy:
            all_healthy = False
    except Exception as e:
        components["qdrant"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        all_healthy = False

    # Check LLM provider (simple check)
    try:
        if settings.openai_api_key.get_secret_value():
            components["llm_provider"] = {
                "status": "configured",
                "model": settings.openai_model_name,
            }
        else:
            components["llm_provider"] = {
                "status": "not_configured",
            }
            all_healthy = False
    except Exception as e:
        components["llm_provider"] = {
            "status": "error",
            "error": str(e),
        }
        all_healthy = False

    # Overall status
    response_status = status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        status_code=response_status,
        content={
            "status": "ready" if all_healthy else "not_ready",
            "components": components,
        },
    )


@router.get("/health/live")
async def liveness_check() -> JSONResponse:
    """
    Liveness check - verifies the application is running.
    
    Returns:
        Liveness status
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "alive",
        },
    )
