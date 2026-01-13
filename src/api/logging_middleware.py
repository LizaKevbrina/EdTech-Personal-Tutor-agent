"""
Logging middleware for request/response tracking.
"""
import time
import uuid
from typing import Callable

import structlog
from fastapi import Request, Response

from src.core.logging import get_logger
from src.core.metrics import http_request_duration_seconds, http_requests_total

logger = get_logger(__name__)


async def logging_middleware(
    request: Request,
    call_next: Callable,
) -> Response:
    """
    Logging middleware that tracks all requests.
    
    Args:
        request: FastAPI request
        call_next: Next middleware/handler
        
    Returns:
        Response
    """
    # Generate request ID
    request_id = str(uuid.uuid4())

    # Bind request context
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )

    # Log request start
    logger.info(
        "request_started",
        client_host=request.client.host if request.client else "unknown",
    )

    # Track request time
    start_time = time.time()

    try:
        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Log response
        logger.info(
            "request_completed",
            status_code=response.status_code,
            duration=f"{duration:.3f}s",
        )

        # Update metrics
        http_request_duration_seconds.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)

        http_requests_total.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response

    except Exception as e:
        duration = time.time() - start_time

        logger.error(
            "request_failed",
            error=str(e),
            duration=f"{duration:.3f}s",
        )

        # Update metrics for error
        http_requests_total.labels(
            method=request.method,
            endpoint=request.url.path,
            status=500,
        ).inc()

        raise

    finally:
        # Clear context
        structlog.contextvars.clear_contextvars()
