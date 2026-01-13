"""
Error handling middleware for FastAPI.
"""
from typing import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse

from src.core.exceptions import (
    EdTechTutorException,
    PIIDetectedError,
    PromptInjectionError,
    RateLimitExceededError,
    TokenBudgetExceededError,
    ValidationError,
)
from src.core.logging import get_logger

logger = get_logger(__name__)


async def error_handler_middleware(
    request: Request,
    call_next: Callable,
) -> Response:
    """
    Global error handling middleware.
    
    Args:
        request: FastAPI request
        call_next: Next middleware/handler
        
    Returns:
        Response
    """
    try:
        response = await call_next(request)
        return response

    except RateLimitExceededError as e:
        logger.warning(
            "rate_limit_exceeded",
            path=request.url.path,
            error=str(e),
        )
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "rate_limit_exceeded",
                "message": str(e),
                "details": e.details,
            },
        )

    except TokenBudgetExceededError as e:
        logger.warning(
            "token_budget_exceeded",
            path=request.url.path,
            error=str(e),
        )
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "token_budget_exceeded",
                "message": str(e),
                "details": e.details,
            },
        )

    except PIIDetectedError as e:
        logger.warning(
            "pii_detected",
            path=request.url.path,
            error=str(e),
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "pii_detected",
                "message": "Personal information detected in input. Please remove sensitive data.",
                "details": {"pii_types": e.details.get("pii_types", [])},
            },
        )

    except PromptInjectionError as e:
        logger.warning(
            "prompt_injection_detected",
            path=request.url.path,
            error=str(e),
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "invalid_input",
                "message": "Invalid input detected. Please rephrase your message.",
            },
        )

    except ValidationError as e:
        logger.warning(
            "validation_error",
            path=request.url.path,
            error=str(e),
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "validation_error",
                "message": str(e),
                "details": e.details,
            },
        )

    except EdTechTutorException as e:
        logger.error(
            "application_error",
            path=request.url.path,
            error=str(e),
            error_code=e.error_code,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": e.error_code,
                "message": str(e),
                "details": e.details,
            },
        )

    except Exception as e:
        logger.error(
            "unexpected_error",
            path=request.url.path,
            error=str(e),
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred. Please try again later.",
            },
        )
