"""
Structured logging configuration with contextual information.
Uses structlog for production-grade logging with JSON output.
"""
import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from src.core.config import settings


def add_app_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Add application context to all log messages.
    
    Args:
        logger: Logger instance
        method_name: Method name being logged
        event_dict: Event dictionary
        
    Returns:
        EventDict: Updated event dictionary with context
    """
    event_dict["app"] = "edtech-tutor"
    event_dict["environment"] = settings.environment
    return event_dict


def add_log_level(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Add log level to event dict.
    
    Args:
        logger: Logger instance
        method_name: Method name being logged
        event_dict: Event dictionary
        
    Returns:
        EventDict: Updated event dictionary
    """
    if method_name == "warn":
        method_name = "warning"
    event_dict["level"] = method_name.upper()
    return event_dict


def drop_color_message_key(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Remove color_message key used by ConsoleRenderer.
    
    Args:
        logger: Logger instance
        method_name: Method name
        event_dict: Event dictionary
        
    Returns:
        EventDict: Cleaned event dictionary
    """
    event_dict.pop("color_message", None)
    return event_dict


def setup_logging() -> None:
    """
    Configure structured logging based on environment settings.
    
    Sets up:
    - JSON logging for production
    - Console logging for development
    - Log level from settings
    - Context processors
    """
    # Shared processors for all configurations
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        add_app_context,
        add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    
    # Configure based on format
    if settings.log_format == "json":
        # JSON logging for production
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            drop_color_message_key,
            structlog.processors.JSONRenderer(),
        ]
        renderer = structlog.processors.JSONRenderer()
    else:
        # Console logging for development
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level),
    )
    
    # Set third-party loggers to WARNING
    for logger_name in ["httpx", "httpcore", "urllib3", "qdrant_client"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        BoundLogger: Configured logger
    """
    return structlog.get_logger(name)


# Example usage with context
def log_with_context(logger: structlog.stdlib.BoundLogger, **context: Any) -> None:
    """
    Log with additional context.
    
    Args:
        logger: Logger instance
        **context: Context key-value pairs
        
    Example:
        >>> logger = get_logger(__name__)
        >>> log_with_context(logger, student_id="123", action="quiz_start")
        >>> logger.info("Quiz started successfully")
    """
    for key, value in context.items():
        structlog.contextvars.bind_contextvars(**{key: value})
