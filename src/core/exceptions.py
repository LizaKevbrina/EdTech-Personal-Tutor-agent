"""
Custom exceptions for the EdTech Tutor Agent.
Provides specific exception types for different error scenarios.
"""
from typing import Any


class EdTechTutorException(Exception):
    """Base exception for all custom exceptions."""
    
    def __init__(
        self,
        message: str,
        *args: Any,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize exception.
        
        Args:
            message: Error message
            *args: Additional arguments
            error_code: Error code for tracking
            details: Additional error details
        """
        super().__init__(message, *args)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}


# Configuration Errors
class ConfigurationError(EdTechTutorException):
    """Raised when configuration is invalid or missing."""
    pass


# Qdrant Errors
class QdrantConnectionError(EdTechTutorException):
    """Raised when cannot connect to Qdrant."""
    pass


class QdrantQueryError(EdTechTutorException):
    """Raised when Qdrant query fails."""
    pass


class CollectionNotFoundError(EdTechTutorException):
    """Raised when Qdrant collection doesn't exist."""
    pass


# LLM Errors
class LLMProviderError(EdTechTutorException):
    """Raised when LLM provider call fails."""
    pass


class LLMTimeoutError(EdTechTutorException):
    """Raised when LLM call times out."""
    pass


class LLMRateLimitError(EdTechTutorException):
    """Raised when LLM rate limit is exceeded."""
    pass


# RAG Errors
class RetrievalError(EdTechTutorException):
    """Raised when document retrieval fails."""
    pass


class EmbeddingError(EdTechTutorException):
    """Raised when embedding generation fails."""
    pass


# Memory Errors
class MemoryError(EdTechTutorException):
    """Raised when memory operations fail."""
    pass


# Tool Errors
class ToolExecutionError(EdTechTutorException):
    """Raised when tool execution fails."""
    pass


class CodeExecutionError(ToolExecutionError):
    """Raised when code execution in sandbox fails."""
    pass


class QuizGenerationError(ToolExecutionError):
    """Raised when quiz generation fails."""
    pass


# Input Validation Errors
class ValidationError(EdTechTutorException):
    """Raised when input validation fails."""
    pass


class PIIDetectedError(ValidationError):
    """Raised when PII is detected in input."""
    pass


class PromptInjectionError(ValidationError):
    """Raised when prompt injection attempt is detected."""
    pass


# Rate Limiting Errors
class RateLimitExceededError(EdTechTutorException):
    """Raised when rate limit is exceeded."""
    pass


class TokenBudgetExceededError(EdTechTutorException):
    """Raised when token budget is exceeded."""
    pass


# Student Progress Errors
class StudentNotFoundError(EdTechTutorException):
    """Raised when student profile doesn't exist."""
    pass


class ProgressUpdateError(EdTechTutorException):
    """Raised when progress update fails."""
    pass
