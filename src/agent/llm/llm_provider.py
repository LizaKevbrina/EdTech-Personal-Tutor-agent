"""
LLM provider wrapper with fallback mechanism, retry logic, and monitoring.
"""
import asyncio
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.config import settings
from src.core.exceptions import LLMProviderError, LLMRateLimitError, LLMTimeoutError
from src.core.logging import get_logger
from src.core.metrics import track_llm_call

logger = get_logger(__name__)


class LLMProvider:
    """
    LLM provider with production features:
    - Primary and fallback models
    - Retry logic with exponential backoff
    - Timeout handling
    - Token usage tracking
    - Error handling and graceful degradation
    """

    def __init__(self) -> None:
        """Initialize LLM provider with primary and fallback models."""
        self._primary_llm: BaseChatModel | None = None
        self._fallback_llm: BaseChatModel | None = None
        self._is_initialized = False

    def initialize(self) -> None:
        """
        Initialize LLM models.
        
        Raises:
            LLMProviderError: If initialization fails
        """
        try:
            # Primary model
            self._primary_llm = ChatOpenAI(
                model=settings.openai_model_name,
                temperature=settings.openai_temperature,
                max_tokens=settings.openai_max_tokens,
                timeout=settings.openai_timeout,
                api_key=settings.openai_api_key.get_secret_value(),
                max_retries=0,  # We handle retries ourselves
            )

            # Fallback model (cheaper/faster)
            self._fallback_llm = ChatOpenAI(
                model=settings.fallback_model_name,
                temperature=settings.openai_temperature,
                max_tokens=settings.openai_max_tokens,
                timeout=settings.openai_timeout,
                api_key=settings.openai_api_key.get_secret_value(),
                max_retries=0,
            )

            self._is_initialized = True

            logger.info(
                "llm_provider_initialized",
                primary_model=settings.openai_model_name,
                fallback_model=settings.fallback_model_name,
            )

        except Exception as e:
            logger.error("llm_provider_initialization_failed", error=str(e))
            raise LLMProviderError(f"Failed to initialize LLM provider: {e}") from e

    @property
    def primary_llm(self) -> BaseChatModel:
        """Get primary LLM instance."""
        if not self._primary_llm:
            raise LLMProviderError("LLM provider not initialized")
        return self._primary_llm

    @property
    def fallback_llm(self) -> BaseChatModel:
        """Get fallback LLM instance."""
        if not self._fallback_llm:
            raise LLMProviderError("LLM provider not initialized")
        return self._fallback_llm

    @retry(
        retry=retry_if_exception_type(LLMRateLimitError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        reraise=True,
    )
    @track_llm_call(
        provider="openai",
        model=settings.openai_model_name,
        operation="completion",
    )
    async def generate(
        self,
        messages: list[BaseMessage],
        use_fallback: bool = False,
    ) -> ChatResult:
        """
        Generate completion with retry and fallback logic.
        
        Args:
            messages: List of messages for the conversation
            use_fallback: Whether to use fallback model directly
            
        Returns:
            ChatResult with generation
            
        Raises:
            LLMProviderError: If all attempts fail
            LLMTimeoutError: If timeout occurs
            LLMRateLimitError: If rate limit exceeded
        """
        llm = self.fallback_llm if use_fallback else self.primary_llm
        model_name = (
            settings.fallback_model_name if use_fallback else settings.openai_model_name
        )

        try:
            logger.debug(
                "llm_generate_started",
                model=model_name,
                messages_count=len(messages),
            )

            result = await llm.agenerate([messages])

            logger.info(
                "llm_generate_completed",
                model=model_name,
                prompt_tokens=result.llm_output.get("token_usage", {}).get(
                    "prompt_tokens", 0
                ),
                completion_tokens=result.llm_output.get("token_usage", {}).get(
                    "completion_tokens", 0
                ),
            )

            return result

        except asyncio.TimeoutError as e:
            logger.error(
                "llm_timeout",
                model=model_name,
                timeout=settings.openai_timeout,
            )
            
            # Try fallback if not already using it
            if not use_fallback:
                logger.info("attempting_fallback_model")
                return await self.generate(messages, use_fallback=True)
            
            raise LLMTimeoutError(
                f"LLM call timed out after {settings.openai_timeout}s",
                details={"model": model_name},
            ) from e

        except Exception as e:
            error_str = str(e).lower()

            # Handle rate limiting
            if "rate limit" in error_str or "429" in error_str:
                logger.warning(
                    "llm_rate_limit",
                    model=model_name,
                    error=str(e),
                )
                raise LLMRateLimitError(
                    "Rate limit exceeded",
                    details={"model": model_name},
                ) from e

            # Handle other errors with fallback
            logger.error(
                "llm_generation_failed",
                model=model_name,
                error=str(e),
            )

            # Try fallback if not already using it
            if not use_fallback:
                logger.info("attempting_fallback_model_after_error")
                try:
                    return await self.generate(messages, use_fallback=True)
                except Exception as fallback_error:
                    logger.error(
                        "fallback_model_failed",
                        error=str(fallback_error),
                    )

            raise LLMProviderError(
                f"LLM generation failed: {e}",
                details={"model": model_name},
            ) from e

    async def generate_with_system_prompt(
        self,
        system_prompt: str,
        user_message: str,
        use_fallback: bool = False,
    ) -> str:
        """
        Convenient method to generate with system prompt.
        
        Args:
            system_prompt: System instruction
            user_message: User message
            use_fallback: Whether to use fallback model
            
        Returns:
            Generated text
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        result = await self.generate(messages, use_fallback=use_fallback)
        return result.generations[0][0].text

    async def count_tokens(self, text: str) -> int:
        """
        Count tokens in text (approximate).
        
        Args:
            text: Text to count tokens
            
        Returns:
            Approximate token count
        """
        # Simple approximation: 1 token ≈ 4 characters
        # For production, use tiktoken library
        return len(text) // 4


# Global instance
llm_provider = LLMProvider()
