"""
Embeddings manager with caching and error handling.
"""
import hashlib
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.core.config import settings
from src.core.exceptions import EmbeddingError
from src.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingsManager:
    """
    Manages embedding generation with caching and error handling.
    """

    def __init__(self) -> None:
        """Initialize embeddings manager."""
        self._embeddings: Embeddings | None = None
        self._cache: dict[str, list[float]] = {}

    def initialize(self) -> None:
        """Initialize embeddings model."""
        try:
            self._embeddings = OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=settings.openai_api_key.get_secret_value(),
                dimensions=settings.embedding_dimensions,
            )

            logger.info(
                "embeddings_initialized",
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
            )

        except Exception as e:
            logger.error("embeddings_initialization_failed", error=str(e))
            raise EmbeddingError(f"Failed to initialize embeddings: {e}") from e

    @property
    def embeddings(self) -> Embeddings:
        """Get embeddings instance."""
        if not self._embeddings:
            raise EmbeddingError("Embeddings not initialized")
        return self._embeddings

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def embed_query(self, text: str, use_cache: bool = True) -> list[float]:
        """
        Generate embedding for query text.
        
        Args:
            text: Query text
            use_cache: Whether to use cache
            
        Returns:
            Embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                logger.debug("embedding_cache_hit", text_length=len(text))
                return self._cache[cache_key]

        try:
            logger.debug("generating_embedding", text_length=len(text))
            
            embedding = await self.embeddings.aembed_query(text)
            
            # Cache result
            if use_cache:
                cache_key = self._get_cache_key(text)
                self._cache[cache_key] = embedding
            
            logger.debug(
                "embedding_generated",
                text_length=len(text),
                embedding_dim=len(embedding),
            )
            
            return embedding

        except Exception as e:
            logger.error(
                "embedding_generation_failed",
                text_length=len(text),
                error=str(e),
            )
            raise EmbeddingError(f"Failed to generate embedding: {e}") from e

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def embed_documents(
        self, texts: list[str], use_cache: bool = True
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.
        
        Args:
            texts: List of document texts
            use_cache: Whether to use cache
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            return []

        try:
            # Check cache for each text
            embeddings: list[list[float]] = []
            texts_to_embed: list[str] = []
            text_indices: list[int] = []

            for i, text in enumerate(texts):
                if use_cache:
                    cache_key = self._get_cache_key(text)
                    if cache_key in self._cache:
                        embeddings.append(self._cache[cache_key])
                        continue

                texts_to_embed.append(text)
                text_indices.append(i)

            # Generate embeddings for uncached texts
            if texts_to_embed:
                logger.debug(
                    "generating_batch_embeddings",
                    count=len(texts_to_embed),
                )

                new_embeddings = await self.embeddings.aembed_documents(texts_to_embed)

                # Cache and insert results
                for text, embedding in zip(texts_to_embed, new_embeddings):
                    if use_cache:
                        cache_key = self._get_cache_key(text)
                        self._cache[cache_key] = embedding

                # Merge with cached results
                result = [None] * len(texts)
                cached_idx = 0
                new_idx = 0

                for i in range(len(texts)):
                    if i in text_indices:
                        result[i] = new_embeddings[new_idx]
                        new_idx += 1
                    else:
                        result[i] = embeddings[cached_idx]
                        cached_idx += 1

                embeddings = result  # type: ignore

            logger.info(
                "batch_embeddings_generated",
                total_count=len(texts),
                cached_count=len(texts) - len(texts_to_embed),
            )

            return embeddings

        except Exception as e:
            logger.error(
                "batch_embedding_failed",
                texts_count=len(texts),
                error=str(e),
            )
            raise EmbeddingError(f"Failed to generate batch embeddings: {e}") from e

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        cache_size = len(self._cache)
        self._cache.clear()
        logger.info("embedding_cache_cleared", cache_size=cache_size)


# Global instance
embeddings_manager = EmbeddingsManager()
