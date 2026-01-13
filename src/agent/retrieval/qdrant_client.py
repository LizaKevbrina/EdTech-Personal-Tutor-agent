"""
Production-ready Qdrant client with retry logic, error handling, and monitoring.
"""
import asyncio
from typing import Any, Literal

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.config import settings
from src.core.exceptions import (
    CollectionNotFoundError,
    QdrantConnectionError,
    QdrantQueryError,
)
from src.core.logging import get_logger
from src.core.metrics import track_qdrant_query

logger = get_logger(__name__)


class QdrantManager:
    """
    Manages Qdrant vector database operations with production features:
    - Connection pooling
    - Retry logic with exponential backoff
    - Error handling and fallbacks
    - Metrics tracking
    - Health checks
    """

    def __init__(self) -> None:
        """Initialize Qdrant client with configuration."""
        self._client: AsyncQdrantClient | None = None
        self._is_healthy = False

    async def initialize(self) -> None:
        """
        Initialize and test Qdrant connection.
        
        Raises:
            QdrantConnectionError: If connection fails
        """
        try:
            self._client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=(
                    settings.qdrant_api_key.get_secret_value()
                    if settings.qdrant_api_key
                    else None
                ),
                timeout=settings.qdrant_timeout,
            )

            # Test connection
            await self._client.get_collections()
            self._is_healthy = True

            logger.info(
                "qdrant_initialized",
                url=settings.qdrant_url,
            )

        except Exception as e:
            logger.error(
                "qdrant_initialization_failed",
                error=str(e),
                url=settings.qdrant_url,
            )
            raise QdrantConnectionError(
                f"Failed to initialize Qdrant client: {e}",
                details={"url": settings.qdrant_url},
            ) from e

    async def close(self) -> None:
        """Close Qdrant client connection."""
        if self._client:
            await self._client.close()
            self._is_healthy = False
            logger.info("qdrant_connection_closed")

    @property
    def client(self) -> AsyncQdrantClient:
        """
        Get Qdrant client instance.
        
        Returns:
            AsyncQdrantClient: Client instance
            
        Raises:
            QdrantConnectionError: If client not initialized
        """
        if not self._client:
            raise QdrantConnectionError("Qdrant client not initialized")
        return self._client

    @property
    def is_healthy(self) -> bool:
        """Check if Qdrant connection is healthy."""
        return self._is_healthy

    async def health_check(self) -> bool:
        """
        Perform health check on Qdrant connection.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            await self.client.get_collections()
            self._is_healthy = True
            return True
        except Exception as e:
            logger.warning("qdrant_health_check_failed", error=str(e))
            self._is_healthy = False
            return False

    @retry(
        retry=retry_if_exception_type((UnexpectedResponse, asyncio.TimeoutError)),
        stop=stop_after_attempt(settings.qdrant_retry_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    @track_qdrant_query(collection="dynamic", operation="search")
    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: float | None = None,
        query_filter: models.Filter | None = None,
    ) -> list[models.ScoredPoint]:
        """
        Search for similar vectors in collection with retry logic.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            query_filter: Optional metadata filter
            
        Returns:
            List of scored points
            
        Raises:
            CollectionNotFoundError: If collection doesn't exist
            QdrantQueryError: If search fails
        """
        try:
            # Validate collection exists
            if not await self._collection_exists(collection_name):
                raise CollectionNotFoundError(
                    f"Collection '{collection_name}' not found",
                    details={"collection": collection_name},
                )

            result = await self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold or settings.rag_score_threshold,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )

            logger.debug(
                "qdrant_search_completed",
                collection=collection_name,
                results_count=len(result),
                limit=limit,
            )

            return result

        except CollectionNotFoundError:
            raise
        except Exception as e:
            logger.error(
                "qdrant_search_failed",
                collection=collection_name,
                error=str(e),
            )
            raise QdrantQueryError(
                f"Search failed in collection '{collection_name}': {e}",
                details={
                    "collection": collection_name,
                    "limit": limit,
                },
            ) from e

    @retry(
        retry=retry_if_exception_type((UnexpectedResponse, asyncio.TimeoutError)),
        stop=stop_after_attempt(settings.qdrant_retry_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    @track_qdrant_query(collection="dynamic", operation="upsert")
    async def upsert(
        self,
        collection_name: str,
        points: list[models.PointStruct],
    ) -> models.UpdateResult:
        """
        Upsert points into collection with retry logic.
        
        Args:
            collection_name: Name of the collection
            points: List of points to upsert
            
        Returns:
            Update result
            
        Raises:
            QdrantQueryError: If upsert fails
        """
        try:
            result = await self.client.upsert(
                collection_name=collection_name,
                points=points,
            )

            logger.info(
                "qdrant_upsert_completed",
                collection=collection_name,
                points_count=len(points),
            )

            return result

        except Exception as e:
            logger.error(
                "qdrant_upsert_failed",
                collection=collection_name,
                error=str(e),
            )
            raise QdrantQueryError(
                f"Upsert failed in collection '{collection_name}': {e}",
                details={"collection": collection_name},
            ) from e

    async def _collection_exists(self, collection_name: str) -> bool:
        """
        Check if collection exists.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            bool: True if exists, False otherwise
        """
        try:
            collections = await self.client.get_collections()
            return any(c.name == collection_name for c in collections.collections)
        except Exception as e:
            logger.warning(
                "collection_check_failed",
                collection=collection_name,
                error=str(e),
            )
            return False

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Literal["Cosine", "Euclid", "Dot"] = "Cosine",
    ) -> None:
        """
        Create a new collection if it doesn't exist.
        
        Args:
            collection_name: Name of the collection
            vector_size: Size of vectors
            distance: Distance metric
            
        Raises:
            QdrantQueryError: If creation fails
        """
        try:
            if await self._collection_exists(collection_name):
                logger.info(
                    "collection_already_exists",
                    collection=collection_name,
                )
                return

            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=getattr(models.Distance, distance.upper()),
                ),
            )

            logger.info(
                "collection_created",
                collection=collection_name,
                vector_size=vector_size,
                distance=distance,
            )

        except Exception as e:
            logger.error(
                "collection_creation_failed",
                collection=collection_name,
                error=str(e),
            )
            raise QdrantQueryError(
                f"Failed to create collection '{collection_name}': {e}",
                details={"collection": collection_name},
            ) from e

    async def get_collection_info(
        self, collection_name: str
    ) -> models.CollectionInfo | None:
        """
        Get collection information.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection info or None if not found
        """
        try:
            return await self.client.get_collection(collection_name)
        except Exception as e:
            logger.warning(
                "get_collection_info_failed",
                collection=collection_name,
                error=str(e),
            )
            return None


# Global instance
qdrant_manager = QdrantManager()
