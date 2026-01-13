"""
Complete RAG pipeline integrating multi-query retrieval,
vector search, and contextualized compression.
"""
from typing import Any

from langchain_core.documents import Document
from qdrant_client import models

from src.agent.retrieval.compressor import compressor
from src.agent.retrieval.embeddings import embeddings_manager
from src.agent.retrieval.multi_query import multi_query_retriever
from src.agent.retrieval.qdrant_client import qdrant_manager
from src.core.config import settings
from src.core.exceptions import RetrievalError
from src.core.logging import get_logger
from src.core.metrics import retrieval_relevance_score

logger = get_logger(__name__)


class RAGPipeline:
    """
    Production-ready RAG pipeline with:
    - Multi-query retrieval
    - Parallel vector search
    - Deduplication
    - Contextualized compression
    - Metadata filtering
    - Quality metrics
    """

    def __init__(
        self,
        collection_name: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        enable_compression: bool | None = None,
    ) -> None:
        """
        Initialize RAG pipeline.
        
        Args:
            collection_name: Qdrant collection to search
            top_k: Number of documents to retrieve per query
            score_threshold: Minimum similarity score
            enable_compression: Whether to compress retrieved documents
        """
        self.collection_name = collection_name
        self.top_k = top_k or settings.rag_top_k
        self.score_threshold = score_threshold or settings.rag_score_threshold
        self.enable_compression = (
            enable_compression
            if enable_compression is not None
            else settings.rag_compression_enabled
        )

    async def retrieve(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None = None,
        enable_multi_query: bool = True,
    ) -> list[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            metadata_filter: Optional metadata filters for Qdrant
            enable_multi_query: Whether to use multi-query retrieval
            
        Returns:
            List of relevant documents
            
        Raises:
            RetrievalError: If retrieval fails
        """
        try:
            logger.info(
                "rag_retrieval_started",
                query=query,
                collection=self.collection_name,
                enable_multi_query=enable_multi_query,
            )

            # Step 1: Generate query variations (if enabled)
            queries = [query]
            if enable_multi_query:
                queries = await multi_query_retriever.generate_queries(query)

            logger.debug("queries_generated", count=len(queries))

            # Step 2: Generate embeddings for all queries
            query_embeddings = await embeddings_manager.embed_documents(queries)

            # Step 3: Parallel search in Qdrant
            qdrant_filter = self._build_qdrant_filter(metadata_filter)

            all_results: list[models.ScoredPoint] = []
            for query_text, embedding in zip(queries, query_embeddings):
                results = await qdrant_manager.search(
                    collection_name=self.collection_name,
                    query_vector=embedding,
                    limit=self.top_k,
                    score_threshold=self.score_threshold,
                    query_filter=qdrant_filter,
                )
                all_results.extend(results)

                logger.debug(
                    "query_search_completed",
                    query=query_text,
                    results_count=len(results),
                )

            # Step 4: Deduplicate results
            unique_docs = self._deduplicate_results(all_results)

            logger.debug(
                "results_deduplicated",
                total_results=len(all_results),
                unique_results=len(unique_docs),
            )

            # Step 5: Convert to Document objects
            documents = self._convert_to_documents(unique_docs)

            # Step 6: Compress documents (if enabled)
            if self.enable_compression and documents:
                documents = await compressor.compress_documents(documents, query)

            # Step 7: Track quality metrics
            if documents:
                avg_score = sum(doc.metadata.get("score", 0) for doc in documents) / len(
                    documents
                )
                retrieval_relevance_score.labels(
                    collection=self.collection_name
                ).observe(avg_score)

            logger.info(
                "rag_retrieval_completed",
                query=query,
                documents_returned=len(documents),
                avg_relevance_score=avg_score if documents else 0,
            )

            return documents

        except Exception as e:
            logger.error(
                "rag_retrieval_failed",
                query=query,
                error=str(e),
            )
            raise RetrievalError(
                f"RAG retrieval failed: {e}",
                details={"query": query, "collection": self.collection_name},
            ) from e

    def _build_qdrant_filter(
        self, metadata_filter: dict[str, Any] | None
    ) -> models.Filter | None:
        """
        Build Qdrant filter from metadata dict.
        
        Args:
            metadata_filter: Metadata filter dict
            
        Returns:
            Qdrant Filter object or None
        """
        if not metadata_filter:
            return None

        conditions = []
        for key, value in metadata_filter.items():
            if isinstance(value, list):
                # Multiple values (OR condition)
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value),
                    )
                )
            else:
                # Single value
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )

        return models.Filter(must=conditions) if conditions else None

    def _deduplicate_results(
        self, results: list[models.ScoredPoint]
    ) -> list[models.ScoredPoint]:
        """
        Deduplicate search results by ID, keeping highest scores.
        
        Args:
            results: List of scored points
            
        Returns:
            Deduplicated list
        """
        seen_ids: dict[str, models.ScoredPoint] = {}

        for result in results:
            point_id = str(result.id)

            if point_id not in seen_ids or result.score > seen_ids[point_id].score:
                seen_ids[point_id] = result

        # Sort by score descending
        unique_results = sorted(
            seen_ids.values(), key=lambda x: x.score, reverse=True
        )

        return unique_results

    def _convert_to_documents(
        self, results: list[models.ScoredPoint]
    ) -> list[Document]:
        """
        Convert Qdrant results to LangChain Document objects.
        
        Args:
            results: List of scored points
            
        Returns:
            List of Document objects
        """
        documents = []

        for result in results:
            # Extract page_content from payload
            page_content = result.payload.get("page_content", "")
            if not page_content:
                page_content = result.payload.get("text", "")

            # Build metadata
            metadata = {
                "id": str(result.id),
                "score": result.score,
                **{k: v for k, v in result.payload.items() if k != "page_content"},
            }

            documents.append(
                Document(
                    page_content=page_content,
                    metadata=metadata,
                )
            )

        return documents

    async def retrieve_by_topic(
        self, query: str, topic: str
    ) -> list[Document]:
        """
        Retrieve documents filtered by topic.
        
        Args:
            query: User query
            topic: Topic to filter by
            
        Returns:
            List of relevant documents
        """
        return await self.retrieve(
            query=query,
            metadata_filter={"topic": topic},
        )


# Factory function for creating pipelines
def create_rag_pipeline(collection_name: str) -> RAGPipeline:
    """
    Create a RAG pipeline for a specific collection.
    
    Args:
        collection_name: Name of Qdrant collection
        
    Returns:
        Configured RAG pipeline
    """
    return RAGPipeline(collection_name=collection_name)
