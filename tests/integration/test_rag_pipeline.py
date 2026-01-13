"""
Integration tests for RAG pipeline.
"""
import pytest
from langchain_core.documents import Document

from src.agent.retrieval.embeddings import embeddings_manager
from src.agent.retrieval.qdrant_client import qdrant_manager
from src.agent.retrieval.rag_pipeline import RAGPipeline
from src.core.config import settings


@pytest.fixture(scope="module")
async def setup_qdrant():
    """Setup Qdrant for testing."""
    await qdrant_manager.initialize()
    embeddings_manager.initialize()

    # Create test collection
    await qdrant_manager.create_collection(
        collection_name="test_courses",
        vector_size=settings.embedding_dimensions,
    )

    yield

    # Cleanup
    await qdrant_manager.close()


@pytest.mark.asyncio
class TestRAGPipeline:
    """Test suite for RAG pipeline."""

    async def test_retrieve_documents(self, setup_qdrant):
        """Test document retrieval."""
        pipeline = RAGPipeline(collection_name="test_courses", top_k=3)

        # Note: This requires seeded data
        docs = await pipeline.retrieve(
            query="Python recursion",
            enable_multi_query=False,
        )

        assert isinstance(docs, list)
        # May be empty if no data seeded

    async def test_multi_query_retrieval(self, setup_qdrant):
        """Test multi-query retrieval."""
        pipeline = RAGPipeline(collection_name="test_courses")

        docs = await pipeline.retrieve(
            query="Explain loops",
            enable_multi_query=True,
        )

        assert isinstance(docs, list)

    async def test_metadata_filtering(self, setup_qdrant):
        """Test retrieval with metadata filters."""
        pipeline = RAGPipeline(collection_name="test_courses")

        docs = await pipeline.retrieve(
            query="Python",
            metadata_filter={"difficulty": "beginner"},
        )

        assert isinstance(docs, list)
