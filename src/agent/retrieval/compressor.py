"""
Contextualized compressor that filters and compresses retrieved documents
to keep only relevant information.
"""
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.llm.llm_provider import llm_provider
from src.core.config import settings
from src.core.exceptions import RetrievalError
from src.core.logging import get_logger

logger = get_logger(__name__)


COMPRESSION_SYSTEM_PROMPT = """You are an AI assistant that extracts only the most relevant information from documents based on a query.

Your task:
1. Read the provided document chunk
2. Identify ONLY the information directly relevant to answering the query
3. Extract and return this relevant information concisely
4. If the chunk contains NO relevant information, return "NOT_RELEVANT"

Guidelines:
- Be precise and concise
- Keep the original meaning and context
- Remove fluff, redundant information, and tangential details
- Preserve key facts, numbers, and specific details
- If unsure about relevance, include it (better to over-include than miss important info)

Query: {query}

Document chunk:
{document}

Relevant extract:"""


class ContextualizedCompressor:
    """
    Compresses retrieved documents by filtering out irrelevant parts.
    
    Uses LLM to intelligently identify and extract only the parts
    of documents that are relevant to the query, reducing token usage
    and improving context quality.
    """

    def __init__(
        self,
        max_context_length: int | None = None,
        use_fallback_model: bool = True,
    ) -> None:
        """
        Initialize compressor.
        
        Args:
            max_context_length: Maximum total context length in tokens
            use_fallback_model: Whether to use cheaper fallback model
        """
        self.max_context_length = (
            max_context_length or settings.rag_max_context_length
        )
        self.use_fallback_model = use_fallback_model

    async def compress_documents(
        self,
        documents: list[Document],
        query: str,
    ) -> list[Document]:
        """
        Compress documents by extracting only relevant information.
        
        Args:
            documents: List of retrieved documents
            query: Original query
            
        Returns:
            List of compressed documents with only relevant content
            
        Raises:
            RetrievalError: If compression fails
        """
        if not documents:
            return []

        if not query.strip():
            logger.warning("empty_query_for_compression")
            return documents

        try:
            logger.debug(
                "compressing_documents",
                query=query,
                documents_count=len(documents),
            )

            compressed_docs: list[Document] = []
            total_tokens = 0

            for doc in documents:
                # Check if we've reached max context length
                if total_tokens >= self.max_context_length:
                    logger.info(
                        "max_context_length_reached",
                        total_tokens=total_tokens,
                        max_length=self.max_context_length,
                    )
                    break

                # Compress individual document
                compressed_content = await self._compress_single_document(
                    doc.page_content, query
                )

                # Skip if not relevant
                if compressed_content == "NOT_RELEVANT":
                    logger.debug(
                        "document_not_relevant",
                        doc_id=doc.metadata.get("id", "unknown"),
                    )
                    continue

                # Estimate tokens (rough approximation)
                doc_tokens = len(compressed_content) // 4
                total_tokens += doc_tokens

                # Create compressed document
                compressed_doc = Document(
                    page_content=compressed_content,
                    metadata={
                        **doc.metadata,
                        "compressed": True,
                        "original_length": len(doc.page_content),
                        "compressed_length": len(compressed_content),
                        "compression_ratio": len(compressed_content)
                        / len(doc.page_content)
                        if len(doc.page_content) > 0
                        else 0,
                    },
                )

                compressed_docs.append(compressed_doc)

            logger.info(
                "documents_compressed",
                original_count=len(documents),
                compressed_count=len(compressed_docs),
                total_tokens=total_tokens,
            )

            return compressed_docs

        except Exception as e:
            logger.error(
                "compression_failed",
                error=str(e),
                documents_count=len(documents),
            )

            # Fallback: return original documents (truncated if needed)
            logger.warning("falling_back_to_original_documents")
            return self._fallback_truncate(documents)

    async def _compress_single_document(
        self, document_text: str, query: str
    ) -> str:
        """
        Compress a single document.
        
        Args:
            document_text: Document content
            query: Query for context
            
        Returns:
            Compressed text or "NOT_RELEVANT"
        """
        try:
            prompt = COMPRESSION_SYSTEM_PROMPT.format(
                query=query, document=document_text
            )

            messages = [SystemMessage(content=prompt)]

            result = await llm_provider.generate(
                messages, use_fallback=self.use_fallback_model
            )

            compressed = result.generations[0][0].text.strip()

            return compressed

        except Exception as e:
            logger.warning(
                "single_document_compression_failed",
                error=str(e),
            )
            # Return original on error
            return document_text

    def _fallback_truncate(self, documents: list[Document]) -> list[Document]:
        """
        Fallback: truncate documents to fit max context length.
        
        Args:
            documents: Original documents
            
        Returns:
            Truncated documents
        """
        truncated_docs: list[Document] = []
        total_tokens = 0

        for doc in documents:
            doc_tokens = len(doc.page_content) // 4

            if total_tokens + doc_tokens > self.max_context_length:
                # Truncate this document
                remaining_tokens = self.max_context_length - total_tokens
                remaining_chars = remaining_tokens * 4

                if remaining_chars > 0:
                    truncated_content = doc.page_content[:remaining_chars] + "..."
                    truncated_doc = Document(
                        page_content=truncated_content,
                        metadata={**doc.metadata, "truncated": True},
                    )
                    truncated_docs.append(truncated_doc)
                break

            truncated_docs.append(doc)
            total_tokens += doc_tokens

        return truncated_docs


# Global instance
compressor = ContextualizedCompressor()
