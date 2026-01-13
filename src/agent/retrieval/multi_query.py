"""
Multi-query retriever that generates multiple variations of a query
for improved retrieval performance.
"""
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.llm.llm_provider import llm_provider
from src.core.config import settings
from src.core.exceptions import RetrievalError
from src.core.logging import get_logger

logger = get_logger(__name__)


MULTI_QUERY_SYSTEM_PROMPT = """You are an AI assistant specialized in generating query variations for information retrieval in an educational context.

Your task is to generate {num_variations} different variations of the given query that maintain the same semantic meaning but use different phrasings, synonyms, and perspectives.

Guidelines:
- Each variation should capture the core intent of the original query
- Use different terminology and sentence structures
- Consider technical and non-technical phrasings
- Include relevant keywords and concepts
- Keep variations concise and focused

Return ONLY the variations, one per line, without numbering or additional text."""


class MultiQueryRetriever:
    """
    Generates multiple query variations for enhanced retrieval.
    
    Based on the Multi-Query Retriever pattern from LangChain,
    this generates diverse query formulations to improve recall.
    """

    def __init__(self, num_variations: int | None = None) -> None:
        """
        Initialize multi-query retriever.
        
        Args:
            num_variations: Number of query variations to generate
        """
        self.num_variations = num_variations or settings.rag_multi_query_variations

    async def generate_queries(self, original_query: str) -> list[str]:
        """
        Generate multiple query variations.
        
        Args:
            original_query: Original user query
            
        Returns:
            List of query variations including original
            
        Raises:
            RetrievalError: If query generation fails
        """
        if not original_query.strip():
            raise RetrievalError("Cannot generate queries from empty input")

        try:
            logger.debug(
                "generating_query_variations",
                original_query=original_query,
                num_variations=self.num_variations,
            )

            system_prompt = MULTI_QUERY_SYSTEM_PROMPT.format(
                num_variations=self.num_variations
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Query: {original_query}"),
            ]

            result = await llm_provider.generate(messages)
            variations_text = result.generations[0][0].text

            # Parse variations
            variations = [
                line.strip()
                for line in variations_text.strip().split("\n")
                if line.strip()
            ]

            # Remove numbering if present
            variations = [
                var.split(".", 1)[-1].strip() if "." in var[:5] else var
                for var in variations
            ]

            # Always include original query first
            all_queries = [original_query] + [
                var for var in variations if var != original_query
            ]

            # Limit to num_variations + 1 (including original)
            all_queries = all_queries[: self.num_variations + 1]

            logger.info(
                "query_variations_generated",
                original_query=original_query,
                variations_count=len(all_queries) - 1,
            )

            return all_queries

        except Exception as e:
            logger.error(
                "query_variation_failed",
                original_query=original_query,
                error=str(e),
            )

            # Fallback: return only original query
            logger.warning("falling_back_to_original_query")
            return [original_query]


# Global instance
multi_query_retriever = MultiQueryRetriever()
