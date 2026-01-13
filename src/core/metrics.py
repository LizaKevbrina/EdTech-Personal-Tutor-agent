"""
Prometheus metrics for monitoring agent performance.
Tracks performance, business, and quality metrics.
"""
from functools import wraps
from time import time
from typing import Any, Callable, TypeVar

from prometheus_client import Counter, Gauge, Histogram, Info

from src.core.config import settings

# Type variable for decorators
F = TypeVar("F", bound=Callable[..., Any])


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

# Request metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

# LLM metrics
llm_call_duration_seconds = Histogram(
    "llm_call_duration_seconds",
    "LLM API call duration in seconds",
    ["provider", "model", "operation"],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0),
)

llm_tokens_used_total = Counter(
    "llm_tokens_used_total",
    "Total tokens used",
    ["provider", "model", "token_type"],  # token_type: prompt, completion
)

llm_calls_total = Counter(
    "llm_calls_total",
    "Total LLM API calls",
    ["provider", "model", "status"],  # status: success, error, timeout
)

# Qdrant metrics
qdrant_query_duration_seconds = Histogram(
    "qdrant_query_duration_seconds",
    "Qdrant query duration in seconds",
    ["collection", "operation"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0),
)

qdrant_queries_total = Counter(
    "qdrant_queries_total",
    "Total Qdrant queries",
    ["collection", "operation", "status"],
)


# ============================================================================
# BUSINESS METRICS
# ============================================================================

student_questions_total = Counter(
    "student_questions_total",
    "Total student questions",
    ["topic", "difficulty"],
)

quiz_completions_total = Counter(
    "quiz_completions_total",
    "Total quiz completions",
    ["topic", "result"],  # result: passed, failed
)

quiz_completion_rate = Gauge(
    "quiz_completion_rate",
    "Quiz completion rate",
    ["topic"],
)

difficult_topics_total = Counter(
    "difficult_topics_total",
    "Topics marked as difficult",
    ["topic"],
)

learning_pace_distribution = Histogram(
    "learning_pace_distribution",
    "Distribution of learning pace",
    ["student_id"],
    buckets=(0.5, 1.0, 1.5, 2.0, 3.0, 5.0),
)

active_students_total = Gauge(
    "active_students_total",
    "Number of currently active students",
)


# ============================================================================
# QUALITY METRICS
# ============================================================================

retrieval_relevance_score = Histogram(
    "retrieval_relevance_score",
    "Document retrieval relevance score",
    ["collection"],
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)

answer_accuracy_score = Gauge(
    "answer_accuracy_score",
    "Answer accuracy based on feedback",
    ["topic"],
)

tool_execution_success_rate = Gauge(
    "tool_execution_success_rate",
    "Tool execution success rate",
    ["tool_name"],
)

ragas_context_relevance = Gauge(
    "ragas_context_relevance",
    "RAGAS context relevance metric",
)

ragas_answer_relevance = Gauge(
    "ragas_answer_relevance",
    "RAGAS answer relevance metric",
)

ragas_faithfulness = Gauge(
    "ragas_faithfulness",
    "RAGAS faithfulness metric",
)


# ============================================================================
# SYSTEM METRICS
# ============================================================================

cache_hit_rate = Gauge(
    "cache_hit_rate",
    "Redis cache hit rate",
)

error_rate = Gauge(
    "error_rate",
    "Overall error rate",
    ["error_type"],
)

rate_limit_exceeded_total = Counter(
    "rate_limit_exceeded_total",
    "Total rate limit violations",
    ["limit_type"],  # limit_type: requests, tokens
)


# ============================================================================
# APPLICATION INFO
# ============================================================================

app_info = Info("app_info", "Application information")
app_info.info({
    "name": settings.api_title,
    "version": settings.api_version,
    "environment": settings.environment,
})


# ============================================================================
# DECORATOR UTILITIES
# ============================================================================

def track_llm_call(provider: str, model: str, operation: str = "completion") -> Callable[[F], F]:
    """
    Decorator to track LLM API calls.
    
    Args:
        provider: LLM provider name (e.g., "openai")
        model: Model name (e.g., "gpt-4")
        operation: Operation type (e.g., "completion", "embedding")
        
    Returns:
        Decorated function
        
    Example:
        >>> @track_llm_call("openai", "gpt-4", "completion")
        >>> async def call_llm():
        >>>     return await llm.ainvoke(...)
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                
                # Track token usage if available
                if hasattr(result, "usage"):
                    llm_tokens_used_total.labels(
                        provider=provider,
                        model=model,
                        token_type="prompt",
                    ).inc(result.usage.prompt_tokens)
                    
                    llm_tokens_used_total.labels(
                        provider=provider,
                        model=model,
                        token_type="completion",
                    ).inc(result.usage.completion_tokens)
                
                return result
            
            except TimeoutError:
                status = "timeout"
                raise
            except Exception:
                status = "error"
                raise
            finally:
                duration = time() - start_time
                
                llm_call_duration_seconds.labels(
                    provider=provider,
                    model=model,
                    operation=operation,
                ).observe(duration)
                
                llm_calls_total.labels(
                    provider=provider,
                    model=model,
                    status=status,
                ).inc()
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                
                # Track token usage if available
                if hasattr(result, "usage"):
                    llm_tokens_used_total.labels(
                        provider=provider,
                        model=model,
                        token_type="prompt",
                    ).inc(result.usage.prompt_tokens)
                    
                    llm_tokens_used_total.labels(
                        provider=provider,
                        model=model,
                        token_type="completion",
                    ).inc(result.usage.completion_tokens)
                
                return result
            
            except TimeoutError:
                status = "timeout"
                raise
            except Exception:
                status = "error"
                raise
            finally:
                duration = time() - start_time
                
                llm_call_duration_seconds.labels(
                    provider=provider,
                    model=model,
                    operation=operation,
                ).observe(duration)
                
                llm_calls_total.labels(
                    provider=provider,
                    model=model,
                    status=status,
                ).inc()
        
        # Return appropriate wrapper based on function type
        if hasattr(func, "__call__") and hasattr(func.__call__, "__await__"):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


def track_qdrant_query(collection: str, operation: str) -> Callable[[F], F]:
    """
    Decorator to track Qdrant queries.
    
    Args:
        collection: Collection name
        operation: Operation type (e.g., "search", "insert", "update")
        
    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time()
            status = "success"
            
            try:
                return await func(*args, **kwargs)
            except Exception:
                status = "error"
                raise
            finally:
                duration = time() - start_time
                
                qdrant_query_duration_seconds.labels(
                    collection=collection,
                    operation=operation,
                ).observe(duration)
                
                qdrant_queries_total.labels(
                    collection=collection,
                    operation=operation,
                    status=status,
                ).inc()
        
        return wrapper  # type: ignore
    
    return decorator
