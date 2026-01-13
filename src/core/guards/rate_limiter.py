"""
Rate limiter for controlling API usage per user.
"""
import time
from typing import Any

from redis import Redis
from redis.exceptions import RedisError

from src.core.config import settings
from src.core.exceptions import RateLimitExceededError, TokenBudgetExceededError
from src.core.logging import get_logger
from src.core.metrics import rate_limit_exceeded_total

logger = get_logger(__name__)


class RateLimiter:
    """
    Redis-based rate limiter with sliding window.
    
    Limits:
    - Requests per hour per user
    - Tokens per day per user
    """

    def __init__(self) -> None:
        """Initialize rate limiter."""
        self._redis: Redis | None = None
        self._enabled = True

    def initialize(self) -> None:
        """Initialize Redis connection."""
        try:
            self._redis = Redis.from_url(
                settings.redis_url,
                password=(
                    settings.redis_password.get_secret_value()
                    if settings.redis_password
                    else None
                ),
                decode_responses=True,
                socket_timeout=settings.redis_timeout,
            )

            # Test connection
            self._redis.ping()

            logger.info("rate_limiter_initialized", redis_url=settings.redis_url)

        except RedisError as e:
            logger.error(
                "rate_limiter_initialization_failed",
                error=str(e),
            )
            # Continue without rate limiting
            self._enabled = False
            logger.warning("rate_limiting_disabled")

    @property
    def redis(self) -> Redis:
        """Get Redis client."""
        if not self._redis:
            raise RuntimeError("Rate limiter not initialized")
        return self._redis

    async def check_request_limit(self, user_id: str) -> None:
        """
        Check if user has exceeded request rate limit.
        
        Args:
            user_id: User identifier
            
        Raises:
            RateLimitExceededError: If limit exceeded
        """
        if not self._enabled:
            return

        try:
            key = f"rate_limit:requests:{user_id}"
            current_time = int(time.time())
            window_start = current_time - 3600  # 1 hour window

            # Remove old entries
            self.redis.zremrangebyscore(key, 0, window_start)

            # Count requests in window
            request_count = self.redis.zcard(key)

            if request_count >= settings.rate_limit_requests_per_hour:
                logger.warning(
                    "request_rate_limit_exceeded",
                    user_id=user_id,
                    count=request_count,
                    limit=settings.rate_limit_requests_per_hour,
                )

                # Update metrics
                rate_limit_exceeded_total.labels(limit_type="requests").inc()

                raise RateLimitExceededError(
                    f"Request rate limit exceeded: {request_count}/{settings.rate_limit_requests_per_hour} per hour",
                    details={
                        "user_id": user_id,
                        "limit": settings.rate_limit_requests_per_hour,
                        "current": request_count,
                    },
                )

            # Add current request
            self.redis.zadd(key, {str(current_time): current_time})
            self.redis.expire(key, 3600)

            logger.debug(
                "request_limit_checked",
                user_id=user_id,
                count=request_count + 1,
                limit=settings.rate_limit_requests_per_hour,
            )

        except RateLimitExceededError:
            raise
        except RedisError as e:
            logger.error("rate_limit_check_failed", error=str(e))
            # Allow request on Redis failure
            pass

    async def check_token_limit(self, user_id: str, tokens_used: int) -> None:
        """
        Check if user has exceeded daily token limit.
        
        Args:
            user_id: User identifier
            tokens_used: Number of tokens to add
            
        Raises:
            TokenBudgetExceededError: If limit exceeded
        """
        if not self._enabled:
            return

        try:
            key = f"rate_limit:tokens:{user_id}"
            current_time = int(time.time())
            day_start = current_time - 86400  # 24 hours

            # Remove old entries
            self.redis.zremrangebyscore(key, 0, day_start)

            # Sum tokens in window
            entries = self.redis.zrange(key, 0, -1, withscores=True)
            total_tokens = sum(int(score) for _, score in entries)

            if total_tokens + tokens_used > settings.rate_limit_tokens_per_day:
                logger.warning(
                    "token_budget_exceeded",
                    user_id=user_id,
                    total=total_tokens + tokens_used,
                    limit=settings.rate_limit_tokens_per_day,
                )

                # Update metrics
                rate_limit_exceeded_total.labels(limit_type="tokens").inc()

                raise TokenBudgetExceededError(
                    f"Daily token budget exceeded: {total_tokens + tokens_used}/{settings.rate_limit_tokens_per_day}",
                    details={
                        "user_id": user_id,
                        "limit": settings.rate_limit_tokens_per_day,
                        "current": total_tokens + tokens_used,
                    },
                )

            # Add tokens
            self.redis.zadd(key, {f"{current_time}:{tokens_used}": tokens_used})
            self.redis.expire(key, 86400)

            logger.debug(
                "token_limit_checked",
                user_id=user_id,
                tokens=total_tokens + tokens_used,
                limit=settings.rate_limit_tokens_per_day,
            )

        except TokenBudgetExceededError:
            raise
        except RedisError as e:
            logger.error("token_limit_check_failed", error=str(e))
            # Allow on Redis failure
            pass

    async def get_usage_stats(self, user_id: str) -> dict[str, Any]:
        """
        Get usage statistics for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Usage stats dictionary
        """
        if not self._enabled:
            return {
                "requests_remaining": -1,
                "tokens_remaining": -1,
                "rate_limiting_enabled": False,
            }

        try:
            current_time = int(time.time())

            # Requests
            requests_key = f"rate_limit:requests:{user_id}"
            window_start = current_time - 3600
            self.redis.zremrangebyscore(requests_key, 0, window_start)
            request_count = self.redis.zcard(requests_key)

            # Tokens
            tokens_key = f"rate_limit:tokens:{user_id}"
            day_start = current_time - 86400
            self.redis.zremrangebyscore(tokens_key, 0, day_start)
            entries = self.redis.zrange(tokens_key, 0, -1, withscores=True)
            total_tokens = sum(int(score) for _, score in entries)

            return {
                "requests_used": request_count,
                "requests_limit": settings.rate_limit_requests_per_hour,
                "requests_remaining": max(
                    0, settings.rate_limit_requests_per_hour - request_count
                ),
                "tokens_used": total_tokens,
                "tokens_limit": settings.rate_limit_tokens_per_day,
                "tokens_remaining": max(
                    0, settings.rate_limit_tokens_per_day - total_tokens
                ),
                "rate_limiting_enabled": True,
            }

        except RedisError as e:
            logger.error("get_usage_stats_failed", error=str(e))
            return {
                "error": "Failed to retrieve usage stats",
                "rate_limiting_enabled": False,
            }

    def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            self._redis.close()
            logger.info("rate_limiter_closed")


# Global instance
rate_limiter = RateLimiter()
