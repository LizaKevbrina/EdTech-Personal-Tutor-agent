"""
Configuration management with Pydantic Settings.
Supports multiple environments (dev, prod) and validation.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment-based configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    environment: Literal["dev", "prod", "test"] = Field(default="dev")
    debug: bool = Field(default=False)
    
    # API Settings
    api_title: str = Field(default="EdTech Tutor Agent")
    api_version: str = Field(default="0.1.0")
    api_prefix: str = Field(default="/api/v1")
    
    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=4)
    
    # Security
    secret_key: SecretStr = Field(default="change-me-in-production")
    allowed_origins: list[str] = Field(default=["http://localhost:3000"])
    
    # LLM Provider (OpenAI by default)
    llm_provider: Literal["openai", "anthropic", "azure"] = Field(default="openai")
    openai_api_key: SecretStr = Field(default="")
    openai_model_name: str = Field(default="gpt-4-turbo-preview")
    openai_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    openai_max_tokens: int = Field(default=1000, ge=1, le=4000)
    openai_timeout: int = Field(default=30, ge=5, le=120)
    
    # Fallback LLM (if primary fails)
    fallback_model_name: str = Field(default="gpt-3.5-turbo")
    
    # Embeddings
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimensions: int = Field(default=1536)
    
    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: SecretStr | None = Field(default=None)
    qdrant_collection_courses: str = Field(default="courses_content")
    qdrant_collection_quiz: str = Field(default="quiz_bank")
    qdrant_collection_history: str = Field(default="student_history")
    qdrant_timeout: int = Field(default=10)
    qdrant_retry_attempts: int = Field(default=3)
    
    # Redis (for rate limiting and caching)
    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_password: SecretStr | None = Field(default=None)
    redis_timeout: int = Field(default=5)
    
    # Rate Limiting
    rate_limit_requests_per_hour: int = Field(default=100)
    rate_limit_tokens_per_day: int = Field(default=10000)
    
    # RAG Configuration
    rag_top_k: int = Field(default=5, ge=1, le=20)
    rag_score_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    rag_multi_query_variations: int = Field(default=3, ge=1, le=5)
    rag_compression_enabled: bool = Field(default=True)
    rag_max_context_length: int = Field(default=8000)
    
    # Memory
    memory_buffer_size: int = Field(default=10, ge=1, le=50)
    memory_summary_enabled: bool = Field(default=True)
    
    # Tools
    code_executor_enabled: bool = Field(default=True)
    code_executor_timeout: int = Field(default=10, ge=5, le=30)
    quiz_generator_enabled: bool = Field(default=True)
    
    # Monitoring
    langsmith_api_key: SecretStr | None = Field(default=None)
    langsmith_project: str = Field(default="edtech-tutor")
    langsmith_enabled: bool = Field(default=False)
    
    sentry_dsn: SecretStr | None = Field(default=None)
    sentry_enabled: bool = Field(default=False)
    sentry_environment: str = Field(default="development")
    
    prometheus_enabled: bool = Field(default=True)
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    log_format: Literal["json", "console"] = Field(default="json")
    
    # Evaluation
    ragas_evaluation_enabled: bool = Field(default=True)
    ragas_sample_size: int = Field(default=50)
    llm_judge_enabled: bool = Field(default=False)
    llm_judge_model: str = Field(default="gpt-4")
    
    @field_validator("openai_api_key", "qdrant_api_key", mode="before")
    @classmethod
    def validate_api_keys(cls, v: str | SecretStr | None) -> SecretStr:
        """Validate API keys are not empty in production."""
        if isinstance(v, SecretStr):
            return v
        if v is None or v == "":
            return SecretStr("")
        return SecretStr(v)
    
    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, v: str | list[str]) -> list[str]:
        """Parse comma-separated origins string."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    def validate_production(self) -> None:
        """Validate critical settings for production environment."""
        if self.environment != "prod":
            return
        
        errors = []
        
        if not self.openai_api_key.get_secret_value():
            errors.append("OPENAI_API_KEY must be set in production")
        
        if self.secret_key.get_secret_value() == "change-me-in-production":
            errors.append("SECRET_KEY must be changed in production")
        
        if self.debug:
            errors.append("DEBUG must be False in production")
        
        if errors:
            raise ValueError(
                f"Production validation failed:\n" + "\n".join(f"- {e}" for e in errors)
            )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings
    """
    settings = Settings()
    
    # Validate production settings
    if settings.environment == "prod":
        settings.validate_production()
    
    return settings


# Global settings instance
settings = get_settings()
