"""
All application configuration lives here.
Values are read from environment variables (or .env) via pydantic-settings.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # RAG pipeline parameters
    RAG_PERSIST_DIR: str = "./rag_store"
    RAG_CHUNK_SIZE: int = 500
    RAG_OVERLAP: int = 80
    RAG_TOP_K: int = 20
    RAG_TOP_N: int = 5

    # Auth — default allows Swagger UI to work out of the box
    API_KEY: str = "dev-key-123"

    # Environment tag
    ENV: str = "development"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read once at startup)."""
    return Settings()
