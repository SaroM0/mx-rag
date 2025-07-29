from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings managed through environment variables.
    """

    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "text-embedding-3-large"
    openai_dimensions: int = 1536
    openai_chunk_size: int = 1000
    openai_max_retries: int = 3
    openai_timeout: float = 60.0
    openai_retry_min_seconds: int = 4
    openai_retry_max_seconds: int = 20

    # Vector Store Configuration
    vectorstore_persist_directory: str = "src/data/chroma"
    vectorstore_collection_name: Optional[str] = None

    # Embedding Configuration
    show_embedding_progress: bool = True
    enable_tiktoken: bool = True

    # Ingestion Configuration
    pdf_directory: str = "src/pdfs"
    chunks_directory: str = "src/data/chunks"
    chunk_size: int = 512
    chunk_overlap: int = 50
    save_chunks: bool = True

    # Chat Configuration
    chat_model_name: str = "gpt-4"
    chat_temperature: float = 0.0
    chat_max_tokens: int = 2000
    chat_top_k: int = 3
    chat_return_source_docs: bool = True
    chat_streaming: bool = False

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached instance of the settings.
    This avoids reading the environment every time the settings are accessed.
    """
    return Settings()
