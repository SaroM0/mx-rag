from functools import lru_cache
from typing import List

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.app.core.config import get_settings


@lru_cache()
def get_embeddings() -> OpenAIEmbeddings:
    """
    Returns a cached instance of OpenAIEmbeddings with optimized configuration.

    Returns:
        OpenAIEmbeddings: Configured embeddings instance.

    Raises:
        ValueError: If OpenAI API key is not set.
    """
    settings = get_settings()

    if not settings.openai_api_key:
        raise ValueError("OpenAI API key not found in settings")

    return OpenAIEmbeddings(
        model=settings.openai_model,
        openai_api_key=settings.openai_api_key,
        dimensions=settings.openai_dimensions,
        chunk_size=settings.openai_chunk_size,
        show_progress_bar=settings.show_embedding_progress,
        retry_min_seconds=settings.openai_retry_min_seconds,
        retry_max_seconds=settings.openai_retry_max_seconds,
        max_retries=settings.openai_max_retries,
        timeout=settings.openai_timeout,
        tiktoken_enabled=settings.enable_tiktoken,
    )


@lru_cache()
def get_vectordb() -> Chroma:
    """
    Returns a cached instance of Chroma vectorstore with optimized configuration.

    This is the single source of truth for vector storage in the application.
    If you need to switch to another store (Pinecone, Weaviate), this is the only place to change.

    Returns:
        Chroma: Configured vector store instance.
    """
    settings = get_settings()
    embeddings = get_embeddings()

    return Chroma(
        persist_directory=settings.vectorstore_persist_directory,
        embedding_function=embeddings,
        collection_name=settings.vectorstore_collection_name or "langchain",
    )


async def aembed_texts(texts: List[str]) -> List[List[float]]:
    """
    Asynchronously embed multiple texts using the configured embeddings.

    Args:
        texts: List of texts to embed.

    Returns:
        List of embeddings vectors.
    """
    embeddings = get_embeddings()
    return await embeddings.aembed_documents(texts)


async def aembed_query(query: str) -> List[float]:
    """
    Asynchronously embed a single query text.

    Args:
        query: Text to embed.

    Returns:
        Query embedding vector.
    """
    embeddings = get_embeddings()
    return await embeddings.aembed_query(query)
