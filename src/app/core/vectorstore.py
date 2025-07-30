from typing import List

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from src.app.core.config import get_settings


def get_embeddings() -> OpenAIEmbeddings:
    """
    Get OpenAI embeddings with configured settings.

    Returns:
        OpenAIEmbeddings: Configured embeddings instance.
    """
    settings = get_settings()
    return OpenAIEmbeddings(
        openai_api_key=settings.openai_api_key,
        model=settings.openai_model,
        dimensions=settings.openai_dimensions,
        chunk_size=settings.openai_chunk_size,
        max_retries=settings.openai_max_retries,
        timeout=settings.openai_timeout,
        retry_min_seconds=settings.openai_retry_min_seconds,
        retry_max_seconds=settings.openai_retry_max_seconds,
        show_progress_bar=settings.show_embedding_progress,
        tiktoken_enabled=settings.enable_tiktoken,
    )


def get_vectordb() -> Chroma:
    """
    Get Chroma vector store with configured settings.

    Returns:
        Chroma: Configured vector store instance.
    """
    settings = get_settings()
    return Chroma(
        embedding_function=get_embeddings(),
        persist_directory=settings.vectorstore_persist_directory,
        collection_name=settings.vectorstore_collection_name,
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
