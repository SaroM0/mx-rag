from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from src.app.core.vectorstore import aembed_texts, get_embeddings, get_vectordb


@pytest.fixture
def mock_embeddings():
    """Fixture for mock embeddings"""
    mock = AsyncMock()
    mock.embed_documents.return_value = [np.random.rand(1536) for _ in range(2)]
    mock.aembed_documents.return_value = [np.random.rand(1536) for _ in range(2)]
    return mock


def test_get_embeddings():
    """Test getting OpenAI embeddings"""
    with patch("openai.AsyncOpenAI"):
        embeddings = get_embeddings()
        assert embeddings is not None


@pytest.mark.asyncio
async def test_get_vectordb(mock_embeddings, mock_chromadb):
    """Test getting Chroma vector store"""
    with (
        patch("src.app.core.vectorstore.get_embeddings", return_value=mock_embeddings),
        patch("chromadb.PersistentClient", return_value=mock_chromadb),
    ):
        vectordb = get_vectordb()
        assert vectordb is not None


@pytest.mark.asyncio
async def test_aembed_texts(mock_embeddings):
    """Test asynchronous text embedding"""
    texts = ["This is a test", "Another test text"]
    mock_embeddings.aembed_documents.return_value = [
        np.random.rand(1536) for _ in range(2)
    ]

    with patch("src.app.core.vectorstore.get_embeddings", return_value=mock_embeddings):
        embeddings = await aembed_texts(texts)
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536
