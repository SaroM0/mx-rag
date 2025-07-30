from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.app.core.config import Settings
from src.app.main import app


@pytest.fixture
def test_settings():
    """Test settings"""
    return Settings(
        openai_api_key="test-key",
        chat_model="gpt-4",
        embedding_model="text-embedding-3-small",
        chat_top_k=3,
        chat_return_source_docs=True,
        data_dir="test_data",
        persist_directory="test_persist",
    )


@pytest.fixture
def mock_openai():
    """Fixture for mocked OpenAI client"""
    mock = AsyncMock()
    mock.embeddings.create.return_value = Mock(
        data=[{"embedding": np.random.rand(1536).tolist()}]
    )
    mock.chat.completions.create.return_value = Mock(
        model_dump=lambda: {
            "choices": [{"message": {"content": "Test response", "role": "assistant"}}],
            "usage": {"total_tokens": 100},
        }
    )
    return mock


@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB client"""
    mock = AsyncMock()
    mock._admin_client = Mock()
    mock._server = Mock()
    mock.bindings = Mock()
    mock.bindings.get_tenant.return_value = {"name": "default_tenant"}
    mock.bindings.get_database.return_value = {"name": "default_database"}
    mock.get_tenant = lambda name: {"name": name}
    mock.get_database = lambda name: {"name": name}

    # Mock collection operations
    mock.get_or_create_collection = Mock()
    mock.get_collection = Mock()
    mock.list_collections = Mock(return_value=[])
    mock.create_collection = Mock()
    mock.delete_collection = Mock()

    # Mock settings
    mock.settings = Mock(
        persist_directory="/tmp/chromadb",
        allow_reset=True,
        is_persistent=True,
        migrations="apply",
        migrations_hash_algorithm="md5",
    )

    # Mock similarity search
    mock.asimilarity_search = AsyncMock(
        return_value=[
            Mock(page_content="Sample content", metadata={"source": "test.pdf"})
        ]
    )

    # Mock database operations
    mock.start = Mock()
    mock.stop = Mock()
    mock.reset = Mock()
    mock.heartbeat = Mock()
    mock.persist = Mock()
    mock.get_version = Mock(return_value="0.4.0")

    return mock


@pytest.fixture
def mock_vectordb(mock_chromadb):
    """Mock vector database"""
    mock = AsyncMock()
    mock.asimilarity_search = AsyncMock(
        return_value=[
            Mock(page_content="Sample content", metadata={"source": "test.pdf"})
        ]
    )
    return mock


@pytest.fixture
def test_client(test_settings, mock_openai, mock_vectordb, mock_chromadb):
    """Test client"""
    with (
        patch("src.app.core.config.get_settings", return_value=test_settings),
        patch("openai.AsyncOpenAI", return_value=mock_openai),
        patch("src.app.core.vectorstore.get_vectordb", return_value=mock_vectordb),
        patch("chromadb.PersistentClient", return_value=mock_chromadb),
        patch("chromadb.api.client.Client", return_value=mock_chromadb),
        patch("chromadb.api.rust.RustBindingsAPI", return_value=mock_chromadb),
    ):
        client = TestClient(app)
        yield client
