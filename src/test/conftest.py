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
        chat_model_name="gpt-4",
        openai_model="text-embedding-3-small",
        chat_top_k=3,
        chat_return_source_docs=True,
        pdf_directory="test_data",
        vectorstore_persist_directory="test_persist",
        chat_temperature=0.0,
        chat_max_tokens=2000,
        chat_streaming=False,
        openai_dimensions=1536,
        openai_chunk_size=1000,
        openai_max_retries=3,
        openai_timeout=60.0,
        openai_retry_min_seconds=4,
        openai_retry_max_seconds=20,
        model_input_cost_per_token=0.15 / 1000,
        model_output_cost_per_token=0.60 / 1000,
        model_cached_input_cost_per_token=0.075 / 1000,
        show_embedding_progress=True,
        enable_tiktoken=True,
        chunks_directory="test_chunks",
        chunk_size=512,
        chunk_overlap=50,
        save_chunks=True,
    )


@pytest.fixture
def mock_openai():
    """Fixture for mocked OpenAI client"""
    mock = AsyncMock()
    mock.embeddings.create.return_value = Mock(
        data=[{"embedding": np.random.rand(1536).tolist()}]
    )

    mock.chat.completions.create.return_value = AsyncMock(
        choices=[{"message": {"content": "Test response"}}]
    )
    return mock


class MockMetadata(dict):
    """Mock metadata dictionary"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get(self, key, default=None):
        return self[key] if key in self else default


class MockDocument:
    """Mock document for similarity search results"""

    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __getitem__(self, key):
        return getattr(self, key)

    def __getattr__(self, key):
        try:
            return self.__dict__[key]
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )


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
            MockDocument(
                page_content="Sample content",
                metadata=MockMetadata(source="test.pdf"),
            )
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
            MockDocument(
                page_content="Sample content",
                metadata=MockMetadata(source="test.pdf"),
            )
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
        patch("chromadb.Client", return_value=mock_chromadb),
    ):
        client = TestClient(app)
        yield client
