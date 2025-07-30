import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.app.core.config import Settings


def test_settings_default_values():
    """Test Settings with default values"""
    settings = Settings(openai_api_key="test-key")  # Required field

    assert settings.openai_model == "text-embedding-3-small"
    assert settings.openai_dimensions == 1536
    assert settings.openai_chunk_size == 1000
    assert settings.openai_max_retries == 3
    assert settings.openai_timeout == 60.0
    assert settings.openai_retry_min_seconds == 4
    assert settings.openai_retry_max_seconds == 20
    assert settings.model_input_cost_per_token == 0.15 / 1000
    assert settings.model_output_cost_per_token == 0.60 / 1000
    assert settings.pdf_directory == "src/pdfs"
    assert settings.chunks_directory == "src/data/chunks"
    assert settings.vectorstore_persist_directory == "src/data/chroma"


def test_settings_from_env():
    """Test Settings loaded from environment variables"""
    test_values = {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_MODEL": "text-embedding-3-large",
        "OPENAI_DIMENSIONS": "2048",
        "OPENAI_CHUNK_SIZE": "2000",
        "OPENAI_MAX_RETRIES": "5",
        "OPENAI_TIMEOUT": "120.0",
        "OPENAI_RETRY_MIN_SECONDS": "8",
        "OPENAI_RETRY_MAX_SECONDS": "40",
        "MODEL_INPUT_COST_PER_TOKEN": "0.0002",
        "MODEL_OUTPUT_COST_PER_TOKEN": "0.0004",
        "PDF_DIRECTORY": "/test/pdfs",
        "CHUNKS_DIRECTORY": "/test/chunks",
        "VECTORSTORE_PERSIST_DIRECTORY": "/test/chroma",
    }

    with patch.dict(os.environ, test_values, clear=True):
        settings = Settings()

        assert settings.openai_api_key == "test-key"
        assert settings.openai_model == "text-embedding-3-large"
        assert settings.openai_dimensions == 2048
        assert settings.openai_chunk_size == 2000
        assert settings.openai_max_retries == 5
        assert settings.openai_timeout == 120.0
        assert settings.openai_retry_min_seconds == 8
        assert settings.openai_retry_max_seconds == 40
        assert settings.model_input_cost_per_token == 0.0002
        assert settings.model_output_cost_per_token == 0.0004
        assert settings.pdf_directory == "/test/pdfs"
        assert settings.chunks_directory == "/test/chunks"
        assert settings.vectorstore_persist_directory == "/test/chroma"


def test_settings_validation():
    """Test Settings validation"""
    # Test empty API key
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            openai_api_key="",
            chat_model="gpt-4",
            embedding_model="text-embedding-3-small",
        )
    assert "openai_api_key" in str(exc_info.value)

    # Test invalid model name
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            openai_api_key="test-key",
            chat_model="invalid-model",
            embedding_model="text-embedding-3-small",
        )
    assert "chat_model" in str(exc_info.value)

    # Test invalid top_k value
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            openai_api_key="test-key",
            chat_model="gpt-4",
            embedding_model="text-embedding-3-small",
            chat_top_k=-1,
        )
    assert "chat_top_k" in str(exc_info.value)
