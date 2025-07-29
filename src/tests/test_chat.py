from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain.schema import Document

from src.app.main import app


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_vectordb():
    """Mock the vector store."""
    with patch("src.app.core.vectorstore.get_vectordb") as mock:
        vectordb = MagicMock()
        vectordb.as_retriever.return_value = MagicMock()
        mock.return_value = vectordb
        yield mock


@pytest.fixture
def mock_chat_llm():
    """Mock the chat language model."""
    with patch("src.app.services.chat_service.get_chat_llm") as mock:
        llm = MagicMock()
        mock.return_value = llm
        yield mock


def test_chat_endpoint_success(test_client, mock_vectordb, mock_chat_llm):
    """Test successful chat request."""
    # Mock data
    test_query = "What is RAG?"
    test_history = [("Hello", "Hi there!")]
    expected_answer = "RAG stands for Retrieval Augmented Generation."
    test_source_ids = ["chunk_1", "chunk_2"]

    # Mock chain response
    mock_chain = MagicMock()
    mock_chain.return_value = {
        "answer": expected_answer,
        "source_documents": [
            Document(page_content="", metadata={"chunk_id": id_})
            for id_ in test_source_ids
        ],
    }

    with patch(
        "src.app.services.chat_service.ConversationalRetrievalChain.from_llm",
        return_value=mock_chain,
    ):
        response = test_client.post(
            "/chat/",
            json={"query": test_query, "history": test_history},
        )

    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == expected_answer
    assert result["source_ids"] == test_source_ids


def test_chat_endpoint_validation(test_client):
    """Test input validation."""
    # Test missing query
    response = test_client.post("/chat/", json={"history": []})
    assert response.status_code == 422

    # Test invalid history format
    response = test_client.post(
        "/chat/",
        json={"query": "test", "history": ["invalid"]},
    )
    assert response.status_code == 422


def test_chat_endpoint_error_handling(test_client, mock_vectordb, mock_chat_llm):
    """Test error handling."""
    # Mock error in chat service
    mock_chain = MagicMock()
    mock_chain.side_effect = ValueError("Test error")

    with patch(
        "src.app.services.chat_service.ConversationalRetrievalChain.from_llm",
        return_value=mock_chain,
    ):
        response = test_client.post(
            "/chat/",
            json={"query": "test", "history": []},
        )

    assert response.status_code == 400
    assert "error" in response.json()["detail"].lower()
