from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def summary_request():
    """Fixture for summary request data"""
    return {
        "query": "",
        "history": [
            ("What is RAG?", "RAG stands for Retrieval Augmented Generation"),
            (
                "Can you explain more?",
                "RAG combines search and language models to provide accurate answers",
            ),
        ],
    }


def test_summary_endpoint_success(test_client: TestClient, summary_request):
    """Test successful summary generation"""
    mock_chain = AsyncMock()
    mock_chain.ainvoke.return_value = "A discussion about RAG and its functionality."

    with patch("langchain_core.runnables.RunnableSequence", return_value=mock_chain):
        response = test_client.post("/summary/", json=summary_request)
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert data["summary"] == "A discussion about RAG and its functionality."
        assert "cost_info" in data


def test_summary_endpoint_validation(test_client: TestClient):
    """Test summary endpoint input validation"""
    invalid_request = {
        "query": "",
        "history": [],  # Empty history should fail validation
    }

    response = test_client.post("/summary/", json=invalid_request)
    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_summary_endpoint_error_handling(
    test_client: TestClient, summary_request, mock_openai
):
    """Test summary endpoint error handling"""
    # Mock OpenAI to raise an error
    mock_openai.create.side_effect = Exception("LLM error")

    response = test_client.post("/summary/", json=summary_request)
    assert response.status_code == 500
    assert "error" in response.json()["detail"].lower()


def test_summary_endpoint_empty_history(test_client: TestClient):
    """Test summary endpoint with empty history"""
    request = {"query": "", "history": []}
    response = test_client.post("/summary/", json=request)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_summary_endpoint_invalid_history(test_client: TestClient):
    """Test summary endpoint with invalid history format"""
    request = {
        "query": "",
        "history": ["invalid", "format"],  # Should be tuples
    }
    response = test_client.post("/summary/", json=request)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_summary_endpoint_openai_error(test_client: TestClient, summary_request):
    """Test summary endpoint when OpenAI API fails"""
    mock_chain = AsyncMock()
    mock_chain.ainvoke.side_effect = Exception("OpenAI API error")

    with patch("langchain_core.runnables.RunnableSequence", return_value=mock_chain):
        response = test_client.post("/summary/", json=summary_request)
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "OpenAI API error" in data["detail"]
