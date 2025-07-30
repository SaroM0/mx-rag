from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def summary_request():
    """Fixture for summary request data"""
    return {
        "history": [
            ("What is RAG?", "RAG stands for Retrieval Augmented Generation"),
            (
                "Can you explain more?",
                "RAG combines search and language models to provide accurate answers",
            ),
        ],
        "query": "",
    }


def test_summary_endpoint_success(test_client: TestClient, summary_request):
    """Test successful summary generation"""
    mock_chain = Mock()
    mock_response = Mock()
    mock_response.content = "A discussion about RAG and its functionality."
    mock_chain.ainvoke = AsyncMock()
    mock_chain.ainvoke.return_value = mock_response

    with patch("src.app.routers.summary.get_chat_llm", return_value=mock_chain):
        print("\nMock chain type:", type(mock_chain))
        print("Mock chain dir:", dir(mock_chain))
        print("Mock chain ainvoke type:", type(mock_chain.ainvoke))
        print("Mock chain ainvoke dir:", dir(mock_chain.ainvoke))
        print("Mock chain ainvoke return value:", mock_chain.ainvoke.return_value)
        print(
            "Mock chain ainvoke return value type:",
            type(mock_chain.ainvoke.return_value),
        )
        print(
            "Mock chain ainvoke return value content:",
            mock_chain.ainvoke.return_value.content,
        )
        response = test_client.post("/summary/", json=summary_request)
        print("Summary response:", response.json())
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert data["summary"] == "A discussion about RAG and its functionality."


def test_summary_endpoint_empty_history(test_client: TestClient):
    """Test summary endpoint with empty history"""
    request = {"history": [], "query": ""}
    response = test_client.post("/summary/", json=request)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert "History cannot be empty" in data["detail"]


def test_summary_endpoint_invalid_history(test_client: TestClient):
    """Test summary endpoint with invalid history format"""
    request = {"history": [{"invalid": "format"}], "query": ""}
    response = test_client.post("/summary/", json=request)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_summary_endpoint_missing_history(test_client: TestClient):
    """Test summary endpoint with missing history"""
    request = {"query": ""}
    response = test_client.post("/summary/", json=request)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert "history cannot be empty" in str(data["detail"]).lower()


def test_summary_endpoint_openai_error(test_client: TestClient, summary_request):
    """Test summary endpoint when OpenAI API fails"""
    mock_chain = Mock()
    mock_chain.ainvoke = AsyncMock()
    mock_chain.ainvoke.side_effect = Exception("Error generating summary")

    with patch("src.app.routers.summary.get_chat_llm", return_value=mock_chain):
        print("\nMock chain type:", type(mock_chain))
        print("Mock chain dir:", dir(mock_chain))
        print("Mock chain ainvoke type:", type(mock_chain.ainvoke))
        print("Mock chain ainvoke dir:", dir(mock_chain.ainvoke))
        print("Mock chain ainvoke side effect:", mock_chain.ainvoke.side_effect)
        print(
            "Mock chain ainvoke side effect type:", type(mock_chain.ainvoke.side_effect)
        )
        response = test_client.post("/summary/", json=summary_request)
        print("Summary error response:", response.json())
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Error generating summary" in data["detail"]
