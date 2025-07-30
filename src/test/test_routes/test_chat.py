import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def chat_request():
    """Fixture for chat request data"""
    return {
        "query": "What is RAG?",
        "history": [("What is RAG?", "RAG stands for Retrieval Augmented Generation")],
    }


def test_chat_endpoint_success(test_client: TestClient, chat_request):
    """Test successful chat endpoint request"""
    response = test_client.post("/chat/", json=chat_request)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "cost_info" in data
    assert "sources" in data


def test_chat_endpoint_invalid_request(test_client: TestClient):
    """Test chat endpoint with invalid request"""
    invalid_request = {
        "query": "",  # Empty query
        "history": [],
    }
    response = test_client.post("/chat/", json=invalid_request)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_chat_endpoint_vectordb_error(
    test_client: TestClient, chat_request, mock_vectordb
):
    """Test chat endpoint when vector store fails"""
    mock_vectordb.asimilarity_search.side_effect = Exception("Vector store error")
    response = test_client.post("/chat/", json=chat_request)
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "Vector store error" in data["detail"]


def test_raw_chat_endpoint_success(test_client: TestClient, chat_request):
    """Test successful raw chat endpoint request"""
    response = test_client.post("/chat/raw", json=chat_request)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "cost_info" in data


def test_raw_chat_endpoint_openai_error(
    test_client: TestClient, chat_request, mock_openai
):
    """Test raw chat endpoint when OpenAI API fails"""
    mock_openai.chat.completions.create.side_effect = Exception("OpenAI API error")
    response = test_client.post("/chat/raw", json=chat_request)
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "OpenAI API error" in data["detail"]
