from unittest.mock import patch

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
    mock_response = {
        "answer": "RAG is a technique...",
        "cost_info": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "input_cost": 0.0002,
            "output_cost": 0.0003,
            "total_cost": 0.0005,
            "is_cached": False,
        },
        "source_documents": [
            {
                "id": "test.pdf",
                "content": "RAG is a technique...",
                "source": "test.pdf",
                "metadata": {"source": "test.pdf"},
            }
        ],
        "processing_time": 1.0,
    }

    with patch("src.app.routers.chat.process_chat", return_value=mock_response):
        print("\nChat request:", chat_request)
        response = test_client.post("/chat/", json=chat_request)
        print("Chat response:", response.json())
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
    print("\nInvalid chat request:", invalid_request)
    print("Chat response (invalid):", response.json())
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_chat_endpoint_vectordb_error(
    test_client: TestClient, chat_request, mock_vectordb
):
    """Test chat endpoint when vector store fails"""
    error = Exception("Vector store error")
    mock_vectordb.asimilarity_search.side_effect = error
    response = test_client.post("/chat/", json=chat_request)
    print("\nVector store error:", str(error))
    print("Chat response (vectordb error):", response.json())
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "Error processing chat request" in data["detail"]


def test_raw_chat_endpoint_success(test_client: TestClient, chat_request):
    """Test successful raw chat endpoint request"""
    mock_response = {
        "answer": "RAG is a technique...",
        "cost_info": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "input_cost": 0.0002,
            "output_cost": 0.0003,
            "total_cost": 0.0005,
            "is_cached": False,
        },
        "processing_time": 1.0,
    }

    with patch("src.app.routers.chat.process_raw_chat", return_value=mock_response):
        print("\nRaw chat request:", chat_request)
        response = test_client.post("/chat/raw", json=chat_request)
        print("Raw chat response:", response.json())
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "cost_info" in data


def test_raw_chat_endpoint_openai_error(
    test_client: TestClient, chat_request, mock_openai
):
    """Test raw chat endpoint when OpenAI API fails"""
    error = Exception("OpenAI API error")
    mock_openai.chat.completions.create.side_effect = error
    response = test_client.post("/chat/raw", json=chat_request)
    print("\nOpenAI API error:", str(error))
    print("Raw chat response (OpenAI error):", response.json())
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "Error processing chat request" in data["detail"]
