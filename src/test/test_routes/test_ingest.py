from unittest.mock import patch

from fastapi.testclient import TestClient


def test_ingest_endpoint_success(test_client: TestClient):
    """Test successful ingestion of PDF files"""
    mock_results = [
        {"file": "doc1.pdf", "chunks": 10, "status": "success"},
        {"file": "doc2.pdf", "chunks": 15, "status": "success"},
    ]

    with patch("src.ingestion.ingest.ingest_directory", return_value=mock_results):
        response = test_client.post("/ingest/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "results" in data
        assert len(data["results"]) == 2
        assert all(doc["status"] == "success" for doc in data["results"])


def test_ingest_endpoint_no_pdfs(test_client: TestClient):
    """Test ingest endpoint when no PDFs are found"""
    mock_results = []

    with patch("src.ingestion.ingest.ingest_directory", return_value=mock_results):
        response = test_client.post("/ingest/")
        assert response.status_code == 404
        data = response.json()
        assert data["status"] == "error"
        assert "No PDF files found" in data["detail"]


def test_ingest_endpoint_processing_error(test_client: TestClient):
    """Test ingest endpoint when PDF processing fails"""
    mock_results = [
        {
            "file": "doc1.pdf",
            "chunks": 0,
            "status": "error",
            "error": "Failed to process PDF",
        },
        {"file": "doc2.pdf", "chunks": 15, "status": "success"},
    ]

    with patch("src.ingestion.ingest.ingest_directory", return_value=mock_results):
        response = test_client.post("/ingest/")
        assert response.status_code == 207  # Partial success
        data = response.json()
        assert data["status"] == "partial"
        assert len(data["results"]) == 2
        assert any(doc["status"] == "error" for doc in data["results"])
        assert any(doc["status"] == "success" for doc in data["results"])


def test_ingest_endpoint_system_error(test_client: TestClient):
    """Test ingest endpoint when system error occurs"""
    with patch(
        "src.ingestion.ingest.ingest_directory", side_effect=Exception("System error")
    ):
        response = test_client.post("/ingest/")
        assert response.status_code == 500
        data = response.json()
        assert data["status"] == "error"
        assert "System error" in data["detail"]
