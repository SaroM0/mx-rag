from unittest.mock import patch

from fastapi.testclient import TestClient


def test_ingest_endpoint_success(test_client: TestClient):
    """Test successful ingestion of PDF files"""
    mock_results = [
        {
            "status": "success",
            "pdf_path": "src/pdfs/doc1.pdf",
            "chunks_processed": 10,
            "chunks_saved": True,
        },
        {
            "status": "success",
            "pdf_path": "src/pdfs/doc2.pdf",
            "chunks_processed": 15,
            "chunks_saved": True,
        },
    ]

    with patch("src.app.routers.ingest.ingest_directory", return_value=mock_results):
        print("\nMock ingest results:", mock_results)
        response = test_client.post("/ingest/")
        print("Ingest response:", response.json())
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "results" in data
        assert len(data["results"]) == 2
        assert all(doc["status"] == "success" for doc in data["results"])


def test_ingest_endpoint_no_pdfs(test_client: TestClient):
    """Test ingest endpoint when no PDFs are found"""
    mock_results = []

    with patch("src.app.routers.ingest.ingest_directory", return_value=mock_results):
        print("\nMock ingest results (empty):", mock_results)
        response = test_client.post("/ingest/")
        print("Ingest response (no PDFs):", response.json())
        assert response.status_code == 404
        data = response.json()
        assert data["status"] == "error"
        assert "No PDF files found" in data["detail"]


def test_ingest_endpoint_processing_error(test_client: TestClient):
    """Test ingest endpoint when PDF processing fails"""
    mock_results = [
        {
            "status": "error",
            "pdf_path": "src/pdfs/doc1.pdf",
            "error": "Failed to process PDF",
        },
        {
            "status": "success",
            "pdf_path": "src/pdfs/doc2.pdf",
            "chunks_processed": 15,
            "chunks_saved": True,
        },
    ]

    with patch("src.app.routers.ingest.ingest_directory", return_value=mock_results):
        print("\nMock ingest results (partial error):", mock_results)
        response = test_client.post("/ingest/")
        print("Ingest response (partial error):", response.json())
        assert response.status_code == 207  # Partial success
        data = response.json()
        assert data["status"] == "partial"
        assert len(data["results"]) == 2
        assert any(doc["status"] == "error" for doc in data["results"])
        assert any(doc["status"] == "success" for doc in data["results"])


def test_ingest_endpoint_system_error(test_client: TestClient):
    """Test ingest endpoint when system error occurs"""
    error = Exception("System error")
    with patch("src.app.routers.ingest.ingest_directory", side_effect=error):
        print("\nMock ingest error:", str(error))
        response = test_client.post("/ingest/")
        print("Ingest response (system error):", response.json())
        assert response.status_code == 500
        data = response.json()
        assert data["status"] == "error"
        assert "Error during ingestion" in data["detail"]
