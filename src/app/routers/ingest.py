from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from src.ingestion.ingest import ingest_directory

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/")
async def ingest_all() -> Dict[str, Any]:
    """
    Ingest all PDF files from the configured directory.

    Returns:
        Dict[str, Any]: Results of ingestion process for each PDF.
    """
    try:
        results = ingest_directory()
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during ingestion: {str(e)}")
