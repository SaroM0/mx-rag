from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from src.ingestion.ingest import ingest_directory

router = APIRouter(tags=["ingest"])


@router.post("/ingest")
async def ingest_pdfs():
    """Ingest PDF files from the data directory"""
    try:
        results = ingest_directory()

        if not results:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "detail": "No PDF files found in the data directory",
                },
            )

        # Check if any files were processed successfully
        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = sum(1 for r in results if r["status"] == "error")

        if success_count == 0 and error_count > 0:
            # All files failed
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "detail": "Failed to process all PDF files",
                    "results": results,
                },
            )
        elif error_count > 0:
            # Partial success
            return JSONResponse(
                status_code=207,
                content={
                    "status": "partial",
                    "message": "Some files failed to process",
                    "results": results,
                },
            )
        else:
            # All successful
            return {
                "status": "success",
                "message": "All files processed successfully",
                "results": results,
            }

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": f"Error during ingestion: {str(e)}"},
        )
