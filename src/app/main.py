from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.app.routers import chat, ingest, summary

app = FastAPI(title="RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)
app.include_router(ingest.router)
app.include_router(summary.router)


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring the application status.
    """
    return {"status": "healthy"}
