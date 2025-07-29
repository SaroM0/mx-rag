from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.app.routers import chat, ingest

app = FastAPI(
    title="MX RAG API",
    description="RAG API for document question answering",
    version="0.1.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(chat.router)
app.include_router(ingest.router)


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring the application status.
    """
    return {"status": "healthy"}
