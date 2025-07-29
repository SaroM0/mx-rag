from fastapi import APIRouter, HTTPException

from src.app.schemas.chat import ChatRequest, ChatResponse
from src.app.services.chat_service import process_chat

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a chat request with RAG (Retrieval Augmented Generation).

    Args:
        request: Chat request containing query and history.

    Returns:
        ChatResponse: Generated answer and source document IDs.
    """
    try:
        return process_chat(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
