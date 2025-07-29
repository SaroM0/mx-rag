from fastapi import APIRouter, HTTPException

from src.app.schemas.chat import ChatRequest, ChatResponse, RawChatResponse
from src.app.services.chat_service import process_chat, process_raw_chat

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


@router.post("/raw", response_model=RawChatResponse)
async def raw_chat(request: ChatRequest) -> RawChatResponse:
    """
    Process a chat request using only the language model without RAG.
    This endpoint provides a baseline for comparison with the RAG-enhanced responses.

    Args:
        request: Chat request containing query and history.

    Returns:
        RawChatResponse: Generated answer without source documents.
    """
    try:
        return process_raw_chat(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
