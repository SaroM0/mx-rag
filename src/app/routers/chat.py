from fastapi import APIRouter, HTTPException

from src.app.schemas.chat import ChatRequest, ChatResponse, RawChatResponse
from src.app.services.chat_service import process_chat, process_raw_chat

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with RAG"""
    if not request.query:
        raise HTTPException(status_code=422, detail="Query cannot be empty")

    try:
        response = await process_chat(request)
        return response
    except Exception as e:
        if "Vector store error" in str(e):
            raise HTTPException(status_code=500, detail=f"Vector store error: {str(e)}")
        elif "OpenAI API error" in str(e):
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
        else:
            raise HTTPException(
                status_code=500, detail=f"Error processing chat request: {str(e)}"
            )


@router.post("/chat/raw", response_model=RawChatResponse)
async def raw_chat(request: ChatRequest):
    """Raw chat endpoint without RAG"""
    if not request.query:
        raise HTTPException(status_code=422, detail="Query cannot be empty")

    try:
        response = await process_raw_chat(request)
        return response
    except Exception as e:
        if "OpenAI API error" in str(e):
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
        else:
            raise HTTPException(
                status_code=500, detail=f"Error processing chat request: {str(e)}"
            )
