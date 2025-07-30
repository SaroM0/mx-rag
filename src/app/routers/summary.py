import time
import traceback

from fastapi import APIRouter, HTTPException

from src.app.core.config import get_settings
from src.app.schemas.chat import ChatRequest
from src.app.schemas.summary import SummaryResponse
from src.app.services.chat_service import calculate_cost_info, get_chat_llm
from src.app.utils.history_utils import convert_history_to_messages

router = APIRouter(tags=["summary"])


@router.post("/summary", response_model=SummaryResponse)
async def generate_summary(request: ChatRequest):
    """Generate a summary of the conversation"""
    if not request.history:
        raise HTTPException(
            status_code=422, detail="History cannot be empty for summary generation"
        )

    try:
        start_time = time.time()
        settings = get_settings()

        # Convert history to messages
        messages = convert_history_to_messages(request.history)
        if not messages:
            raise HTTPException(status_code=422, detail="Invalid chat history format")

        # Create and run the chain
        print("\nget_chat_llm:", get_chat_llm)
        llm = get_chat_llm()
        try:
            print("\nRunning chain with messages:", messages)
            print("LLM type:", type(llm))
            print("LLM dir:", dir(llm))
            print("LLM ainvoke type:", type(llm.ainvoke))
            print("Messages type:", type(messages))
            print("Messages content:", messages)
            response = await llm.ainvoke(messages)
            print("Chain response:", response)
            print("Chain response type:", type(response))
            print("Chain response dir:", dir(response))
            summary = response.content
            print("Summary:", summary)
        except Exception as e:
            print("Chain error:", str(e))
            print("Chain error type:", type(e))
            print("Chain error dir:", dir(e))
            print("Chain error traceback:")
            traceback.print_exc()
            raise HTTPException(
                status_code=500, detail=f"Error generating summary: {str(e)}"
            )

        # Calculate processing time and costs
        processing_time = time.time() - start_time
        cost_info = calculate_cost_info(
            input_text=str(messages),
            output_text=summary,
        )

        return SummaryResponse(
            summary=summary, processing_time=processing_time, cost_info=cost_info
        )

    except HTTPException:
        raise
    except Exception as e:
        print("Router error:", str(e))
        print("Router error type:", type(e))
        print("Router error dir:", dir(e))
        print("Router error traceback:")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Error generating summary: {str(e)}"
        )
