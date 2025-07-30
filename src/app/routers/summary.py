import time

from fastapi import APIRouter, HTTPException

from src.app.core.config import get_settings
from src.app.llm.summary_chain import create_summary_chain
from src.app.schemas.chat import ChatRequest
from src.app.schemas.summary import SummaryResponse
from src.app.services.chat_service import calculate_cost_info
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
        chain = create_summary_chain(settings)
        try:
            summary = await chain.ainvoke({"messages": messages})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

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
        raise HTTPException(
            status_code=500, detail=f"Error generating summary: {str(e)}"
        )
