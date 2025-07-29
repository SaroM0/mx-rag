import time

from fastapi import APIRouter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from src.app.core.config import get_settings
from src.app.schemas.chat import ChatRequest, CostInfo
from src.app.schemas.summary import SummaryResponse
from src.app.services.chat_service import get_chat_llm

router = APIRouter(prefix="/summary", tags=["summary"])

SUMMARY_PROMPT = """
Given the following chat history between a user and an AI assistant:
{chat_history}

Generate a brief and concise summary (2-3 sentences) highlighting the key points and topics discussed.
Focus on the main questions asked and solutions provided.
"""


@router.post("/", response_model=SummaryResponse)
async def summarize_conversation(request: ChatRequest) -> SummaryResponse:
    """
    Generate a concise summary of the conversation history.

    Args:
        request: ChatRequest containing the conversation history

    Returns:
        SummaryResponse containing the summary, processing time and cost information
    """
    start_time = time.time()
    settings = get_settings()

    # Initialize the summarization chain
    prompt = PromptTemplate.from_template(SUMMARY_PROMPT)
    summary_chain = LLMChain(llm=get_chat_llm(), prompt=prompt)

    # Format chat history into a readable string
    chat_history = "\n".join(
        f"{'User' if i % 2 == 0 else 'Assistant'}: {msg}"
        for i, msg in enumerate(request.history)
    )

    # Generate summary
    summary = await summary_chain.arun(chat_history=chat_history)

    # Calculate processing time
    processing_time = time.time() - start_time

    # Calculate token usage and cost (simplified version)
    # In a real implementation, you would get this from the LLM response
    estimated_tokens = len(summary.split()) * 1.3  # rough estimate
    input_tokens = int(estimated_tokens * 0.7)
    output_tokens = int(estimated_tokens * 0.3)

    cost_info = CostInfo(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_cost=input_tokens * settings.model_input_cost_per_token,
        output_cost=output_tokens * settings.model_output_cost_per_token,
        total_cost=(input_tokens * settings.model_input_cost_per_token)
        + (output_tokens * settings.model_output_cost_per_token),
    )

    return SummaryResponse(
        summary=summary.strip(), processing_time=processing_time, cost_info=cost_info
    )
