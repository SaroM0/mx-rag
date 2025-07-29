from pydantic import BaseModel, Field

from src.app.schemas.chat import CostInfo


class SummaryResponse(BaseModel):
    """
    Response schema for conversation summary.
    """
    summary: str = Field(..., description="Brief summary of the conversation")
    processing_time: float = Field(
        ...,
        description="Time taken to process the request in seconds",
    )
    cost_info: CostInfo = Field(
        ...,
        description="Cost information for this request",
    )