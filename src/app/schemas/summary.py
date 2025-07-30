from pydantic import BaseModel, Field

from src.app.schemas.chat import CostInfo


class SummaryResponse(BaseModel):
    """Summary response model."""

    summary: str = Field(description="Generated summary")
    processing_time: float = Field(description="Processing time in seconds")
    cost_info: CostInfo = Field(description="Cost information")
