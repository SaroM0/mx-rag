from typing import List, Tuple

from pydantic import BaseModel, Field


class CostInfo(BaseModel):
    """Cost information for a chat interaction."""

    input_tokens: int = Field(description="Number of input tokens")
    output_tokens: int = Field(description="Number of output tokens")
    total_tokens: int = Field(description="Total number of tokens")
    input_cost: float = Field(description="Cost for input tokens")
    output_cost: float = Field(description="Cost for output tokens")
    total_cost: float = Field(description="Total cost")
    is_cached: bool = Field(
        default=False, description="Whether the response was cached"
    )


class SourceDocument(BaseModel):
    """Source document information."""

    id: str = Field(description="Document ID")
    content: str = Field(description="Document content")
    source: str = Field(description="Document source")
    metadata: dict = Field(default_factory=dict, description="Document metadata")


class ChatRequest(BaseModel):
    """Chat request model."""

    query: str = Field(description="User query")
    history: List[Tuple[str, str]] = Field(
        default_factory=list, description="Chat history"
    )


class ChatResponse(BaseModel):
    """Chat response model."""

    answer: str = Field(description="Generated answer")
    sources: List[SourceDocument] = Field(
        default_factory=list, description="Source documents"
    )
    processing_time: float = Field(description="Processing time in seconds")
    cost_info: CostInfo = Field(description="Cost information")


class RawChatResponse(BaseModel):
    """Raw chat response model without RAG."""

    answer: str = Field(description="Generated answer")
    processing_time: float = Field(description="Processing time in seconds")
    cost_info: CostInfo = Field(description="Cost information")
