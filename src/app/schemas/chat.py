from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field


class CostInfo(BaseModel):
    """
    Cost information for model usage.
    """

    input_tokens: int = Field(..., description="Number of input tokens processed")
    output_tokens: int = Field(..., description="Number of output tokens generated")
    total_tokens: int = Field(..., description="Total tokens used")
    input_cost: float = Field(..., description="Cost for input tokens in USD")
    output_cost: float = Field(..., description="Cost for output tokens in USD")
    total_cost: float = Field(..., description="Total cost in USD")


class ChatRequest(BaseModel):
    """
    Chat request schema.
    """

    query: str = Field(..., description="The user's question")
    history: List[Tuple[str, str]] = Field(
        default=[],
        description="List of (human, ai) message tuples representing chat history",
    )


class SourceDocument(BaseModel):
    """
    Source document schema.
    """

    id: str = Field(..., description="Unique identifier for the source")
    content: str = Field(..., description="Content of the source document")
    source: str = Field(..., description="Source file name")
    metadata: Dict[str, Any] = Field(
        default={},
        description="Additional metadata about the source",
    )


class ChatResponse(BaseModel):
    """
    Chat response schema.
    """

    answer: str = Field(..., description="The generated answer")
    sources: List[SourceDocument] = Field(
        ...,
        description="List of source documents used to generate the answer",
    )
    processing_time: float = Field(
        ...,
        description="Time taken to process the request in seconds",
    )
    cost_info: CostInfo = Field(
        ...,
        description="Cost information for this request",
    )


class RawChatResponse(BaseModel):
    """
    Raw chat response schema without RAG sources.
    """

    answer: str = Field(..., description="The generated answer")
    processing_time: float = Field(
        ...,
        description="Time taken to process the request in seconds",
    )
    cost_info: CostInfo = Field(
        ...,
        description="Cost information for this request",
    )
