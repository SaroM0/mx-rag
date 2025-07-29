from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field


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
