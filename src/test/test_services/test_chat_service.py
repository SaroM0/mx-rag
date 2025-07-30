from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.app.schemas.chat import ChatRequest, ChatResponse, RawChatResponse
from src.app.services.chat_service import (
    calculate_cost_info,
    calculate_tokens,
    convert_history_to_messages,
    process_chat,
    process_raw_chat,
)


def test_calculate_tokens():
    """Test token calculation"""
    text = "Hello, this is a test message"
    tokens = calculate_tokens(text)
    assert tokens > 0
    assert isinstance(tokens, int)


def test_calculate_cost_info():
    """Test cost calculation"""
    input_text = "What is RAG?"
    output_text = "RAG is a technique for enhancing LLM responses."

    cost_info = calculate_cost_info(input_text, output_text)

    assert cost_info.input_tokens > 0
    assert cost_info.output_tokens > 0
    assert cost_info.total_cost > 0


def test_convert_history_to_messages():
    """Test chat history conversion"""
    history = [
        ("What is RAG?", "RAG stands for Retrieval Augmented Generation"),
        ("How does it work?", "RAG combines search and language models"),
    ]

    messages = convert_history_to_messages(history)

    assert len(messages) == 4  # Each tuple converts to 2 messages
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "What is RAG?"
    assert messages[1].content == "RAG stands for Retrieval Augmented Generation"


@pytest.mark.asyncio
async def test_process_chat(mock_openai, mock_vectordb):
    """Test chat processing"""
    request = ChatRequest(
        query="What is RAG?",
        history=[("What is RAG?", "RAG stands for Retrieval Augmented Generation")],
    )

    # Mock LLM response
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = Mock(content="RAG is a technique...")

    with patch("src.app.services.chat_service.get_chat_llm", return_value=mock_llm):
        response = await process_chat(request)

        assert isinstance(response, ChatResponse)
        assert response.answer == "RAG is a technique..."
        assert response.cost_info is not None
        assert response.sources is not None


@pytest.mark.asyncio
async def test_process_raw_chat(mock_openai):
    """Test raw chat processing"""
    request = ChatRequest(
        query="What is RAG?",
        history=[("What is RAG?", "RAG stands for Retrieval Augmented Generation")],
    )

    # Mock LLM response
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = Mock(content="RAG is a technique...")

    with patch("src.app.services.chat_service.get_chat_llm", return_value=mock_llm):
        response = await process_raw_chat(request)

        assert isinstance(response, RawChatResponse)
        assert response.answer == "RAG is a technique..."
        assert response.cost_info is not None
