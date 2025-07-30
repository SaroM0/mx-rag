import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.app.utils.history_utils import convert_history_to_messages


def test_convert_history_to_messages():
    """Test converting chat history to LangChain messages"""
    # Test with empty history
    assert convert_history_to_messages([]) == []

    # Test with single message pair
    history = [("Hello", "Hi there!")]
    messages = convert_history_to_messages(history)
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "Hello"
    assert messages[1].content == "Hi there!"

    # Test with multiple message pairs
    history = [
        ("What is RAG?", "RAG stands for Retrieval Augmented Generation"),
        ("Can you explain more?", "RAG combines search and language models"),
    ]
    messages = convert_history_to_messages(history)
    assert len(messages) == 4
    assert all(isinstance(m, (HumanMessage, AIMessage)) for m in messages)
    assert [m.content for m in messages] == [
        "What is RAG?",
        "RAG stands for Retrieval Augmented Generation",
        "Can you explain more?",
        "RAG combines search and language models",
    ]


def test_convert_history_to_messages_invalid_input():
    """Test converting invalid chat history"""
    # Test with None
    with pytest.raises(TypeError):
        convert_history_to_messages(None)

    # Test with invalid history format
    with pytest.raises(TypeError):
        convert_history_to_messages(["not a tuple"])

    # Test with invalid message types
    with pytest.raises(TypeError):
        convert_history_to_messages([(None, "Hi"), ("Hello", None)])
