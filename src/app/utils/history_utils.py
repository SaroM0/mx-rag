from typing import List, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def convert_history_to_messages(history: List[Tuple[str, str]]) -> List[BaseMessage]:
    """
    Convert chat history tuples to LangChain message objects.

    Args:
        history: List of (human_message, ai_message) tuples.

    Returns:
        List[BaseMessage]: List of LangChain messages.

    Raises:
        TypeError: If history is None or contains invalid message types.
    """
    if history is None:
        raise TypeError("History cannot be None")

    messages = []
    for i, pair in enumerate(history):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(
                f"History item {i} must be a tuple of (human_message, ai_message)"
            )

        human_msg, ai_msg = pair
        if not isinstance(human_msg, str) or not isinstance(ai_msg, str):
            raise TypeError(f"History item {i} contains non-string messages")

        messages.extend([HumanMessage(content=human_msg), AIMessage(content=ai_msg)])
    return messages
