from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

from src.app.core.config import Settings
from src.app.services.chat_service import get_chat_llm

SUMMARY_PROMPT = """You are a helpful AI assistant tasked with summarizing a conversation.
Please provide a concise summary of the following conversation, focusing on the key points and decisions made.

Conversation:
{messages}

Summary:"""


def create_summary_chain(settings: Settings) -> RunnableSequence:
    """
    Create a chain for generating conversation summaries.

    Args:
        settings: Application settings.

    Returns:
        RunnableSequence: Chain for generating summaries.
    """
    prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT)
    llm = get_chat_llm()
    chain = prompt | llm
    return chain
