from typing import List, Tuple

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from langchain_community.chat_models import ChatOpenAI

from src.app.core.config import get_settings
from src.app.core.vectorstore import get_vectordb
from src.app.schemas.chat import ChatRequest, ChatResponse, SourceDocument


def convert_history_to_messages(history: List[Tuple[str, str]]) -> List[BaseMessage]:
    """
    Convert chat history from tuples to LangChain message format.

    Args:
        history: List of (human, ai) message tuples.

    Returns:
        List of LangChain messages alternating between human and AI.
    """
    messages = []
    for human_msg, ai_msg in history:
        messages.extend([HumanMessage(content=human_msg), AIMessage(content=ai_msg)])
    return messages


def get_chat_llm() -> ChatOpenAI:
    """
    Create a configured ChatOpenAI instance.

    Returns:
        ChatOpenAI: Configured language model instance.

    Raises:
        ValueError: If OpenAI API key is not set.
    """
    settings = get_settings()

    if not settings.openai_api_key:
        raise ValueError("OpenAI API key not found in settings")

    return ChatOpenAI(
        model_name=settings.chat_model_name,
        openai_api_key=settings.openai_api_key,
        temperature=settings.chat_temperature,
        max_tokens=settings.chat_max_tokens,
        streaming=settings.chat_streaming,
    )


def create_chat_prompt() -> ChatPromptTemplate:
    """
    Create the chat prompt template with system message and placeholders.
    """
    system_template = (
        "You are a helpful AI assistant. Use the provided context to answer "
        "questions accurately and concisely. If you're not sure about something, "
        "say so rather than making assumptions.\n\n"
        "Context: {context}\n\n"
        "Current conversation:\n{chat_history}\n"
        "Human: {question}\n"
        "Assistant: "
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
        ]
    )


def process_chat(payload: ChatRequest) -> ChatResponse:
    """
    Process a chat request using RAG (Retrieval Augmented Generation).

    This function:
    1. Retrieves relevant documents from the vector store
    2. Uses them as context for the language model
    3. Generates a response based on the context and chat history

    Args:
        payload: Chat request containing the query and chat history.

    Returns:
        ChatResponse: Generated answer and source document IDs.

    Raises:
        ValueError: If the chat model or vector store configuration is invalid.
        Exception: For other processing errors.
    """
    try:
        settings = get_settings()
        vectordb = get_vectordb()

        # Configure retriever
        retriever = vectordb.as_retriever(search_kwargs={"k": settings.chat_top_k})

        # Get language model
        llm = get_chat_llm()

        # Format chat history
        chat_history = "\n".join(
            [f"Human: {h}\nAssistant: {a}" for h, a in (payload.history or [])]
        )

        # Create chain with custom prompt
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=settings.chat_return_source_docs,
            combine_docs_chain_kwargs={"prompt": create_chat_prompt()},
            rephrase_question=False,
            return_generated_question=False,
        )

        # Process the query with history
        result = chain.invoke(
            {
                "question": payload.query,
                "chat_history": chat_history,
            }
        )

        # Extract answer and sources
        answer = result["answer"].strip()
        sources = [
            SourceDocument(
                id=doc.metadata.get("chunk_id", "unknown"),
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                metadata={
                    k: v
                    for k, v in doc.metadata.items()
                    if k not in ["chunk_id", "source"]
                },
            )
            for doc in result.get("source_documents", [])
        ]

        return ChatResponse(answer=answer, sources=sources)

    except ValueError as e:
        # Handle configuration errors
        raise ValueError(f"Configuration error: {str(e)}")
    except Exception as e:
        # Handle other errors
        raise Exception(f"Error processing chat request: {str(e)}")


async def aprocess_chat(payload: ChatRequest) -> ChatResponse:
    """
    Asynchronous version of process_chat.
    Currently a wrapper as LangChain's ConversationalRetrievalChain
    doesn't support async yet. When it does, this will be updated.

    Args:
        payload: Chat request containing the query and chat history.

    Returns:
        ChatResponse: Generated answer and source document IDs.
    """
    return process_chat(payload)
