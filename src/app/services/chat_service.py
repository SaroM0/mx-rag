import time
from typing import List, Tuple

import tiktoken
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from langchain_community.chat_models import ChatOpenAI

from src.app.core.config import get_settings
from src.app.core.vectorstore import get_vectordb
from src.app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    CostInfo,
    RawChatResponse,
    SourceDocument,
)


def calculate_tokens(text: str) -> int:
    """
    Calculate the number of tokens in a text string.

    Args:
        text: The text to calculate tokens for.

    Returns:
        int: Number of tokens.
    """
    encoding = tiktoken.get_encoding(
        "cl100k_base"
    )  # Using OpenAI's recommended encoding
    return len(encoding.encode(text))


def calculate_cost_info(
    input_text: str, output_text: str, is_cached: bool = False
) -> CostInfo:
    """
    Calculate cost information for model usage.

    Args:
        input_text: The input text sent to the model
        output_text: The output text received from the model
        is_cached: Whether the input was cached (for RAG queries)

    Returns:
        CostInfo: Cost information including token counts and prices
    """
    settings = get_settings()

    input_tokens = calculate_tokens(input_text)
    output_tokens = calculate_tokens(output_text)
    total_tokens = input_tokens + output_tokens

    # Calculate costs based on token counts
    input_cost_per_token = (
        settings.model_cached_input_cost_per_token
        if is_cached
        else settings.model_input_cost_per_token
    )

    input_cost = input_tokens * input_cost_per_token
    output_cost = output_tokens * settings.model_output_cost_per_token
    total_cost = input_cost + output_cost

    return CostInfo(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=total_cost,
    )


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
        ChatResponse: Generated answer, source document IDs, and processing time.

    Raises:
        ValueError: If the chat model or vector store configuration is invalid.
        Exception: For other processing errors.
    """
    start_time = time.time()
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

        # Calculate costs - for RAG, we consider the input as cached
        input_text = f"{chat_history}\nHuman: {payload.query}"
        cost_info = calculate_cost_info(
            input_text=input_text,
            output_text=answer,
            is_cached=True,  # RAG queries use cached input pricing
        )

        processing_time = time.time() - start_time
        return ChatResponse(
            answer=answer,
            sources=sources,
            processing_time=processing_time,
            cost_info=cost_info,
        )

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


def process_raw_chat(payload: ChatRequest) -> RawChatResponse:
    """
    Process a chat request using only the language model without RAG.
    This provides a baseline for comparison with the RAG-enhanced responses.

    Args:
        payload: Chat request containing the query and chat history.

    Returns:
        RawChatResponse: Generated answer and processing time.

    Raises:
        ValueError: If the chat model configuration is invalid.
        Exception: For other processing errors.
    """
    start_time = time.time()
    try:
        # Get language model
        llm = get_chat_llm()

        # Format chat history
        messages = convert_history_to_messages(payload.history or [])

        # Add the current query
        messages.append(HumanMessage(content=payload.query))

        # Get direct response from the model
        response = llm.invoke(messages)
        answer = response.content.strip()

        # Calculate costs - for raw queries, input is not cached
        input_text = f"{payload.query}"
        if payload.history:
            input_text = (
                "\n".join([f"Human: {h}\nAssistant: {a}" for h, a in payload.history])
                + f"\nHuman: {payload.query}"
            )

        cost_info = calculate_cost_info(
            input_text=input_text,
            output_text=answer,
            is_cached=False,  # Raw queries use standard input pricing
        )

        processing_time = time.time() - start_time
        return RawChatResponse(
            answer=answer, processing_time=processing_time, cost_info=cost_info
        )

    except ValueError as e:
        # Handle configuration errors
        raise ValueError(f"Configuration error: {str(e)}")
    except Exception as e:
        # Handle other errors
        raise Exception(f"Error processing chat request: {str(e)}")


async def aprocess_raw_chat(payload: ChatRequest) -> RawChatResponse:
    """
    Asynchronous version of process_raw_chat.
    Currently a wrapper as the underlying operations are synchronous.

    Args:
        payload: Chat request containing the query and chat history.

    Returns:
        RawChatResponse: Generated answer without source documents.
    """
    return process_raw_chat(payload)
