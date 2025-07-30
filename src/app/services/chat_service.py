import time
from typing import List, Tuple

import tiktoken
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

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
    Calculate the number of tokens in a text using tiktoken.

    Args:
        text: Text to calculate tokens for.

    Returns:
        Number of tokens.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def calculate_cost_info(
    input_text: str, output_text: str, is_cached: bool = False
) -> CostInfo:
    """
    Calculate cost information for a chat interaction.

    Args:
        input_text: Input text.
        output_text: Output text.
        is_cached: Whether the response was cached.

    Returns:
        CostInfo: Cost information.
    """
    settings = get_settings()
    input_tokens = calculate_tokens(input_text)
    output_tokens = calculate_tokens(output_text)
    total_tokens = input_tokens + output_tokens

    input_cost = input_tokens * settings.model_input_cost_per_token
    output_cost = output_tokens * settings.model_output_cost_per_token
    total_cost = input_cost + output_cost

    return CostInfo(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=total_cost,
        is_cached=is_cached,
    )


def convert_history_to_messages(history: List[Tuple[str, str]]) -> List[BaseMessage]:
    """
    Convert chat history to a list of messages.

    Args:
        history: List of (human, ai) message tuples.

    Returns:
        List of messages.
    """
    messages = []
    for human_msg, ai_msg in history:
        messages.append(HumanMessage(content=human_msg))
        messages.append(AIMessage(content=ai_msg))
    return messages


def get_chat_llm() -> ChatOpenAI:
    """
    Get configured chat LLM.

    Returns:
        ChatOpenAI: Configured chat model.
    """
    settings = get_settings()
    return ChatOpenAI(
        openai_api_key=settings.openai_api_key,
        model_name=settings.chat_model_name,
        temperature=settings.chat_temperature,
        max_tokens=settings.chat_max_tokens,
    )


def create_chat_prompt() -> ChatPromptTemplate:
    """
    Create chat prompt template.

    Returns:
        ChatPromptTemplate: Configured prompt template.
    """
    template = """
    You are a helpful AI assistant. Use the following context to answer the question.
    If you don't know the answer, just say that you don't know.

    Context:
    {context}

    Chat History:
    {history}

    Human: {question}
    Assistant: """

    return ChatPromptTemplate.from_template(template)


async def process_chat(payload: ChatRequest) -> ChatResponse:
    """
    Process a chat request with RAG.

    Args:
        payload: Chat request.

    Returns:
        ChatResponse: Generated answer and source documents.
    """
    start_time = time.time()
    settings = get_settings()

    try:
        # Get relevant documents
        vectordb = get_vectordb()
        docs = await vectordb.asimilarity_search(payload.query, k=settings.chat_top_k)

        # Format context and history
        context = "\n\n".join(doc.page_content for doc in docs)
        history_str = "\n".join(
            f"Human: {h[0]}\nAssistant: {h[1]}" for h in payload.history
        )

        # Create messages
        prompt = create_chat_prompt()
        messages = prompt.format_messages(
            context=context,
            question=payload.query,
            history=history_str,
        )

        # Get response
        llm = get_chat_llm()
        response = await llm.ainvoke(messages)

        # Calculate processing time and costs
        processing_time = time.time() - start_time
        cost_info = calculate_cost_info(
            input_text=str(messages),
            output_text=response.choices[0]["message"]["content"],
        )

        # Format source documents
        source_docs = []
        if settings.chat_return_source_docs:
            for doc in docs:
                source_docs.append(
                    SourceDocument(
                        id=doc.metadata.get("source", ""),
                        content=doc.page_content,
                        source=doc.metadata.get("source", ""),
                        metadata=doc.metadata,
                    )
                )

        return ChatResponse(
            answer=response.choices[0]["message"]["content"],
            source_documents=source_docs,
            processing_time=processing_time,
            cost_info=cost_info,
        )

    except Exception as e:
        raise Exception(f"Error processing chat request: {str(e)}")


async def process_raw_chat(payload: ChatRequest) -> RawChatResponse:
    """
    Process a chat request without RAG.

    Args:
        payload: Chat request.

    Returns:
        RawChatResponse: Generated answer.
    """
    start_time = time.time()

    try:
        # Convert history to messages
        messages = convert_history_to_messages(payload.history)
        messages.append(HumanMessage(content=payload.query))

        # Get response
        llm = get_chat_llm()
        response = await llm.ainvoke(messages)

        # Calculate processing time and costs
        processing_time = time.time() - start_time
        cost_info = calculate_cost_info(
            input_text=str(messages),
            output_text=response.choices[0]["message"]["content"],
        )

        return RawChatResponse(
            answer=response.choices[0]["message"]["content"],
            processing_time=processing_time,
            cost_info=cost_info,
        )

    except Exception as e:
        raise Exception(f"Error processing chat request: {str(e)}")
