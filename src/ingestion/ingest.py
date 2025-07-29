import json
import os
from pathlib import Path
from typing import Any, Dict, List

import fitz
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from src.app.core.config import get_settings
from src.app.core.vectorstore import get_embeddings


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        str: Extracted text content.

    Raises:
        FileNotFoundError: If the PDF file doesn't exist.
        ValueError: If the PDF file is invalid or empty.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            raise ValueError(f"PDF file is empty: {pdf_path}")

        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error processing PDF {pdf_path}: {str(e)}")
    finally:
        if "doc" in locals():
            doc.close()


def save_chunk(chunk: Document, chunk_id: int, output_dir: str) -> None:
    """
    Save a text chunk to a JSON file.

    Args:
        chunk: Document containing the text chunk.
        chunk_id: Unique identifier for the chunk.
        output_dir: Directory to save the chunk file.
    """
    chunk_data = {
        "text": chunk.page_content,
        "metadata": {"chunk_id": str(chunk_id), **chunk.metadata},
    }

    chunk_path = Path(output_dir) / f"chunk_{chunk_id}.json"
    with open(chunk_path, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, ensure_ascii=False, indent=2)


def process_pdf(pdf_path: str) -> List[Document]:
    """
    Process a PDF file and split it into chunks.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List[Document]: List of document chunks.
    """
    settings = get_settings()
    text = extract_text_from_pdf(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # Add rich metadata
    file_name = Path(pdf_path).name
    metadata = {
        "source": file_name,
        "file_path": pdf_path,
        "file_type": "pdf",
    }

    # Split text into chunks
    chunks = splitter.create_documents([text], metadatas=[metadata])

    # Add chunk-specific metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "chunk_id": str(i),
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
        )

    return chunks


def ingest_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Ingest a PDF file into Chroma in one atomic operation.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict[str, Any]: Ingestion statistics.
    """
    try:
        settings = get_settings()

        # 1. Generate chunks
        chunks = process_pdf(pdf_path)

        # 2. Prepare embeddings and IDs
        embeddings = get_embeddings()
        ids = [chunk.metadata["chunk_id"] for chunk in chunks]

        # 3. (Re)create collection and add all chunks atomically
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            ids=ids,
            collection_name=settings.vectorstore_collection_name or "langchain",
            persist_directory=settings.vectorstore_persist_directory,
        )

        # 4. Ensure persistence to disk
        vectordb.persist()

        # 5. Save chunks to JSON if configured
        if settings.save_chunks:
            os.makedirs(settings.chunks_directory, exist_ok=True)
            for i, chunk in enumerate(chunks):
                save_chunk(chunk, i, settings.chunks_directory)

        return {
            "status": "success",
            "pdf_path": pdf_path,
            "chunks_processed": len(chunks),
            "chunks_saved": settings.save_chunks,
        }

    except Exception as e:
        return {"status": "error", "pdf_path": pdf_path, "error": str(e)}


def ingest_directory() -> List[Dict[str, Any]]:
    """
    Ingest all PDF files from the configured directory.

    Returns:
        List[Dict[str, Any]]: List of ingestion results for each PDF.
    """
    settings = get_settings()
    pdf_dir = Path(settings.pdf_directory)

    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    results = []
    for pdf_file in pdf_dir.glob("*.pdf"):
        result = ingest_pdf(str(pdf_file))
        results.append(result)

    return results
