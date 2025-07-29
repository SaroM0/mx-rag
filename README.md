# MX-RAG: Document Question-Answering System

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-green.svg)](https://fastapi.tiangolo.com/)

A powerful Retrieval-Augmented Generation (RAG) system built with FastAPI and LangChain, designed to provide intelligent question-answering capabilities over PDF documents. The system leverages OpenAI's embeddings and chat models to process and respond to queries with source attribution.

## Features

- **Document Processing**
  - PDF text extraction and intelligent chunking
  - Vector embeddings for semantic search
  - Automatic metadata tracking
  - Optional chunk storage for reference

- **Chat Capabilities**
  - RAG-enabled responses with source attribution
  - Direct LLM chat option
  - Conversation history support
  - Chat summarization

- **Cost Management**
  - Detailed token usage tracking
  - Cost optimization with cached embeddings
  - Transparent cost breakdown in responses

- **Production Ready**
  - Docker containerization
  - AWS CDK infrastructure
  - Health monitoring
  - CORS support
  - Comprehensive error handling

## Prerequisites

- Python 3.9+
- OpenAI API key
- Docker (optional)
- AWS CLI (for deployment)
- Node.js & npm (for CDK)

## Installation

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mx-rag.git
   cd mx-rag
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a .env file:
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key and other settings
   ```

### Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t mx-rag .
   ```

2. Run with Docker Compose:
   ```bash
   docker-compose up -d
   ```

## Usage

### Starting the Server

#### Local:
```bash
uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Docker:
```bash
docker-compose up -d
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Ingest Documents
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json"
```

#### Chat with RAG
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does the document say about X?",
    "history": []
  }'
```

#### Direct Chat (without RAG)
```bash
curl -X POST http://localhost:8000/chat/raw \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Y?",
    "history": []
  }'
```

#### Summarize Conversation
```bash
curl -X POST http://localhost:8000/summary \
  -H "Content-Type: application/json" \
  -d '{
    "history": [
      ["What is X?", "X is..."],
      ["Tell me more", "Well..."]
    ]
  }'
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| OPENAI_API_KEY | Your OpenAI API key | Required |
| OPENAI_MODEL | Embedding model | text-embedding-3-large |
| CHAT_MODEL_NAME | Chat model | gpt-4 |
| VECTORSTORE_PERSIST_DIRECTORY | ChromaDB storage location | src/data/chroma |
| PDF_DIRECTORY | PDF files location | src/pdfs |
| CHUNK_SIZE | Document chunk size | 512 |
| CHUNK_OVERLAP | Overlap between chunks | 50 |

See `src/app/core/config.py` for all available settings.

## Project Structure

```
mx-rag/
├── docker-compose.yml    # Docker composition
├── Dockerfile           # Container definition
├── infra/              # AWS CDK Infrastructure
├── src/
│   ├── app/            # FastAPI Application
│   │   ├── core/       # Core configurations
│   │   ├── routers/    # API endpoints
│   │   ├── schemas/    # Pydantic models
│   │   └── services/   # Business logic
│   ├── data/           # Vector store & chunks
│   ├── ingestion/      # PDF processing
│   └── pdfs/           # Source PDFs
```

## AWS Deployment

1. Install AWS CDK dependencies:
   ```bash
   cd infra
   npm install
   ```

2. Deploy the infrastructure:
   ```bash
   cdk deploy
   ```

## Performance Considerations

- Use async endpoints for better concurrency
- Monitor token usage and costs
- Consider caching frequently accessed documents
- Adjust chunk size based on your use case

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [LangChain](https://python.langchain.com/)
- [OpenAI](https://openai.com/)
- [ChromaDB](https://www.trychroma.com/)
