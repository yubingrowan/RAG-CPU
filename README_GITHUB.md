# RAG Infrastructure System

A complete Retrieval-Augmented Generation (RAG) system with hybrid search (BM25 + dense vector), reranking, conversation memory, REST API, and Streamlit UI.

## Features

- **Hybrid Retrieval**: BM25 sparse search + Vector dense search with RRF fusion
- **Reranking**: Cross-encoder reranking for improved relevance
- **Conversation Memory**: Redis-based conversation history caching
- **Async Indexing**: Kafka-based asynchronous document indexing
- **LangGraph Workflows**: Structured query and index orchestration
- **Web UI**: Streamlit chat interface with query and index modes
- **REST API**: FastAPI server for programmatic access
- **PDF Support**: Document parsing for PDF files

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                        │
│  ┌──────────────┐           ┌──────────────┐              │
│  │ Streamlit UI │           │   FastAPI    │              │
│  │  (Chat Box)  │           │  REST API    │              │
│  └──────┬───────┘           └──────┬───────┘              │
│         │                            │                       │
│         └────────────┬───────────────┘                       │
│                      │                                       │
┌─────────────────────▼───────────────────────────────────────┐
│                    LangGraph Workflows                       │
│  ┌──────────────┐           ┌──────────────┐               │
│  │ Query Graph  │           │ Index Graph  │               │
│  └──────┬───────┘           └──────┬───────┘               │
│         │                            │                        │
│         │                            │                        │
│  ┌──────▼──────┐            ┌───────▼──────┐               │
│  │ RAG Pipeline │            │ Indexing     │               │
│  └──────┬──────┘            │  Pipeline    │               │
│         │                   └──────┬───────┘               │
│         │                          │                        │
│  ┌──────▼──────┐    ┌────────────┐  │                        │
│  │ Hybrid      │    │ Reranker   │  │                        │
│  │ Retrieval    │───▶│ (Cross-Enc)│  │                        │
│  └──────┬──────┘    └────────────┘  │                        │
│         │                           │                        │
│         ├────────────┬────────────┐ │                        │
│         │            │            │ │                        │
│  ┌──────▼──────┐ ┌──▼──────┐ ┌──▼───────┐ │                 │
│  │   BM25      │ │ Vector  │ │  LLM     │ │                 │
│  │ Retrieval   │ │   DB    │ │ (llama)  │ │                 │
│  └─────────────┘ └─────────┘ └──────────┘ │                 │
│         │            │                 │                     │
│         └─────┬──────┘                 │                     │
│               │                        │                     │
│  ┌────────────▼────────────┐   ┌──────▼──────┐            │
│  │   Redis (Conversation)  │   │   Kafka     │            │
│  └─────────────────────────┘   │ (Async)     │            │
│                                 └─────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Core Services
- **llama.cpp**: LLM inference server (Qwen2.5-7B-Instruct Q4_K_M)
- **Redis**: Conversation history caching
- **Kafka**: Message queue for async tasks
- **Qdrant**: Vector database for dense retrieval
- **Streamlit**: Web UI for chat and document indexing

### Models
- **LLM**: Qwen2.5-7B-Instruct (Q4_K_M quantization)
- **Embedding**: nomic-embed-text-v1.5 (768 dims)
- **Reranker**: ms-marco-TinyBERT-L-2-v2 (cross-encoder)

## Installation

### Prerequisites
- Docker & Docker Compose (for Redis, Kafka, Qdrant)
- Python 3.10+
- CMake (for building llama.cpp)

### 1. Build llama.cpp

**Important**: This project uses a modified version of llama.cpp with context size optimizations.

```bash
cd /home/developer/llama.cpp
rm -rf build

# Build with CPU optimizations for Intel i7-1260P
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_AVX=ON \
  -DGGML_AVX2=ON \
  -DGGML_FMA=ON \
  -DGGML_F16C=ON \
  -DGGML_SSE3=ON \
  -DGGML_SSE4_1=ON \
  -DGGML_SSE4_2=ON

cmake --build build --config Release -j 12
```

**llama.cpp Modifications:**

The source code was modified to fix context size limitations:

**File**: `src/llama-context.cpp`
**Lines**: 181 & 187
**Change**: 
```cpp
// Original (256 token limit):
cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);
cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

// Modified (1024 token limit):
cparams.n_ctx = GGML_PAD(cparams.n_ctx, 1024);
cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 1024);
```

**Reason**: The hardcoded GGML_PAD value of 256 was causing severe context truncation despite passing `-c 1024` to the server. This modification allows full utilization of the 1024 token context window.

### 2. Download Models

Models should be placed in `ai-infra-rag/models/`:

```bash
# LLM Model (Qwen2.5-7B-Instruct Q4_K_M)
# Download from: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF
# File: qwen2.5-7b-instruct-q4_k_m.gguf

# Embedding Model
# Download from: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
# Directory: nomic-embed-text-v1.5/

# Reranker Model
# Download from: https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L-2-v2
# Directory: ms-marco-TinyBERT-L-2-v2/
```

### 3. Install Python Dependencies

```bash
cd /home/developer/ai-infra-rag
pip3 install -r requirements.txt
```

### 4. Start Services

```bash
cd /home/developer/ai-infra-rag
./scripts/start_all.sh
```

This will:
- Start Docker services (Redis, Kafka, Qdrant)
- Create Kafka topics
- Start llama.cpp server
- Start FastAPI API server on `http://localhost:8000`
- Start Streamlit UI on `http://localhost:8501`

Or start services individually:
```bash
# Docker services only
docker-compose up -d

# llama.cpp only
./scripts/start_llama_server.sh

# API server only
python3 src/api/api_server.py

# Streamlit UI only
streamlit run ui/chat_ui.py --server.port 8501
```

## Usage

### Streamlit Chat UI

Access the web interface at `http://localhost:8501`

**Features:**
- **Query Mode**: Chat with RAG system, view sources
- **Index Mode**: Add documents via text input or file upload
- **Settings**: Configure reranking, temperature, clear chat

### REST API

```bash
# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "session_id": "user_123",
    "use_rerank": true,
    "top_k": 5
  }'

# Index document
curl -X POST http://localhost:8000/index/document \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Machine learning is a subset of AI...",
    "metadata": {"category": "AI", "source": "docs"}
  }'
```

### Python API

```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

# Query
result = pipeline.query(
    query="What is machine learning?",
    session_id="user_123",
    use_rerank=True,
    top_k=5
)

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

### LangGraph Workflows

```python
from workflows.query_graph import QueryWorkflow
import asyncio

workflow = QueryWorkflow()
result = asyncio.run(workflow.run(
    query="What is machine learning?",
    session_id="user_123",
    use_rerank=True,
    top_k=5
))
```

## Project Structure

```
ai-infra-rag/
├── src/
│   ├── rag_pipeline.py            # RAG pipeline
│   ├── indexing_pipeline.py       # Document indexing
│   ├── retrieval/
│   │   ├── vector_db.py           # Qdrant client
│   │   ├── bm25_retrieval.py      # BM25 retrieval
│   │   ├── hybrid_retrieval.py    # Hybrid search
│   │   └── reranker.py            # Reranking
│   ├── embedding/
│   │   └── embedding.py           # Embedding service
│   ├── llm/
│   │   └── llama_client.py        # LLM HTTP client
│   ├── cache/
│   │   └── conversation_cache.py  # Redis cache
│   ├── parsing/
│   │   └── pdf_parser.py         # PDF parsing
│   ├── workflows/
│   │   ├── query_graph.py        # LangGraph query workflow
│   │   ├── index_graph.py        # LangGraph index workflow
│   │   └── kafka_index_consumer.py # Kafka consumer
│   └── api/
│       └── api_server.py          # FastAPI server
├── ui/
│   └── chat_ui.py                # Streamlit UI
├── scripts/
│   ├── start_all.sh               # Start all services
│   ├── start_llama_server.sh      # Start llama.cpp
│   ├── index_docs.py              # Index documents
│   └── evaluate_rag.py            # RAGAS evaluation
├── docs/
│   └── *.md                      # Documentation
├── models/                        # Model files (not included)
├── data/                          # Data files (not included)
├── docker-compose.yml             # Docker services
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## Performance

**CPU Inference (Intel i7-1260P):**
- Prompt Processing: 29.48 tokens/second
- Text Generation: 6.38 tokens/second
- Memory Usage: 6.5GB/7.6GB (85% utilization)
- Response Time: 25-90 seconds per query

**Context Configuration:**
- Total Context: 1024 tokens
- Prompt: ~512 tokens
- Generation: 512 tokens (max_tokens)

## Key Features

### Hybrid Retrieval with RRF
- Combines BM25 sparse search and vector dense search
- Uses Reciprocal Rank Fusion (RRF) for result fusion
- RRF formula: `score(d) = Σ (1 / (k + rank_i(d)))` with k=60
- Parent-based deduplication to avoid multiple chunks from same document

### Reranking
- Cross-encoder reranking for improved relevance
- ms-marco-TinyBERT-L-2-v2 model
- Configurable via API parameter

### Conversation Memory
- Redis-based caching with 1-hour TTL
- Session-based conversation history
- Configurable max history turns

### Async Indexing
- Kafka-based asynchronous document processing
- LangGraph workflow orchestration
- Non-blocking document indexing

## Configuration

### Hybrid Search Weights
```python
# RRF is used by default (no weights needed)
# Results include bm25_rank, vector_rank, and rrf_score
```

### Context Window
```python
pipeline = RAGPipeline()
pipeline.max_context_length = 4000  # characters
pipeline.max_history = 5  # conversation turns
```

### LLM Settings
```python
# In llama_client.py
context_limit = 1024  # tokens
max_tokens = 512  # tokens
```

## API Endpoints

- `GET /` - Health check
- `GET /health` - Service health status
- `POST /query` - Execute RAG query
- `POST /index/document` - Index single document
- `POST /index/batch` - Index batch of documents
- `POST /session/clear` - Clear conversation history

## Troubleshooting

### llama.cpp Context Size Issue
If you encounter context truncation, ensure:
1. llama.cpp is built with the modified source code
2. Server is started with `-c 1024` parameter
3. Python client uses `context_limit=1024`

### Memory Issues
- Reduce `max_tokens` in RAG pipeline
- Reduce `top_k` for retrieval
- Close unnecessary services

### Performance Issues
- Ensure llama.cpp is built with CPU optimizations
- Increase thread count in startup script
- Consider GPU acceleration for production

## License

This project is for educational and research purposes.

## Acknowledgments

- llama.cpp: https://github.com/ggerganov/llama.cpp
- Qwen: https://github.com/QwenLM/Qwen2.5
- Qdrant: https://github.com/qdrant/qdrant
- LangGraph: https://github.com/langchain-ai/langgraph
