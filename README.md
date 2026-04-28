# RAG Infrastructure System

A complete Retrieval-Augmented Generation (RAG) system with hybrid search (BM25 + dense vector), reranking, conversation memory, and REST API.

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

### Python Modules
- `llama_client.py`: HTTP client for llama.cpp
- `embedding.py`: Text embedding service
- `vector_db.py`: Qdrant vector database client
- `bm25_retrieval.py`: BM25 sparse retrieval
- `hybrid_retrieval.py`: Combined BM25 + vector search
- `reranker.py`: Cross-encoder reranking
- `conversation_cache.py`: Redis-based conversation memory
- `rag_pipeline.py`: Complete RAG pipeline
- `indexing_pipeline.py`: Document indexing
- `api_server.py`: FastAPI REST API
- `pdf_parser.py`: PDF document parsing
- `query_graph.py`: LangGraph query workflow
- `index_graph.py`: LangGraph index workflow
- `kafka_index_consumer.py`: Kafka consumer for async indexing

## Setup

### Prerequisites
- Docker & Docker Compose (for Redis, Kafka, Qdrant)
- Python 3.10+
- CMake (for building llama.cpp)

### Installation

1. **Build llama.cpp (Optimized for CPU)**
```bash
cd /home/developer/llama.cpp
rm -rf build
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

**Note**: The above build configuration enables all CPU instruction sets for optimal performance on Intel i7-1260P. For other CPUs, adjust accordingly.

2. **Download models**
```bash
# Models should be in ai-infra-rag/models/:
# - qwen2.5-7b-instruct-q4_k_m.gguf (LLM)
# - nomic-embed-text-v1.5 (Embedding)
# - ms-marco-TinyBERT-L-2-v2 (Reranker)
```

3. **Install Python dependencies**
```bash
cd /home/developer/ai-infra-rag
pip3 install -r requirements.txt
```

4. **Start all services**
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
cd /home/developer/ai-infra-rag
docker-compose up -d

# llama.cpp only
./scripts/start_llama_server.sh

# API server only
python3 src/api/api_server.py

# Streamlit UI only
streamlit run ui/chat_ui.py --server.port 8501
```

## Usage

### Index Documents

```python
from indexing_pipeline import IndexingPipeline

pipeline = IndexingPipeline()

# Index single document
doc_id = pipeline.add_single_document(
    text="Machine learning is a subset of AI...",
    metadata={"category": "AI", "source": "docs"}
)

# Index batch
documents = [
    {"text": "Deep learning uses neural networks...", "metadata": {"category": "AI"}},
    {"text": "NLP deals with text and speech...", "metadata": {"category": "NLP"}}
]
stats = pipeline.index_documents(documents)
```

### Query via Python

```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

result = pipeline.query(
    query="What is machine learning?",
    session_id="user_123",
    use_rerank=True,
    top_k=5
)

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

### Query via API

```bash
# Start API server (or use ./scripts/start_all.sh)
python3 src/api/api_server.py

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "session_id": "user_123",
    "use_rerank": true,
    "top_k": 5
  }'
```

### Query via LangGraph Workflow

```python
from workflows.query_graph import QueryWorkflow
import asyncio

workflow = QueryWorkflow()
result = asyncio.run(workflow.run(
    query="What is machine learning?",
    session_id="user_123",
    use_rerank=True,
    top_k=5,
    temperature=0.3
))

print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])}")
```

### Index via LangGraph Workflow

```python
from workflows.index_graph import IndexWorkflow
import asyncio

workflow = IndexWorkflow()
result = asyncio.run(workflow.run(documents=[
    {"text": "Machine learning is a subset of AI...", "metadata": {"source": "docs"}}
]))

print(f"Status: {result['status']}")
print(f"Indexed: {result['indexed_count']} chunks")
```

### Index via Kafka (Async)

```python
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

task = {
    "task_id": "task_001",
    "documents": [
        {"text": "Document content...", "metadata": {"source": "file.txt"}}
    ],
    "timestamp": "2026-04-16T00:00:00"
}

producer.send('index_tasks', value=task)
producer.flush()
```

### Use Streamlit Chat UI

```bash
# Start Streamlit UI (or use ./scripts/start_all.sh)
streamlit run ui/chat_ui.py --server.port 8501

# Access at http://localhost:8501
# Features:
# - Query Mode: Chat with RAG system, view sources
# - Index Mode: Add documents via text input or file upload
# - Settings: Configure reranking, temperature, clear chat
```

### API Endpoints

- `GET /` - Health check
- `GET /health` - Service health status
- `POST /query` - Execute RAG query
- `POST /index/document` - Index single document
- `POST /index/batch` - Index batch of documents
- `POST /session/clear` - Clear conversation history

## Configuration

### Hybrid Search Weights
```python
retriever = HybridRetriever(bm25_weight=0.5, vector_weight=0.5)
```

### Reranking
```python
result = pipeline.query(query, session_id, use_rerank=True)
```

### Context Window
```python
pipeline = RAGPipeline()
pipeline.max_context_length = 4000  # characters
pipeline.max_history = 5  # conversation turns
```

## Project Structure

```
ai-infra-rag/
├── src/
│   ├── __init__.py
│   ├── rag_pipeline.py            # RAG pipeline
│   ├── indexing_pipeline.py       # Document indexing
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_db.py           # Qdrant client
│   │   ├── bm25_retrieval.py      # BM25 retrieval
│   │   ├── hybrid_retrieval.py    # Hybrid search
│   │   └── reranker.py            # Reranking
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── embedding.py           # Embedding service
│   ├── llm/
│   │   ├── __init__.py
│   │   └── llama_client.py        # LLM HTTP client
│   ├── cache/
│   │   ├── __init__.py
│   │   └── conversation_cache.py  # Redis cache
│   ├── parsing/
│   │   ├── __init__.py
│   │   └── pdf_parser.py         # PDF document parsing
│   ├── workflows/
│   │   ├── __init__.py
│   │   ├── query_graph.py        # LangGraph query workflow
│   │   ├── index_graph.py        # LangGraph index workflow
│   │   └── kafka_index_consumer.py # Kafka async index consumer
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── chunker.py            # Document chunking
│   └── api/
│       ├── __init__.py
│       └── api_server.py          # FastAPI server
├── ui/
│   └── chat_ui.py                # Streamlit chat UI
├── scripts/
│   ├── start_all.sh               # Start all services
│   ├── start_llama_server.sh      # Start llama.cpp only
│   ├── index_docs.py              # Index documents script
│   ├── evaluate_rag.py            # RAGAS evaluation script
│   └── simple_eval.py            # Simple evaluation script
├── docs/
│   ├── *.md                      # Documentation files
│   └── pdf/                      # PDF documents for indexing
├── models/
│   ├── qwen2.5-7b-instruct-q4_k_m.gguf  # LLM model
│   ├── nomic-embed-text-v1.5/         # Embedding model
│   └── ms-marco-TinyBERT-L-2-v2/      # Reranker model
├── data/
│   ├── bm25_index.json            # BM25 index persistence
│   ├── eval_dataset.json         # Evaluation dataset
│   └── eval_results*.json        # Evaluation results
├── docker-compose.yml             # Docker compose config
├── requirements.txt               # Python dependencies
└── README.md                     # This file

External:
├── llama.cpp/                     # llama.cpp source (built separately)
```

## Performance Notes

- **Embedding**: nomic-embed-text-v1.5 (768 dims, 8192 context)
- **LLM**: Qwen2.5-7B Q4_K_M (~4.4GB, CPU inference)
- **Rerank**: ms-marco-TinyBERT-L-2-v2 (fast cross-encoder)
- **Retrieval**: Hybrid BM25 + vector with configurable weights
- **Cache**: Redis with 1-hour TTL for conversations

## Architecture Notes

### Current Design Decisions

**parent_id Usage:**
- parent_id is stored in metadata for each chunk
- Currently not used in RAG retrieval or answer generation
- Potential uses: document filtering, source tracing, full document reconstruction
- To reconstruct full documents: query all chunks by parent_id and merge (requires handling overlap)

**BM25 Implementation:**
- Uses custom inverted index (stored in `data/bm25_index.json`)
- Not using Qdrant sparse vectors
- BM25 calculates similarity using term frequency statistics (doc_freqs, idf, avg_doc_len)
- Comparison with Qdrant sparse vectors:
  - Inverted index: More accurate for keyword matching, custom BM25 formula
  - Sparse vectors: Faster (HNSW index), unified with dense vectors
- Current decision: Keep inverted index for accuracy, sufficient for current scale

**Chunking Strategy:**
- Chunk size: 512 tokens
- Overlap: 50 tokens
- Overlap is stored in chunk text (not deduplicated in context)
- To reconstruct: query by parent_id, sort by chunk_index, handle overlap

**Asynchronous Design:**
- Qdrant uses AsyncQdrantClient (I/O operations)
- BM25, embedding, reranking remain synchronous (CPU-bound)
- API server directly awaits async pipeline methods
- CPU-bound operations wrapped with asyncio.to_thread where needed

**Hybrid Retrieval Implementation:**
- Changed from score-based fusion to RRF (Reciprocal Rank Fusion)
- RRF formula: score(d) = Σ (1 / (k + rank_i(d)))
- Benefits:
  - Industry standard for multi-stage retrieval
  - Handles missing scores gracefully (documents only in one retrieval result)
  - No need for score normalization
  - Robust to different score scales
- RRF constant k = 60 (standard value)
- Removed weight parameters (bm25_weight, vector_weight) from API
- Results now include bm25_rank, vector_rank, and rrf_score in addition to backward-compatible combined_score
- Parent-based deduplication: Results are deduplicated by parent_id to avoid returning multiple chunks from the same document
- Results include parent_id and metadata for tracking document origins

## Optimization Log

### 2026-04-17 - Llama.cpp Context Size Fix & Token Allocation Optimization

**Problem Identified:**
- Llama.cpp server was running with 256 tokens context limit despite `-c 1024` startup parameter
- Python client had mismatched context_limit values (1024 vs 256 vs 2048)
- RAG responses were severely truncated due to limited context space
- max_tokens was set too low (128), limiting answer completeness

**Root Cause Analysis:**
- Source code in `src/llama-context.cpp` had hardcoded GGML_PAD values of 256
- Python client methods had inconsistent context_limit defaults:
  - `truncate_prompt()`: 1024
  - `complete()`: 1024  
  - `chat()`: 2048 (this was the missed one!)
- RAG pipeline hardcoded max_tokens to 128

**Fixes Applied:**

1. **Llama.cpp Source Code Fix:**
   ```cpp
   // In src/llama-context.cpp, lines 181 & 187:
   cparams.n_ctx = GGML_PAD(cparams.n_ctx, 1024);  // Changed from 256
   cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 1024);  // Changed from 256
   ```

2. **Python Client Consistency:**
   - Updated `llama_client.py` all methods to use `context_limit=1024` by default
   - Enhanced error logging for better debugging of LLM responses

3. **RAG Pipeline Token Allocation:**
   - Increased `max_tokens` from 128 to 512 in `rag_pipeline.py`
   - Updated all context_limit parameters to 1024

4. **Optimized Build Configuration:**
   ```bash
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

**Token Allocation Strategy:**
```
Total Context: 1024 tokens
- Prompt: ~512 tokens (retrieved docs + system prompt + history)
- Generation: 512 tokens (max_tokens)
- Smart truncation: When prompt exceeds 512, keep newest 512 tokens
```

**Performance Impact:**
- **Before**: 128 tokens max generation, 256 total context
- **After**: 512 tokens max generation, 1024 total context
- **Response Length**: 763 chars (before) vs 1820 chars (after) - 2.4x improvement
- **Context Utilization**: Full 1024 tokens vs partial 256 tokens

**CPU Performance Analysis (Intel i7-1260P):**
- **Prompt Processing**: 29.48 tokens/second
- **Text Generation**: 6.38 tokens/second  
- **Memory Usage**: 6.5GB/7.6GB (85% utilization)
- **Response Time**: 25-90 seconds per query (acceptable for CPU-only)

**Trade-offs & Considerations:**
- Longer response times due to increased token generation
- Higher memory usage with larger context
- Need to balance prompt length vs generation length within 1024 limit
- CPU inference remains the primary bottleneck (6.38 tokens/sec)

**Optimization Recommendations:**
1. **Prompt Optimization**: Reduce retrieved documents (top_k: 3->2) to leave more space for generation
2. **Token Balance**: Consider max_tokens=384 to leave 640 tokens for prompt
3. **Memory Management**: Monitor memory usage, consider closing unnecessary services
4. **Future Scaling**: GPU acceleration for significant performance improvements

### 2026-04-16 - Performance Optimization (CPU Inference)

**Changes:**
1. Reduced max_tokens from 512 to 256 in RAGPipeline to speed up LLM generation
2. Increased llama.cpp thread count to 8 in startup script for parallel processing
3. Reduced default top_k from 5 to 3 to reduce retrieval and reranking overhead

**Benefits:**
- Faster LLM inference due to reduced token generation
- Better CPU utilization with multi-threading
- Reduced retrieval and reranking time with fewer documents
- Overall improved response latency without hardware upgrades

**Trade-offs:**
- Shorter LLM responses (256 tokens vs 512)
- Fewer retrieved documents (3 vs 5) may reduce answer completeness
- Still using Q4_K_M quantization (could try Q3_K_M for further speedup if needed)

### 2026-04-16 - Enhanced Prompt Engineering

**Changes:**
1. Improved RAG pipeline prompt with detailed instructions and examples
2. Added core principles section for clear guidelines
3. Added specific instructions for source attribution, handling insufficient information, answer structure, technical accuracy, context integration, and conversation consistency
4. Added examples of good and bad responses to guide LLM behavior

**Benefits:**
- More structured and consistent answers from LLM
- Better source citation and attribution
- Clearer handling of insufficient context information
- Improved technical accuracy preservation
- Better integration of information from multiple documents
- Enhanced conversation consistency across turns

### 2026-04-16 - LangGraph Workflows and Chat UI

**Changes:**
1. Implemented LangGraph query workflow (`query_graph.py`) for RAG query orchestration
2. Implemented LangGraph index workflow (`index_graph.py`) for document indexing orchestration
3. Implemented Kafka async index consumer (`kafka_index_consumer.py`) for background indexing
4. Created Streamlit chat UI (`chat_ui.py`) with query and index modes
5. Added PDF parsing support with `pdf_parser.py`
6. Updated IndexingPipeline to support PDF documents
7. Improved prompt engineering with structured instructions and source citation guidance

**Benefits:**
- LangGraph provides structured workflow orchestration for complex RAG operations
- Kafka async indexing enables scalable document processing without blocking
- Streamlit UI provides user-friendly interface for both querying and indexing
- PDF support expands document types that can be indexed
- Improved prompt engineering enhances answer quality and encourages source citation

**Features:**
- Query Mode: Chat interface with RAG, source viewing, conversation history
- Index Mode: Text input and file upload for document indexing
- Kafka async indexing: Send tasks to Kafka topic, consumer processes asynchronously
- LangGraph workflows: State-based orchestration with validation and error handling

### 2026-04-15 - Parent ID Deduplication

**Changes:**
1. Modified BM25 retrieval to return metadata including parent_id
2. Modified hybrid retrieval to extract parent_id from both BM25 and Vector results
3. Implemented deduplication logic to remove duplicate chunks from same parent document
4. Results now include parent_id and metadata fields

**Benefits:**
- Avoids returning multiple chunks from the same document
- Improves result diversity and relevance
- Better utilizes the chunking metadata
- Keeps highest-scoring chunk per parent document

### 2026-04-16 - Max Tokens Increase

**Changes:**
1. Increased max_tokens from 128 to 512 in RAGPipeline
2. Allows LLM to generate more complete and detailed responses

**Benefits:**
- Longer, more comprehensive answers
- Better coverage of retrieved information
- Reduced answer truncation
- Improved user experience

### 2026-04-15 - RRF Implementation

**Changes:**
1. Replaced score-based fusion with Reciprocal Rank Fusion (RRF)
2. RRF formula: score(d) = Σ (1 / (k + rank_i(d))) with k=60
3. Removed bm25_weight and vector_weight parameters from API
4. Results now include bm25_rank, vector_rank, and rrf_score

**Benefits:**
- Handles missing scores gracefully (documents only in one retrieval result)
- Industry standard for multi-stage retrieval
- No score normalization needed
- Robust to different score scales between BM25 and vector retrieval

### 2026-04-15 - Architecture Documentation

**Documented:**
1. parent_id metadata usage and limitations
2. BM25 inverted index implementation vs Qdrant sparse vectors
3. Chunking overlap handling
4. Asynchronous design decisions

### 2026-04-13 - Initial Performance Optimization

**Issues Identified:**
1. Document content confusion: vector_databases.md contained GIS vector database references, causing LLM to answer about geographic systems instead of embedding databases
2. max_tokens too low (128): Limited answer length, incomplete responses
3. Qdrant API compatibility: Used scroll + manual similarity calculation instead of native search

**Fixes Applied:**
1. **Document Content**: Completely rewrote vector_databases.md to focus exclusively on embedding vector databases for RAG systems, removing all GIS references
2. **max_tokens**: Increased from 128 to 256 for more complete answers (CPU inference tradeoff: ~20-40s per query)
3. **Qdrant API**: Successfully implemented native search using `query_points` API with HNSW index. Replaced scroll + manual calculation approach. Performance should improve significantly for larger datasets.

**Performance Impact:**
- Current vector search: scroll + manual calculation (100-500ms for 10K vectors)
- Target with native search: HNSW index (5-20ms for 10K vectors)
- BM25: Local calculation (fast, no changes needed)

**Next Steps:**
- Implement Qdrant native search API
- Add document chunking strategy (current: full document as one chunk)
- Add caching layer for embeddings and retrieval results

## Future Enhancements

- [x] Document parsing (PDF, DOCX) - PDF parsing implemented with pypdf
- [x] Kafka consumer for async indexing - Implemented with LangGraph workflows
- [ ] Vector retrieval result caching (Redis)
- [ ] Streaming responses
- [ ] Multi-modal support
- [x] Evaluation metrics - RAGAS evaluation implemented
- [x] Frontend UI (Streamlit/React) - Streamlit chat UI implemented
