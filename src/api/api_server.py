#!/usr/bin/env python3
"""
FastAPI Server for RAG System
REST API endpoints for document indexing and querying
"""

import sys
import os
import asyncio
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_pipeline import RAGPipeline
from indexing_pipeline import IndexingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("api_server")

app = FastAPI(title="RAG API", version="1.0.0")

# Initialize components
rag_pipeline = RAGPipeline()
indexing_pipeline = IndexingPipeline()


# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    session_id: str
    use_rerank: bool = True
    top_k: int = 5
    temperature: float = 0.3


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str


class DocumentRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None


class DocumentBatchRequest(BaseModel):
    documents: List[DocumentRequest]


class IndexResponse(BaseModel):
    vector_db_count: int
    bm25_count: int


class SessionClearRequest(BaseModel):
    session_id: str


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "RAG API is running"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    llama_healthy = rag_pipeline.llama_client.health_check()
    return {
        "status": "ok" if llama_healthy else "degraded",
        "llama_server": "ok" if llama_healthy else "error"
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Execute RAG query
    
    Args:
        request: Query request with query text, session ID, and options
    
    Returns:
        Query response with answer and sources
    """
    logger.info(
        "Received /query request: session=%s top_k=%d use_rerank=%s temperature=%.2f",
        request.session_id,
        request.top_k,
        request.use_rerank,
        request.temperature
    )
    logger.info("Query text: %s", request.query)
    try:
        result = await rag_pipeline.query(
            query=request.query,
            session_id=request.session_id,
            use_rerank=request.use_rerank,
            top_k=request.top_k,
            temperature=request.temperature
        )
        logger.info(
            "Query result: session=%s answer_len=%d sources=%d",
            request.session_id,
            len(result["answer"]),
            len(result["sources"])
        )
        logger.info("Query answer: %s", result["answer"])
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=result["session_id"]
        )
    except Exception as e:
        logger.exception("/query failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/document", response_model=str)
async def index_document(request: DocumentRequest):
    """
    Index a single document

    Args:
        request: Document with text and optional metadata

    Returns:
        Document ID
    """
    try:
        doc_id = await asyncio.to_thread(
            indexing_pipeline.add_single_document,
            text=request.text,
            metadata=request.metadata
        )
        return doc_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/batch", response_model=IndexResponse)
async def index_batch(request: DocumentBatchRequest):
    """
    Index multiple documents

    Args:
        request: List of documents with text and optional metadata

    Returns:
        Indexing statistics
    """
    try:
        stats = await indexing_pipeline.index_documents(request.documents)
        return IndexResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/clear")
async def clear_session(request: SessionClearRequest):
    """
    Clear conversation history for a session

    Args:
        request: Session ID to clear

    Returns:
        Success message
    """
    try:
        await asyncio.to_thread(
            rag_pipeline.clear_session,
            request.session_id
        )
        return {"status": "ok", "message": f"Session {request.session_id} cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
