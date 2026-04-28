#!/usr/bin/env python3
"""
Reranker using ms-marco-TinyBERT-L-2-v2
Re-rank retrieval results for better relevance
"""

import logging
import time
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

logger = logging.getLogger("Reranker")


class Reranker:
    """Cross-encoder reranker for result reordering"""
    
    def __init__(self, model_path: str = "/home/developer/ai-infra-rag/models/ms-marco-TinyBERT-L-2-v2"):
        self.model = CrossEncoder(model_path)
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query relevance
        
        Args:
            query: Search query
            documents: List of documents with 'text' field
            top_k: Number of top results to return
        
        Returns:
            Reranked documents with scores
        """
        # Prepare query-document pairs
        pairs = [[query, doc['text']] for doc in documents]
        
        # Predict relevance scores
        scores = self.model.predict(pairs)
        
        # Attach scores to documents
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # Sort by rerank score
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked[:top_k]
    
    def rerank_with_metadata(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rerank results preserving original metadata
        
        Args:
            query: Search query
            results: List of results with 'text' and other fields
            top_k: Number of top results to return
        
        Returns:
            Reranked results with all original metadata
        """
        start_time = time.perf_counter()
        logger.info(
            "Reranking start: query=%r documents=%d top_k=%d",
            query,
            len(results),
            top_k
        )

        # Extract text for reranking
        texts = [r['text'] for r in results]

        # Prepare pairs
        pairs = [[query, text] for text in texts]

        # Predict scores
        scores = self.model.predict(pairs)

        # Attach scores and sort
        for result, score in zip(results, scores):
            result['rerank_score'] = float(score)

        reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
        rerank_time = time.perf_counter() - start_time
        logger.info(
            "Reranking complete: returned=%d rerank_time=%.3fs",
            len(reranked[:top_k]),
            rerank_time
        )

        return reranked[:top_k]
