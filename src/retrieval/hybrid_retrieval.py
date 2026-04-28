#!/usr/bin/env python3
"""
Hybrid Retrieval (BM25 + Dense Vector)
Combines sparse and dense retrieval for better results
"""

from typing import List, Dict, Any
import asyncio
import logging
import os
import time
from embedding.embedding import EmbeddingService
from retrieval.vector_db import VectorDB
from retrieval.bm25_retrieval import BM25Retriever

logger = logging.getLogger("HybridRetriever")


class HybridRetriever:
    """Hybrid retrieval combining BM25 and vector search"""
    
    def __init__(self, rrf_k: int = 60):
        """
        Initialize hybrid retriever

        Args:
            rrf_k: RRF constant (typically 60)
        """
        self.rrf_k = rrf_k
        self.embedding_service = EmbeddingService()
        self.vector_db = VectorDB()
        # Increase k1 to give more weight to term frequency
        self.bm25_retriever = BM25Retriever(k1=2.0, b=0.75)

        # Load BM25 index if exists
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        bm25_index_path = os.path.join(script_dir, "data", "bm25_index.json")
        if os.path.exists(bm25_index_path):
            self.bm25_retriever.load(bm25_index_path)
            logger.info("HybridRetriever loaded BM25 index with %d documents", len(self.bm25_retriever.corpus))
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        bm25_top_k: int = 50,
        vector_top_k: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining BM25 and vector retrieval

        Args:
            query: Search query
            top_k: Final number of results
            bm25_top_k: Number of BM25 results to consider
            vector_top_k: Number of vector results to consider

        Returns:
            Combined and ranked results
        """
        search_start = time.perf_counter()
        logger.info(
            "Hybrid search start: query=%r top_k=%d bm25_top_k=%d vector_top_k=%d",
            query,
            top_k,
            bm25_top_k,
            vector_top_k
        )

        # BM25 retrieval (CPU-bound, keep synchronous)
        bm25_start = time.perf_counter()
        bm25_results = self.bm25_retriever.search(query, top_k=bm25_top_k)
        bm25_time = time.perf_counter() - bm25_start
        logger.debug("BM25 results=%d bm25_time=%.3fs", len(bm25_results), bm25_time)

        # Vector retrieval (async I/O)
        # Run embedding in thread pool to avoid blocking
        query_embedding_start = time.perf_counter()
        query_embedding = await asyncio.to_thread(self.embedding_service.embed, query)
        query_embedding_time = time.perf_counter() - query_embedding_start
        logger.debug("Query embedding complete: embed_time=%.3fs embedding_len=%d", query_embedding_time, len(query_embedding))

        vector_start = time.perf_counter()
        vector_results = await self.vector_db.search(query_embedding, limit=vector_top_k)
        vector_time = time.perf_counter() - vector_start
        logger.debug("Vector results=%d vector_time=%.3fs", len(vector_results), vector_time)

        # RRF: Combine rankings using Reciprocal Rank Fusion
        rrf_scores = {}
        
        # Add BM25 rankings
        for rank, result in enumerate(bm25_results, 1):
            doc_id = result['id']
            rrf_score = 1.0 / (self.rrf_k + rank)
            rrf_scores[doc_id] = {
                'id': doc_id,
                'text': result['text'],
                'bm25_score': result['score'],
                'vector_score': 0,
                'bm25_rank': rank,
                'vector_rank': None,
                'rrf_score': rrf_score,
                'parent_id': result.get('metadata', {}).get('parent_id'),
                'metadata': result.get('metadata', {})
            }
        
        # Add Vector rankings and combine
        for rank, result in enumerate(vector_results, 1):
            doc_id = result['id']
            rrf_score = 1.0 / (self.rrf_k + rank)
            if doc_id in rrf_scores:
                rrf_scores[doc_id]['vector_score'] = result['score']
                rrf_scores[doc_id]['vector_rank'] = rank
                rrf_scores[doc_id]['rrf_score'] += rrf_score
                # Merge metadata if not present
                if not rrf_scores[doc_id].get('parent_id'):
                    rrf_scores[doc_id]['parent_id'] = result.get('metadata', {}).get('parent_id')
                if not rrf_scores[doc_id].get('metadata'):
                    rrf_scores[doc_id]['metadata'] = result.get('metadata', {})
            else:
                rrf_scores[doc_id] = {
                    'id': doc_id,
                    'text': result['text'],
                    'bm25_score': 0,
                    'vector_score': result['score'],
                    'bm25_rank': None,
                    'vector_rank': rank,
                    'rrf_score': rrf_score,
                    'parent_id': result.get('metadata', {}).get('parent_id'),
                    'metadata': result.get('metadata', {})
                }
        
        # Sort by RRF score
        ranked_results = sorted(
            rrf_scores.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )
        
        # Add combined_score for backward compatibility
        for result in ranked_results:
            result['combined_score'] = result['rrf_score']
        
        # Deduplicate by parent_id
        seen_parents = {}
        deduplicated_results = []
        for result in ranked_results:
            parent_id = result.get('parent_id')
            if not parent_id:
                # If no parent_id, keep the result
                deduplicated_results.append(result)
            elif parent_id not in seen_parents:
                # First time seeing this parent, keep it
                seen_parents[parent_id] = True
                deduplicated_results.append(result)
            # else: skip duplicate from same parent
        
        total_time = time.perf_counter() - search_start
        logger.info(
            "Hybrid search complete: returned=%d top_k=%d total_time=%.3fs bm25_time=%.3fs vector_time=%.3fs",
            len(deduplicated_results[:top_k]),
            top_k,
            total_time,
            bm25_time,
            vector_time
        )
        return deduplicated_results[:top_k]


if __name__ == "__main__":
    # Test hybrid retriever
    retriever = HybridRetriever()
    
    # Note: This requires documents to be indexed first
    # For testing, you would need to run the indexing pipeline first
    
    query = "machine learning algorithms"
    results = retriever.search(query, top_k=5)
    
    print("Hybrid Search Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['combined_score']:.4f} (BM25: {result['bm25_score']:.4f}, Vector: {result['vector_score']:.4f})")
        print(f"   Text: {result['text'][:100]}...")
