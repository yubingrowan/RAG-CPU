#!/usr/bin/env python3
"""
Query the RAG system
"""

import sys
import os
import asyncio

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, src_dir)

from rag_pipeline import RAGPipeline


async def query_rag(question: str, session_id: str = "default", use_rerank: bool = True, top_k: int = 5):
    """
    Query the RAG system
    
    Args:
        question: User question
        session_id: Session identifier
        use_rerank: Whether to use reranking
        top_k: Number of documents to retrieve
    """
    pipeline = RAGPipeline()
    
    result = await pipeline.query(
        query=question,
        session_id=session_id,
        use_rerank=use_rerank,
        top_k=top_k,
        temperature=0.7,
        max_tokens=32  # Reduce for faster testing
    )
    
    print(f"Answer: {result['answer']}")
    print(f"\nSources ({len(result['sources'])}):")
    for i, source in enumerate(result['sources'], 1):
        print(f"  {i}. Score: {source.get('rerank_score', source.get('combined_score', 0)):.4f}")
        print(f"     Source: {source.get('metadata', {}).get('source', 'unknown')}")
        print(f"     Text: {source['text'][:150]}...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Query RAG system')
    parser.add_argument('question', help='Question to ask')
    parser.add_argument('--session', default='default', help='Session ID')
    parser.add_argument('--no-rerank', action='store_true', help='Disable reranking')
    parser.add_argument('--top-k', type=int, default=5, help='Number of documents to retrieve')
    
    args = parser.parse_args()
    
    asyncio.run(query_rag(
        question=args.question,
        session_id=args.session,
        use_rerank=not args.no_rerank,
        top_k=args.top_k
    ))
