#!/usr/bin/env python3
"""
Vector Database Client (Qdrant) - Async
"""

import logging
import time
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Dict, Optional, Any
import uuid

logger = logging.getLogger("VectorDB")


class VectorDB:
    """Qdrant vector database client - async"""
    
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = AsyncQdrantClient(url=f"http://{host}:{port}")
        self.collection_name = "documents"
    
    async def create_collection(
        self,
        vector_size: int = 768,
        distance: Distance = Distance.COSINE
    ) -> None:
        """Create collection for document embeddings"""
        await self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance)
        )
    
    async def insert_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Insert documents with vectors
        
        Args:
            documents: List of dicts with 'text', 'vector', 'metadata'
        
        Returns:
            List of point IDs
        """
        points = []
        for doc in documents:
            point_id = str(uuid.uuid4())
            points.append(PointStruct(
                id=point_id,
                vector=doc["vector"],
                payload={
                    "text": doc["text"],
                    **doc.get("metadata", {})
                }
            ))
        
        await self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return [p.id for p in points]
    
    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.0,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query_vector: Query embedding
            limit: Number of results
            score_threshold: Minimum similarity score
            filter_conditions: Optional filter dict
        
        Returns:
            List of results with text, score, metadata
        """
        query_filter = None
        if filter_conditions:
            conditions = [
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
                for key, value in filter_conditions.items()
            ]
            query_filter = Filter(must=conditions)

        search_start = time.perf_counter()
        # Use Qdrant native search with HNSW index
        results = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter
        )
        elapsed = time.perf_counter() - search_start
        logger.debug(
            "VectorDB search complete: limit=%d score_threshold=%.3f elapsed=%.3fs points=%d",
            limit,
            score_threshold,
            elapsed,
            len(results.points)
        )

        return [
            {
                "id": r.id,
                "text": r.payload.get("text", ""),
                "score": r.score,
                "metadata": {k: v for k, v in r.payload.items() if k != "text"}
            }
            for r in results.points
        ]
    
    async def delete_document(self, point_id: str) -> None:
        """Delete a document by ID"""
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=[point_id]
        )
    
    async def get_collection_info(self) -> Dict:
        """Get collection statistics"""
        return await self.client.get_collection(self.collection_name)


if __name__ == "__main__":
    # Test the client
    db = VectorDB()
    
    # Create collection (768 dims for nomic-embed-text-v1.5)
    db.create_collection(vector_size=768)
    
    print("Collection created successfully")
    print(db.get_collection_info())
