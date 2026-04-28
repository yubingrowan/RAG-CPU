#!/usr/bin/env python3
"""
Embedding Service using nomic-embed-text-v1.5
"""

import logging
import time
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np

logger = logging.getLogger("EmbeddingService")


class EmbeddingService:
    """Text embedding service using nomic-embed-text-v1.5"""
    
    def __init__(self, model_path: str = "/home/developer/ai-infra-rag/models/nomic-embed-text-v1.5"):
        self.model = SentenceTransformer(model_path)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text(s)
        
        Args:
            text: Single text string or list of texts
        
        Returns:
            Embedding vector(s)
        """
        embed_start = time.perf_counter()
        embeddings = self.model.encode(text)
        elapsed = time.perf_counter() - embed_start
        if isinstance(text, str):
            result = embeddings.tolist()
            logger.debug(
                "Embedding generated: text_len=%d embed_len=%d elapsed=%.3fs",
                len(text),
                len(result),
                elapsed
            )
            return result
        else:
            result = embeddings.tolist()
            logger.debug(
                "Batch embedding generated: batch_size=%d embed_dim=%d elapsed=%.3fs",
                len(text),
                len(result[0]) if result else 0,
                elapsed
            )
            return result
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of texts
            batch_size: Batch size for processing
        
        Returns:
            List of embedding vectors
        """
        embed_start = time.perf_counter()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True
        )
        elapsed = time.perf_counter() - embed_start
        result = embeddings.tolist()
        logger.debug(
            "Batch embedding complete: batch_size=%d embed_dim=%d elapsed=%.3fs",
            len(texts),
            len(result[0]) if result else 0,
            elapsed
        )
        return result
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score (0-1)
        """
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


if __name__ == "__main__":
    # Test the embedding service
    service = EmbeddingService()
    
    print(f"Embedding dimension: {service.embedding_dim}")
    
    # Test single embedding
    text = "This is a test sentence."
    emb = service.embed(text)
    print(f"Embedding length: {len(emb)}")  # list length
    import numpy as np
    print(f"Embedding shape (as numpy): {np.array(emb).shape}")  # numpy shape
    
    # Test batch embedding
    texts = ["Hello world", "Test sentence", "Another example"]
    embeddings = service.embed_batch(texts)
    print(f"Batch embeddings: (as numpy): {np.array(embeddings).shape}")
    
    # Test similarity
    sim = service.similarity("Hello world", "Hello there")
    print(f"Similarity: {sim}")
