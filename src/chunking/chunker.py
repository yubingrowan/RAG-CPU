#!/usr/bin/env python3
"""
Document Chunker
Splits documents into smaller chunks with overlap and parent-child relationships
"""

from typing import List, Dict, Any
import re


class DocumentChunker:
    """Document chunker with overlap and parent-child tracking"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize chunker
        
        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Overlap tokens between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents
        
        Args:
            documents: List of dicts with 'text' and optional 'metadata'
        
        Returns:
            List of chunked documents with parent-child metadata
        """
        all_chunks = []
        
        for doc in documents:
            parent_id = doc.get('id') or str(hash(doc.get('text', '')))
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            
            chunks = self.chunk_text(text, parent_id, metadata)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def chunk_text(self, text: str, parent_id: str, parent_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with overlap
        
        Args:
            text: Text to chunk
            parent_id: ID of parent document
            parent_metadata: Metadata from parent document
        
        Returns:
            List of chunks with metadata
        """
        if parent_metadata is None:
            parent_metadata = {}
        
        # Split by paragraphs first
        paragraphs = self._split_paragraphs(text)
        
        # Build chunks
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            # Check if adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'text': current_chunk.strip(),
                    'metadata': {
                        **parent_metadata,
                        'parent_id': parent_id,
                        'chunk_index': chunk_index,
                        'chunk_count': -1  # Will update later
                    }
                })
                chunk_index += 1
                
                # Start new chunk with overlap
                current_chunk = self._get_overlap_text(current_chunk)
            
            current_chunk += para + "\n\n"
        
        # Add last chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': {
                    **parent_metadata,
                    'parent_id': parent_id,
                    'chunk_index': chunk_index,
                    'chunk_count': -1
                }
            })
        
        # Update chunk_count
        for chunk in chunks:
            chunk['metadata']['chunk_count'] = len(chunks)
        
        return chunks
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split by double newlines
        paragraphs = re.split(r'\n\n+', text)
        # Filter empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlapping text from end of chunk"""
        if self.chunk_overlap == 0:
            return ""
        
        # Simple character-based overlap
        words = text.split()
        if len(words) <= self.chunk_overlap:
            return text
        
        overlap_words = words[-self.chunk_overlap:]
        return " ".join(overlap_words) + " "


if __name__ == "__main__":
    # Test chunker
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
    
    test_text = """
    Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by retrieving relevant information from external knowledge bases before generating responses.

    This approach helps LLMs provide more accurate, up-to-date, and contextually relevant answers. It reduces hallucinations by grounding responses in factual information.

    Vector databases store document embeddings for efficient similarity search. Popular options include Qdrant, Pinecone, and Weaviate.
    """
    
    chunks = chunker.chunk_text(test_text, "test_doc_123")
    
    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i}:")
        print(f"Length: {len(chunk['text'])}")
        print(f"Text: {chunk['text'][:100]}...")
        print(f"Metadata: {chunk['metadata']}")
