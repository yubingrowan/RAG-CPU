#!/usr/bin/env python3
"""
Document Indexing Pipeline
Processes documents and indexes them into vector DB and BM25
"""

import json
import uuid
import os
import sys
from typing import List, Dict, Any
from pathlib import Path

from embedding.embedding import EmbeddingService
from retrieval.vector_db import VectorDB
from retrieval.bm25_retrieval import BM25Retriever
from chunking.chunker import DocumentChunker
from parsing.pdf_parser import PDFParser


class IndexingPipeline:
    """Pipeline for indexing documents"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_db = VectorDB()
        self.bm25_retriever = BM25Retriever()
        self.chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
        self.pdf_parser = PDFParser()
        # Use absolute path for BM25 index
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.bm25_index_path = os.path.join(script_dir, "data", "bm25_index.json")

        # Load BM25 index if exists
        os.makedirs(os.path.dirname(self.bm25_index_path), exist_ok=True)
        self.bm25_retriever.load(self.bm25_index_path)
        if self.bm25_retriever.corpus:
            print(f"Loaded BM25 index with {len(self.bm25_retriever.corpus)} documents")

        # Initialize vector DB collection (only if not exists)
        # Note: For async client, we'll check collection in a separate async method
        # For now, skip collection check in __init__ to avoid async issues
    
    async def index_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Index documents into both vector DB and BM25 with chunking

        Args:
            documents: List of dicts with 'text' and optional 'metadata'

        Returns:
            Stats about indexed documents
        """
        # Chunk documents first
        chunks = self.chunker.chunk_documents(documents)
        print(f"Chunked {len(documents)} documents into {len(chunks)} chunks")

        # Prepare documents for vector DB
        vector_docs = []
        bm25_docs = []

        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            text = chunk['text']
            metadata = chunk['metadata']

            # Generate embedding (CPU-bound, keep in thread pool)
            embedding = self.embedding_service.embed(text)

            # Add to vector DB batch
            vector_docs.append({
                'vector': embedding,
                'text': text,
                'metadata': {**metadata, 'chunk_id': chunk_id}
            })

            # Add to BM25
            bm25_docs.append({
                'id': chunk_id,
                'text': text,
                'metadata': metadata
            })

        # Insert into vector DB (async I/O)
        if vector_docs:
            await self.vector_db.insert_documents(vector_docs)
            print(f"Indexed {len(vector_docs)} chunks into vector DB")

        # Index in BM25 (CPU-bound, keep synchronous)
        if bm25_docs:
            self.bm25_retriever.index_documents(bm25_docs)
            self.bm25_retriever.save(self.bm25_index_path)

        result = {
            'total_documents': len(documents),
            'total_chunks': len(chunks),
            'vector_db_count': len(vector_docs),
            'bm25_count': len(bm25_docs),
            'vector_db_indexed': len(vector_docs),
            'bm25_indexed': len(bm25_docs)
        }

        return result
    
    def parse_pdf_directory(self, pdf_dir: str) -> List[Dict[str, Any]]:
        """
        Parse all PDF files from a directory
        
        Args:
            pdf_dir: Path to directory containing PDF files
            
        Returns:
            List of documents with 'text' and 'metadata'
        """
        pdf_path = Path(pdf_dir)
        if not pdf_path.exists():
            print(f"PDF directory not found: {pdf_dir}")
            return []
        
        documents = []
        pdf_files = list(pdf_path.glob("*.pdf"))
        
        print(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
        
        for pdf_file in pdf_files:
            # Skip Zone.Identifier files (Windows)
            if "Zone.Identifier" in pdf_file.name:
                continue
            
            print(f"Parsing {pdf_file.name}...")
            result = self.pdf_parser.parse_with_metadata(str(pdf_file))
            
            if result:
                documents.append({
                    'text': result['text'],
                    'metadata': result['metadata']
                })
        
        print(f"Successfully parsed {len(documents)} PDF documents")
        return documents
    
    def index_from_file(self, file_path: str) -> Dict[str, int]:
        """
        Index documents from a JSON file
        
        Args:
            file_path: Path to JSON file with documents
        
        Returns:
            Stats about indexed documents
        """
        with open(file_path, 'r') as f:
            documents = json.load(f)
        
        return self.index_documents(documents)
    
    def add_single_document(self, text: str, metadata: Dict = None) -> str:
        """
        Add a single document
        
        Args:
            text: Document text
            metadata: Optional metadata
        
        Returns:
            Document ID
        """
        doc_id = str(uuid.uuid4())
        
        # Generate embedding and insert into vector DB
        embedding = self.embedding_service.embed(text)
        self.vector_db.insert_documents([{
            'vector': embedding,
            'text': text,
            'metadata': {**(metadata or {}), 'doc_id': doc_id}
        }])
        
        # Add to BM25
        self.bm25_retriever.add_document(doc_id, text)
        
        return doc_id


if __name__ == "__main__":
    # Test the indexing pipeline
    pipeline = IndexingPipeline()
    
    # Test documents
    test_docs = [
        {
            'text': 'Machine learning is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to learn from data.',
            'metadata': {'category': 'AI', 'source': 'test'}
        },
        {
            'text': 'Deep learning is a subset of machine learning that uses neural networks with multiple layers to model complex patterns in data.',
            'metadata': {'category': 'AI', 'source': 'test'}
        },
        {
            'text': 'Natural language processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language.',
            'metadata': {'category': 'NLP', 'source': 'test'}
        }
    ]
    
    stats = pipeline.index_documents(test_docs)
    print(f"Indexing stats: {stats}")
