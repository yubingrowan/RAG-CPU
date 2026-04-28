#!/usr/bin/env python3
"""
Index documents from a directory into the RAG system
"""

import sys
import os
import asyncio

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, src_dir)

from indexing_pipeline import IndexingPipeline


async def index_directory(directory: str = "docs"):
    """
    Index all markdown, text, and PDF files from a directory

    Args:
        directory: Path to directory containing documents
    """
    project_dir = os.path.dirname(os.path.dirname(__file__))
    doc_dir = os.path.join(project_dir, directory)

    if not os.path.exists(doc_dir):
        print(f"Error: Directory {doc_dir} does not exist")
        return

    pipeline = IndexingPipeline()

    # Check if this is a PDF directory
    pdf_dir = os.path.join(doc_dir, "pdf")
    if os.path.exists(pdf_dir):
        print(f"Found PDF directory, parsing PDF files...")
        pdf_docs = pipeline.parse_pdf_directory(pdf_dir)
        if pdf_docs:
            print(f"\nIndexing {len(pdf_docs)} PDF documents...")
            stats = await pipeline.index_documents(pdf_docs)
            print(f"\nPDF indexing complete!")
            print(f"Total documents: {stats['total_documents']}")
            print(f"Total chunks: {stats['total_chunks']}")
            print(f"Vector DB indexed: {stats['vector_db_indexed']}")
            print(f"BM25 indexed: {stats['bm25_indexed']}")
            return

    # Regular text/markdown files
    docs = []
    for filename in os.listdir(doc_dir):
        if filename.endswith('.md') or filename.endswith('.txt'):
            filepath = os.path.join(doc_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                docs.append({
                    'text': text,
                    'metadata': {'source': filename, 'path': filepath}
                })
                print(f"Loaded: {filename}")

    if not docs:
        print(f"No documents found in {doc_dir}")
        return

    print(f"\nIndexing {len(docs)} documents...")
    stats = await pipeline.index_documents(docs)

    print(f"\nIndexing complete!")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Vector DB indexed: {stats['vector_db_indexed']}")
    print(f"BM25 indexed: {stats['bm25_indexed']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Index documents into RAG system')
    parser.add_argument('--dir', default='docs', help='Directory containing documents (default: docs)')

    args = parser.parse_args()

    asyncio.run(index_directory(args.dir))
