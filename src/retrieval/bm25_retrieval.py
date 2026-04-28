#!/usr/bin/env python3
"""
BM25 Retrieval
Keyword-based sparse retrieval using BM25 algorithm
"""

import logging
import time
import math
from collections import defaultdict
from typing import List, Dict, Tuple, Any
import re
import json
import os

logger = logging.getLogger("BM25Retriever")


class BM25Retriever:
    """BM25 sparse retrieval"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = []
        self.idf = {}
        self.avg_doc_len = 0
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenizer"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        return tokens
    
    def index_documents(self, documents: List[Dict[str, str]]) -> None:
        """
        Index documents for BM25 retrieval
        
        Args:
            documents: List of dicts with 'id' and 'text'
        """
        self.corpus = documents
        n_docs = len(documents)
        
        # Calculate document frequencies
        self.doc_freqs = []
        doc_lengths = []
        
        for doc in documents:
            tokens = self.tokenize(doc['text'])
            doc_lengths.append(len(tokens))
            
            freq = defaultdict(int)
            for token in tokens:
                freq[token] += 1
            self.doc_freqs.append(freq)
        
        # Calculate average document length
        self.avg_doc_len = sum(doc_lengths) / n_docs if n_docs > 0 else 0
        
        # Calculate IDF
        token_doc_freq = defaultdict(int)
        for freq in self.doc_freqs:
            for token in freq:
                token_doc_freq[token] += 1
        
        for token, freq in token_doc_freq.items():
            self.idf[token] = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1)
    
    def _score_bm25(self, query_tokens: List[str], doc_index: int) -> float:
        """
        Calculate BM25 score for a document

        Args:
            query_tokens: Tokenized query
            doc_index: Index of document in corpus

        Returns:
            BM25 score
        """
        doc_freq = self.doc_freqs[doc_index]
        doc_len = len(self.tokenize(self.corpus[doc_index]['text']))
        score = 0.0

        for token in query_tokens:
            if token not in doc_freq:
                continue

            tf = doc_freq[token]
            idf = self.idf.get(token, 0)

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += idf * (numerator / denominator)

        return score

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search the corpus using BM25.
        """
        search_start = time.perf_counter()
        query_tokens = self.tokenize(query)
        scores = []
        for i in range(len(self.corpus)):
            score = self._score_bm25(query_tokens, i)
            scores.append({"document": self.corpus[i], "score": score})

        # Sort by score in descending order
        scores.sort(key=lambda x: x["score"], reverse=True)

        results = [{
            "id": s["document"].get("id", str(i)),
            "text": s["document"]["text"],
            "score": s["score"],
            "metadata": s["document"].get("metadata", {})
        } for i, s in enumerate(scores[:top_k])]

        elapsed = time.perf_counter() - search_start
        logger.debug(
            "BM25 search complete: query=%r tokens=%d returned=%d elapsed=%.3fs",
            query,
            len(query_tokens),
            len(results),
            elapsed
        )
        return results
    
    def add_document(self, doc_id: str, text: str) -> None:
        """
        Add a single document to the index
        
        Args:
            doc_id: Document ID
            text: Document text
        """
        self.corpus.append({'id': doc_id, 'text': text})
        tokens = self.tokenize(text)
        
        freq = defaultdict(int)
        for token in tokens:
            freq[token] += 1
        self.doc_freqs.append(freq)
        
        # Recalculate average doc length
        doc_lengths = [len(self.tokenize(d['text'])) for d in self.corpus]
        self.avg_doc_len = sum(doc_lengths) / len(doc_lengths)
        
        # Recalculate IDF
        n_docs = len(self.corpus)
        token_doc_freq = defaultdict(int)
        for freq in self.doc_freqs:
            for token in freq:
                token_doc_freq[token] += 1
        
        for token, freq in token_doc_freq.items():
            self.idf[token] = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1)

    def save(self, file_path: str) -> None:
        """
        Save BM25 index to file

        Args:
            file_path: Path to save the index
        """
        data = {
            'k1': self.k1,
            'b': self.b,
            'corpus': self.corpus,
            'doc_freqs': [dict(d) for d in self.doc_freqs],
            'idf': self.idf,
            'avg_doc_len': self.avg_doc_len
        }
        with open(file_path, 'w') as f:
            json.dump(data, f)

    def load(self, file_path: str) -> None:
        """
        Load BM25 index from file

        Args:
            file_path: Path to load the index from
        """
        if not os.path.exists(file_path):
            return

        with open(file_path, 'r') as f:
            data = json.load(f)

        self.k1 = data['k1']
        self.b = data['b']
        self.corpus = data['corpus']
        self.doc_freqs = [defaultdict(int, d) for d in data['doc_freqs']]
        self.idf = data['idf']
        self.avg_doc_len = data['avg_doc_len']


if __name__ == "__main__":
    # Test BM25 retriever
    retriever = BM25Retriever()
    
    documents = [
        {'id': '1', 'text': 'Machine learning is a subset of artificial intelligence'},
        {'id': '2', 'text': 'Deep learning uses neural networks with multiple layers'},
        {'id': '3', 'text': 'Natural language processing deals with text and speech'},
        {'id': '4', 'text': 'Computer vision enables machines to understand images'}
    ]
    
    retriever.index_documents(documents)
    
    results = retriever.search('neural networks', top_k=2)
    
    print("BM25 Search Results:")
    for result in results:
        print(f"ID: {result['id']}, Score: {result['score']:.4f}, Text: {result['text']}")
