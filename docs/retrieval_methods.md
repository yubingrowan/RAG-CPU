# Retrieval Methods for RAG

## Overview

Retrieval is a critical component of RAG systems. The quality of retrieved documents directly impacts the quality of generated responses. This guide covers various retrieval methods and their tradeoffs.

## Dense Retrieval

### How It Works

Dense retrieval converts text into vector embeddings using neural networks and finds similar documents using vector similarity search.

### Advantages
- Captures semantic meaning
- Handles synonyms and paraphrases
- Works well with modern embedding models
- Scalable with vector databases

### Disadvantages
- Requires embedding computation
- May miss exact keyword matches
- Can be computationally expensive
- Depends on quality of embedding model

### Implementation

```python
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
query_embedding = model.encode("machine learning")
doc_embeddings = model.encode(documents)

# Find similar documents
similarities = cosine_similarity(query_embedding, doc_embeddings)
```

## Sparse Retrieval (BM25)

### How It Works

BM25 is a ranking function that uses term frequency and inverse document frequency to score documents based on keyword matches.

### Advantages
- Fast and efficient
- Good for exact keyword matches
- No embedding computation needed
- Well-understood and proven

### Disadvantages
- Misses semantic similarities
- Requires good tokenization
- Limited to keyword matching
- Vocabulary mismatch issues

### Implementation

```python
from rank_bm25 import BM25Okapi

# Tokenize documents
tokenized_corpus = [doc.split() for doc in documents]

# Create BM25 index
bm25 = BM25Okapi(tokenized_corpus)

# Search
tokenized_query = query.split()
doc_scores = bm25.get_scores(tokenized_query)
```

## Hybrid Retrieval

### Combining Dense and Sparse

Hybrid retrieval combines the strengths of both methods:
- Dense retrieval captures semantic meaning
- Sparse retrieval captures exact matches
- Results are combined using weighted scoring

### Combination Methods

**Score Fusion**
- Normalize scores from both methods
- Apply weights: `final_score = α * dense_score + β * sparse_score`
- Tune weights on validation set

**Reciprocal Rank Fusion (RRF)**
- Ranks results from each method
- Combines ranks rather than scores
- Formula: `RRF(d) = Σ 1/(k + rank_i(d))`

**Learned Combination**
- Use machine learning to learn optimal combination
- Requires training data
- Can adapt to specific use cases

### Implementation

```python
# Dense retrieval
dense_results = vector_search(query_embedding, top_k=50)

# Sparse retrieval
sparse_results = bm25_search(query, top_k=50)

# Combine with RRF
combined = reciprocal_rank_fusion([dense_results, sparse_results])
```

## Reranking

### Why Rerank?

Initial retrieval may return many results, some of which are not very relevant. Reranking uses a more sophisticated model to reorder results for better quality.

### Cross-Encoder Rerankers

Cross-encoder models take query-document pairs and output relevance scores directly.

**Advantages**
- More accurate than bi-encoders
- Captures fine-grained relevance
- Better for ranking top results

**Disadvantages**
- Slower than bi-encoders
- Requires scoring each pair
- Not suitable for large-scale retrieval

### Implementation

```python
from sentence_transformers import CrossEncoder

# Load reranker
reranker = CrossEncoder('ms-marco-TinyBERT-L-2-v2')

# Rerank results
pairs = [[query, doc] for doc in retrieved_docs]
scores = reranker.predict(pairs)

# Sort by scores
reranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
```

## Advanced Techniques

### Query Expansion

Expand the query with related terms to improve recall:
- Synonyms
- Related concepts
- Domain-specific terminology
- Pseudo-relevance feedback

### Dense Passage Retrieval (DPR)

Uses specialized models trained specifically for retrieval:
- Separate encoders for queries and passages
- Trained on large QA datasets
- Optimized for relevance

### ColBERT

Uses late interaction between query and document tokens:
- More fine-grained matching
- Captures term-level interactions
- Better for long documents

### Multi-Vector Search

Store multiple embeddings per document:
- Title embedding
- Content embedding
- Section embeddings
- Combine results from all

## Evaluation

### Metrics

**Retrieval Metrics**
- Precision@k: Fraction of relevant in top-k
- Recall@k: Fraction of relevant retrieved
- MRR (Mean Reciprocal Rank): Average reciprocal rank of first relevant
- NDCG (Normalized Discounted Cumulative Gain): Accounts for ranking quality

**End-to-End Metrics**
- Answer quality
- User satisfaction
- Task completion rate

### Evaluation Datasets

**Benchmark Datasets**
- MS MARCO
- TREC
- BEIR benchmark
- Domain-specific datasets

**Custom Evaluation**
- Create test queries
- Human annotation of relevance
- Measure on your specific use case

## Best Practices

### Choose the Right Method

- **BM25**: Good for keyword-heavy queries, fast retrieval
- **Dense**: Good for semantic queries, synonyms
- **Hybrid**: Best of both worlds, recommended for most use cases
- **Reranking**: Use when quality is critical, can afford extra latency

### Optimize for Your Use Case

- Analyze your query patterns
- Understand your document characteristics
- Measure performance on your data
- Iterate and tune parameters

### Consider Tradeoffs

- **Latency vs Accuracy**: Reranking improves accuracy but adds latency
- **Cost vs Quality**: Better models cost more
- **Recall vs Precision**: Balance based on your needs
- **Index Size vs Quality**: More dimensions may improve quality but increase storage

## Implementation Tips

### Chunking Strategy

- Choose chunk size based on your content
- Consider overlap between chunks
- Use semantic chunking when possible
- Test different strategies

### Index Configuration

- Choose appropriate index type (HNSW for most cases)
- Tune index parameters (ef_construction, M)
- Monitor index build time and size
- Plan for updates and deletions

### Query Optimization

- Use query expansion for better recall
- Implement caching for frequent queries
- Batch queries when possible
- Monitor query latency

## Common Pitfalls

### Over-Reliance on One Method

- Don't rely solely on dense or sparse retrieval
- Hybrid approaches generally perform better
- Test multiple methods on your data

### Ignoring Domain Specificity

- Generic models may not work well in specialized domains
- Consider fine-tuning on domain data
- Use domain-specific vocabulary

### Poor Evaluation

- Don't assume a method works without testing
- Evaluate on your actual use case
- Monitor performance in production

### Neglecting Updates

- Knowledge bases change over time
- Plan for incremental updates
- Consider freshness requirements
