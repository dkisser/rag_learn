# Qdrant — Vector Database Overview

> **Source**: https://qdrant.tech/documentation/overview/
> **抓取日期**: 2026-07-18
> **作者/机构**: Qdrant (qdrant.tech)

## What is Qdrant?

Open-source vector database and similarity search engine written in Rust. Provides a production-grade service for storing, searching, and managing points (vectors + payload).

## Architecture

Client-server. Official clients for Python, JavaScript/TypeScript, Rust, Go, .NET, Java. HTTP and gRPC APIs.

### Deployment

- Qdrant OSS (self-hosted)
- Managed Cloud
- Hybrid Cloud
- Private Cloud

---

## Data Structure

**Collections** are named sets of points. Each point has:
- A vector (dense or sparse)
- Optional JSON payload
- Unique ID (uint64 or UUID)

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="my_docs",
    vectors_config={
        "dense": VectorParams(size=768, distance=Distance.COSINE),
        "sparse": VectorParams(size=30000, distance=Distance.DOT),
    },
)
```

---

## Supported Vector Types

- **Dense** — semantic meaning (BERT, OpenAI, Cohere)
- **Sparse** — lexical matches (BM25, SPLADE)

Enables **Hybrid Retrieval**.

---

## HNSW Indexing

**HNSW (Hierarchical Navigable Small World)** graph for fast ANN search. Data organized into **segments**, automatically compacted in background.

Tunable: `m` (edges per node), `ef_construct` (build candidate list), `ef` (search candidate list).

---

## Payload Filtering

Payload indexes extend the HNSW graph, enabling **single-pass filtered vector search**:

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = client.search(
    collection_name="my_docs",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[FieldCondition(key="category", match=MatchValue(value="technology"))]
    ),
    top_k=10,
)
```

---

## Distance Metrics

Cosine, Dot product, Euclidean, Manhattan.

---

## Hybrid Search

```python
results = client.search(
    collection_name="my_docs",
    query_vector=("dense", dense_vector),
    sparse_vector=("sparse", sparse_indices, sparse_values),
    top_k=10,
)
```

Qdrant fuses via RRF by default.

---

## Scaling

- **Sharding** — horizontal scale
- **Replication** — fault tolerance (≥2 for production)
- **Strict mode** — prevents inefficient queries, rate limits

---

## Comparison

| Feature | Qdrant | FAISS | Milvus | Weaviate |
|---------|--------|-------|--------|----------|
| Production server | Yes | Library only | Yes | Yes |
| Written in | Rust | C++ | Go/C++ | Go |
| Payload filtering | First-class | No | Yes | Yes |
| Sparse vectors | Yes | No | Yes | Yes |
| Hybrid search | Built-in | Manual | Built-in | Built-in |

---

## Key Takeaways

- Production-grade vector DB with first-class payload filtering and hybrid search.
- HNSW for fast ANN; payload indexes extend the graph for single-pass filtered search.
- Dense + sparse supported natively.
- Deployment: OSS, Cloud, Hybrid, Private.
