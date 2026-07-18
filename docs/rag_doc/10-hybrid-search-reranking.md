# Hybrid Search and Reranking in RAG

> **Source**: https://weaviate.io/blog/hybrid-search-explained
> **补充来源**: https://www.pinecone.io/learn/series/rag/rerankers/
> **抓取日期**: 2026-07-18
> **作者/机构**: Weaviate Blog / Pinecone Learning Center

## Why Hybrid Search?

Combine two complementary paradigms:
1. **Sparse retrieval** (BM25, SPLADE) — exact keyword matching.
2. **Dense retrieval** (neural embeddings) — semantic similarity.

> "Hybrid search works by combining the results of sparse vector search and dense vector search into a single, ranked list."

---

## Reciprocal Rank Fusion (RRF)

### Formula

```
RRF_score(d) = Σ  1 / (k + rank_i(d))
```

- `rank_i(d)` = rank in retriever `i`'s list (1-indexed)
- `k` ≈ 60 (constant; smooths contributions)

### Worked Example

BM25: A, B, C. Dense: B, C, A. `k = 0`:

| Doc | BM25 Rank | Dense Rank | RRF Score |
|-----|-----------|------------|-----------|
| A   | 1         | 3          | 1.333     |
| B   | 2         | 1          | 1.500     |
| C   | 3         | 2          | 0.833     |

**Final: B → A → C**

### Tuning k

| k | Behavior |
|---|----------|
| 0 | Pure inverse-rank, sensitive |
| 60 | Default; robust |
| ∞ | All ranks equal |

---

## Weaviate Implementation

```python
response = article.query.hybrid(
    query="fisherman that catches salmon",
    alpha=0.5,
    fusion_type="rankedFusion",
    return_metadata=MetadataQuery(score=True, explain_score=True),
)
```

`alpha`: 0 = keyword, 1 = vector, 0.5 = equal. `rankedFusion` = RRF.

---

## Reranking — Second Stage

Cross-encoders rerank candidates with full token-level attention.

### Bi-Encoder vs Cross-Encoder

| Property | Bi-Encoder | Cross-Encoder |
|----------|------------|---------------|
| Encoding | Independent | Joint |
| Speed | Fast (precompute) | Slow (per query-doc) |
| Information loss | High | Low |
| Latency | <100 ms | Seconds over thousands |
| Use | First-stage | Top-k reranking (k ≤ 100) |

---

## Two-Stage Pipeline

```
User query
   ↓
[Stage 1] Hybrid retrieval (BM25 + dense)
   ↓
Top-100 candidates (high recall)
   ↓
[Stage 2] Cross-encoder reranker
   ↓
Top-10 reranked (high precision)
   ↓
LLM context
```

---

## Pinecone Reranker Example

```python
from pinecone.grpc import PineconeGRPC

pc = PineconeGRPC(api_key="PINECONE_API_KEY")

def embed(batch, input_type):
    res = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=batch,
        parameters={"input_type": input_type, "truncate": "END"},
    )
    return [x["values"] for x in res.data]

docs = get_docs(query, top_k=50)

rerank_docs = pc.inference.rerank(
    model="bge-reranker-v2-m3",
    query=query,
    documents=docs,
    top_n=10,
    return_documents=True,
)
```

---

## Popular Rerankers

| Model | Provider |
|-------|----------|
| `bge-reranker-v2-m3` | BAAI (multilingual, open-source) |
| `bge-reranker-large` | BAAI (English, very strong) |
| `cohere-rerank-3` | Cohere (commercial API) |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | SBERT (lightweight) |
| `jina-reranker` | Jina AI (commercial API) |

---

## Production Recipe

1. Run BM25 + dense in parallel.
2. Fuse top-100 via RRF (`k=60`).
3. Rerank top-50 with cross-encoder.
4. Pass final top-10 to LLM.

---

## Key Takeaways

- **Hybrid search** = sparse (BM25) + dense (vector) retrieval. RRF is the standard fusion.
- RRF formula: `Σ 1/(k + rank)`, `k = 60`.
- **Reranking** with a cross-encoder adds precision on top of hybrid retrieval.
- Two-stage pipeline (hybrid → rerank → LLM) is the canonical RAG architecture.
