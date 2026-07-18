# BM25 — Best Matching 25

> **Source**: https://en.wikipedia.org/wiki/Okapi_BM25
> **补充来源**: https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables
> **抓取日期**: 2026-07-18
> **作者/机构**: Wikipedia / Elastic Blog

## Overview

**BM25 (Best Matching 25)** is a ranking function used by search engines to estimate document relevance to a query. Originated from the probabilistic retrieval framework of Robertson, Spärck Jones, and colleagues in the 1970s–80s. "Okapi" comes from the first system that implemented it (City University, London, 1980s).

Default lexical similarity in Elasticsearch (since v5.0), Lucene, OpenSearch, Tantivy.

---

## Formula

```
score(D, Q) = Σ  IDF(q_i) · [ f(q_i, D) · (k_1 + 1) ]
                         ────────────────────────────
                         [ f(q_i, D) + k_1 · (1 - b + b · |D| / avgdl) ]
```

Sum runs over each query term `q_i`.

---

## Parameters

| Parameter | Typical Value | Meaning |
|-----------|---------------|---------|
| `k_1` | 1.2 – 2.0 | Term-frequency saturation. Higher = slower saturation. |
| `b`   | 0.75         | Length normalization. `b=0` disables, `b=1` fully applies. |
| `|D|` | n/a          | Document length in words. |
| `avgdl` | n/a         | Average document length. |

### k_1 controls term-frequency saturation

TF rises quickly while `tf() ≤ k_1` and grows more slowly when `tf() > k_1`. Prevents a term appearing 100 times from dominating.

Worked example (`b = 0`, `k_1 = 10`):
- 1 occurrence → 0.074
- 2 occurrences → 0.136
- 3 occurrences → 0.188

### b controls length normalization

Worked example (`b = 1`, `k_1 = 5`): A 1-word doc containing "shane" (1/1) ranks above a 6-word doc containing "Shane Shane Shane Connelly Connelly Connelly" (3/6).

---

## IDF Component

```
IDF(q_i) = ln( (N - n(q_i) + 0.5) / (n(q_i) + 0.5) + 1 )
```

Where `N` = total docs, `n(q_i)` = docs containing `q_i`. Common terms get low IDF; rare terms get high IDF.

---

## Variants

- **BM11** — `b = 1`
- **BM15** — `b = 0`
- **BM25F** — Multi-field weighted (headlines, body, anchor text)
- **BM25+** — Adds δ (default 1.0) for proper lower-bound on TF normalization

---

## Why BM25 in RAG

1. Exact term matching — catches rare keywords, IDs, proper nouns.
2. Zero training — works out of the box.
3. Cheap — 10–100× smaller index than dense vectors.
4. Complementary to dense — hybrid retrievers consistently outperform either alone.

Typical hybrid score uses **Reciprocal Rank Fusion (RRF)** with `k ≈ 60`.

---

## Reference Table

| Behavior | Setting |
|----------|---------|
| Length neutrality | `b = 0` |
| Full length penalty | `b = 1` |
| Balanced default | `b = 0.75`, `k_1 = 1.2` |
| Strong TF (long docs) | `k_1 = 2.0` |
| Strong TF (short docs) | `k_1 = 0.5–1.0` |

## Key Takeaways

- BM25 = workhorse lexical retrieval — sparse, fast, effective.
- Two knobs: `k_1` (TF saturation), `b` (length normalization).
- For RAG: BM25 + dense retrieval are complementary — use both.
