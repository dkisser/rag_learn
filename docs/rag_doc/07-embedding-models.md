# Pretrained Models — Sentence Transformers

> **Source**: https://www.sbert.net/docs/pretrained_models.html
> **抓取日期**: 2026-07-18
> **作者/机构**: UKPLab / Sentence-Transformers (SBERT.net)

## Overview

The sentence-transformers library provides pretrained models for generating dense vector embeddings of sentences, paragraphs, and images. All models are hosted on the Hugging Face Hub under the `sentence-transformers/` organization.

---

## General Purpose Models

Trained on 1 billion+ training pairs. `all-mpnet-base-v2` offers best quality; `all-MiniLM-L6-v2` is 5× faster.

| Model | Dimensions | Speed | Quality |
|-------|-----------|-------|---------|
| `all-mpnet-base-v2` | 768 | Baseline | Best |
| `all-MiniLM-L12-v2` | 384 | ~2× faster | Strong |
| `all-MiniLM-L6-v2` | 384 | ~5× faster | Good |

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embeddings = model.encode([
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
])
similarities = model.similarity(embeddings, embeddings)
```

---

## Multi-QA Models

Trained on 215M (question, answer) pairs from StackExchange, Yahoo Answers, Bing/Google.

### Dot-Product

| Model | Perf (6 Datasets) | Queries/sec GPU/CPU |
|-------|-------------------|---------------------|
| `multi-qa-mpnet-base-dot-v1` | 57.60 | 4,000 / 170 |
| `multi-qa-distilbert-dot-v1` | 52.51 | 7,000 / 350 |
| `multi-qa-MiniLM-L6-dot-v1` | 49.19 | 18,000 / 750 |

### Cosine-Similarity (Normalized)

| Model | Perf (6 Datasets) | Queries/sec GPU/CPU |
|-------|-------------------|---------------------|
| `multi-qa-mpnet-base-cos-v1` | 57.46 | 4,000 / 170 |
| `multi-qa-distilbert-cos-v1` | 52.83 | 7,000 / 350 |
| `multi-qa-MiniLM-L6-cos-v1` | 51.83 | 18,000 / 750 |

---

## MSMARCO Passage Models

Trained on 500k real Bing queries with relevant passages.

### Dot-Product

| Model | MSMARCO MRR@10 | Perf (6 Datasets) | Queries/sec GPU/CPU |
|-------|----------------|-------------------|---------------------|
| `msmarco-bert-base-dot-v5` | 38.08 | 52.11 | 4,000 / 170 |
| `msmarco-distilbert-dot-v5` | 37.25 | 49.47 | 7,000 / 350 |
| `msmarco-distilbert-base-tas-b` | 34.43 | 49.25 | 7,000 / 350 |

### Cosine-Similarity

| Model | MSMARCO MRR@10 | Perf (6 Datasets) | Queries/sec GPU/CPU |
|-------|----------------|-------------------|---------------------|
| `msmarco-distilbert-cos-v5` | 33.79 | 44.98 | 7,000 / 350 |
| `msmarco-MiniLM-L12-cos-v5` | 32.75 | 43.89 | 11,000 / 400 |
| `msmarco-MiniLM-L6-cos-v5` | 32.27 | 42.16 | 18,000 / 750 |

---

## Choosing a Model

| Use Case | Recommended Model |
|----------|------------------|
| General semantic similarity | `all-mpnet-base-v2` or `all-MiniLM-L6-v2` |
| Question-answer RAG | `multi-qa-mpnet-base-cos-v1` |
| Web passage retrieval | `msmarco-distilbert-base-tas-b` |
| Mobile / edge | `all-MiniLM-L6-v2` |

## Key Takeaways

- **all-MiniLM-L6-v2** — 5× faster, good quality. Default starting point.
- **all-mpnet-base-v2** — Best quality general purpose.
- **multi-qa-*** — Optimized for question-answering / RAG.
- **msmarco-*** — Optimized for Bing-style web passage retrieval.
