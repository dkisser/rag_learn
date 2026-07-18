# ColPali: Efficient Document Retrieval with Vision Language Models

> **来源**:
> - https://arxiv.org/abs/2407.01449
> - https://arxiv.org/html/2407.01449v6
>
> **抓取日期**: 2026-07-18
> **作者/机构**: Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, Pierre Colombo (ICLR 2025)

---

## Abstract

Documents are visually rich structures that convey information through text, but also figures, page layouts, tables, or even fonts. Since modern retrieval systems mainly rely on the textual information they extract from document pages to index documents — often through lengthy and brittle processes — they struggle to exploit key visual cues efficiently. This limits their capabilities in many practical document retrieval applications such as Retrieval Augmented Generation (RAG).

To benchmark current systems on visually rich document retrieval, we introduce the **Visual Document Retrieval Benchmark (ViDoRe)**, composed of various page-level retrieval tasks spanning multiple domains, languages, and practical settings. The inherent complexity and performance shortcomings of modern systems motivate a new concept: **doing document retrieval by directly embedding the images of the document pages**.

We release **ColPali**, a Vision Language Model trained to produce high-quality multi-vector embeddings from images of document pages. Combined with a late interaction matching mechanism, ColPali largely outperforms modern document retrieval pipelines while being drastically simpler, faster and end-to-end trainable. We release models, data, code and benchmarks under open licenses.

---

## Key Contributions

### 1. ViDoRe Benchmark

A comprehensive **Visual Document Retrieval Benchmark** covering:
- Multiple **domains** (academic, practical)
- Multiple **languages** (English, French)
- Multiple **modalities** (text, figures, tables, infographics)

It includes datasets like:
- DocVQA
- InfoVQA
- TAT-DQA
- arXivQA
- TabFQuAD
- Plus practical topic-specific corpora

### 2. ColPali Model

Adapts **PaliGemma-3B** (a VLM combining **SigLIP** vision encoder with **Gemma-2B** language model) to produce **ColBERT-style multi-vector embeddings**. Uses late interaction mechanism where each query token computes max dot product with all document tokens.

---

## Architecture Details

- Projects output token embeddings to **128-dimensional** space
- Late interaction operator:

```
LI(q, d) = Σ_i max_j ⟨E_q(i) | E_d(j)⟩
```

- Trained with **contrastive loss** using in-batch negatives

---

## Key Results (nDCG@5 from Table 2)

| Method | Average nDCG@5 |
|--------|---------------|
| **ColPali** | **81.3%** |
| BiSigLIP (fine-tuned SigLIP) | 58.6% |
| Best baseline (Unstructured + Captioning + BGE-M3) | 67.0% |
| Contrastive VLMs like SigLIP | 51.4% |

ColPali particularly excels on visually complex tasks:
- **ArxivQA**: 79.1%
- **TabFQuAD**: 83.9%
- **InfoVQA**: 81.8%

substantially beating baselines that require complex OCR/captioning pipelines.

---

## Why It Matters for RAG

Traditional RAG pipelines that rely on textual extraction struggle with:
- **Layout complexity**: Multi-column layouts, mixed text/figures
- **Visual elements**: Charts, infographics, tables with rich formatting
- **Brittle extraction pipelines**: OCR errors, parsing failures on complex documents
- **End-to-end complexity**: Multiple preprocessing steps, each adding potential failures

ColPali offers a paradigm shift:
1. **Direct visual embedding**: Index document page images directly, no extraction needed
2. **End-to-end trainable**: The retriever learns from data, not from hand-engineered features
3. **Drastically simpler pipeline**: One model replaces the OCR + chunking + embedding + captioning chain
4. **Faster**: Eliminates multi-stage preprocessing
5. **Better**: Significant improvements on visually rich documents (67.0% → 81.3% nDCG@5 average)

---

## Late Interaction Mechanism

The late interaction mechanism is borrowed from ColBERT. Instead of compressing the entire document into a single embedding vector (bi-encoder approach), ColPali:

1. **Encodes each query token** separately → multiple query embeddings
2. **Encodes each document token** separately → multiple document embeddings
3. **Computes similarity at inference time**: For each query token, find max dot product with all document tokens, then sum across query tokens

```
score(query, document) = Σ_{i ∈ query tokens} max_{j ∈ document tokens} (E_q[i] · E_d[j])
```

This preserves fine-grained matching signals that single-vector embeddings lose through compression, while still being efficient enough for production retrieval.

---

## Training

- **Base model**: PaliGemma-3B (SigLIP vision encoder + Gemma-2B LM)
- **Output dimension**: 128 (reduced from base LM hidden size)
- **Training objective**: Contrastive loss with in-batch negatives
- **Open source**: Models, data, code, and benchmarks released under open licenses

---

## Citation

```
@inproceedings{faysse2025colpali,
  title={ColPali: Efficient Document Retrieval with Vision Language Models},
  author={Faysse, Manuel and Sibille, Hugues and Wu, Tony and Omrani, Bilel and Viaud, Gautier and Hudelot, C{\'e}line and Colombo, Pierre},
  booktitle={ICLR},
  year={2025}
}
```

**ArXiv ID**: 2407.01449
**Latest revision**: v6 (Feb 28, 2025)
**Venue**: ICLR 2025

---

## Links

- Paper: https://arxiv.org/abs/2407.01449
- HTML version: https://arxiv.org/html/2407.01449v6
- PDF: https://arxiv.org/pdf/2407.01449
