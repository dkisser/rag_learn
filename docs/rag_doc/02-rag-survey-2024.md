# Retrieval-Augmented Generation for Large Language Models: A Survey

> **来源**: https://arxiv.org/abs/2312.10997
> **抓取日期**: 2026-07-18
> **作者/机构**: Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang (Fudan University, Tongji University)
> **版本**: v5 (Revised March 27, 2024); initial submission December 18, 2023

## Abstract

Large Language Models (LLMs) showcase impressive capabilities but encounter challenges like hallucination, outdated knowledge, and non-transparent, untraceable reasoning processes. Retrieval-Augmented Generation (RAG) has emerged as a promising solution by incorporating knowledge from external databases. This enhances the accuracy and credibility of the generation, particularly for knowledge-intensive tasks, and allows for continuous knowledge updates and integration of domain-specific information. RAG synergistically merges LLMs' intrinsic knowledge with the vast, dynamic repositories of external databases.

This comprehensive survey paper offers a detailed examination of the progression of RAG paradigms, encompassing the **Naive RAG**, the **Advanced RAG**, and the **Modular RAG**. It meticulously scrutinizes the tripartite foundation of RAG frameworks, which includes the **retrieval**, the **generation**, and the **augmentation** techniques. Furthermore, the paper delineates the state-of-the-art retrieval technologies, particularly focusing on the integration of dense retrieval, sparse retrieval, and hybrid retrieval methods.

## 1. Introduction

LLMs such as GPT-4, Llama, and Claude have demonstrated remarkable general-purpose capabilities across many tasks. However, they still face persistent challenges:

- **Hallucination**: LLMs may generate plausible but factually incorrect content.
- **Outdated knowledge**: The model's knowledge is frozen at training time.
- **Opaque reasoning**: The reasoning path cannot be inspected or verified.

RAG mitigates these issues by **retrieving** relevant context from an external knowledge base at inference time and **augmenting** the LLM's prompt with this retrieved information, providing the model with grounded, current, and traceable evidence.

## 2. The Evolution of RAG Paradigms

The survey formalizes a clear evolutionary path of RAG systems through three paradigms:

### 2.1 Naive RAG

The simplest RAG pipeline, following a straightforward three-step process:

1. **Indexing**: Documents are split into chunks, encoded with an embedding model, and stored in a vector index.
2. **Retrieval**: The user query is embedded, and the top-K most similar chunks are retrieved via vector similarity.
3. **Generation**: The retrieved chunks are concatenated with the query as context, and the LLM generates an answer.

**Limitations of Naive RAG**:
- **Retrieval quality issues**: Low precision (irrelevant results) and low recall (missing relevant docs).
- **Generation problems**: Hallucination, context dilution when too many chunks are returned, and inability to handle multi-hop questions.
- **Augmentation obstacles**: Naive concatenation may lose important context or introduce redundancy.

### 2.2 Advanced RAG

Advanced RAG introduces **pre-retrieval** and **post-retrieval** optimizations on top of the Naive pipeline:

**Pre-Retrieval Optimization**:
- **Query rewriting / query expansion**: Reformulating the user query (e.g., with an LLM or HyDE) to better match indexed documents.
- **Query routing**: Directing the query to the appropriate index or pipeline based on intent.
- **Index optimization**: Better chunking strategies (semantic chunking, sliding window), metadata enrichment, and hierarchical indexing (small-to-big retrieval).

**Post-Retrieval Optimization**:
- **Re-ranking**: Using a cross-encoder or LLM to re-rank the top-K results for higher precision.
- **Context compression**: Trimming retrieved passages to fit within the LLM context window.
- **Contextual fusion**: Combining or deduplicating information across retrieved chunks.

**Chunking Optimizations**:
- **Sliding window**: Overlapping chunks to preserve context across boundaries.
- **Semantic chunking**: Grouping sentences based on embedding similarity rather than fixed token counts.

### 2.3 Modular RAG

The most flexible paradigm, Modular RAG decomposes the pipeline into **independently tunable modules** that can be composed, replaced, or chained. Key innovations include:

- **Search modules**: BM25, dense retrieval, web search, hybrid search, SQL queries, knowledge graph queries.
- **Memory modules**: Conversational memory, episodic memory, and long-term memory.
- **Routing modules**: Directing queries to the appropriate retrieval pipeline.
- **Predict modules**: Using LLMs to generate search queries, answers, or filter results.
- **Task adapter modules**: Customizing the RAG pipeline to specific downstream tasks.

**Patterns of Modular RAG**:
- **Re-Retrieval**: Iteratively retrieving additional context when initial results are insufficient.
- **Step-Back Prompting**: Asking the LLM to abstract the question before retrieving.
- **Chain-of-Thought Retrieval**: Using the LLM's reasoning chain to drive retrieval queries.
- **Self-RAG**: The model itself decides when and what to retrieve using reflection tokens.
- **Corrective RAG (CRAG)**: Detecting retrieval failures and falling back to web search.

## 3. The Tripartite Foundation of RAG

The survey identifies three core technical areas that underpin all RAG systems:

### 3.1 Retrieval

Retrieval methods are categorized as:

| Type | Examples | Strengths |
|------|----------|-----------|
| **Sparse retrieval** | BM25, TF-IDF | Efficient, interpretable, strong for keyword matching |
| **Dense retrieval** | DPR, BGE, ColBERT, E5 | Semantic matching, handles paraphrase |
| **Hybrid retrieval** | BM25 + dense + cross-encoder reranker | Combines lexical and semantic strengths |

**Key retrieval challenges**:
- **Recall vs. precision trade-off**.
- **Out-of-domain generalization**: Retrievers may underperform in specialized domains.
- **Index freshness**: Maintaining up-to-date embeddings for evolving corpora.

**RAG evaluation frameworks**: The survey discusses benchmarks like RGB (Retrieval-Augmented Generation Benchmark), RAGAS, RECALL, and ARES for evaluating retrieval and end-to-end RAG quality.

### 3.2 Generation

The generator (typically an LLM) must integrate retrieved context with its own knowledge. Key considerations:

- **Context window limits**: Modern LLMs support 8K-200K tokens, but longer contexts increase cost and latency.
- **Lost-in-the-middle effect**: LLMs tend to ignore information in the middle of long contexts.
- **Faithfulness**: The model should not contradict retrieved evidence.
- **Citation generation**: Producing inline references to source documents.

### 3.3 Augmentation

Augmentation strategies determine **how** retrieved information is fed to the LLM:

- **Naive concatenation**: Append retrieved chunks directly to the prompt.
- **Step-wise augmentation**: Iteratively retrieve and re-prompt.
- **Recursive augmentation**: Chain multiple retrieval steps for multi-hop reasoning.
- **Conditional augmentation**: Adapt the retrieval strategy based on query type.

## 4. RAG vs. Fine-Tuning

The survey explicitly contrasts RAG with fine-tuning:

| Dimension | RAG | Fine-tuning |
|-----------|-----|-------------|
| **Knowledge update** | Update the index (no retraining) | Retrain model on new data |
| **External knowledge** | Naturally integrated | Hard-coded into weights |
| **Data freshness** | Real-time | Frozen at training time |
| **Cost** | Lower (no retraining) | Higher (GPU training) |
| **Interpretability** | High (traceable to sources) | Low (implicit in weights) |
| **Customization** | Per-query adaptive | Global style/format change |
| **Latency** | Higher (retrieval overhead) | Lower (single inference) |

**Best practice**: RAG and fine-tuning are **complementary**, not competing. RAG grounds the model in factual evidence; fine-tuning shapes the model's style, format, and reasoning patterns. Many production systems combine both.

## 5. Evaluation

The survey reviews RAG evaluation along multiple axes:

- **Retrieval quality**: Recall@K, MRR, NDCG, context relevance.
- **Generation quality**: Faithfulness, answer relevance, correctness.
- **End-to-end RAG metrics**: RAGAS (faithfulness, answer relevance, context relevance, context recall).
- **Human evaluation**: Fluency, factuality, citation accuracy.

## 6. Future Directions

The survey identifies several open problems:

1. **Long-context RAG**: Better handling of very long retrieved contexts (e.g., via compression, hierarchical retrieval).
2. **Multi-modal RAG**: Extending retrieval to images, video, audio, code.
3. **GraphRAG**: Knowledge-graph-based retrieval for relational reasoning.
4. **Self-RAG & agentic RAG**: LLMs autonomously deciding when and what to retrieve.
5. **Efficient indexing**: Reducing embedding and storage costs for billion-scale corpora.
6. **Robustness**: Defending against adversarial retrieved content.

## 7. Significance

This survey became the **canonical reference** for RAG research and practice. Its **Naive → Advanced → Modular** taxonomy is now standard terminology in the field, and the tripartite **retrieval / generation / augmentation** framing shapes how new RAG techniques are designed and evaluated.

## 8. Citation

```
@article{gao2024rag,
  title={Retrieval-Augmented Generation for Large Language Models: A Survey},
  author={Gao, Yunfan and Xiong, Yun and Gao, Xinyu and Jia, Kangxiang and Pan, Jinliu and Bi, Yuxi and Dai, Yi and Sun, Jiawei and Wang, Meng and Wang, Haofen},
  journal={arXiv preprint arXiv:2312.10997},
  year={2024}
}
```
