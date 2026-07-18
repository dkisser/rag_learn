# Modular RAG

> **来源**: https://arxiv.org/abs/2312.10997 (Retrieval-Augmented Generation for Large Language Models: A Survey, Section II-C)
> **HTML 版本**: https://arxiv.org/html/2312.10997v5
> **抓取日期**: 2026-07-18
> **作者/机构**: Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang (Fudan University et al.)

## Paper Context

"Retrieval-Augmented Generation for Large Language Models: A Survey" (arXiv:2312.10997) is the most cited academic survey of RAG. It introduces the now-standard **three-paradigm taxonomy**:

- **Naive RAG** — the original "Retrieve-Read" pipeline.
- **Advanced RAG** — adds pre-retrieval and post-retrieval optimizations (chunking, query rewriting, re-ranking, compression).
- **Modular RAG** — the most general paradigm, breaking RAG into reusable, composable modules that can be rearranged and extended.

This document covers the **Modular RAG** section (Section II-C) in depth.

## Section II-C Modular RAG

> "The modular RAG architecture advances beyond the former two RAG paradigms, offering enhanced adaptability and versatility. It incorporates diverse strategies for improving its components, such as adding a search module for similarity searches and refining the retriever through fine-tuning. Innovations like restructured RAG modules and rearranged RAG pipelines have been introduced to tackle specific challenges. The shift towards a modular RAG approach is becoming prevalent, supporting both sequential processing and integrated end-to-end training across its components. Despite its distinctiveness, Modular RAG builds upon the foundational principles of Advanced and Naive RAG, illustrating a progression and refinement within the RAG family."

### Core claim

Modular RAG is **not a replacement** for Naive or Advanced RAG — it is a **generalization** that subsumes both. Any Naive or Advanced RAG pipeline is a Modular RAG pipeline with a specific module arrangement. The contribution of Modular RAG is **introducing a vocabulary and design space** for assembling RAG pipelines from interchangeable parts.

## II-C1 New Modules

Modular RAG introduces a set of **specialized modules** beyond the Retrieve and Read of Naive RAG.

### Search module
Adapts to specific scenarios by enabling **direct searches** across heterogeneous data sources: search engines, databases, knowledge graphs. The module uses **LLM-generated code and query languages** (e.g., SQL, Cypher) to query structured sources. This is what makes Modular RAG useful for enterprise data, where not everything lives in a vector store.

### RAG-Fusion
Addresses limitations of single-query retrieval by:
1. **Multi-query expansion** — the LLM rewrites the user's query into multiple perspectives.
2. **Parallel vector searches** — each perspective is embedded and searched independently.
3. **Reciprocal rank fusion / intelligent re-ranking** — results are merged and re-ranked, surfacing both explicitly relevant and "transformative" (cross-perspective) knowledge.

RAG-Fusion is one of the most cited concrete Modular RAG patterns.

### Memory module
Leverages the LLM's own memory (conversation history, user profile, prior retrievals) to **guide retrieval**. Creates an unbounded memory pool that iteratively self-enhances, aligning retrieval more closely with the user's data distribution over time. This is the foundation for **conversational RAG** and **personalized RAG**.

### Routing
Navigates through **diverse data sources**, selecting the optimal pathway for a query:
- A summarization pathway for "what is X?" questions.
- A specific database search for factual lookups.
- A merged pathway that combines multiple streams for complex questions.

The router is typically an LLM with a structured output describing the chosen pathway.

### Predict module
Aims to reduce redundancy and noise by **generating context directly through the LLM**, rather than retrieving it. This is useful when:
- The corpus is small and the LLM already knows the answer.
- The question is so common that retrieval adds latency without value.
- The user wants the LLM to reason from parametric knowledge.

The Predict module effectively gives Modular RAG an "off-ramp" from retrieval.

### Task Adapter module
Tailors RAG to **various downstream tasks**:
- **Automated prompt retrieval** for zero-shot inputs — the adapter selects the best system prompt given the query.
- **Task-specific retrievers** built via **few-shot query generation** — the adapter generates synthetic queries for a downstream task and uses them to fine-tune a retriever.

Task Adapters turn a general-purpose RAG system into a task-specialized one with minimal engineering.

## II-C2 New Patterns

Modular RAG's defining feature is the ability to **substitute or reconfigure modules** to address specific challenges. This goes beyond the fixed Retrieve-Read structure of Naive and Advanced RAG.

### Rewrite-Retrieve-Read
Uses the LLM to **rewrite the retrieval query** before sending it to the retriever. A feedback loop updates the rewriter over time based on retrieval quality. Improves performance on tasks where the user's natural-language query is a poor search query.

### Generate-Read
Replaces retrieval with **LLM-generated content**. The LLM produces a candidate passage; the generator then conditions on that passage (not a retrieved one). Useful when retrieval is impossible or expensive.

### Recite-Read
Emphasizes **retrieval from model weights** — the LLM recites relevant facts from its parametric memory, then reads them as if they were retrieved passages. Enhances the model's ability to handle knowledge-intensive tasks without an external retriever.

### Hybrid retrieval
Integrates **keyword, semantic, and vector searches** to cater to diverse query types. A single user query might benefit from BM25 for exact entity matching plus dense retrieval for paraphrased intent.

### Sub-queries and HyDE
- **Sub-queries**: complex questions are decomposed into sub-questions; each is retrieved independently; answers are then composed.
- **HyDE (Hypothetical Document Embeddings)**: the LLM generates a hypothetical answer, embeds it, and retrieves documents similar to that hypothetical answer. Often outperforms embedding the raw query.

### Demonstrate-Search-Predict (DSP)
A framework that **demonstrates** the desired reasoning pattern with examples, **searches** for relevant evidence at each step, and **predicts** the next reasoning step. Used heavily for multi-hop QA and chain-of-thought RAG.

### Retrieve-Read-Retrieve-Read (ITER-RETGEN)
An **iterative** pattern: generate an initial answer, retrieve documents to refine it, generate again, retrieve again, ... Used for complex tasks where one retrieval pass is insufficient.

### FLARE (Forward-Looking Active Retrieval)
Evaluates the **necessity of retrieval on the fly** based on the next-token uncertainty. When the model is uncertain about the next token, it triggers a retrieval; when confident, it continues without one. This is the conceptual precursor to Self-RAG.

### Self-RAG
Adds **reflection tokens** to the LM vocabulary, allowing the LM itself to decide when to retrieve, whether retrieved passages are relevant, and whether its own generation is supported. Self-RAG is treated in the survey as a canonical Modular RAG example because it generalizes the fixed Retrieve-Read flow with adaptive retrieval.

### Fine-tuning and RL integration
Modular RAG can be **combined with fine-tuning or reinforcement learning** more easily than the fixed pipelines. The retriever, rewriter, and even the generator can be jointly fine-tuned end-to-end because their boundaries are explicit.

## Paradigm Progression

| Aspect | Naive RAG | Advanced RAG | Modular RAG |
|---|---|---|---|
| Architecture | Linear Retrieve -> Read | Retrieve -> (optimize) -> Read | Reconfigurable graph of modules |
| Optimization | None | Pre-/post-retrieval heuristics | Module substitution, new modules, joint training |
| Flexibility | Low | Medium | High |
| Best for | Simple Q&A | Q&A with noisy corpora | Diverse tasks, complex pipelines |
| Examples | RAG (Lewis 2020) | Re-ranking, query rewrite | RAG-Fusion, Self-RAG, FLARE, HyDE, DSP, GraphRAG |

The progression is **cumulative**: each paradigm inherits everything from the previous one and adds new capabilities.

## Key Terminology

- **Modular RAG** — RAG decomposed into independent, replaceable modules.
- **RAG-Fusion** — multi-query expansion with reciprocal rank fusion.
- **Rewrite-Retrieve-Read** — query rewriting before retrieval.
- **Generate-Read** — LLM-generated content replacement pattern.
- **FLARE** — Forward-Looking Active Retrieval (uncertainty-triggered retrieval).
- **Self-RAG** — reflection tokens for retrieval and critique decisions.
- **DSP** — Demonstrate-Search-Predict framework.
- **HyDE** — Hypothetical Document Embeddings.
- **ITER-RETGEN** — iterative retrieve-read-retrieve-read.

## Why Modular RAG Matters

1. **Composability** — engineers can mix and match modules without rewriting pipelines.
2. **Specialization** — different tasks benefit from different module arrangements.
3. **Research substrate** — the survey's vocabulary has become the standard way to describe new RAG variants (CRAG, Self-RAG, GraphRAG, Adaptive RAG are all Modular RAG instances).
4. **Production maintainability** — explicit module boundaries make it easier to test, swap, and monitor individual components.
5. **End-to-end training** — modules can be jointly trained because their interfaces are well-defined.

## Paper Specifications

- **arXiv ID**: 2312.10997 [cs.CL]
- **Title**: Retrieval-Augmented Generation for Large Language Models: A Survey
- **Authors**: Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang
- **v1 submitted**: 18 December 2023; **v5 revised**: 27 March 2024

## Relationship to Other Variants

Every modern RAG variant in this document set is a Modular RAG instance:

- **Self-RAG** (doc 16) — adds reflection tokens to the LM and reorganizes the flow into Retrieve-Grade-Generate loops.
- **CRAG** (doc 17) — adds a Retrieval Evaluator module plus a Web Search module, with confidence-based routing.
- **GraphRAG** (doc 18) — adds an indexing-time module that builds a knowledge graph and community summaries, then routes retrieval through them.
- **Agentic RAG** (doc 19) — uses LangGraph state machines to instantiate Modular RAG patterns at runtime with the LLM as a routing agent.

The Modular RAG paradigm is the **umbrella** under which all of these sit.
