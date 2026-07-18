# Corrective Retrieval Augmented Generation (CRAG)

> **来源**: https://arxiv.org/abs/2401.15884
> **抓取日期**: 2026-07-18
> **作者/机构**: Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling (University of Science and Technology of China)

## Abstract

Large language models (LLMs) inevitably exhibit hallucinations since the accuracy of generated texts cannot be secured solely by the parametric knowledge they encapsulate. Although retrieval-augmented generation (RAG) is a practicable complement to LLMs, it relies heavily on the relevance of retrieved documents, raising concerns about how the model behaves if retrieval goes wrong.

To this end, we propose the **Corrective Retrieval Augmented Generation (CRAG)** to improve the robustness of generation. In CRAG, a lightweight **retrieval evaluator** assesses the overall quality of retrieved documents for a given query, returning a confidence degree that is then used to trigger different **knowledge retrieval actions**. To enhance robustness, CRAG incorporates **large-scale web searches** as an extension to the static retrieval corpus. Additionally, a **decompose-then-recompose algorithm** is applied to selectively focus on key information and filter out irrelevant content in retrieved documents. CRAG is **plug-and-play** and can be seamlessly integrated with various RAG-based approaches.

## Motivation

Vanilla RAG assumes the retriever always returns useful passages. In practice this fails in three regimes:

1. The corpus does not cover the query (out-of-domain).
2. The retriever surfaces irrelevant passages (top-k noise).
3. Passages are partially relevant — only a few sentences are useful.

CRAG explicitly addresses all three by adding a confidence-based corrective layer between the retriever and the generator.

## Method Overview

CRAG is composed of three pluggable components:

### 1. Retrieval Evaluator

A small classifier that, given `(query, retrieved document)`, outputs one of three confidence levels:

| Confidence | Action triggered |
|---|---|
| **Correct** | Use the retrieved documents as-is. |
| **Incorrect** | Discard retrieval; fall back to web search results. |
| **Ambiguous** | Combine retrieved documents with web search results. |

The evaluator is implemented as a fine-tuned T5-large (or any lightweight sequence classifier) trained on **AutoGPT-generated labels** — query–document pairs labeled "Correct" / "Incorrect" / "Ambiguous" by an LLM. This avoids the cost of human annotation while still capturing nuanced relevance.

### 2. Web Search Extension

When the evaluator signals "Incorrect" or "Ambiguous", CRAG issues a web query to a search API (e.g., Google, Bing, Tavily) and pulls the top results. Web results are concatenated with the original retrieved set (in the Ambiguous case) or replace it entirely (in the Incorrect case). This gives the system access to fresh, large-scale knowledge beyond the static corpus.

### 3. Decompose-Then-Recompose (DTR) Knowledge Refinement

Even "Correct" retrieved documents contain noise. DTR processes each document in three steps:

1. **Decompose** — split the document into fine-grained strips (sentences or knowledge triples).
2. **Filter** — score each strip with the same retrieval evaluator; discard strips marked irrelevant.
3. **Recompose** — concatenate the surviving strips into a clean, query-focused passage that is sent to the generator.

In the paper, DTR strips are scored by a lightweight **fine-tuned T5-based "strip scorer"**, not the larger evaluator — keeping latency low.

## CRAG Pipeline (high-level)

```
Query
  |
  v
Retriever (static corpus)  ->  Top-k Documents
  |
  v
Retrieval Evaluator  ->  {Correct, Incorrect, Ambiguous}
  |
  +-- Correct     -> DTR Refinement  --+
  +-- Incorrect   -> Web Search        --+--> Recomposed Context --> LLM --> Answer
  +-- Ambiguous   -> (DTR + Web)       --+
```

## Experimental Setup

- **Datasets** (4 total, covering short- and long-form generation):
  - PopQA, TriviaQA, HotpotQA (short-form QA)
  - Biography generation (long-form, evaluated with self-BLEU and FactScore)
  - PubHealth (fact verification)
- **Baselines**:
  - Standard RAG (no correction).
  - RAG with re-ranking (e.g., UPR, MonoT5).
  - Self-RAG (Asai et al., 2023).
- **LLM backbone**: ChatGPT (GPT-3.5-turbo) and Llama2-chat-7B/13B.

## Key Results

- **Short-form QA**: CRAG consistently outperforms vanilla RAG and several re-ranking baselines on PopQA and TriviaQA, with the largest gains on questions where the static corpus has weak coverage.
- **Long-form generation**: CRAG improves FactScore (factuality) on biography generation by filtering irrelevant biographical snippets before they enter the prompt.
- **Plug-and-play**: When bolted onto ChatGPT or Llama2-chat, CRAG improves both, demonstrating that the corrective layer generalizes across backbones.
- **Web fallback effectiveness**: The web search branch rescues performance on out-of-domain queries where static retrieval gives near-zero recall.

## Why CRAG Works

The authors attribute the gains to three corrective mechanisms acting in series:

1. **Reject wrong retrieval** — preventing hallucinations caused by feeding irrelevant context into the generator.
2. **Augment with web knowledge** — extending the knowledge frontier beyond the static corpus.
3. **Denoise retrieved context** — DTR removes the long-tail of irrelevant sentences that top-k retrievers are forced to include when the corpus is noisy.

The combination is more robust than any single intervention (e.g., re-ranking alone).

## Limitations

- **Latency**: the web-search branch adds a network round trip; CRAG is significantly slower than static RAG in the "Incorrect" branch.
- **Evaluator error propagation**: a miscalibrated evaluator sends the wrong context to the generator. The paper mitigates this with the Ambiguous branch but does not eliminate the risk.
- **Web quality**: web search results themselves can be unreliable; CRAG does not verify web content, only blends it with retrieval.
- **Cost**: DTR requires running a small scorer over every document strip, which is non-trivial for very long documents.

## Paper Specifications

- **arXiv ID**: 2401.15884 [cs.CL]
- **v1 submitted**: 29 January 2024; **v3 revised**: 7 October 2024
- **Authors**: Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling (USTC)

## Relationship to Other RAG Variants

CRAG sits in the **Modular RAG** family alongside:

- **Self-RAG** (arxiv:2310.11511) — uses reflection tokens inside the LM instead of an external evaluator.
- **Adaptive RAG** (Jeong et al., 2024) — adds a query-complexity router that decides whether to retrieve at all.
- **Self-Reflective RAG with LangGraph** (Ankush Gola, LangChain blog 2024) — an open-source LangGraph implementation that combines CRAG-style grading with Self-RAG-style state machines.

CRAG's distinctive angle is the **explicit triage** into three actions via a single confidence score, plus the **web-search fallback**, which most other corrective variants lack.
