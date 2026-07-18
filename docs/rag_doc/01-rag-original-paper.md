# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

> **来源**: https://arxiv.org/abs/2005.11401
> **抓取日期**: 2026-07-18
> **作者/机构**: Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (Facebook AI Research, University College London, New York University)
> **发表会议**: NeurIPS 2020

## Abstract

Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely manipulate knowledge is still limited, and hence on knowledge-intensive tasks, their performance lags behind task-specific architectures. Additionally, providing provenance for their decisions and updating their world knowledge remain open research problems. Pre-trained models with a differentiable access mechanism to explicit non-parametric memory can overcome this issue, but have so far been only investigated for extractive downstream tasks.

We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG) — models which combine pre-trained parametric and non-parametric memory for language generation. We introduce RAG models where the parametric memory is a pre-trained seq2seq model and the non-parametric memory is a dense vector index of Wikipedia, accessed with a pre-trained neural retriever. We compare two RAG formulations, one which conditions on the same retrieved passages across the whole generated sequence, the other can use different passages per token. We fine-tune and evaluate our models on a wide range of knowledge-intensive NLP tasks and set the state-of-the-art on three open domain QA tasks, outperforming parametric seq2seq models and task-specific retrieve-and-extract architectures. For language generation tasks, we find that RAG models generate more specific, diverse and factual language than a state-of-the-art parametric-only seq2seq baseline.

## 1. Introduction

Pre-trained neural language models have become the foundational building blocks of modern natural language processing (NLP) pipelines. These models, including BERT, GPT, T5, and BART, are trained on large generic corpora and store factual knowledge implicitly in their parameters. However, this parametric storage has well-known limitations:

- **Limited knowledge access**: The model's ability to precisely access and manipulate the factual knowledge stored in its parameters is bounded.
- **No provenance**: It is difficult to attribute model outputs to specific training sources.
- **Static world knowledge**: Updating the model's knowledge requires retraining, which is expensive.

To address these issues, the paper proposes augmenting pre-trained models with a **non-parametric memory** that can be queried via a differentiable retrieval mechanism — and applies this idea to **generation** tasks (not just extractive QA, as prior work had done).

## 2. Key Contributions

1. **First general-purpose fine-tuning recipe for retrieval-augmented generation** combining parametric (seq2seq) and non-parametric (dense vector index of Wikipedia) memory.
2. **Two RAG formulations** are introduced and compared:
   - **RAG-Sequence**: Conditions on the **same** retrieved document across the whole generated sequence.
   - **RAG-Token**: Can use a **different** retrieved document at each generation step (token).
3. **State-of-the-art results** on three open-domain QA tasks (Natural Questions, TriviaQA, WebQuestions, CuratedTrec, Jeopardy, etc.).
4. **Generative quality improvements**: RAG models produce more **specific, diverse, and factual** language than parametric-only seq2seq baselines.

## 3. Method

### 3.1 Models

RAG consists of two components:

- **Generator (parametric)** $p_\theta(y_i|x, y_{<i})$: A pre-trained BART-style sequence-to-sequence model.
- **Retriever (non-parametric)** $p_\eta(z|x)$: A pre-trained dense passage retriever (DPR) that returns the top-K documents $z$ from a corpus (Wikipedia) given a query $x$.

The retriever is built on a bi-encoder architecture using BERT-based encoders for both the query and the document. Documents are pre-encoded and indexed in a FAISS vector store for efficient approximate nearest-neighbor search.

### 3.2 RAG-Sequence

In **RAG-Sequence**, the same retrieved document conditions the entire generated output:

$$p_{\text{RAG-Sequence}}(y|x) \approx \prod_{i}^{N} \sum_{z \in \text{top-}k(p(\cdot|x))} p_\eta(z|x)\, p_\theta(y_i|x, z, y_{<i})$$

That is, the model marginalizes over the retrieved documents by computing a weighted sum of generation probabilities for each candidate document, with the weight being the retriever's relevance score. The **same** document $z$ is used for all tokens $y_i$ in the answer.

### 3.3 RAG-Token

In **RAG-Token**, the model may condition on a **different** document at each generation step:

$$p_{\text{RAG-Token}}(y|x) \approx \prod_{i}^{N} \sum_{z \in \text{top-}k(p(\cdot|x))} p_\eta(z|x)\, p_\theta(y_i|x, z_i, y_{<i})$$

This formulation allows the generator to draw on multiple pieces of evidence while producing the output, which can be advantageous when the answer requires synthesizing information across passages.

### 3.4 Training

The generator $p_\theta$ is fine-tuned end-to-end with the retriever's documents treated as latent variables. Gradients are back-propagated through the retrieved documents to update the generator's parameters. The retriever itself is **not** updated during RAG training (its query encoder is frozen), though the documents are dynamically retrieved at each step.

The retriever $p_\eta(z|x) \propto \exp(d(z)^\top q(x))$, where $d(\cdot)$ is the document encoder and $q(\cdot)$ is the query encoder, both built from BERT.

### 3.5 Decoding

At inference, the authors consider two decoding strategies:

- **Greedy decoding** for QA-style tasks where a single deterministic answer is desired.
- **Sampled decoding** (with diverse beam search and rejection sampling) for generation tasks where diversity is desired (e.g., FEVER fact verification, MS-MARCO generation, Jeopardy question generation).

## 4. Experimental Results

### 4.1 Open-Domain Question Answering

RAG was evaluated on four open-domain QA datasets: **Natural Questions (NQ)**, **TriviaQA**, **WebQuestions (WQ)**, and **CuratedTrec (CT)**. RAG achieved state-of-the-art results on all four, outperforming:

- **Parametric-only seq2seq baselines** (T5, BART fine-tuned).
- **Task-specific retrieve-and-extract architectures** such as REALM, T5 + DPR extractive reader, and Fusion-in-Decoder.

Key results (test set Exact Match):

| Model | NQ | TriviaQA | WQ | CT |
|-------|----|----------|-----|----|
| T5-11B | 32.8 | 47.4 | 30.6 | 41.5 |
| RAG-Token (BART) | 44.1 | 56.1 | 33.4 | 47.5 |
| RAG-Sequence (BART) | 44.5 | 56.8 | 34.2 | 46.9 |

### 4.2 Knowledge-Intensive Generation

The paper evaluated RAG on **FEVER** (fact verification), **MS-MARCO** (answer generation from web passages), and **Jeopardy Question Generation**. On FEVER, RAG-Token with BART achieved 86.3% label accuracy, an absolute improvement of +7.6% over a parametric-only baseline.

For Jeopardy generation, human evaluators rated RAG-generated questions as **more factual** and **more specific** than those from BART alone, while being comparable in fluency.

### 4.3 Ablation Studies

Ablations showed:

- **Hot-swapping the index**: Updating the document index (e.g., with newer Wikipedia snapshots) immediately reflects in model outputs, demonstrating knowledge freshness without retraining.
- **Document count K**: Top-K=5 was generally a sweet spot. Larger K provided diminishing returns.
- **Retriever type**: DPR outperformed BM25 for RAG, indicating the value of dense retrieval for generation tasks.

## 5. Discussion & Impact

The RAG paper established several lasting ideas for the field:

1. **Retrieval-augmented generation as a general pattern** for grounding LLM outputs in external, updatable knowledge.
2. **Differentiable end-to-end fine-tuning** that combines non-parametric memory with parametric generators.
3. **Knowledge provenance**: Because outputs can be tied to retrieved documents, attribution becomes possible.
4. **Dynamic knowledge updates**: The retriever's index can be updated without retraining the generator, addressing the static-knowledge problem of LLMs.

The paper has become one of the most cited works in modern NLP and forms the foundation of the **RAG paradigm** that is now standard in production LLM systems.

## 6. Citation

```
@inproceedings{lewis2020rag,
  title={Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks},
  author={Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio and Karpukhin, Vladimir and Goyal, Naman and K{\"u}ttler, Heinrich and Lewis, Mike and Yih, Wen-tau and Rockt{\"a}schel, Tim and Riedel, Sebastian and Kiela, Douwe},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```
