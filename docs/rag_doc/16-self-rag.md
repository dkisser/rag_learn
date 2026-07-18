# Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection

> **来源**: https://arxiv.org/abs/2310.11511
> **抓取日期**: 2026-07-18
> **作者/机构**: Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi (University of Washington, Allen Institute for AI, IBM Research AI)

## Abstract

Despite their remarkable capabilities, large language models (LLMs) often produce responses containing factual inaccuracies due to their sole reliance on the parametric knowledge they encapsulate. Retrieval-Augmented Generation (RAG), an ad hoc approach that augments LMs with retrieval of relevant knowledge, decreases such issues. However, indiscriminately retrieving and incorporating a fixed number of retrieved passages, regardless of whether retrieval is necessary, or passages are relevant, diminishes LM versatility or can lead to unhelpful response generation.

We introduce a new framework called **Self-Reflective Retrieval-Augmented Generation (Self-RAG)** that enhances an LM's quality and factuality through retrieval and self-reflection. Our framework trains a single arbitrary LM that **adaptively retrieves passages on-demand**, and generates and reflects on retrieved passages and its own generations using special tokens, called **reflection tokens**. Generating reflection tokens makes the LM controllable during the inference phase, enabling it to tailor its behavior to diverse task requirements.

Experiments show that Self-RAG (7B and 13B parameters) significantly outperforms state-of-the-art LLMs and retrieval-augmented models on a diverse set of tasks. Specifically, Self-RAG outperforms ChatGPT and retrieval-augmented Llama2-chat on Open-domain QA, reasoning and fact verification tasks, and it shows significant gains in improving factuality and citation accuracy for long-form generations relative to these models.

## Motivation

Standard RAG pipelines have three structural limitations that Self-RAG explicitly targets:

1. **Indiscriminate retrieval** — they retrieve a fixed number of top-k passages even when retrieval is not necessary (e.g., "write a thank-you note").
2. **No relevance filtering** — retrieved passages are concatenated to the prompt without checking whether they actually support the answer.
3. **No self-critique** — the model cannot detect or signal when its generation is unsupported by the evidence, leading to hallucinations.

Self-RAG addresses each of these by training the LM itself to make retrieval and critique decisions on the fly.

## Core Idea: Reflection Tokens

The defining innovation of Self-RAG is a vocabulary of **special "reflection tokens"** that the LM emits during generation. These tokens are categorized into two families:

### Retrieval tokens
- **Retrieve**: whether to retrieve new context at all. Values: `yes` / `no` / `continue` (when continuing an unfinished generation).

### Critique tokens (evaluated for each retrieved passage or generated segment)
- **ISREL** (Is Relevant) — whether the passage is relevant to the query. Values: `relevant` / `irrelevant`.
- **ISSUP** (Is Supported) — whether the generation is supported by the passage. Values: `fully supported` / `partially supported` / `no support`.
- **ISUSE** (Is Useful) — overall usefulness of the response on a 1–5 scale (`5, 4, 3, 2, 1`).

By inserting these tokens into the training corpus and fine-tuning the LM on them, Self-RAG turns retrieval quality assessment, hallucination detection, and usefulness scoring into a **next-token-prediction problem** the LM can solve at inference time.

## Self-RAG Algorithm at Inference Time

```
Input: question x
1.  Emit Retrieve token. If "no" -> directly generate y from parametric memory.
2.  If "yes" -> retrieve D passages with retriever R(x, D, k).
3.  For each passage d:
       - Generate segment y_t conditioned on x and d.
       - Emit ISREL(d), ISSUP(y_t, d), ISUSE(y_t).
4.  Use the four critique scores to:
       - Filter out irrelevant passages (hard threshold on ISREL).
       - Re-rank candidate segments (weighted sum of ISSUP and ISUSE).
5.  Output the highest-scoring segment y*.
```

A practical softmax over the discrete critique-token values gives a continuous confidence score, enabling soft filtering or re-ranking.

## Training Recipe

Self-RAG is trained in two phases:

1. **Critique-token supervised fine-tuning (SFT).**
   - The authors construct "critique-augmented" training data by prompting GPT-4 to insert reflection tokens into outputs of an instruction-tuned LM (e.g., Llama2-chat).
   - They then fine-tune a base LM on these labeled examples so that it learns to emit the tokens in the right places.

2. **Rejection-sampling fine-tuning for the retriever.**
   - The retriever is updated to surface passages that the trained Self-RAG critic marks as relevant.
   - Updates are computed only from passages that the critic judged relevant (rejection sampling), avoiding the noise of irrelevant ones.

The whole pipeline is end-to-end: at deployment, a single Self-RAG model performs retrieval decisions, generation, and self-critique with no external judge.

## Architectural and Empirical Highlights

- **On-demand retrieval** cuts unnecessary context consumption: on tasks where parametric knowledge suffices, the model emits `no` and skips the retriever entirely.
- **Single-model deployment** — unlike approaches that chain a separate evaluator, Self-RAG has no auxiliary models at inference.
- **Controllability at inference** — system-level prompts can bias the LM toward more retrieval, more concise answers, or stricter citation by conditioning on the desired reflection-token distribution.
- **7B / 13B parameters** — competitive with ChatGPT (which is several orders of magnitude larger at deployment) on multiple benchmarks.

## Reported Results (from the paper)

| Benchmark | Task type | Self-RAG vs. baselines |
|---|---|---|
| PopQA, TriviaQA | Open-domain QA | Outperforms ChatGPT and retrieval-augmented Llama2-chat |
| ARC-Challenge, PubHealth | Reasoning + fact verification | Beats ChatGPT and Llama2-chat |
| ASQA, FactScore | Long-form generation with citations | Significant gains in factuality and citation accuracy |

Key claim from the abstract: *"Self-RAG (7B and 13B parameters) significantly outperforms state-of-the-art LLMs and retrieval-augmented models on a diverse set of tasks."*

## Limitations and Discussion

- Critique tokens add to the output sequence length and thus latency and token cost; the overhead is non-trivial for long generations.
- The reflection vocabulary is discrete, which means soft, gradient-based adaptation of critique behavior is harder than with continuous scoring heads.
- Quality of the retriever still bounds the ceiling — Self-RAG cannot fabricate evidence that is not in the corpus.
- The SFT data is bootstrapped from GPT-4, so Self-RAG inherits that model's calibration biases.

## Paper Specifications

- **arXiv ID**: 2310.11511 [cs.CL]
- **Submitted**: 17 October 2023
- **Pages**: 30, **Figures**: 2, **Tables**: 12
- **License**: CC BY 4.0
- **DOI**: 10.48550/arXiv.2310.11511

## Related Work

Self-RAG sits at the intersection of:
- Classic RAG (Lewis et al., 2020).
- Adaptive retrieval (Asai et al., "When to Retrieve", 2023; Schick et al., Toolformer, 2023).
- Self-critique / self-evaluation (Saunders et al., 2022; Welleck et al., 2023).
- Modular RAG patterns (Gao et al., 2023, arXiv:2312.10997) where Self-RAG is cited as a key example of flexible orchestration.

Self-RAG's distinct contribution is making reflection tokens a first-class part of the LM vocabulary, enabling a single model to decide when to retrieve, what to retrieve, and whether to trust what it produced.
