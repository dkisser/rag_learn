# Query Rewriting for Retrieval-Augmented Large Language Models

> **来源**: https://arxiv.org/abs/2305.14283
> **抓取日期**: 2026-07-18
> **作者/机构**: Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, Nan Duan — Shanghai Jiao Tong University, Microsoft Research Asia, Microsoft Azure AI

## Abstract

Large Language Models (LLMs) play powerful, black-box readers in the retrieve-then-read pipeline, making remarkable progress in knowledge-intensive tasks. This work introduces a new framework, Rewrite-Retrieve-Read instead of the previous retrieve-then-read for the retrieval-augmented LLMs from the perspective of the query rewriting. Unlike prior studies focusing on adapting either the retriever or the reader, our approach pays attention to the adaptation of the search query itself, for there is inevitably a gap between the input text and the needed knowledge in retrieval. We first prompt an LLM to generate the query, then use a web search engine to retrieve contexts. Furthermore, to better align the query to the frozen modules, we propose a trainable scheme for our pipeline. A small language model is adopted as a trainable rewriter to cater to the black-box LLM reader. The rewriter is trained using the feedback of the LLM reader by reinforcement learning. Evaluation is conducted on downstream tasks, open-domain QA and multiple-choice QA. Experiments results show consistent performance improvement, indicating that our framework is proven effective and scalable, and brings a new framework for retrieval-augmented LLM.

## 1 Introduction

Large Language Models (LLMs) have shown remarkable abilities for human language processing and extraordinary scalability and adaptability in few- or zero-shot settings. However, the training process depends on large-scale high-quality corpora but without the perception of the real world. Thus, LLMs still have to face the issue of hallucination and temporal misalignment. This affects the reliability of LLMs and hinders wider practical application, because the consistency between the LLM responses with the real world needs further validation.

Existing work has proved that incorporating external knowledge (i.e., non-parametric knowledge) with internal knowledge (i.e., parametric knowledge) can effectively alleviate hallucination, especially for knowledge-intensive tasks. In fact, retrieval-augmented LLMs have been shown so effective that they have been regarded as a standard solution to alleviate the factuality drawbacks in naive LLM generations. Retrieval augmentation is applied to select relative passages as external contexts for the language model, which is retrieve-then-read framework. Take the open-domain Question-Answering task (open-domain QA) as an example, a retriever first searches for related documents for a question. Then the LLM receives the question and the documents, then predicts an answer.

As most LLMs are only accessible through inference APIs, they play the part of black-box frozen readers in the pipeline. Existing approaches overlook the adaptation of the query, i.e., the input of the retrieve-then-read pipeline. The retrieval query is either original from datasets or directly determined by the black-box generation, thus is always fixed. However, there is inevitably a gap between the input text and the knowledge that is really needed to query. This limits performance and places a burden on retrieval capability enhancement and prompt engineering.

In consideration of this issue, this paper proposes Rewrite-Retrieve-Read, a new framework for retrieval augmentation, which can be further tuned for adapting to LLMs. In front of the retriever, a step of rewriting the input is added, filling the gap between the given input and retrieval need, as is shown in Figure 1. We adopt the off-the-shelf tool, an internet search engine, as the retriever, which avoids the maintenance of the search index and can access up-to-date knowledge. Different from previous studies that require the memory of multiple interaction rounds between the retriever and the LLM for each sample, the motivation of our rewriting step is to clarify the retrieval need from the input text.

We also propose a trainable scheme for our rewrite-retrieve-read framework. The black-box retriever and the reader form a frozen system. To further smooth the steps of our pipeline, we apply a small, trainable language model to perform the rewriting step, denoted as the rewriter. The rewriter is trained by reinforcement learning using the LLM performance as a reward, learning to adapt the retrieval query to improve the reader on downstream tasks.

Our proposed methods are evaluated on knowledge-intensive downstream tasks including open-domain QA (HotpotQA, AmbigNQ, PopQA) and multiple choice QA (MMLU). The experiments are implemented on T5-large as the rewriter, ChatGPT and Vicuna-13B as the LLM reader. The results show that query rewriting consistently improves the retrieve-augmented LLM performance. The results also indicate that the smaller language model can be competent for query rewriting.

## Figure 1: Overview of the pipeline

From left to right, the figure shows (a) standard retrieve-then-read method, (b) LLM as a query rewriter for the rewrite-retrieve-read pipeline, and (c) the pipeline with a trainable rewriter.

The illustrated example uses the input:

> What profession does Nicholas Ray and Elia Kazan have in common?

The rewritten queries are `Nicholas Ray profession` and `Elia Kazan profession`. Web search retrieves descriptions identifying both people as directors, enabling the black-box LLM reader to produce `director`. The figure marks both the retriever hit and reader answer as correct.

## 3 Methodology

We present Rewrite-Retrieve-Read, a pipeline that improves the retrieval-augmented LLM from the perspective of query rewriting.

### 3.1 Rewrite-Retrieve-Read

A task with retrieval augmentation can be denoted as follows. Given a dataset of a knowledge-intensive task (e.g., open-domain QA),

$$
D = \{(x, y)_i\}, \quad i = 0, 1, 2, \ldots, N,
$$

$x$ (e.g., a question) is the input to the pipeline, $y$ is the expected output (e.g., the correct answer). Our pipeline consists of three steps.

1. **Query rewrite:** generate a query $\tilde{x}$ for required knowledge based on the original input $x$.
2. **Retrieve:** search for related context, $doc$.
3. **Read:** comprehend the input along with contexts $[doc, x]$ and predict the output $\hat{y}$.

A straightforward but effective method is to ask an LLM to rewrite queries to search for information that is potentially needed. We use a few-shot prompt to encourage the LLM to think, and the output can be none, one or more queries to search.

### 3.2 Trainable Scheme

Besides, total reliance on a frozen LLM has shown some drawbacks. Reasoning errors or invalid search hinders the performance. On the other hand, retrieved knowledge may sometimes mislead and compromise the language model. To better align to the frozen modules, it is feasible to add a trainable model and adapt it by taking the LLM reader feedback as a reward.

Based on our framework, we further propose to utilize a trainable small language model to take over the rewriting step. The trainable model is initialized with the pre-trained T5-large (770M), denoted as trainable rewriter, $G_\theta$. The rewriter is first trained on pseudo data to warm up, then continually trained by reinforcement learning.

#### 3.2.1 Rewriter Warm-up

The task, query rewriting, is quite different from the pre-training objective of sequence-to-sequence generative models like T5. First, we construct a pseudo dataset for the query rewriting task. Inspired by recent distillation methods, we prompt the LLM to rewrite the original questions $x$ in the training set and collect the generated queries $\tilde{x}$ as pseudo labels. The collected samples are then filtered: Those that get correct predictions from the LLM reader are selected into the warm-up dataset, denoted as

$$
D_{Train} = \{(x, \tilde{x}) \mid \hat{y} = y\}.
$$

The rewriter $G_\theta$ is fine-tuned on $D_{Train}$ with the standard log-likelihood as the training objective:

$$
L_{warm} = -\sum_t \log p_\theta(\hat{\tilde{x}}_t \mid \tilde{x}_{<t}, x). \tag{1}
$$

The rewriter model after warm-up shows modest performance, which depends on the pseudo data quality and rewriter capability. Highly relying on the human-written prompt line, $\tilde{x}$ can be sub-optimal. The relatively small scale of the rewriter size is also a limitation of the performance after the warm-up. Then we turn to reinforcement learning to align the rewriter to the following retriever and LLM reader.

#### 3.2.2 Reinforcement Learning

To further fine-tune the rewriter to cater to the LLM reader, we adopt a policy gradient reinforcement learning framework. The rewriter optimization is formulated as a Markov Decision Process 5-tuple $\langle S, A, P, R, \gamma \rangle$.

1. The state space $S$ is a finite set limited by the vocabulary and the sequence length.
2. The action space $A$ is equal to the vocabulary.
3. The transition probability $P$ is determined by the policy network, which is the rewriter model $G_\theta$.
4. The reward function $R$ gives a reward value that depends on the current state. The policy gradient is derived from rewards, used as the training objective.
5. $\gamma$ denotes the discount factor.

More specifically, the rewriter $G_\theta$ after the warm-up is the initial policy model $\pi_0$. At each step $t$, the action $a_t$ is to generate the next token $\hat{\tilde{x}}_t$ based on the observation of the present state, $s_t = [x, \hat{\tilde{x}}_{<t}]$. When the generation is stopped by the End-Of-Sentence token, one episode is ended. After finishing the retrieval and reading, a reward is computed by evaluating the final output, i.e., a score for the LLM reader prediction.

We adopt Proximal Policy Optimization (PPO). The reward function reflects the quality of the generated queries, which needs to be consistent with the final evaluation of the task. $\hat{\tilde{x}}$ is fed to the retriever and the reader for a final prediction $\hat{y}$. A part of the reward function is the measures of $\hat{y}$ compared to the golden label $y$ (e.g., exact match and F1 of the predicted answers), denoted as $R_{lm}$. Besides, a KL-divergence regularization is added to prevent the model from deviating too far from the initialization:

$$
R(s_t, a_t) = R_{lm}(\hat{\tilde{x}}, y) - \beta KL(\pi_\theta \parallel \pi_0). \tag{4}
$$

## 4 Implementation

### Rewriter

For the frozen pipeline, we prompt an LLM to rewrite the query with few-shot in-context learning. Our prompt follows the formulation of `[instruction, demonstrations, input]`, where the input is $x$. The instruction is straightforward and demonstrations are 1–3 random examples from training sets and are kept constant across all runs, mainly for the task-specific output format illustration, i.e., a short phrase as an answer for HotpotQA, and an option as an answer for MMLU. For the training scheme, we fine-tune a T5 as the rewriter.

### Retriever

We use the Bing search engine as the retriever. It requires no candidate index construction like a dense retriever, nor candidates like a textbook. But it allows for a wide knowledge scope and up-to-time factuality. With Bing API, the retrieval is performed in two approaches.

1. For all retrieved web pages, we concatenate the snippets that are related sentences selected by Bing. This method is similar to using a search engine in a browser, input a query and press Enter, then collect the texts shown on the search result page.
2. For retrieved web pages, we request the URLs and parser to get all the texts. This is similar to clicking on items on the search result page. Then we use BM25 to keep those with higher relevance scores with the query, reducing the document length.

### Reader

The reader is a frozen LLM, where we adopt ChatGPT (`gpt-3.5-turbo`) and Vicuna-13B. It performs reading comprehension and prediction with few-shot in-context learning. In our prompt, following the brief instruction and the demonstrations, the input is $x$ or $[doc, \hat{\tilde{x}}]$ with retrieval augmentation.

### Prompt lines used for the LLMs

**Direct prompt**

```text
Answer the question in the following format, end the answer with '**'.
{demonstration}
Question: {x}
Answer:
```

**Reader prompt in retrieval-augment pipelines**

```text
Answer the question in the following format, end the answer with '**'.
{demonstration}
Question: {doc} {x}
Answer:
```

**Open-domain QA — LLM as a frozen rewriter**

```text
Think step by step to answer this question, and provide search engine queries
for knowledge that you need. Split the queries with ';' and end the queries
with '**'.
{demonstration}
Question: {x}
Answer:
```

**Multiple-choice QA — LLM as a frozen rewriter**

```text
Provide a better search query for web search engine to answer the given
question, end the queries with '**'.
{demonstration}
Question: {x}
Answer:
```

## 5 Experiments

Three open-domain QA datasets are used for evaluation: HotpotQA consists of complex questions that require multi-hop reasoning; AmbigNQ provides a disambiguated version of Natural Questions; PopQA includes long-tail distributions as it contains more low-popularity knowledge than other popular QA tasks. The evaluation metrics are Exact Match (EM) and F1 scores.

For multiple-choice QA, evaluation is conducted on Massive Multi-task Language Understanding (MMLU), an exam question dataset including four categories: Humanities, STEM, Social Sciences, and Other.

The following settings are implemented: **Direct**, the standard in-context learning without any augmentations; **Retrieve-then-read**, the standard retrieval-augmented method; **LLM as a frozen rewriter**, where a frozen LLM reasons and generates queries by few-shot in-context learning; and **Trainable rewriter**, where output queries from the fine-tuned rewriter are used by the retriever and reader.

### Open-domain QA results

| Dataset | Method | EM | F1 |
|---|---|---:|---:|
| HotpotQA | Direct | 32.36 | 43.05 |
| HotpotQA | Retrieve-then-read | 30.47 | 41.34 |
| HotpotQA | LLM rewriter | 32.80 | 43.85 |
| HotpotQA | Trainable rewriter | **34.38** | **45.97** |
| AmbigNQ | Direct | 42.10 | 53.05 |
| AmbigNQ | Retrieve-then-read | 45.80 | 58.50 |
| AmbigNQ | LLM rewriter | 46.40 | 58.74 |
| AmbigNQ | Trainable rewriter | **47.80** | **60.71** |
| PopQA | Direct | 41.94 | 44.61 |
| PopQA | Retrieve-then-read | 43.20 | 47.53 |
| PopQA | LLM rewriter | **46.00** | **49.74** |
| PopQA | Trainable rewriter | 45.72 | 49.51 |

For the three datasets, query rewriting consistently brings performance gain with both a frozen rewriter and a trainable rewriter. On AmbigNQ and PopQA, the standard retrieval augments the reader, indicating useful external knowledge is retrieved. On HotpotQA, the standard retrieval hurts the reader. This shows that using complex questions as queries cannot compensate for the parametric knowledge, but bring noises instead. This suggests that multi-hop questions are not suitable queries for the web search engine. The scores increase by adding the rewriting step.

### MMLU results

| Reader and method | Humanities | STEM | Other | Social Sciences |
|---|---:|---:|---:|---:|
| ChatGPT — Direct | 75.6 | 58.8 | 69.0 | 71.6 |
| ChatGPT — Retrieve-then-read | 76.7 | 63.3 | 70.0 | 78.2 |
| ChatGPT — LLM rewriter | 77.0 | 63.5 | 72.6 | 76.4 |
| Vicuna-13B — Direct | 39.8 | 34.9 | 50.2 | 46.6 |
| Vicuna-13B — Retrieve-then-read | 40.2 | 39.8 | 55.2 | 50.6 |
| Vicuna-13B — LLM rewriter | 42.0 | 41.5 | 57.1 | 52.2 |
| Vicuna-13B — Trainable rewriter | **43.2** | 40.9 | **59.3** | 51.2 |

With ChatGPT as a reader, query rewriting improves the scores in most settings, except for the social sciences category. With Vicuna as a reader, the method achieves more gains on the four categories compared to ChatGPT. This agrees with the intuition that a more powerful reader has more parametric memories, thus is more difficult to compensate with external knowledge.

## 6 Analysis

The training process includes two stages, warm-up and reinforcement learning. The validation curves show upward trends with some fluctuations on all the datasets. For multi-hop questions in HotpotQA, the standard retrieval is relatively weaker. Complex questions can be not specific search queries and show a larger gap from rewritten queries. On AmbigNQ and PopQA, the method surpasses the baselines after several iterations (3 or 4). This indicates that the RL training stage can compensate for the insufficiency of the distillation on the pseudo data during warm-up training.

The proposed method is a pipeline framework, instead of an end-to-end system. The query rewriting first affects the retrieved context, then the context makes a difference to the output of the reader. Hence, QA metrics are indirect measurements. After text normalization, the hit rate is computed to measure whether the retrieved context contains the correct answers.

| AmbigNQ method | EM | F1 | Hit ratio |
|---|---:|---:|---:|
| No retrieval | 42.10 | 53.05 | — |
| Upper bound | 58.40 | 69.45 | 100 |
| Retrieve-then-read, snippet | 38.70 | 50.50 | 61.1 |
| Retrieve-then-read, BM25 | 45.80 | 58.50 | 76.4 |
| LLM rewriter, snippet | 39.80 | 52.64 | 63.5 |
| LLM rewriter, BM25 | 46.40 | 58.74 | 77.5 |
| Trainable rewriter, BM25 | **47.80** | **60.71** | **82.2** |

Content selection with BM25 recalls better documents than snippets, while query rewriting makes progress on both settings. The improvement in the hit rate of the retriever is more significant than the improvement in the reader.

## 7 Conclusion

This paper introduces the Rewrite-Retrieve-Read pipeline, where a query rewriting step is added for the retrieval-augmented LLM. This approach is applicable for adopting a frozen large language model as the reader and a real-time web search engine as the retriever. Further, we propose to apply a tuneable small language model as the rewriter, which can be trained to cater to the frozen retriever and reader. The training implementation consists of two stages, warm-up and reinforcement learning. Evaluation and analyses on open-domain QA and multiple-choice QA show the effectiveness of query rewriting. Our work proposes a novel retrieval-augmented black-box LLM framework, proves that the retrieval augmentation can be enhanced from the aspect of query rewriting, and provides a new method for integrating trainable modules into black-box LLMs.

## Limitations

There is still a trade-off between generalization and specialization among downstream tasks. Adding a training process, the scalability to direct transfer is compromised, compared to few-shot in-context learning. The research line of LLM agent has shown impressive performance but relies on multiple calls to the LLM for each sample, where the LLM plays as an agent to flexibly call the retriever multiple times, reads the context in earlier hops, and generates follow-up questions. Different from these studies, our motivation is to enhance the one-turn retriever-then-read framework with a trainable query rewriter. Using a web search engine as the retriever also leads to some limitations. Neural dense retrievers that are based on professional, filtered knowledge bases may potentially achieve better and controllable retrieval.
