# Precise Zero-Shot Dense Retrieval without Relevance Labels

> **来源**: https://arxiv.org/abs/2212.10496
> **抓取日期**: 2026-07-18
> **作者/机构**: Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan — Carnegie Mellon University and University of Waterloo

## Abstract

While dense retrieval has been shown effective and efficient across tasks and languages, it remains difficult to create effective fully zero-shot dense retrieval systems when no relevance label is available. In this paper, we recognize the difficulty of zero-shot learning and encoding relevance. Instead, we propose to pivot through Hypothetical Document Embeddings (HyDE). Given a query, HyDE first zero-shot instructs an instruction-following language model (e.g. InstructGPT) to generate a hypothetical document. The document captures relevance patterns but is unreal and may contain false details. Then, an unsupervised contrastively learned encoder (e.g. Contriever) encodes the document into an embedding vector. This vector identifies a neighborhood in the corpus embedding space, where similar real documents are retrieved based on vector similarity. This second step grounds the generated document to the actual corpus, with the encoder’s dense bottleneck filtering out the incorrect details. Our experiments show that HyDE significantly outperforms the state-of-the-art unsupervised dense retriever Contriever and shows strong performance comparable to fine-tuned retrievers, across various tasks (e.g. web search, QA, fact verification) and languages (e.g. sw, ko, ja).

## 1 Introduction

Dense retrieval, the method of retrieving documents using semantic embedding similarities, has been shown successful across tasks like web search, question answering, and fact verification. A variety of methods such as negative mining, distillation and task-specific pre-training have been proposed to improve the effectiveness of supervised dense retrieval models.

On the other hand, zero-shot dense retrieval still remains difficult. Many recent works consider the alternative transfer learning setup, where the dense retrievers are trained on a high-resource dataset and then evaluated on queries from new tasks. The MS-MARCO collection, a massive judged dataset with a large number of judged query-document pairs, is arguably the most commonly used. In practice, however, the existence of such a large dataset cannot always be assumed. Even MS-MARCO restricts commercial use and cannot be adopted in a variety of real-world search scenarios.

In this paper, we aim to build effective fully zero-shot dense retrieval systems that require no relevance supervision, work out-of-box and generalize across tasks. As supervision is not available, we start by examining self-supervised representation learning methods. Modern deep learning enables two distinct learning algorithms. At the token level, generative large language models (LLM) pre-trained on large corpus have demonstrated strong natural language understanding (NLU) and generation (NLG) capabilities. At the document level, text (chunk) encoders pre-trained with contrastive objectives learn to encode document-document similarity into inner-product. On top of these, one extra insight into LLM is borrowed: the LLMs further trained to follow instructions can zero-shot generalize to diverse unseen instructions.

With these ingredients, we propose to pivot through Hypothetical Document Embeddings (HyDE), and decompose dense retrieval into two tasks, a generative task performed by an instruction-following language model and a document-document similarity task performed by a contrastive encoder.

First, we feed the query to the generative model and instruct it to “write a document that answers the question”, i.e. a hypothetical document. We expect the generative process to capture “relevance” by giving an example; the generated document is not real, can contain factual errors but is like a relevant document. In the second step, we use an unsupervised contrastive encoder to encode this document into an embedding vector. Here, we expect the encoder’s dense bottleneck to serve a lossy compressor, where the extra (hallucinated) details are filtered out from the embedding. We use this vector to search against the corpus embeddings. The most similar real documents are retrieved and returned. The retrieval leverages document-document similarity encoded in the inner-product during contrastive training.

Note that, interestingly, with HyDE factorization, the query-document similarity score is no longer explicitly modeled nor computed. Instead, the retrieval task is cast into two NLU and NLG tasks. HyDE appears unsupervised. No model is trained in HyDE: both the generative model and the contrastive encoder remain intact. Supervision signals were only involved in instruction learning of our backbone LLM.

In our experiments, we show HyDE using InstructGPT and Contriever as backbone models significantly outperforms the previous state-of-the-art Contriever-only zero-shot no-relevance system on 11 query sets, covering tasks like Web Search, Question Answering, Fact Verification and languages like Swahili, Korean, Japanese.

## Figure 1: An illustration of the HyDE model

The figure traces three representative queries through the same pipeline:

- For `how long does it take to remove wisdom tooth`, the instruction is `write a passage to answer the question`; the generated document begins “It usually takes between 30 minutes and two hours to remove a wisdom tooth...”, while a retrieved real document begins “How wisdom teeth are removed... Some ... a few minutes, whereas others can take 20 minutes or longer....”
- For `How has the COVID-19 pandemic impacted mental health?`, the instruction is `write a scientific paper passage to answer the question`; the generated document mentions that “depression and anxiety had increased by 20% since the start of the pandemic...”, and retrieval grounds it in real documents discussing depressive symptoms in COVID-19 patients.
- For a Korean query asking when humans began to use fire, the instruction asks for a detailed passage in Korean. The generated answer and the retrieved real document may disagree in exact dates, illustrating how the dense bottleneck can discard incorrect details while preserving a useful semantic neighborhood.

HyDE serves all types of queries without changing the underlying GPT-3 and Contriever/mContriever models.

## 3 Methodology

### 3.1 Preliminaries

Dense retrieval models similarity between query and document with inner product similarity. Given a query $q$ and document $d$, it uses two encoder functions $enc_q$ and $enc_d$ to map them into $d$-dimension vectors $v_q$, $v_d$, whose inner product is used as similarity measurement:

$$
sim(q,d) = \langle enc_q(q), enc_d(d) \rangle = \langle v_q, v_d \rangle. \tag{1}
$$

For zero-shot retrieval, we consider $L$ query sets $Q_1, Q_2, \ldots, Q_L$ and their corresponding search corpus document sets $D_1, D_2, \ldots, D_L$. Denote the $j$-th query from the $i$-th query set $Q_i$ as $q_{ij}$. We need to fully define mapping functions $enc_q$ and $enc_d$ without access to any query set $Q_i$, document set $D_i$, or any relevance judgment $r_{ij}$.

The difficulty of zero-shot dense retrieval lies precisely in Equation 1: it requires learning of two embedding functions (for query and document respectively) into the same embedding space where inner product captures relevance. Without relevance judgments/scores to fit, learning becomes intractable.

### 3.2 HyDE

HyDE circumvents the aforementioned learning problem by performing search in document-only embedding space that captures document-document similarity. This can be easily learned using unsupervised contrastive learning. We set document encoder $enc_d$ directly as a contrastive encoder $enc_{con}$:

$$
f = enc_d = enc_{con}. \tag{2}
$$

This function is also denoted as $f$ for simplicity. This unsupervised contrastive encoder will be shared by all incoming document corpora:

$$
v_d = f(d), \qquad \forall d \in D_1 \cup D_2 \cup \ldots \cup D_L. \tag{3}
$$

To build the query vector, we consider in addition an instruction-following LM, $InstructLM$. It takes a query $q$ and a textual instruction $INST$ and follows them to perform the task specified by $INST$. For simplicity, denote:

$$
g(q, INST) = InstructLM(q, INST). \tag{4}
$$

Now we can use $g$ to map queries to “hypothetical” documents by sampling from $g$, setting $INST$ to be “write a paragraph that answers the question”. The generated document is not real, can and is likely to be ungrounded factually. We only require it to capture relevance pattern. This is done by generating documents, i.e. providing examples. Critically, here we offload relevance modeling from representation learning model to an NLG model that generalizes significantly more easily, naturally, and effectively. Generating examples also replaces explicit modeling of relevance scores.

We can now encode the generated document using the document encoder $f$:

$$
E[v_{q_{ij}}] = E[f(g(q_{ij}, INST_i))]. \tag{5}
$$

Formally, $g$ defines a probability distribution based on the chain rule. In this paper, we simply consider the expectation value, assuming the distribution of $v_{q_{ij}}$ is unimodal, i.e. the query is not ambiguous. The study of ambiguous queries and diversity is left to future work. We estimate Equation 5 by sampling $N$ documents from $g$, $[\hat d_1, \hat d_2, \ldots, \hat d_N]$:

$$
\hat v_{q_{ij}} = \frac{1}{N}\sum_{\hat d_k \sim g(q_{ij}, INST_i)} f(\hat d_k)
= \frac{1}{N}\sum_{k=1}^{N} f(\hat d_k). \tag{6–7}
$$

We also consider the query as a possible hypothesis:

$$
\hat v_{q_{ij}} = \frac{1}{N+1}\left[\sum_{k=1}^{N}f(\hat d_k) + f(q_{ij})\right]. \tag{8}
$$

Inner product is computed between $\hat v_{q_{ij}}$ and the set of all document vectors $\{f(d) \mid d \in D_i\}$. The most similar documents are retrieved. Here the encoder function $f$ serves as a lossy compressor that outputs dense vectors, where the extra details are filtered and left out from the vector. It further grounds the hypothetical vector to the actual corpus and the real documents.

## 4 Experiments

### 4.1 Setup

We implement HyDE using InstructGPT, a GPT-3 model from the instruct series (`text-davinci-003`) and Contriever models. We sample from InstructGPT using the OpenAI playground default temperature of 0.7 for open-ended generations. We use the English-only Contriever model for English retrieval tasks and multilingual mContriever for non-English tasks. We conducted retrieval experiments with the Pyserini toolkit.

We consider web search query sets TREC DL19 and DL20; they are based on the MS-MARCO dataset. We also use a diverse collection of six low-resource datasets from the BEIR dataset. For non-English retrieval, we consider Swahili, Korean, Japanese, and Bengali from the Mr.TyDi dataset.

Contriever models, Contriever and mContriever, serve as our major baseline. They are trained using unsupervised contrastive learning. HyDE retrievers share the exact same embedding spaces with them. The only difference is how the query vector is built. These comparisons allow us to easily examine the effect of HyDE. The classical heuristic-based lexical retriever BM25 is also included.

### 4.2 Web Search

| System | DL19 MAP | DL19 nDCG@10 | DL19 Recall@1k | DL20 MAP | DL20 nDCG@10 | DL20 Recall@1k |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 30.1 | 50.6 | 75.0 | 28.6 | 48.0 | 78.6 |
| Contriever | 24.0 | 44.5 | 74.6 | 24.0 | 42.1 | 75.4 |
| **HyDE** | **41.8** | **61.3** | **88.0** | **38.2** | **57.9** | **84.4** |
| DPR | 36.5 | 62.2 | 76.9 | 41.8 | 65.3 | 81.4 |
| ANCE | 37.1 | 64.5 | 75.5 | 40.8 | 64.6 | 77.6 |
| ContrieverFT | 41.7 | 62.1 | 83.6 | 43.6 | 63.2 | 85.8 |

HyDE brings sizable improvements to Contriever across the board for both precision-oriented and recall metrics. While unsupervised Contriever can underperform the classical BM25 approach, HyDE outperforms BM25 by large margins. HyDE remains competitive even when compared to fine-tuned models.

### 4.3 Low Resource Retrieval

| System | SciFact | ArguAna | TREC-COVID | FiQA | DBPedia | TREC-NEWS |
|---|---:|---:|---:|---:|---:|---:|
| BM25 nDCG@10 | 67.9 | 39.7 | 59.5 | 23.6 | 31.8 | 39.5 |
| Contriever nDCG@10 | 64.9 | 37.9 | 27.3 | 24.5 | 29.2 | 34.8 |
| **HyDE nDCG@10** | **69.1** | **46.6** | **59.3** | **27.3** | **36.8** | **44.0** |
| BM25 Recall@100 | 92.5 | 93.2 | 49.8 | 54.0 | 46.8 | 44.7 |
| Contriever Recall@100 | 92.6 | 90.1 | 17.2 | 56.2 | 45.3 | 42.3 |
| **HyDE Recall@100** | **96.4** | **97.9** | **41.4** | **62.1** | **47.2** | **50.9** |

HyDE again brings sizable improvements to Contriever across the board in terms of both nDCG and recall. HyDE is only outperformed by BM25 on one dataset, TREC-Covid, with a tiny 0.2 margin; in comparison, the underlying Contriever underperforms by more than 50%. HyDE generally shows better performance than ANCE and DPR, even though the two are fine-tuned on MS-MARCO.

### 4.4 Multilingual Retrieval

| System | Swahili | Korean | Japanese | Bengali |
|---|---:|---:|---:|---:|
| BM25 | 38.9 | 28.5 | 21.2 | 41.8 |
| mContriever | 38.3 | 22.3 | 19.5 | 35.3 |
| **HyDE** | **41.7** | **30.6** | **30.7** | **41.3** |
| mContrieverFT | 51.2 | 34.2 | 32.4 | 42.3 |

Multilingual setup poses several additional challenges to HyDE. The small-sized contrastive encoder gets saturated as the number of languages scales. Meanwhile, the generative LLM faces an opposite issue: with languages not as high resource as English or French, the high-capacity LLM can get under-trained. Nevertheless, HyDE is still able to improve the mContriever model.

## 5 Analysis

### 5.1 Effect of Different Generative Models

| Model | DL19 nDCG@10 | DL20 nDCG@10 |
|---|---:|---:|
| Contriever | 44.5 | 42.1 |
| ContrieverFT | 62.1 | 63.2 |
| HyDE with Contriever and Flan-T5 (11B) | 48.9 | 52.9 |
| HyDE with Contriever and Cohere (52B) | 53.8 | 53.8 |
| HyDE with Contriever and GPT (175B) | **61.3** | **57.9** |
| HyDE with ContrieverFT and Flan-T5 (11B) | 60.2 | 62.1 |
| HyDE with ContrieverFT and Cohere (52B) | 61.4 | 63.1 |
| HyDE with ContrieverFT and GPT (175B) | **67.4** | **63.5** |

Generally, all models bring improvement to the unsupervised Contriever, with larger models bringing larger improvements.

### 5.2 HyDE with Fine-tuned Encoder

HyDE with a fine-tuned encoder is not the intended usage: HyDE is more powerful and irreplaceable when few relevance labels are present. Less powerful instruction LMs can negatively impact the overall performance of the fine-tuned retriever. The performance degradations remain small. On the other hand, the InstructGPT model is able to further bring up the performance, especially on DL19. This suggests that there may still exist certain factors not captured by the fine-tuned encoder but only by the generative model.

## 6 Conclusion

The concept of relevance in HyDE is captured by an NLG model and the language generation process. We demonstrate in many cases, HyDE can be as effective as dense retrievers that learn to model numerical relevance scores. So, is numerical relevance just a statistical artifact of language understanding? Will a weak retriever theoretically suffice as the NLU and NLG models rapidly become stronger? Rushing to conclusions is not smart; more works need to be done to get answers. With this paper, we just want to raise these questions.

Concretely in this paper, we introduce a new paradigm of interactions between LLM and dense encoder/retriever. We demonstrate (part of) relevance modeling and instruction understanding can be delegated to the more powerful and flexible LLM. As a consequence, the need for relevance labels is removed. We are excited to see how this can be generalized further to more sophisticated tasks like multi-hop retrieval/QA and conversational search.

We argue HyDE is also of practical use though not necessarily over the entire lifespan of a search system. At the very beginning of the life of the search system, serving queries using HyDE offers performance comparable to a fine-tuned model, which no other relevance-free model can offer. As the search log grows, a supervised dense retriever can be gradually rolled out. As the dense retriever grows stronger, more queries will be routed to it, with only less common and emerging ones going to HyDE backend.
