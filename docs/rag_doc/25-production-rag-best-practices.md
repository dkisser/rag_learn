# Production RAG Best Practices

> **来源**:
> - https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-customize-rag.html
> - https://www.anthropic.com/news/contextual-retrieval
> - https://www.pinecone.io/learn/series/rag/rerankers/
> - https://www.pinecone.io/learn/series/rag/rag-evaluation/
>
> **抓取日期**: 2026-07-18
> **作者/机构**: Amazon Web Services, Anthropic, Pinecone

This document synthesizes best practices for production RAG (Retrieval Augmented Generation) systems, drawing on AWS, Anthropic, and Pinecone guidance.

---

## Why RAG? (From AWS SageMaker Documentation)

Foundation models are usually trained offline, making the model agnostic to any data that is created after the model was trained. Additionally, foundation models are trained on very general domain corpora, making them less effective for domain-specific tasks. You can use Retrieval Augmented Generation (RAG) to retrieve data from outside a foundation model and augment your prompts by adding the relevant retrieved data in context. (For more information about RAG model architectures, see [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401).)

With RAG, the external data used to augment your prompts can come from multiple data sources, such as document repositories, databases, or APIs. The first step is to convert your documents and any user queries into a compatible format to perform relevancy search. To make the formats compatible, a document collection, or knowledge library, and user-submitted queries are converted to numerical representations using embedding language models. *Embedding* is the process by which text is given numerical representation in a vector space.

RAG model architectures compare the embeddings of user queries within the vector of the knowledge library. The original user prompt is then appended with relevant context from similar documents within the knowledge library. This augmented prompt is then sent to the foundation model. **You can update knowledge libraries and their relevant embeddings asynchronously.**

---

## AWS RAG Architecture Guidance

The retrieved document should be large enough to contain useful context to help augment the prompt, but small enough to fit into the maximum sequence length of the prompt. You can use task-specific JumpStart models, such as the General Text Embeddings (GTE) model from Hugging Face, to provide the embeddings for your prompts and knowledge library documents. After comparing the prompt and document embeddings to find the most relevant documents, construct a new prompt with the supplemental context. Then, pass the augmented prompt to a text generation model of your choosing.

### AWS-Supported RAG Patterns (SageMaker Example Notebooks)

- Retrieval-Augmented Generation: Question Answering using **LangChain and Cohere's** Generate and Embedding Models from SageMaker JumpStart
- Retrieval-Augmented Generation: Question Answering using **Llama-2, Pinecone** and Custom Dataset
- Retrieval-Augmented Generation: Question Answering based on Custom Dataset with **Open-sourced LangChain** Library
- Retrieval-Augmented Generation: Question Answering based on Custom Dataset
- Retrieval-Augmented Generation: Question Answering using **Llama-2 and Text Embedding** Models
- Amazon SageMaker JumpStart — Text Embedding and Sentence Similarity

---

## Anthropic Contextual Retrieval: A Major Finding

Anthropic's research on Contextual Retrieval revealed several production-grade best practices.

### Core Architecture Components

The standard RAG pipeline follows these steps: breaking documents into chunks, creating TF-IDF encodings and semantic embeddings, using BM25 for exact matches, using embeddings for semantic similarity, combining results through rank fusion, then adding top-K chunks to prompts. Combining both approaches **"more accurately retrieves the most applicable chunks."**

### Embedding Model Selection

The research found that **"Voyage and Gemini have the best embeddings"** among tested options. Contextual Retrieval improved performance across all embedding models tested, though **"some models may benefit more than others."** The article specifically recommends **Gemini Text 004** and **Voyage embeddings** for top performance.

### Vector Database & Storage

Vector databases store embeddings for semantic similarity search at runtime. The system retrieves the most relevant chunks based on similarity to user queries.

### Chunking Strategies

Chunk size, boundaries, and overlap all affect retrieval performance. The article recommends breaking documents into chunks of **"no more than a few hundred tokens."** Anthropic found **"using 20 to be the most performant"** compared to 5 or 10 chunks. Adding more chunks increases the chance of including relevant information but risks distracting the model.

### Key Recommendations

- **Combine approaches**: "Embeddings + BM25 is better than embeddings on their own"
- **Use prompt caching**: Reduces Contextual Retrieval costs significantly (~$1.02 per million document tokens)
- **Consider reranking**: "Reranking is better than no reranking"
- **Run evals**: Test with your specific domain data
- **Custom prompts**: Tailor contextualizer prompts to your use case for better results

### Performance Summary

Best results came from combining:
- Contextual embeddings (Voyage or Gemini)
- Contextual BM25
- Reranking
- Passing 20 chunks to the model

Achieving a **67% reduction in retrieval failures**.

---

## Two-Stage Retrieval with Rerankers (Pinecone Best Practice)

### Two-Stage Retrieval

The recommended pattern is a two-stage system:
1. **Fast first-stage retriever** (bi-encoder/embedding model) retrieves a broad set of candidate documents
2. **Second-stage reranker** reorders those documents for maximum relevance

### Reranking Strategy

- **Retrieve more documents than needed** (e.g., `top_k=25`) to maximize retrieval recall
- **Use a reranker** to reorder and keep only the most relevant (e.g., `top_n=3-5`) for the LLM
- This approach maximizes both retrieval recall and LLM recall by minimizing noise in the context window

### Why Rerankers Outperform Bi-Encoders

Rerankers (cross-encoders) process the query and document together at inference time, **avoiding the information compression** that occurs when bi-encoders must embed all possible document meanings into a single vector.

### Important Limitation

The "Lost in the Middle" problem: **context stuffing** (filling the context window with more documents) degrades LLM performance. Position matters — keep relevant chunks at the beginning or end of the prompt.

---

## Evaluation Framework (Pinecone)

Use a combination of retrieval and end-to-end metrics:

### Retrieval Metrics

- **Precision@k** — Of the top-k retrieved documents, how many are relevant?
- **Recall@k** — Of all relevant documents, how many appear in top-k?
- **F1@k** — Harmonic mean of precision and recall at k
- **MRR** (Mean Reciprocal Rank) — How high does the first relevant result appear?
- **AP** (Average Precision) — Considers rank of all relevant results
- **DCG / NDCG** (Discounted Cumulative Gain) — Rewards putting most relevant results higher

### End-to-End / LLM-as-Judge Metrics

- **Answer Relevance** — Does the response address the question?
- **Faithfulness / Groundedness** — Is the answer supported by retrieved context?
- **Context Relevance** — Are retrieved contexts relevant to the query?
- **Citation Accuracy** — Are cited sources correct?

### Popular Frameworks

- **Arize Phoenix**
- **ARES** (Automated RAG Evaluation System)
- **RAGAS** (Retrieval Augmented Generation Assessment)
- **TruLens** — RAG Triad (Context Relevance, Groundedness, Answer Relevance)
- **Galileo**

---

## Comprehensive Best Practices Checklist

### 1. Chunking

- Use chunks of **a few hundred tokens** (Anthropic recommendation)
- Experiment with chunk size, boundary strategy, and overlap
- Consider semantic chunking (sentence/paragraph boundaries) over fixed-size
- For complex documents (tables, code), use structure-aware chunking
- Include overlap to avoid losing context at chunk boundaries

### 2. Embeddings

- Choose strong embedding models (**Voyage**, **Gemini Text 004**, OpenAI `text-embedding-3-large`, Cohere `embed-v3`)
- Match embedding model to your domain
- Consider **multilingual models** if documents span languages
- For long documents, use models with larger context windows
- Re-embed documents when changing embedding models

### 3. Retrieval Strategy

- **Combine BM25 + dense embeddings** (hybrid retrieval) — strictly better than either alone
- Retrieve a wide candidate set (e.g., top-25 to top-100)
- Use **two-stage retrieval**: fast bi-encoder → accurate cross-encoder reranker
- Apply **metadata filtering** (date, source, document type) before or after vector search
- Consider **multi-query retrieval** (generate query variants, merge results)
- For complex queries, use **HyDE** (Hypothetical Document Embeddings) or **step-back prompting**

### 4. Contextual Retrieval (Anthropic)

- Add **contextual prefixes** to chunks before embedding (e.g., "This chunk is from section X of document Y about Z...")
- Combine with **contextual BM25** for hybrid gains
- Use **prompt caching** to reduce cost (~$1.02 per million document tokens)
- **67% retrieval failure reduction** when properly implemented

### 5. Generation

- Pass only top-3 to top-5 chunks to the LLM (avoid "lost in the middle")
- Include **explicit citations** in the prompt
- Add **system prompt instructions** to ground answers in retrieved context
- Use **structured output** (JSON schemas) when possible
- Implement **answer verification** (does the answer actually use the retrieved context?)

### 6. Evaluation

- Build a **representative test set** early
- Measure both **retrieval metrics** (Precision@k, NDCG) and **end-to-end metrics** (faithfulness, relevance)
- Use **LLM-as-judge** for subjective dimensions
- Track **business metrics** alongside technical metrics
- Run **regression tests** on every prompt or pipeline change
- Monitor **production traffic** and sample for ongoing evaluation

### 7. Production Operations

- **Asynchronous embedding pipeline** for knowledge base updates
- **Version your indexes** to enable rollback
- **Monitor latency** at each stage (retrieval, reranking, generation)
- **Log retrieval results** for offline analysis and debugging
- **Cache** frequent queries at retrieval or generation level
- **Set up alerts** for embedding drift, retrieval quality degradation
- **Implement fallbacks** (e.g., return empty context + "I don't know" when retrieval confidence is low)

### 8. Cost Optimization

- Use **smaller embedding models** for initial candidate retrieval, larger models for reranking
- Apply **prompt caching** (especially for contextual prefixes)
- **Batch embedding jobs** during index updates
- Consider **quantized embeddings** (int8) for storage cost reduction
- Use **metadata filters** to reduce search space before vector similarity

### 9. Security & Privacy

- **Encrypt** embeddings at rest and in transit
- Apply **document-level access control** via metadata filtering
- **Audit** retrieval logs for compliance
- **Redact PII** before embedding
- Use **private deployment** for sensitive knowledge bases

### 10. Advanced Patterns

- **GraphRAG**: Combine knowledge graphs with vector retrieval for entity-relationship reasoning
- **ColPali / multi-modal RAG**: Index document page images directly for visually rich documents
- **Agentic RAG**: Let an agent decide when/what to retrieve, with multiple retrieval tools
- **Self-RAG**: Have the LLM critique its own retrieval and regenerate
- **CRAG** (Corrective RAG): Detect and correct low-quality retrievals before generation
- **Multi-hop RAG**: Decompose complex questions, retrieve iteratively

---

## Summary of Key Production Insights

| Best Practice | Source | Impact |
|---------------|--------|--------|
| Chunk size "no more than a few hundred tokens" | Anthropic | Reduces noise, improves focus |
| Pass 20 chunks before reranking | Anthropic | Better recall before reranking |
| Hybrid (BM25 + embeddings) > either alone | Anthropic | Strict improvement |
| Top embedding models: Voyage, Gemini | Anthropic | Best raw retrieval quality |
| Two-stage retrieval (retrieve 25, rerank to top 5) | Pinecone | Maximizes recall + precision |
| Cross-encoder rerankers > bi-encoders alone | Pinecone | Avoids embedding compression |
| Combine Contextual Retrieval + reranking + 20 chunks | Anthropic | 67% reduction in retrieval failures |
| Asynchronous embedding updates | AWS | Decouples ingestion from serving |
| LLM-as-judge for end-to-end metrics | Pinecone | Scalable subjective evaluation |
| Lost in the middle problem | Pinecone | Don't stuff the context window |

The best production RAG systems are **evaluated continuously**, **use hybrid retrieval**, **incorporate reranking**, **respect chunk size limits**, and **are monitored** for both technical and business metrics.
