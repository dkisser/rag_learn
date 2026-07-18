# Retrieval-Augmented Generation (RAG) Architecture Overview

> **来源**: https://aws.amazon.com/what-is/retrieval-augmented-generation/ + https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-customize-rag.html
> **抓取日期**: 2026-07-18
> **作者/机构**: Amazon Web Services (AWS) — official documentation on Retrieval-Augmented Generation

## 1. Definition

**Retrieval-Augmented Generation (RAG)** is an architecture that augments a foundation model's prompt with **relevant data retrieved from outside the model** at inference time. The retrieved context provides the model with up-to-date, domain-specific information that was not part of its original training set.

> AWS definition (SageMaker JumpStart):
> "You can use Retrieval Augmented Generation (RAG) to retrieve data from outside a foundation model and augment your prompts by adding the relevant retrieved data in context."

## 2. The RAG Architecture Diagram (ASCII)

```
                       ┌──────────────────────────────────────────────┐
                       │           EXTERNAL KNOWLEDGE BASE            │
                       │  (Documents, databases, APIs, web pages)     │
                       └────────────────────┬─────────────────────────┘
                                            │
                                            ▼
        ┌───────────────────────────────────────────────────────────────┐
        │                  INGESTION PIPELINE (offline)                 │
        │                                                               │
        │   Documents ──► Chunking ──► Embedding Model ──► Vector DB    │
        └───────────────────────────────────────────────────────────────┘
                                            ▲
                                            │
                       ┌────────────────────┴─────────────────────────┐
                       │              USER QUERY  (Q)                 │
                       └────────────────────┬─────────────────────────┘
                                            ▼
        ┌───────────────────────────────────────────────────────────────┐
        │                  RETRIEVAL PIPELINE (online)                  │
        │                                                               │
        │   Q ──► Embedding Model ──► Vector Search ──► Top-K Chunks    │
        └───────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
        ┌───────────────────────────────────────────────────────────────┐
        │              AUGMENTATION & GENERATION PIPELINE               │
        │                                                               │
        │   [Top-K Context] + Q  ──►  Prompt  ──►  Foundation Model    │
        │                                       ──►  Generated Answer   │
        └───────────────────────────────────────────────────────────────┘
```

## 3. Core Components

A RAG system is composed of three logical pipelines and several key building blocks.

### 3.1 Data Ingestion Pipeline (Offline / Indexing)

Converts external data into a form the system can search over.

| Step | Description |
|------|-------------|
| **Document loading** | Load documents from various sources (S3, databases, wikis, web crawlers, PDFs, Office files). |
| **Chunking** | Split documents into smaller passages (typically 100–500 tokens) with optional overlap to preserve context across boundaries. |
| **Embedding** | Convert each chunk into a dense vector using an embedding model (e.g., Amazon Titan Embeddings, Hugging Face GTE, Cohere Embed). |
| **Indexing** | Store vectors in a vector database (FAISS, Pinecone, Weaviate, Chroma, OpenSearch, pgvector) along with metadata. |

> AWS:
> "The first step is to convert your documents and any user queries into a compatible format to perform relevancy search. To make the formats compatible, a document collection, or knowledge library, and user-submitted queries are converted to numerical representations using embedding language models."

### 3.2 Retrieval Pipeline (Online / Query Time)

Finds the most relevant chunks for a given query.

| Step | Description |
|------|-------------|
| **Query embedding** | Encode the user query into the same vector space as the indexed documents. |
| **Similarity search** | Compute the similarity (cosine, dot product, Euclidean) between the query vector and all chunk vectors. Use approximate nearest neighbor (ANN) algorithms for speed. |
| **Re-ranking (optional)** | Apply a cross-encoder or LLM-based re-ranker to the top-K candidates for higher precision. |
| **Filtering (optional)** | Apply metadata filters (date, source, document type) to narrow the candidate set. |

### 3.3 Augmentation & Generation Pipeline

Combines retrieved context with the query and produces the final answer.

| Step | Description |
|------|-------------|
| **Prompt construction** | Concatenate retrieved chunks with the user query inside a prompt template. |
| **LLM inference** | Send the augmented prompt to the foundation model (BART, T5, Llama, Claude, GPT, etc.). |
| **Post-processing** | Parse the output, attach citations, format the response, and return to the user. |

## 4. The Embedding Process

**Embedding** is the transformation that makes retrieval possible.

> AWS:
> "Embedding is the process by which text is given numerical representation in a vector space."

Key properties:

- **Dimensionality**: Modern embeddings are typically 384 to 4096 dimensions.
- **Semantic locality**: Semantically similar texts map to nearby vectors (high cosine similarity).
- **Asymmetric encoding**: Query and document encoders may differ (as in DPR) for better retrieval quality.

Example:

```
"How do I reset my password?"  ──►  [0.12, -0.45, 0.78, ..., 0.33]   (1024-dim)
"To change your password, ..." ──►  [0.15, -0.42, 0.81, ..., 0.36]   (high similarity)
"What is the capital of France?" ──► [0.89, 0.21, -0.34, ..., -0.12]  (low similarity)
```

## 5. Augmentation Patterns

Several patterns exist for integrating retrieved context with the LLM:

| Pattern | Description |
|---------|-------------|
| **Naive concatenation** | Append all retrieved chunks to the prompt; the LLM reads them as flat context. |
| **Re-ranking + top-K** | Re-rank chunks and keep only the best N to fit the context window. |
| **Iterative retrieval** | Retrieve → generate → retrieve again with refined query (multi-hop). |
| **Step-back prompting** | Retrieve background context, then retrieve specific evidence. |
| **Self-RAG** | LLM emits reflection tokens that decide when/what to retrieve. |
| **Corrective RAG (CRAG)** | Detect low-quality retrievals and fall back to web search or alternative sources. |
| **GraphRAG** | Retrieve subgraphs from a knowledge graph to ground relational reasoning. |

## 6. Data Updates

One of RAG's key advantages is that the knowledge base can be updated **without retraining the model**:

> AWS:
> "You can update knowledge libraries and their relevant embeddings asynchronously."

Common update strategies:

- **Real-time streaming**: New documents are chunked, embedded, and indexed immediately.
- **Periodic batch updates**: A scheduler rebuilds the index on a fixed cadence (hourly, daily).
- **Incremental updates**: Only new or modified chunks are added; outdated ones are soft-deleted or versioned.
- **Hybrid**: Critical updates are streamed in real-time; less critical ones batched.

## 7. Common Pitfalls & Design Considerations

### 7.1 Retrieval Quality

- **Top-K choice**: Too small → misses relevant context; too large → noisy context and overflows LLM window.
- **Chunking strategy**: Naive fixed-size chunking can break semantic units (e.g., split a paragraph mid-thought).
- **Embedding model mismatch**: Using different encoders for indexing and querying ruins retrieval quality.

### 7.2 Generation Quality

- **Lost-in-the-middle**: LLMs pay more attention to context at the start and end of the prompt. Place the most relevant chunks there.
- **Context window overflow**: Long retrieved contexts can exceed the LLM's max input length. Apply re-ranking + compression.
- **Faithfulness**: LLMs may still hallucinate or contradict retrieved evidence. Explicit prompting ("answer only based on the context below") helps.

### 7.3 Operational

- **Latency**: Retrieval adds 100ms–1s to inference. For latency-sensitive apps, use smaller embedding models or precomputed caches.
- **Cost**: Embedding API calls, vector DB hosting, and LLM tokens all scale with usage.
- **Security**: Retrieved content can be poisoned; production systems should validate and sanitize inputs.

## 8. AWS RAG Architecture (SageMaker JumpStart)

The official AWS reference architecture for RAG on SageMaker JumpStart:

> "RAG model architectures compare the embeddings of user queries within the vector of the knowledge library. The original user prompt is then appended with relevant context from similar documents within the knowledge library. This augmented prompt is then sent to the foundation model."

Typical AWS stack:

```
S3 (documents) ──► SageMaker Processing Job (chunking + embedding)
                                          │
                                          ▼
                              OpenSearch / FAISS (vector index)
                                          │
User Query ──► API Gateway ──► Lambda ──► SageMaker Endpoint (embedding)
                                          │
                                          ▼
                            Vector Search (top-K chunks)
                                          │
                                          ▼
                          Prompt Construction + Foundation Model
                                          │
                                          ▼
                                       Response
```

## 9. Component Summary Table

| Component | Purpose | AWS Examples |
|-----------|---------|--------------|
| **Document store** | Source documents | S3, FSx |
| **Embedding model** | Text → vectors | Amazon Titan, Cohere Embed, Hugging Face GTE |
| **Vector database** | Similarity search | OpenSearch, Aurora pgvector, FAISS on SageMaker |
| **Retriever** | Top-K chunk selection | Custom logic on Lambda / SageMaker |
| **Foundation model** | Answer generation | Llama-2, Mistral, Cohere Command, Anthropic Claude on Bedrock |
| **Orchestrator** | Pipeline glue | LangChain, LlamaIndex, custom code |
| **Observability** | Monitoring | CloudWatch, SageMaker Clarify, custom logs |

## 10. Summary

- A RAG system has **three pipelines**: ingestion (offline), retrieval (online), and generation (online).
- The core trick is **embedding**: converting text into vectors so similarity can be computed mathematically.
- The architecture is **modular**: any component can be swapped independently (e.g., change embedding model without retraining the LLM).
- **Knowledge updates** are decoupled from model training — index updates are nearly free, model retraining is not.
- The pattern is **technology-agnostic**: it works with any LLM, any embedding model, and any vector database.

## 11. References

- AWS — What is Retrieval-Augmented Generation? — https://aws.amazon.com/what-is/retrieval-augmented-generation/
- AWS SageMaker JumpStart — Customize RAG — https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-customize-rag.html
- Lewis et al. 2020 — RAG original paper — https://arxiv.org/abs/2005.11401
- Gao et al. 2024 — RAG Survey — https://arxiv.org/abs/2312.10997
