# Retrieval-Augmented Generation (RAG) on AWS — Architecture & When to Choose RAG vs Fine-tuning

> **来源**: https://aws.amazon.com/what-is/retrieval-augmented-generation/ + https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-customize-rag.html
> **抓取日期**: 2026-07-18
> **作者/机构**: Amazon Web Services (AWS) — official documentation
> **关联上下文**: 该内容综合自 AWS 官方 RAG 解释页面与 Amazon SageMaker JumpStart RAG 章节，是 AWS 关于 RAG 与 fine-tuning 选择的权威介绍

## 1. What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation (RAG) is an architecture pattern that augments a foundation model's prompt with **relevant data retrieved from an external knowledge source** at inference time. The retrieved context grounds the model's response in up-to-date, domain-specific information that may not have been present in the original training data.

### Why RAG Matters

Foundation models are typically:

- **Trained offline**: The model is agnostic to any data created after the training cut-off.
- **Trained on general corpora**: Less effective for domain-specific tasks without augmentation.

RAG addresses both limitations by retrieving external data on demand and injecting it into the prompt.

> AWS SageMaker documentation (JumpStart RAG chapter):
> "You can use Retrieval Augmented Generation (RAG) to retrieve data from outside a foundation model and augment your prompts by adding the relevant retrieved data in context."

## 2. How RAG Works — The Three-Stage Pipeline

### Stage 1: Data Ingestion (Create External Data)

External data sources — document repositories, databases, APIs, knowledge bases — are converted into a form the model can consume:

1. **Chunking**: Documents are split into smaller, semantically coherent passages (typically 100–500 tokens).
2. **Embedding**: Each chunk is converted into a **dense vector** representation using an embedding model (e.g., Amazon Titan Embeddings, OpenAI Ada, Hugging Face GTE, Cohere Embed).
3. **Indexing**: The vectors are stored in a **vector database** (e.g., Amazon OpenSearch, Pinecone, Weaviate, Chroma, FAISS) that supports fast similarity search.

> AWS: "Embedding is the process by which text is given numerical representation in a vector space. RAG model architectures compare the embeddings of user queries within the vector of the knowledge library."

### Stage 2: Retrieval

When a user submits a query:

1. The query is encoded into a vector using the **same embedding model** used during indexing.
2. The vector database performs a **similarity search** (typically cosine similarity or dot product) to find the top-K most relevant chunks.
3. The retrieved chunks are returned as candidate context.

### Stage 3: Generation (Augmenting the Prompt)

The retrieved context is combined with the original user query through **prompt engineering**:

```
[Retrieved Context]
{chunk_1}
{chunk_2}
...
{chunk_K}

[User Query]
{query}

[Generated Answer]
```

The augmented prompt is then sent to the foundation model, which generates a response that is grounded in both its parametric knowledge and the retrieved evidence.

## 3. RAG vs. Fine-Tuning — When to Use Each

RAG and fine-tuning are **complementary techniques** that solve different problems. The choice depends on what you need to achieve.

### 3.1 Comparison Table

| Dimension | RAG | Fine-Tuning |
|-----------|-----|-------------|
| **Primary use case** | Injecting external / up-to-date knowledge | Teaching a specific style, format, or task pattern |
| **Knowledge update** | Update the vector index — no retraining needed | Retrain the model on new data — expensive |
| **Data freshness** | Near real-time (depends on indexing pipeline) | Frozen at training time |
| **Cost profile** | Low upfront; ongoing embedding + retrieval costs | High upfront (GPU training); low per-query cost |
| **Latency** | Higher (extra retrieval round-trip) | Lower (single forward pass) |
| **Provenance / citations** | Native — outputs trace to source documents | Difficult — knowledge is implicit in weights |
| **Customization of style** | Limited (prompt-level only) | Strong (model learns the style directly) |
| **Hallucination control** | Strong — model is grounded in retrieved facts | Weaker — model may still hallucinate |
| **Privacy / proprietary data** | Index data without training the base model | Requires training on the proprietary data |
| **Domain adaptation depth** | Surface-level (facts in context) | Deep (model internalizes patterns) |
| **Scaling to large knowledge** | Easy — just index more documents | Expensive — retrain to absorb new knowledge |
| **Reversibility** | Easy — swap or remove documents | Difficult — must retrain to undo |

### 3.2 When to Choose RAG

RAG is the right choice when:

- **Knowledge changes frequently** (news, product catalogs, internal documentation, customer support knowledge bases).
- You need **traceable outputs** with citations to source documents (legal, medical, compliance use cases).
- The knowledge base is **large or dynamic**, making retraining impractical.
- You need to **avoid hallucinations** by grounding responses in retrieved evidence.
- **Time-to-deployment** matters — RAG can be operational in hours; fine-tuning takes days/weeks.
- The use case is **knowledge-intensive Q&A**, retrieval-augmented summarization, or chat over documents.

### 3.3 When to Choose Fine-Tuning

Fine-tuning is the right choice when:

- You need the model to learn a **specific writing style, tone, or output format** (e.g., legal contracts, marketing copy, code generation in a particular style).
- The task requires **internalized reasoning patterns** that benefit from gradient updates.
- The model must perform well on **specialized tasks with stable patterns** (e.g., classification, extraction with consistent schemas).
- You want to **reduce latency** by removing the retrieval step.
- You have a **stable, well-curated training dataset** that doesn't change frequently.
- You need the model to learn **vocabulary, abbreviations, or jargon** not in the base model.

### 3.4 When to Combine Both (RAG + Fine-Tuning)

Many production systems combine both:

1. **Fine-tune** a base model to learn your domain's **style, format, and reasoning patterns**.
2. **Use RAG** to ground the fine-tuned model in **up-to-date, factual, attributable** knowledge.

This hybrid approach typically yields the best results for enterprise applications that require both consistency and accuracy.

## 4. Best Practices for Production RAG

### 4.1 Embedding Model Selection

- Use **domain-appropriate embeddings**: General embeddings (e.g., Titan, Cohere) work for most cases; specialized embeddings (e.g., biomedical, legal) outperform general ones in their domains.
- Use **the same model for indexing and retrieval** — mismatched encoders produce poor similarity scores.

### 4.2 Chunking Strategy

- **Chunk size**: Too small = loss of context; too large = noisy retrieval. Typical sweet spot is 200–500 tokens.
- **Chunk overlap**: 10–20% overlap helps preserve context across chunk boundaries.
- **Semantic chunking**: Group related sentences rather than splitting arbitrarily.

### 4.3 Retrieval Optimization

- **Top-K**: Retrieve more than you need, then **re-rank** to keep the best K.
- **Re-ranking**: A cross-encoder or LLM-based re-ranker significantly improves precision over pure vector search.
- **Hybrid search**: Combine dense (semantic) retrieval with sparse (BM25 keyword) retrieval for best recall.

### 4.4 Prompt Engineering

- **Explicit instructions**: Tell the LLM to answer only based on the retrieved context and to say "I don't know" when uncertain.
- **Context positioning**: Place the most relevant chunks at the beginning and end of the context to mitigate the "lost-in-the-middle" effect.
- **Citation prompts**: Ask the LLM to cite which chunks support each claim.

### 4.5 Monitoring

Track retrieval quality, answer faithfulness, latency, and user feedback. Continuous evaluation is essential because both the data and the user's needs evolve over time.

## 5. AWS Services for RAG

AWS offers a managed RAG stack:

- **Amazon Bedrock Knowledge Bases**: Fully managed RAG with built-in ingestion, chunking, embedding, retrieval, and prompt augmentation.
- **Amazon Kendra**: Enterprise semantic search service that can serve as the retrieval backend.
- **Amazon OpenSearch Service**: Managed vector database for similarity search.
- **Amazon SageMaker JumpStart**: Pre-built RAG example notebooks with LangChain + Cohere / Llama-2 / Open Source models.
- **Amazon Bedrock Agents**: Orchestrate multi-step RAG workflows with action execution.

## 6. Summary

- **RAG** = retrieve external knowledge and inject it into the prompt at inference time.
- **Fine-tuning** = update model weights to internalize patterns, style, or knowledge.
- They are **complementary**: RAG for knowledge freshness and grounding; fine-tuning for style and task consistency.
- **Hybrid RAG + fine-tuning** is the production best practice for most enterprise applications.
- AWS provides a **fully managed RAG stack** via Bedrock Knowledge Bases, Kendra, OpenSearch, and SageMaker JumpStart.

## 7. References

- AWS — What is Retrieval-Augmented Generation? — https://aws.amazon.com/what-is/retrieval-augmented-generation/
- AWS SageMaker JumpStart — Retrieval-Augmented Generation — https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-customize-rag.html
- Lewis et al. 2020 — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" — https://arxiv.org/abs/2005.11401
