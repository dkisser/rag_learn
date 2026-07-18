# RAG 技术资料索引

> 25 篇从权威源（arxiv / AWS / LangChain / LlamaIndex / Pinecone / Qdrant / Weaviate / Microsoft Research / Hugging Face / Elastic / SBERT）抓取的 RAG 文档。
> 每篇均保留原文英文内容 + 公式/代码/对比表 + 来源 URL 元数据。

## 一、基础与范式 (Foundations & Paradigms)

| # | 文件 | 来源 | 内容要点 |
|---|------|------|----------|
| 01 | [01-rag-original-paper.md](01-rag-original-paper.md) | arxiv:2005.11401 (Lewis 2020 NeurIPS) | RAG 原始论文，RAG-Sequence / RAG-Token 公式，BART + DPR |
| 02 | [02-rag-survey-2024.md](02-rag-survey-2024.md) | arxiv:2312.10997 (Gao 2024) | 综述：Naive → Advanced → Modular RAG 范式演进 |
| 03 | [03-rag-vs-finetuning.md](03-rag-vs-finetuning.md) | AWS 官方 + SageMaker | RAG vs Fine-tuning 12 维对比表 + 选型决策 |
| 04 | [04-rag-architecture-overview.md](04-rag-architecture-overview.md) | AWS 官方 RAG explainer | Ingestion / Retrieval / Generation 三管道架构 |
| 05 | [05-document-loaders.md](05-document-loaders.md) | Unstructured.io + LangChain patterns | 文档解析与加载器选型（PDF/HTML/Markdown/表格） |

## 二、索引与检索 (Indexing & Retrieval)

| # | 文件 | 来源 | 内容要点 |
|---|------|------|----------|
| 06 | [06-chunking-strategies.md](06-chunking-strategies.md) | Pinecone Learning | 5 种 chunking 策略对比（固定/递归/语义/结构/LLM 上下文） |
| 07 | [07-embedding-models.md](07-embedding-models.md) | SBERT 官方 | sentence-transformers 预训练模型矩阵（all-mpnet/multi-qa/msmarco） |
| 08 | [08-bm25-sparse-retrieval.md](08-bm25-sparse-retrieval.md) | Wikipedia + Elastic Blog | BM25 完整公式、k1/b 参数、IDF 推导、变体 |
| 09 | [09-vector-databases.md](09-vector-databases.md) | Qdrant 官方 | HNSW 索引、payload filter、混合检索、生产部署 |
| 10 | [10-hybrid-search-reranking.md](10-hybrid-search-reranking.md) | Weaviate + Pinecone | RRF 公式、bi-encoder vs cross-encoder、两阶段流水线 |

## 三、查询理解与上下文工程 (Query Understanding)

| # | 文件 | 来源 | 内容要点 |
|---|------|------|----------|
| 11 | [11-query-rewriting.md](11-query-rewriting.md) | arxiv:2305.14283 (Ma 2023) | Rewrite-Retrieve-Read 框架，监督预热 + PPO 训练 |
| 12 | [12-hyde.md](12-hyde.md) | arxiv:2212.10496 (Gao 2022) | HyDE 假设文档嵌入，无监督检索 |
| 13 | [13-multi-query-retrieval.md](13-multi-query-retrieval.md) | LangChain 官方教程 | MultiQueryRetriever，多查询并集去重 |
| 14 | [14-contextual-compression.md](14-contextual-compression.md) | LangChain 官方教程 | ContextualCompressionRetriever，抽取器/过滤器/重排器 |
| 15 | [15-step-back-prompting.md](15-step-back-prompting.md) | arxiv:2310.06117 (Zheng 2023) | Step-back 抽象问题 + RAG 提升复杂推理 |

## 四、高级范式与变体 (Advanced Paradigms)

| # | 文件 | 来源 | 内容要点 |
|---|------|------|----------|
| 16 | [16-self-rag.md](16-self-rag.md) | arxiv:2310.11511 (Asai 2023) | Self-RAG 反思 token，Retrieve/ISREL/ISSUP/ISUSE |
| 17 | [17-corrective-rag.md](17-corrective-rag.md) | arxiv:2401.15884 (Yan 2024) | CRAG 检索评估器 + Web 搜索回退 + DTR 算法 |
| 18 | [18-graphrag.md](18-graphrag.md) | Microsoft Research Blog | GraphRAG：LLM 知识图谱 + Leiden 社区检测 + 全局查询 |
| 19 | [19-agentic-rag.md](19-agentic-rag.md) | LangChain Blog (LangGraph) | Agentic RAG 状态机，Pydantic 评分器 + 工具调用 |
| 20 | [20-modular-rag.md](20-modular-rag.md) | arxiv:2312.10997 Section II-C | Modular RAG 六大新模块 + Rewrite/Retrieve/HyDE/ITER-RETGEN 等模式 |

## 五、工程化与评估 (Engineering & Evaluation)

| # | 文件 | 来源 | 内容要点 |
|---|------|------|----------|
| 21 | [21-rag-evaluation-ragas.md](21-rag-evaluation-ragas.md) | docs.ragas.io 官方 | RAGAS 五指标公式：Faithfulness / Context Precision / Recall / Answer Relevancy / Noise Sensitivity |
| 22 | [22-trulens-evaluation.md](22-trulens-evaluation.md) | trulens.org + GitHub README | TruLens RAG Triad + Agentic 评估器 + OpenTelemetry tracing |
| 23 | [23-colpali-multimodal-rag.md](23-colpali-multimodal-rag.md) | arxiv:2407.01449 (Faysse, ICLR 2025) | ColPali：PaliGemma-3B + ColBERT-style 视觉文档检索 |
| 24 | [24-rag-frameworks-comparison.md](24-rag-frameworks-comparison.md) | LangChain + LlamaIndex 官方 | 两框架设计哲学对比 + 选型决策 |
| 25 | [25-production-rag-best-practices.md](25-production-rag-best-practices.md) | AWS + Anthropic + Pinecone | 生产级 RAG 最佳实践（分块/嵌入/双阶段/Contextual Retrieval/成本/安全） |

---

## 关键概念交叉索引

- **RAG 三范式演进**：01 (原始) → 02 (综述) → 20 (Modular)
- **检索流水线**：08 (BM25) + 09 (向量库) → 10 (混合 + rerank)
- **查询增强**：11 (Rewrite) / 12 (HyDE) / 13 (Multi-query) / 15 (Step-back) → 14 (Compression)
- **反思与纠错**：16 (Self-RAG) / 17 (CRAG)
- **结构化与图谱**：18 (GraphRAG)
- **智能体化**：19 (Agentic RAG)
- **评估体系**：21 (RAGAS) / 22 (TruLens)
- **多模态**：23 (ColPali)
- **工程落地**：24 (框架选型) / 25 (生产实践)
