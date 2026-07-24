# Graph Report - .  (2026-07-24)

## Corpus Check
- 93 files · ~60,213 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 623 nodes · 1382 edges · 28 communities (21 shown, 7 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 141 edges (avg confidence: 0.63)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Application Configuration Tests
- RAG Concepts Retrieval
- Evaluation CLI Sampling
- Streaming Pipeline Tests
- Batch Evaluation Metrics
- Loading Chroma Retrieval
- DeepSeek LLM E2E Tests
- Evaluation Tracing
- Retriever Protocol Collections
- Milvus Retrieval
- Logging Configuration
- Query Transformation Methods
- End-to-End UI Tests
- Self-RAG CRAG
- Graphify Skill References
- Project Architecture Docs
- ColPali Multimodal RAG
- Coffee Knowledge
- Trainable Query Rewriting
- Configuration Secrets
- Package Entry Point
- Tea Fixture
- Recipe Fixture
- No-H1 Fixture
- Short Section Fixture
- H1 Fixture

## God Nodes (most connected - your core abstractions)
1. `Hit` - 51 edges
2. `Catalog` - 36 edges
3. `DeepSeekLLM` - 35 edges
4. `RAGEvent` - 32 edges
5. `Collection` - 26 edges
6. `JSONLEmitter` - 25 edges
7. `StreamPerf` - 25 edges
8. `BaseRetriever` - 22 edges
9. `answer_stream()` - 21 edges
10. `main()` - 19 edges

## Surprising Connections (you probably didn't know these)
- `Alt Sample: Coffee Notes` --semantically_similar_to--> `咖啡冲煮技巧 (Coffee Brewing Guide)`  [INFERRED] [semantically similar]
  tests/fixtures/sample_docs_alt/01-coffee.md → docs/shanzhongshi/咖啡冲煮技巧.md
- `StubRetriever` --uses--> `Collection`  [INFERRED]
  tests/test_app_launch.py → src/rag_learn/collections.py
- `_AltStub` --uses--> `Collection`  [INFERRED]
  tests/test_e2e.py → src/rag_learn/collections.py
- `_FakeChoice` --uses--> `Collection`  [INFERRED]
  tests/test_e2e.py → src/rag_learn/collections.py
- `_FakeChunk` --uses--> `Collection`  [INFERRED]
  tests/test_e2e.py → src/rag_learn/collections.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **graphify Skill Pipeline & References** — _claude_skills_graphify_skill_graphify, _claude_skills_graphify_references_extraction_spec_spec, _claude_skills_graphify_references_query_query, _claude_skills_graphify_references_update_update [EXTRACTED 1.00]
- **rag-learn Retrieval + Streaming Pipeline** — _claude_claude_chroma_retriever, _claude_claude_milvus_lite_retriever, _claude_claude_deepseek_llm, _claude_claude_gradio_ui [INFERRED 0.75]
- **Loader Chunking Test Fixtures** — tests_fixtures_sample_docs_doc_no_h1, tests_fixtures_sample_docs_doc_short_section, tests_fixtures_sample_docs_doc_with_h1 [INFERRED 0.75]
- **RAG Retrieval Pipeline Components** — docs_rag_doc_04_rag_architecture_overview_retrieval_pipeline, docs_rag_doc_08_bm25_sparse_retrieval_bm25, docs_rag_doc_09_vector_databases_hnsw, docs_rag_doc_10_hybrid_search_reranking_rrf, docs_rag_doc_10_hybrid_search_reranking_cross_encoder [INFERRED 0.85]
- **Advanced Modular RAG Variants** — docs_rag_doc_18_graphrag_graphrag, docs_rag_doc_19_agentic_rag_self_rag, docs_rag_doc_19_agentic_rag_crag, docs_rag_doc_20_modular_rag_modular_rag, docs_rag_doc_19_agentic_rag_agentic_rag [EXTRACTED 0.95]
- **RAG Evaluation Frameworks** — docs_rag_doc_21_rag_evaluation_ragas_ragas_metrics, docs_rag_doc_22_trulens_evaluation_trulens, docs_rag_doc_25_production_rag_best_practices_production [EXTRACTED 0.95]
- **Adaptive/Corrective RAG Family (Self-RAG, CRAG, Adaptive-RAG)** — docs_rag_doc_16_self_rag, docs_rag_doc_17_crag, docs_rag_doc_17_crag_modular_rag_family [EXTRACTED 1.00]
- **Query Transformation Techniques (Rewrite, HyDE, Step-Back)** — docs_rag_doc_11_query_rewriting_query_rewriting_step, docs_rag_doc_12_hyde_hypothetical_document, docs_rag_doc_15_step_back_abstraction [INFERRED 0.85]
- **Retrieval Quality Self-Assessment** — docs_rag_doc_16_self_rag_critique_tokens, docs_rag_doc_17_crag_retrieval_evaluator [INFERRED 0.85]

## Communities (28 total, 7 thin omitted)

### Community 0 - "Application Configuration Tests"
Cohesion: 0.06
Nodes (62): Blocks, main(), CLI shim: `python main.py` → launch the Gradio RAG compare app., build_app(), _drain_to_chatbot(), _format_perf(), launch(), _migrate_legacy_chroma() (+54 more)

### Community 1 - "RAG Concepts Retrieval"
Cohesion: 0.05
Nodes (61): BART generator, DPR (Dense Passage Retrieval), RAG Original Paper (Lewis 2020), RAG-Sequence formulation, RAG-Token formulation, Advanced RAG paradigm, Modular RAG paradigm, Naive RAG paradigm (+53 more)

### Community 2 - "Evaluation CLI Sampling"
Cohesion: 0.06
Nodes (45): ArgumentParser, _build_parser(), main(), CLI entry point for batch RAG evaluation., format_csv_row(), parse_csv_row(), CSV row parsing and formatting for batch evaluation., Parse a CSV row into question, collection, and optional ground truth.      Retur (+37 more)

### Community 3 - "Streaming Pipeline Tests"
Cohesion: 0.09
Nodes (33): _format_chunks(), ListEmitter, MetricsEmitter, Protocol, StreamPerf, answer_stream(), build_prompt(), _make_perf() (+25 more)

### Community 4 - "Batch Evaluation Metrics"
Cohesion: 0.11
Nodes (45): _aggregate(), _compute_supervised(), _compute_unsupervised(), _dedupe(), _ground_truth_to_dict(), _load_events(), main(), _make_judge_fn() (+37 more)

### Community 5 - "Loading Chroma Retrieval"
Cohesion: 0.07
Nodes (37): _default_factory(), Path, Chunk, _chunk_size(), _chunk_text(), iter_markdown(), load_documents(), Path (+29 more)

### Community 6 - "DeepSeek LLM E2E Tests"
Cohesion: 0.08
Nodes (27): RuntimeError, DeepSeekLLM, Any, DeepSeek LLM client; uses the OpenAI SDK with DeepSeek's base URL., _FakeChoice, _FakeChunk, _FakeStream, Counts how many tokens were requested and emits a canned answer. (+19 more)

### Community 7 - "Evaluation Tracing"
Cohesion: 0.11
Nodes (32): event_from_dict(), _event_to_dict(), GroundTruth, JSONLEmitter, NullEmitter, Any, Path, RAG event model, emitters, and JSONL serialization. (+24 more)

### Community 8 - "Retriever Protocol Collections"
Cohesion: 0.09
Nodes (31): KeyError, Collection, CollectionNotFoundError, 一个独立的知识库：slug + 显示元数据 + 文档目录 + retriever 工厂。      `retriever` 是懒加载属性：首次访问时由 `ret, 请求的 collection 不在 Catalog 里。, BaseRetriever, Protocol, fake_docs() (+23 more)

### Community 9 - "Milvus Retrieval"
Cohesion: 0.10
Nodes (25): _load_collection_subprocess(), MilvusRetriever, Any, Path, Milvus Lite (embedded) adapter implementing BaseRetriever.  Uses pymilvus.model., Subprocess entry point: open MilvusClient and load the collection.      Runs in, Run target(*args) in an isolated subprocess.      Returns True iff the subproces, Isolated wrapper around MilvusClient.load_collection.      Returns True on succe (+17 more)

### Community 10 - "Logging Configuration"
Cohesion: 0.14
Nodes (21): Handler, create_handlers(), get_log_level(), Path, Process-wide logging configuration., Walk up from ``start_path`` until we find pyproject.toml.      Robust to worktre, Resolve a log-level name to a ``logging`` level integer.      Reads from the ``L, Create console and file handlers, ensuring the log directory exists. (+13 more)

### Community 11 - "Query Transformation Methods"
Cohesion: 0.11
Nodes (20): Bing Web Search Retriever, ChatGPT (gpt-3.5-turbo) Reader, HotpotQA Benchmark, Input-Output Query Gap Problem, Query Rewriting Step, Rewrite-Retrieve-Read Framework, Contriever Contrastive Encoder, Dense Bottleneck Filter (Lossy Compressor) (+12 more)

### Community 12 - "End-to-End UI Tests"
Cohesion: 0.20
Nodes (16): _AltStub, _make_config(), Any, Path, End-to-end smoke: multi-collection catalog flows through the pipeline with a moc, Selecting a different collection must drive retrieval to that side., Submitting an empty question returns empty outputs without raising., An unknown collection slug shows a warning and leaves chunks empty. (+8 more)

### Community 13 - "Self-RAG CRAG"
Cohesion: 0.16
Nodes (14): Self-RAG (Self-Reflective RAG), Adaptive Retrieval Motivation, Critique-Token Supervised Fine-Tuning, Critique Tokens (ISREL, ISSUP, ISUSE), On-Demand Retrieval Decision, Reflection Tokens Vocabulary, Retrieve Token (yes/no/continue), Corrective RAG (CRAG) (+6 more)

### Community 14 - "Graphify Skill References"
Cohesion: 0.22
Nodes (9): Add URL & Watch Folder Reference, Exports & Benchmark Reference, Extraction Subagent Spec, GitHub Clone & Cross-Repo Merge Reference, Commit Hook & CLAUDE.md Integration Reference, Query / Path / Explain Reference, Video/Audio Transcribe Reference, Incremental Update Reference (+1 more)

### Community 15 - "Project Architecture Docs"
Cohesion: 0.48
Nodes (7): Chroma Retriever, DeepSeek LLM, Gradio UI, Milvus Lite Retriever, rag-learn Project, Batch Evaluation CLI (rag_learn.eval.cli), rag-learn README Overview

### Community 16 - "ColPali Multimodal RAG"
Cohesion: 0.33
Nodes (6): ColPali (Vision-Language Document Retrieval), Direct Visual Embedding (No OCR), ColBERT-style Late Interaction, Multi-Vector Page Embeddings, PaliGemma-3B Base VLM, ViDoRe Visual Document Retrieval Benchmark

### Community 17 - "Coffee Knowledge"
Cohesion: 0.33
Nodes (6): 咖啡豆参数信息 (Coffee Bean Params), SHAN.IN COFFEE / 山隐画见, 咖啡冲煮技巧 (Coffee Brewing Guide), Roast-Level Brewing Parameters, 合肥山隐智农科技有限公司, Alt Sample: Coffee Notes

### Community 18 - "Trainable Query Rewriting"
Cohesion: 0.67
Nodes (3): PPO Reinforcement Learning for Rewriter, Rewriter Warm-up via Distillation, Trainable Rewriter (T5-large)

## Knowledge Gaps
- **46 isolated node(s):** `rag-learn`, `config.load_config / ConfigError`, `DEEPSEEK_API_KEY env var`, `Extraction Subagent Spec`, `Query / Path / Explain Reference` (+41 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Hit` connect `Streaming Pipeline Tests` to `Application Configuration Tests`, `Evaluation CLI Sampling`, `Batch Evaluation Metrics`, `Loading Chroma Retrieval`, `DeepSeek LLM E2E Tests`, `Evaluation Tracing`, `Retriever Protocol Collections`, `Milvus Retrieval`, `End-to-End UI Tests`?**
  _High betweenness centrality (0.155) - this node is a cross-community bridge._
- **Why does `DeepSeekLLM` connect `DeepSeek LLM E2E Tests` to `Application Configuration Tests`, `Streaming Pipeline Tests`, `Batch Evaluation Metrics`, `End-to-End UI Tests`?**
  _High betweenness centrality (0.077) - this node is a cross-community bridge._
- **Why does `Catalog` connect `Application Configuration Tests` to `Retriever Protocol Collections`, `End-to-End UI Tests`, `Loading Chroma Retrieval`, `DeepSeek LLM E2E Tests`?**
  _High betweenness centrality (0.054) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `Hit` (e.g. with `ChromaRetriever` and `MilvusRetriever`) actually correct?**
  _`Hit` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `Catalog` (e.g. with `BaseRetriever` and `ChromaRetriever`) actually correct?**
  _`Catalog` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `DeepSeekLLM` (e.g. with `_AltStub` and `_FakeChoice`) actually correct?**
  _`DeepSeekLLM` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `RAGEvent` (e.g. with `StreamPerf` and `_FakeLLM`) actually correct?**
  _`RAGEvent` has 4 INFERRED edges - model-reasoned connections that need verification._