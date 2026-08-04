# Graph Report - rag_learn  (2026-08-04)

## Corpus Check
- 86 files · ~38,887 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 944 nodes · 2288 edges · 41 communities (31 shown, 10 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 214 edges (avg confidence: 0.61)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `99f3ce56`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Eval Event Tracing
- Batch Loader & Dedup
- Gradio App Surface
- Collections Catalog
- Document Loader & Chunking
- Eval CLI Dispatch
- Batch Metric Computation
- LLM Judge & DeepSeek Client
- Milvus Retriever Adapter
- Rate Limiting Primitives
- Batch Eval Runner
- E2E Test Stubs
- README Concept References
- Logging Configuration
- Graphify Skill Docs
- HybridRetriever
- CLAUDE.md Gotchas
- Config Errors
- DeepSeek LLM Anchor
- Package Distribution
- README Anchor
- Coffee Sample Doc
- Tea Sample Doc
- Recipe Sample Doc
- test_app_launch.py
- doc_beans.md
- doc_with_h1.md
- doc_short_section.md
- RateLimiter
- loader.py
- BM25Index
- conftest.py
- test_e2e.py
- retriever/base.py
- test_tracing.py
- Reranker
- tracing.py
- _make_judge_fn

## God Nodes (most connected - your core abstractions)
1. `Hit` - 88 edges
2. `Catalog` - 57 edges
3. `Config` - 56 edges
4. `answer_stream()` - 51 edges
5. `Collection` - 42 edges
6. `DeepSeekLLM` - 36 edges
7. `load_config()` - 34 edges
8. `RAGEvent` - 34 edges
9. `JSONLEmitter` - 32 edges
10. `StreamPerf` - 27 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `launch()`  [EXTRACTED]
  main.py → src/rag_learn/app.py
- `FakeRetriever` --uses--> `Collection`  [INFERRED]
  tests/test_collections.py → src/rag_learn/collections.py
- `_AltStub` --uses--> `Collection`  [INFERRED]
  tests/test_e2e.py → src/rag_learn/collections.py
- `_FakeStream` --uses--> `Collection`  [INFERRED]
  tests/test_e2e.py → src/rag_learn/collections.py
- `_NoopRetriever` --uses--> `Collection`  [INFERRED]
  tests/test_pipeline_catalog_fanout.py → src/rag_learn/collections.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **rag-learn Retrieval + Streaming Pipeline** — _claude_claude_chroma_retriever, _claude_claude_milvus_lite_retriever, _claude_claude_deepseek_llm, _claude_claude_gradio_ui [INFERRED 0.75]
- **graphify Skill Pipeline & References** — _claude_skills_graphify_skill_graphify, _claude_skills_graphify_references_extraction_spec_spec, _claude_skills_graphify_references_query_query, _claude_skills_graphify_references_update_update [EXTRACTED 1.00]
- **RAG retrieval flow** — readme_rag_learn, readme_chroma, readme_persistentclient, readme_all_minilm_l6_v2, readme_gradio, readme_gr_chatbot, readme_deepseek, readme_deepseekllm_stream [EXTRACTED 1.00]
- **batch evaluation pipeline** — readme_rag_learn_eval_cli, readme_csv_template, readme_ratelimiter, readme_shanzhongshi_qa_csv, readme_2026_07_22_batch_evaluation_design_md [EXTRACTED 1.00]

## Communities (41 total, 10 thin omitted)

### Community 0 - "Eval Event Tracing"
Cohesion: 0.06
Nodes (53): _drain_to_chatbot(), _format_chunks(), _format_routing(), Gradio UI: collection dropdown + single answer panel with chunks and perf metric, Render a one-line caption summarizing routing decisions.      ``routing`` is the, Consume an answer_stream's iterator and return the joined text., Config, _build_catalog_summary() (+45 more)

### Community 1 - "Batch Loader & Dedup"
Cohesion: 0.18
Nodes (13): _make_catalog(), _make_config(), _NoopRetriever, _PerQueryFakeRetriever, Path, Regression tests for cross-call metadata isolation in pipeline.answer_stream.  T, When the catalog branch fires, the caller's metadata dict is unchanged., Specific-intent path also must not mutate the caller's dict. (+5 more)

### Community 2 - "Gradio App Surface"
Cohesion: 0.14
Nodes (24): ConfigError, load_config(), _parse_bool(), _parse_optional_float(), Process-wide configuration loaded once from environment variables., Raised when required configuration is missing or invalid., 解析可选的有限浮点数，留空表示禁用该配置。, 新增的 6 个 routing 字段默认值(开关默认关,其余走 dataclass 默认). (+16 more)

### Community 3 - "Collections Catalog"
Cohesion: 0.07
Nodes (46): KeyError, Catalog, Collection, CollectionNotFoundError, Eager 触发每个 collection 的 retriever 懒加载。fail-open.          Returns list of (colle, 一个独立的知识库：slug + 显示元数据 + 文档目录 + retriever 工厂。      `retriever` 是懒加载属性：首次访问时由 `ret, 请求的 collection 不在 Catalog 里。, 不可变集合注册表：slug → Collection 双向索引。 (+38 more)

### Community 4 - "Document Loader & Chunking"
Cohesion: 0.12
Nodes (20): iter_markdown(), load_documents(), Path, Read all *.md in docs_dir and return a flat list of chunks., Return list of (filename, raw_text) sorted by filename, deterministic., Index once for the vector store and once for the BM25 keyword index., Each H1's text must appear inside the resulting chunk so the embedding     can m, Multi-H1 fixture (bean-card layout): every chunk must mention only its     own b (+12 more)

### Community 5 - "Eval CLI Dispatch"
Cohesion: 0.07
Nodes (37): ArgumentParser, _build_parser(), main(), CLI entry point for batch RAG evaluation., format_csv_row(), parse_csv_row(), CSV row parsing and formatting for batch evaluation., Parse a CSV row into question, collection, and optional ground truth.      Retur (+29 more)

### Community 6 - "Batch Metric Computation"
Cohesion: 0.05
Nodes (94): _aggregate(), _compute_supervised(), _compute_unsupervised(), _dedupe(), _ground_truth_to_dict(), _load_events(), main(), _make_judge_fn() (+86 more)

### Community 7 - "LLM Judge & DeepSeek Client"
Cohesion: 0.06
Nodes (45): Blocks, build_app(), _format_perf(), Any, Construct the Gradio UI but do not launch it.      Args:         catalog: The co, DeepSeekLLM, Any, DeepSeek LLM client; uses the OpenAI SDK with DeepSeek's base URL. (+37 more)

### Community 8 - "Milvus Retriever Adapter"
Cohesion: 0.10
Nodes (25): _load_collection_subprocess(), MilvusRetriever, Any, Path, Milvus Lite (embedded) adapter implementing BaseRetriever.  Uses pymilvus.model., Subprocess entry point: open MilvusClient and load the collection.      Runs in, Run target(*args) in an isolated subprocess.      Returns True iff the subproces, Isolated wrapper around MilvusClient.load_collection.      Returns True on succe (+17 more)

### Community 9 - "Rate Limiting Primitives"
Cohesion: 0.24
Nodes (5): CrossEncoderReranker, Local cross-encoder reranker using sentence-transformers CrossEncoder., 为每个问题-命中对打分，按降序返回过滤后的命中。, Unit tests for CrossEncoderReranker using mocked CrossEncoder., TestCrossEncoderReranker

### Community 10 - "Batch Eval Runner"
Cohesion: 0.17
Nodes (15): Chunk, _chunk_size(), _chunk_text(), Markdown discovery + chunking for ingestion., Top-level entry: chunk a single file's content., Split a markdown file by H1 headings.      Each pre-doc starts with its H1 line, Greedy paragraph-then-sentence packing up to CHUNK_SIZE with OVERLAP., split_into_chunks() (+7 more)

### Community 11 - "E2E Test Stubs"
Cohesion: 0.08
Nodes (34): INTENT_LABELS, classify_intent(), decompose_query(), _drain_stream(), _LLMStream, _parse_intent(), _parse_subqueries(), Protocol (+26 more)

### Community 12 - "README Concept References"
Cohesion: 0.10
Nodes (24): 2026-07-22-batch-evaluation-design.md, all-MiniLM-L6-v2, Chroma, evaluation CSV template, data/chroma/, DeepSeek, DeepSeekLLM.stream, gr.Chatbot (+16 more)

### Community 13 - "Logging Configuration"
Cohesion: 0.12
Nodes (23): Handler, main(), CLI shim: `python main.py` → launch the Gradio RAG compare app., create_handlers(), get_log_level(), Path, Process-wide logging configuration., Walk up from ``start_path`` until we find pyproject.toml.      Robust to worktre (+15 more)

### Community 14 - "Graphify Skill Docs"
Cohesion: 0.22
Nodes (9): Add URL & Watch Folder Reference, Exports & Benchmark Reference, Extraction Subagent Spec, GitHub Clone & Cross-Repo Merge Reference, Commit Hook & CLAUDE.md Integration Reference, Query / Path / Explain Reference, Video/Audio Transcribe Reference, Incremental Update Reference (+1 more)

### Community 15 - "HybridRetriever"
Cohesion: 0.24
Nodes (19): HybridRetriever, Run vector and keyword retrieval side-by-side; fuse with RRF.      Implements ``, _FakeBM25, _FakeVectorRetriever, _hit(), Tests for HybridRetriever (vector + BM25, fused via RRF)., Implements BaseRetriever via duck-typing., Shared hits must outrank hits that only appear in one list. (+11 more)

### Community 16 - "CLAUDE.md Gotchas"
Cohesion: 0.67
Nodes (4): Chroma Retriever, Gradio UI, Milvus Lite Retriever, rag-learn Project

### Community 26 - "test_app_launch.py"
Cohesion: 0.06
Nodes (48): ListEmitter, StreamPerf, answer_stream(), Run all retrievers in parallel (threads); return their Hits per side.      Each, Parallel retrieve → build prompt per side → stream tokens per side.      Returns, _retrieve(), Hit, _FakeLLM (+40 more)

### Community 27 - "doc_beans.md"
Cohesion: 0.50
Nodes (3): 耶加 TOH亚军地块-中度烘焙, 苏帕摩-中度烘焙, 达摩-中浅烘焙

### Community 31 - "RateLimiter"
Cohesion: 0.10
Nodes (26): BaseException, RuntimeError, is_rate_limit_error(), Any, Limiter, RateLimiter, Thin wrapper composing pyrate-limiter + threading.Semaphore + tenacity.  Used by, Return True iff ``exc`` represents an HTTP 429 from any layer. (+18 more)

### Community 32 - "loader.py"
Cohesion: 0.18
Nodes (22): _load_existing_keys(), Path, Read a Q&A CSV, run each question through RAG, emit events, and evaluate., Return (collection, question) pairs already emitted to ``events_file``.      Rea, run_qa_csv(), _find_events_file(), MonkeyPatch, Path (+14 more)

### Community 33 - "BM25Index"
Cohesion: 0.18
Nodes (13): BM25Index, In-memory BM25 keyword index., Return up to ``k`` hits sorted by descending BM25 score., Path, _chunks(), Tests for the BM25 keyword index., test_build_replaces_existing_index(), test_search_finds_keyword_match_and_orders_by_score() (+5 more)

### Community 34 - "conftest.py"
Cohesion: 0.07
Nodes (47): PromptMode, MetricsEmitter, Protocol, _answer_catalog_recall(), build_prompt(), _candidate_k(), _flat_retrieve(), _make_perf() (+39 more)

### Community 35 - "test_e2e.py"
Cohesion: 0.32
Nodes (5): LogCaptureFixture, build_reranker(), Build a reranker from config, or return None if disabled or unavailable., Tests for build_reranker factory., TestBuildReranker

### Community 37 - "test_tracing.py"
Cohesion: 0.67
Nodes (3): fixtures_dir(), Path, sample_hits()

### Community 40 - "Reranker"
Cohesion: 0.15
Nodes (10): Protocol, Reranker contract shared by all implementations., Return a new list of hits sorted by descending relevance.          The returned, Reranker, Cross-encoder reranker implementation backed by sentence-transformers., Factory for building a reranker from configuration., Reranker components for refining retrieval results., Tests for the reranker module. (+2 more)

### Community 41 - "tracing.py"
Cohesion: 0.24
Nodes (10): launch(), Production entry: load config, build catalog + LLM, migrate, ingest, serve., build_catalog(), Build the default catalog, optionally wiring hybrid retrieval and filtering., _load_catalog(), _make_llm(), _process_row(), Any (+2 more)

### Community 49 - "_make_judge_fn"
Cohesion: 0.06
Nodes (46): _build_builtin(), _default_factory(), _make_factory(), Path, Collection domain object: a single knowledge base (name, docs, retriever)., Build a retriever factory that captures the hybrid config., Path, Walk up from this file until we find pyproject.toml.      Robust to worktree dir (+38 more)

## Knowledge Gaps
- **34 isolated node(s):** `rag-learn`, `苏帕摩-中度烘焙`, `耶加 TOH亚军地块-中度烘焙`, `达摩-中浅烘焙`, `Tiny` (+29 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Hit` connect `test_app_launch.py` to `Eval Event Tracing`, `BM25Index`, `conftest.py`, `Collections Catalog`, `retriever/base.py`, `test_tracing.py`, `Batch Metric Computation`, `Eval CLI Dispatch`, `Reranker`, `Rate Limiting Primitives`, `Batch Eval Runner`, `Milvus Retriever Adapter`, `loader.py`, `LLM Judge & DeepSeek Client`, `Batch Loader & Dedup`, `HybridRetriever`, `_make_judge_fn`?**
  _High betweenness centrality (0.217) - this node is a cross-community bridge._
- **Why does `answer_stream()` connect `test_app_launch.py` to `Eval Event Tracing`, `Batch Loader & Dedup`, `conftest.py`, `Collections Catalog`, `Batch Metric Computation`, `LLM Judge & DeepSeek Client`, `Reranker`, `tracing.py`, `E2E Test Stubs`, `_make_judge_fn`?**
  _High betweenness centrality (0.080) - this node is a cross-community bridge._
- **Why does `Catalog` connect `Collections Catalog` to `Eval Event Tracing`, `loader.py`, `conftest.py`, `Batch Loader & Dedup`, `LLM Judge & DeepSeek Client`, `tracing.py`, `_make_judge_fn`, `test_app_launch.py`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `Hit` (e.g. with `_make_event()` and `_hit()`) actually correct?**
  _`Hit` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `Catalog` (e.g. with `StubRetriever` and `FakeRetriever`) actually correct?**
  _`Catalog` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 24 inferred relationships involving `Config` (e.g. with `StubRetriever` and `_AltStub`) actually correct?**
  _`Config` has 24 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `Collection` (e.g. with `StubRetriever` and `FakeRetriever`) actually correct?**
  _`Collection` has 19 INFERRED edges - model-reasoned connections that need verification._