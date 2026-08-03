# Graph Report - rag_learn  (2026-08-03)

## Corpus Check
- 85 files · ~38,223 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 916 nodes · 2232 edges · 45 communities (34 shown, 11 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 207 edges (avg confidence: 0.61)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `72157241`
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
- app.py
- pipeline.py
- Reranker
- test_pipeline_parallel.py
- build_reranker
- runner.py
- test_cli.py

## God Nodes (most connected - your core abstractions)
1. `Hit` - 87 edges
2. `Catalog` - 57 edges
3. `Config` - 57 edges
4. `answer_stream()` - 51 edges
5. `Collection` - 42 edges
6. `DeepSeekLLM` - 36 edges
7. `RAGEvent` - 34 edges
8. `JSONLEmitter` - 32 edges
9. `load_config()` - 27 edges
10. `StreamPerf` - 27 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `launch()`  [EXTRACTED]
  main.py → src/rag_learn/app.py
- `FakeRetriever` --uses--> `Collection`  [INFERRED]
  tests/test_collections.py → src/rag_learn/collections.py
- `_AltStub` --uses--> `Collection`  [INFERRED]
  tests/test_e2e.py → src/rag_learn/collections.py
- `_FakeChoice` --uses--> `Collection`  [INFERRED]
  tests/test_e2e.py → src/rag_learn/collections.py
- `_FakeChunk` --uses--> `Collection`  [INFERRED]
  tests/test_e2e.py → src/rag_learn/collections.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **rag-learn Retrieval + Streaming Pipeline** — _claude_claude_chroma_retriever, _claude_claude_milvus_lite_retriever, _claude_claude_deepseek_llm, _claude_claude_gradio_ui [INFERRED 0.75]
- **graphify Skill Pipeline & References** — _claude_skills_graphify_skill_graphify, _claude_skills_graphify_references_extraction_spec_spec, _claude_skills_graphify_references_query_query, _claude_skills_graphify_references_update_update [EXTRACTED 1.00]
- **RAG retrieval flow** — readme_rag_learn, readme_chroma, readme_persistentclient, readme_all_minilm_l6_v2, readme_gradio, readme_gr_chatbot, readme_deepseek, readme_deepseekllm_stream [EXTRACTED 1.00]
- **batch evaluation pipeline** — readme_rag_learn_eval_cli, readme_csv_template, readme_ratelimiter, readme_shanzhongshi_qa_csv, readme_2026_07_22_batch_evaluation_design_md [EXTRACTED 1.00]

## Communities (45 total, 11 thin omitted)

### Community 0 - "Eval Event Tracing"
Cohesion: 0.16
Nodes (15): _build_catalog_summary(), Render a one-line-per-collection string for the decomposer prompt.      ``only``, _NoopRetriever, Path, Tests for catalog fan-out scoping (A) and the split k parameters (B).  A — the d, Legacy compare mode keys ('chroma'/'milvus') are not catalog names., Returns ``n_hits`` unique hits per query and records the k it got., _RecordingRetriever (+7 more)

### Community 1 - "Batch Loader & Dedup"
Cohesion: 0.18
Nodes (13): _make_catalog(), _make_config(), _NoopRetriever, _PerQueryFakeRetriever, Path, Regression tests for cross-call metadata isolation in pipeline.answer_stream.  T, When the catalog branch fires, the caller's metadata dict is unchanged., Specific-intent path also must not mutate the caller's dict. (+5 more)

### Community 2 - "Gradio App Surface"
Cohesion: 0.16
Nodes (18): ConfigError, load_config(), _parse_bool(), Process-wide configuration loaded once from environment variables., Raised when required configuration is missing or invalid., 新增的 6 个 routing 字段默认值(开关默认关,其余走 dataclass 默认)., test_load_config_boolean_parsing(), test_load_config_hybrid_overrides() (+10 more)

### Community 3 - "Collections Catalog"
Cohesion: 0.05
Nodes (70): Blocks, KeyError, build_app(), launch(), _migrate_legacy_chroma(), Any, Production entry: load config, build catalog + LLM, migrate, ingest, serve., 一次性：把 data/chroma/ 根下的遗留文件搬到 data/chroma/rag_doc/。      触发条件：data/chroma/rag_doc (+62 more)

### Community 4 - "Document Loader & Chunking"
Cohesion: 0.11
Nodes (20): iter_markdown(), load_documents(), Path, Read all *.md in docs_dir and return a flat list of chunks., Return list of (filename, raw_text) sorted by filename, deterministic., Index once for the vector store and once for the BM25 keyword index., Each H1's text must appear inside the resulting chunk so the embedding     can m, Multi-H1 fixture (bean-card layout): every chunk must mention only its     own b (+12 more)

### Community 5 - "Eval CLI Dispatch"
Cohesion: 0.05
Nodes (60): ArgumentParser, _build_parser(), main(), CLI entry point for batch RAG evaluation., format_csv_row(), parse_csv_row(), CSV row parsing and formatting for batch evaluation., Parse a CSV row into question, collection, and optional ground truth.      Retur (+52 more)

### Community 6 - "Batch Metric Computation"
Cohesion: 0.06
Nodes (91): _aggregate(), _compute_supervised(), _compute_unsupervised(), _dedupe(), _ground_truth_to_dict(), _load_events(), main(), _make_judge_fn() (+83 more)

### Community 7 - "LLM Judge & DeepSeek Client"
Cohesion: 0.15
Nodes (17): DeepSeekLLM, Any, _FakeChoice, _FakeChunk, _FakeClient, _FakeStream, _RaisingChat, _RaisingClient (+9 more)

### Community 8 - "Milvus Retriever Adapter"
Cohesion: 0.10
Nodes (25): _load_collection_subprocess(), MilvusRetriever, Any, Path, Milvus Lite (embedded) adapter implementing BaseRetriever.  Uses pymilvus.model., Subprocess entry point: open MilvusClient and load the collection.      Runs in, Run target(*args) in an isolated subprocess.      Returns True iff the subproces, Isolated wrapper around MilvusClient.load_collection.      Returns True on succe (+17 more)

### Community 9 - "Rate Limiting Primitives"
Cohesion: 0.13
Nodes (20): Retriever contract shared by all adapter implementations., ChromaRetriever, Path, Chroma adapter implementing BaseRetriever via PersistentClient + default embedde, build_retriever(), Path, Factory that picks the right retriever implementation based on config., Build a retriever instance.      When ``hybrid_enabled`` is True, wraps ``Chroma (+12 more)

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
Cohesion: 0.18
Nodes (20): HybridRetriever, Run vector and keyword retrieval side-by-side; fuse with RRF.      Implements ``, Fetch top-k from each retriever and fuse with Reciprocal Rank Fusion., _FakeBM25, _FakeVectorRetriever, _hit(), Tests for HybridRetriever (vector + BM25, fused via RRF)., Implements BaseRetriever via duck-typing. (+12 more)

### Community 16 - "CLAUDE.md Gotchas"
Cohesion: 0.67
Nodes (4): Chroma Retriever, Gradio UI, Milvus Lite Retriever, rag-learn Project

### Community 26 - "test_app_launch.py"
Cohesion: 0.06
Nodes (56): PromptMode, _format_chunks(), ListEmitter, build_prompt(), Return ``(system_msg, user_msg)`` with display-safe chunk lengths., Hit, _FakeLLM, _FakeReranker (+48 more)

### Community 27 - "doc_beans.md"
Cohesion: 0.50
Nodes (3): 耶加 TOH亚军地块-中度烘焙, 苏帕摩-中度烘焙, 达摩-中浅烘焙

### Community 31 - "RateLimiter"
Cohesion: 0.09
Nodes (27): BaseException, RuntimeError, is_rate_limit_error(), Any, Limiter, RateLimiter, Thin wrapper composing pyrate-limiter + threading.Semaphore + tenacity.  Used by, Return True iff ``exc`` represents an HTTP 429 from any layer. (+19 more)

### Community 32 - "loader.py"
Cohesion: 0.16
Nodes (17): make_routing_config(), Factory for a Config with the intent/decompose routing knobs set.      Keyword o, _hit(), _make_catalog(), _NoopRetriever, _PerQueryFakeRetriever, Path, Tests for the ``routing_sink`` out-parameter of ``pipeline.answer_stream``.  The (+9 more)

### Community 33 - "BM25Index"
Cohesion: 0.18
Nodes (13): BM25Index, In-memory BM25 keyword index., Return up to ``k`` hits sorted by descending BM25 score., Path, _chunks(), Tests for the BM25 keyword index., test_build_replaces_existing_index(), test_search_finds_keyword_match_and_orders_by_score() (+5 more)

### Community 35 - "test_e2e.py"
Cohesion: 0.13
Nodes (21): _AltStub, _FakeChoice, _FakeChunk, _FakeStream, _make_config(), Any, Path, End-to-end smoke: multi-collection catalog flows through the pipeline with a moc (+13 more)

### Community 36 - "retriever/base.py"
Cohesion: 0.22
Nodes (6): BaseRetriever, Protocol, test_hit_equality(), test_hit_is_frozen(), test_protocol_recognises_conforming_class(), test_protocol_rejects_non_conforming()

### Community 37 - "test_tracing.py"
Cohesion: 0.67
Nodes (3): fixtures_dir(), Path, sample_hits()

### Community 38 - "app.py"
Cohesion: 0.14
Nodes (17): _drain_to_chatbot(), _format_routing(), Gradio UI: collection dropdown + single answer panel with chunks and perf metric, Render a one-line caption summarizing routing decisions.      ``routing`` is the, Consume an answer_stream's iterator and return the joined text., Any, Intent classification and query decomposition for catalog-coverage queries.  Two, Immutable record of the routing decisions taken for ONE question.      ``pipelin (+9 more)

### Community 39 - "pipeline.py"
Cohesion: 0.14
Nodes (19): MetricsEmitter, Protocol, _answer_catalog_recall(), answer_stream(), _candidate_k(), _flat_retrieve(), _make_perf(), _merge_dedup() (+11 more)

### Community 40 - "Reranker"
Cohesion: 0.11
Nodes (15): Protocol, Reranker contract shared by all implementations., Return a new list of hits sorted by descending relevance.          The returned, Reranker, CrossEncoderReranker, Cross-encoder reranker implementation backed by sentence-transformers., Local cross-encoder reranker using sentence-transformers CrossEncoder., Score each (query, hit) pair and return hits sorted by descending score. (+7 more)

### Community 43 - "test_pipeline_parallel.py"
Cohesion: 0.22
Nodes (9): _format_perf(), Performance timing data class shared across the package., StreamPerf, _FakeLLM, _FakeRetriever, test_answer_stream_calls_each_retriever_and_each_llm(), test_answer_stream_collects_tokens_in_order(), test_answer_stream_empty_hits_still_yields_tokens() (+1 more)

### Community 45 - "build_reranker"
Cohesion: 0.39
Nodes (6): LogCaptureFixture, Config, build_reranker(), Build a reranker from config, or return None if disabled or unavailable., Tests for build_reranker factory., TestBuildReranker

### Community 47 - "runner.py"
Cohesion: 0.32
Nodes (6): _load_catalog(), _make_llm(), _process_row(), Any, Run a prepared Q&A CSV through the RAG pipeline and evaluate it.  Pacing is dele, DeepSeek LLM client; uses the OpenAI SDK with DeepSeek's base URL.

## Knowledge Gaps
- **34 isolated node(s):** `rag-learn`, `苏帕摩-中度烘焙`, `耶加 TOH亚军地块-中度烘焙`, `达摩-中浅烘焙`, `Tiny` (+29 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Hit` connect `test_app_launch.py` to `Eval Event Tracing`, `BM25Index`, `Batch Loader & Dedup`, `Collections Catalog`, `retriever/base.py`, `test_tracing.py`, `app.py`, `pipeline.py`, `Reranker`, `Rate Limiting Primitives`, `Batch Metric Computation`, `Batch Eval Runner`, `Milvus Retriever Adapter`, `Eval CLI Dispatch`, `test_pipeline_parallel.py`, `HybridRetriever`, `loader.py`, `test_e2e.py`?**
  _High betweenness centrality (0.215) - this node is a cross-community bridge._
- **Why does `answer_stream()` connect `pipeline.py` to `Eval Event Tracing`, `Batch Loader & Dedup`, `loader.py`, `Collections Catalog`, `retriever/base.py`, `test_e2e.py`, `Batch Metric Computation`, `app.py`, `LLM Judge & DeepSeek Client`, `Reranker`, `test_pipeline_parallel.py`, `E2E Test Stubs`, `build_reranker`, `runner.py`, `test_app_launch.py`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Why does `Catalog` connect `Collections Catalog` to `Eval Event Tracing`, `Batch Loader & Dedup`, `loader.py`, `test_e2e.py`, `Eval CLI Dispatch`, `app.py`, `pipeline.py`, `runner.py`, `test_app_launch.py`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `Hit` (e.g. with `_make_event()` and `_hit()`) actually correct?**
  _`Hit` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `Catalog` (e.g. with `StubRetriever` and `FakeRetriever`) actually correct?**
  _`Catalog` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 24 inferred relationships involving `Config` (e.g. with `StubRetriever` and `_AltStub`) actually correct?**
  _`Config` has 24 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `Collection` (e.g. with `StubRetriever` and `FakeRetriever`) actually correct?**
  _`Collection` has 19 INFERRED edges - model-reasoned connections that need verification._