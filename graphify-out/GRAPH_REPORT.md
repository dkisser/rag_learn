# Graph Report - rag_learn  (2026-08-01)

## Corpus Check
- 85 files · ~38,049 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 908 nodes · 2216 edges · 49 communities (39 shown, 10 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 207 edges (avg confidence: 0.61)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `1d7a5ef1`
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
- test_pipeline.py
- CrossEncoderReranker
- test_pipeline_parallel.py
- build_retriever
- build_reranker
- config.py
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
- `StubRetriever` --uses--> `Collection`  [INFERRED]
  tests/test_app_launch.py → src/rag_learn/collections.py
- `FakeRetriever` --uses--> `Collection`  [INFERRED]
  tests/test_collections.py → src/rag_learn/collections.py
- `_AltStub` --uses--> `Collection`  [INFERRED]
  tests/test_e2e.py → src/rag_learn/collections.py
- `_NoopRetriever` --uses--> `Collection`  [INFERRED]
  tests/test_pipeline_catalog_fanout.py → src/rag_learn/collections.py
- `_RecordingRetriever` --uses--> `Collection`  [INFERRED]
  tests/test_pipeline_catalog_fanout.py → src/rag_learn/collections.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **rag-learn Retrieval + Streaming Pipeline** — _claude_claude_chroma_retriever, _claude_claude_milvus_lite_retriever, _claude_claude_deepseek_llm, _claude_claude_gradio_ui [INFERRED 0.75]
- **graphify Skill Pipeline & References** — _claude_skills_graphify_skill_graphify, _claude_skills_graphify_references_extraction_spec_spec, _claude_skills_graphify_references_query_query, _claude_skills_graphify_references_update_update [EXTRACTED 1.00]
- **RAG retrieval flow** — readme_rag_learn, readme_chroma, readme_persistentclient, readme_all_minilm_l6_v2, readme_gradio, readme_gr_chatbot, readme_deepseek, readme_deepseekllm_stream [EXTRACTED 1.00]
- **batch evaluation pipeline** — readme_rag_learn_eval_cli, readme_csv_template, readme_ratelimiter, readme_shanzhongshi_qa_csv, readme_2026_07_22_batch_evaluation_design_md [EXTRACTED 1.00]

## Communities (49 total, 10 thin omitted)

### Community 0 - "Eval Event Tracing"
Cohesion: 0.14
Nodes (8): Hit, _FakeLLM, _FakeReranker, _FakeRetriever, test_answer_stream_no_emitter_does_not_record(), test_answer_stream_no_reranker_uses_vector_order(), test_answer_stream_reranker_reorders_and_truncates_to_k(), test_answer_stream_with_config_over_fetches_candidates()

### Community 1 - "Batch Loader & Dedup"
Cohesion: 0.25
Nodes (22): main(), JSONLEmitter, Path, _make_event(), MonkeyPatch, Path, Tests for the batch evaluation CLI., Empty source_files means 'rely on model knowledge' — answer_f1 still applies. (+14 more)

### Community 2 - "Gradio App Surface"
Cohesion: 0.18
Nodes (16): ConfigError, load_config(), Raised when required configuration is missing or invalid., 新增的 6 个 routing 字段默认值(开关默认关,其余走 dataclass 默认)., test_load_config_boolean_parsing(), test_load_config_hybrid_overrides(), test_load_config_missing_api_key_raises(), test_load_config_overrides() (+8 more)

### Community 3 - "Collections Catalog"
Cohesion: 0.06
Nodes (46): KeyError, Catalog, Collection, CollectionNotFoundError, Eager 触发每个 collection 的 retriever 懒加载。fail-open.          Returns list of (colle, 一个独立的知识库：slug + 显示元数据 + 文档目录 + retriever 工厂。      `retriever` 是懒加载属性：首次访问时由 `ret, 请求的 collection 不在 Catalog 里。, 不可变集合注册表：slug → Collection 双向索引。 (+38 more)

### Community 4 - "Document Loader & Chunking"
Cohesion: 0.09
Nodes (31): _chunk_size(), _chunk_text(), iter_markdown(), load_documents(), Path, Markdown discovery + chunking for ingestion., Top-level entry: chunk a single file's content., Read all *.md in docs_dir and return a flat list of chunks. (+23 more)

### Community 5 - "Eval CLI Dispatch"
Cohesion: 0.07
Nodes (38): ArgumentParser, _build_parser(), main(), CLI entry point for batch RAG evaluation., format_csv_row(), parse_csv_row(), CSV row parsing and formatting for batch evaluation., Parse a CSV row into question, collection, and optional ground truth.      Retur (+30 more)

### Community 6 - "Batch Metric Computation"
Cohesion: 0.11
Nodes (40): _aggregate(), _compute_supervised(), _compute_unsupervised(), _ground_truth_to_dict(), _make_judge_fn(), Any, Batch evaluation CLI for RAG events stored in JSONL., Run unsupervised judge metrics under a shared ``RateLimiter``.      All metrics (+32 more)

### Community 7 - "LLM Judge & DeepSeek Client"
Cohesion: 0.11
Nodes (22): DeepSeekLLM, Any, DeepSeek LLM client; uses the OpenAI SDK with DeepSeek's base URL., test_e2e_full_pipeline_runs(), _FakeChat, _FakeChoice, _FakeChunk, _FakeClient (+14 more)

### Community 8 - "Milvus Retriever Adapter"
Cohesion: 0.10
Nodes (24): _load_collection_subprocess(), MilvusRetriever, Any, Path, Subprocess entry point: open MilvusClient and load the collection.      Runs in, Run target(*args) in an isolated subprocess.      Returns True iff the subproces, Isolated wrapper around MilvusClient.load_collection.      Returns True on succe, _run_isolated() (+16 more)

### Community 9 - "Rate Limiting Primitives"
Cohesion: 0.27
Nodes (8): ChromaRetriever, Path, chroma_dir(), Path, test_chroma_retriever_ensure_indexed_then_search(), test_chroma_retriever_is_base_retriever(), test_chroma_retriever_is_idempotent(), test_chroma_retriever_second_collection_reuses_persisted()

### Community 10 - "Batch Eval Runner"
Cohesion: 0.18
Nodes (22): _load_existing_keys(), Path, Read a Q&A CSV, run each question through RAG, emit events, and evaluate., Return (collection, question) pairs already emitted to ``events_file``.      Rea, run_qa_csv(), _find_events_file(), MonkeyPatch, Path (+14 more)

### Community 11 - "E2E Test Stubs"
Cohesion: 0.08
Nodes (35): INTENT_LABELS, classify_intent(), decompose_query(), _drain_stream(), _LLMStream, _parse_intent(), _parse_subqueries(), Protocol (+27 more)

### Community 12 - "README Concept References"
Cohesion: 0.10
Nodes (24): 2026-07-22-batch-evaluation-design.md, all-MiniLM-L6-v2, Chroma, evaluation CSV template, data/chroma/, DeepSeek, DeepSeekLLM.stream, gr.Chatbot (+16 more)

### Community 13 - "Logging Configuration"
Cohesion: 0.14
Nodes (21): Handler, create_handlers(), get_log_level(), Path, Process-wide logging configuration., Walk up from ``start_path`` until we find pyproject.toml.      Robust to worktre, Resolve a log-level name to a ``logging`` level integer.      Reads from the ``L, Create console and file handlers, ensuring the log directory exists. (+13 more)

### Community 14 - "Graphify Skill Docs"
Cohesion: 0.22
Nodes (9): Add URL & Watch Folder Reference, Exports & Benchmark Reference, Extraction Subagent Spec, GitHub Clone & Cross-Repo Merge Reference, Commit Hook & CLAUDE.md Integration Reference, Query / Path / Explain Reference, Video/Audio Transcribe Reference, Incremental Update Reference (+1 more)

### Community 15 - "HybridRetriever"
Cohesion: 0.08
Nodes (40): Chunk, BM25Index, _IndexedChunk, BM25 keyword index built on top of jieba + rank-bm25.  Used by ``HybridRetriever, Internal record: original chunk + its tokenized text., Tokenize text with jieba, dropping whitespace and pure-punctuation tokens., In-memory BM25 keyword index., (Re)build the index from a fresh chunk list. (+32 more)

### Community 16 - "CLAUDE.md Gotchas"
Cohesion: 0.67
Nodes (4): Chroma Retriever, Gradio UI, Milvus Lite Retriever, rag-learn Project

### Community 26 - "test_app_launch.py"
Cohesion: 0.13
Nodes (29): answer_stream(), Parallel retrieve → build prompt per side → stream tokens per side.      Returns, _FakeReranker, _make_catalog(), _make_config(), _NoopRetriever, _PerQueryFakeRetriever, Path (+21 more)

### Community 27 - "doc_beans.md"
Cohesion: 0.50
Nodes (3): 耶加 TOH亚军地块-中度烘焙, 苏帕摩-中度烘焙, 达摩-中浅烘焙

### Community 31 - "RateLimiter"
Cohesion: 0.10
Nodes (26): BaseException, RuntimeError, is_rate_limit_error(), Any, Limiter, RateLimiter, Thin wrapper composing pyrate-limiter + threading.Semaphore + tenacity.  Used by, Return True iff ``exc`` represents an HTTP 429 from any layer. (+18 more)

### Community 32 - "loader.py"
Cohesion: 0.06
Nodes (54): _format_routing(), Render a one-line caption summarizing routing decisions.      ``routing`` is the, _build_builtin(), _default_factory(), _make_factory(), Path, Collection domain object: a single knowledge base (name, docs, retriever)., Build a retriever factory that captures the hybrid config. (+46 more)

### Community 33 - "BM25Index"
Cohesion: 0.17
Nodes (23): _migrate_legacy_chroma(), 一次性：把 data/chroma/ 根下的遗留文件搬到 data/chroma/rag_doc/。      触发条件：data/chroma/rag_doc, _make_config(), Any, MonkeyPatch, Path, I/O failure during migration must not crash startup., Satisfies BaseRetriever Protocol without touching Chroma. (+15 more)

### Community 34 - "conftest.py"
Cohesion: 0.15
Nodes (18): _dedupe(), _load_events(), Path, Load events from either a single .jsonl file or a directory of them.      When `, _safe_judge(), event_from_dict(), _event_to_dict(), GroundTruth (+10 more)

### Community 35 - "test_e2e.py"
Cohesion: 0.16
Nodes (20): Blocks, build_app(), Any, Construct the Gradio UI but do not launch it.      Args:         catalog: The co, _AltStub, _make_config(), Any, Path (+12 more)

### Community 36 - "retriever/base.py"
Cohesion: 0.12
Nodes (12): Cross-encoder reranker implementation backed by sentence-transformers., BaseRetriever, Protocol, Retriever contract shared by all adapter implementations., Chroma adapter implementing BaseRetriever via PersistentClient + default embedde, Factory that picks the right retriever implementation based on config., Hybrid retriever: Chroma (vector) + BM25 (keyword), fused via RRF., Milvus Lite (embedded) adapter implementing BaseRetriever.  Uses pymilvus.model. (+4 more)

### Community 37 - "test_tracing.py"
Cohesion: 0.23
Nodes (17): _make_event(), Path, With file_name set, JSONLEmitter writes to that exact file regardless of timesta, file_name may include subdirectories; parent dirs are created., Backwards compat: no file_name → keeps the rag_events_<date>.jsonl behavior., test_event_round_trip_via_jsonl(), test_event_round_trip_without_ground_truth(), test_ground_truth_defaults() (+9 more)

### Community 38 - "app.py"
Cohesion: 0.16
Nodes (12): main(), CLI shim: `python main.py` → launch the Gradio RAG compare app., _drain_to_chatbot(), _format_chunks(), _format_perf(), launch(), Gradio UI: collection dropdown + single answer panel with chunks and perf metric, Production entry: load config, build catalog + LLM, migrate, ingest, serve. (+4 more)

### Community 39 - "pipeline.py"
Cohesion: 0.18
Nodes (13): _answer_catalog_recall(), _flat_retrieve(), _make_perf(), _merge_dedup(), _now_hms_ms(), Any, RAG pipeline prompt construction, parallel retrieval, and streaming perf.  `answ, Run all retrievers in parallel (threads); return their Hits per side.      Each (+5 more)

### Community 40 - "Reranker"
Cohesion: 0.21
Nodes (8): Protocol, Reranker contract shared by all implementations., Return a new list of hits sorted by descending relevance.          The returned, Reranker, Factory for building a reranker from configuration., Reranker components for refining retrieval results., Structural tests for the Reranker Protocol., TestRerankerProtocol

### Community 41 - "test_pipeline.py"
Cohesion: 0.35
Nodes (10): PromptMode, build_prompt(), Return ``(system_msg, user_msg)`` with display-safe chunk lengths., _hits(), test_build_prompt_empty_hits_has_empty_prompt_branch(), test_build_prompt_includes_question(), test_build_prompt_lists_each_chunk_with_source(), test_build_prompt_numbering_starts_at_one() (+2 more)

### Community 42 - "CrossEncoderReranker"
Cohesion: 0.25
Nodes (5): CrossEncoderReranker, Local cross-encoder reranker using sentence-transformers CrossEncoder., Score each (query, hit) pair and return hits sorted by descending score., Unit tests for CrossEncoderReranker using mocked CrossEncoder., TestCrossEncoderReranker

### Community 43 - "test_pipeline_parallel.py"
Cohesion: 0.33
Nodes (6): _FakeLLM, _FakeRetriever, test_answer_stream_calls_each_retriever_and_each_llm(), test_answer_stream_collects_tokens_in_order(), test_answer_stream_empty_hits_still_yields_tokens(), test_answer_stream_returns_both_sides()

### Community 44 - "build_retriever"
Cohesion: 0.33
Nodes (8): build_retriever(), Path, Build a retriever instance.      When ``hybrid_enabled`` is True, wraps ``Chroma, Path, Tests for the build_retriever factory., test_hybrid_default_rrf_k(), test_returns_chroma_when_hybrid_disabled(), test_returns_hybrid_when_enabled()

### Community 45 - "build_reranker"
Cohesion: 0.32
Nodes (5): LogCaptureFixture, build_reranker(), Build a reranker from config, or return None if disabled or unavailable., Tests for build_reranker factory., TestBuildReranker

### Community 46 - "config.py"
Cohesion: 0.25
Nodes (6): _parse_bool(), Path, Process-wide configuration loaded once from environment variables., Walk up from this file until we find pyproject.toml.      Robust to worktree dir, _repo_root(), Tests for the reranker module.

### Community 47 - "runner.py"
Cohesion: 0.47
Nodes (5): _load_catalog(), _make_llm(), _process_row(), Any, Run a prepared Q&A CSV through the RAG pipeline and evaluate it.  Pacing is dele

## Knowledge Gaps
- **34 isolated node(s):** `rag-learn`, `苏帕摩-中度烘焙`, `耶加 TOH亚军地块-中度烘焙`, `达摩-中浅烘焙`, `Tiny` (+29 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Hit` connect `Eval Event Tracing` to `Batch Loader & Dedup`, `Collections Catalog`, `Document Loader & Chunking`, `Eval CLI Dispatch`, `Batch Metric Computation`, `Milvus Retriever Adapter`, `Rate Limiting Primitives`, `Batch Eval Runner`, `HybridRetriever`, `test_app_launch.py`, `loader.py`, `BM25Index`, `conftest.py`, `test_e2e.py`, `retriever/base.py`, `test_tracing.py`, `app.py`, `pipeline.py`, `Reranker`, `test_pipeline.py`, `CrossEncoderReranker`, `test_pipeline_parallel.py`?**
  _High betweenness centrality (0.213) - this node is a cross-community bridge._
- **Why does `answer_stream()` connect `test_app_launch.py` to `loader.py`, `Eval Event Tracing`, `conftest.py`, `Collections Catalog`, `test_e2e.py`, `retriever/base.py`, `app.py`, `pipeline.py`, `LLM Judge & DeepSeek Client`, `Reranker`, `E2E Test Stubs`, `test_pipeline_parallel.py`, `runner.py`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Why does `Catalog` connect `Collections Catalog` to `loader.py`, `BM25Index`, `test_e2e.py`, `app.py`, `pipeline.py`, `Batch Eval Runner`, `runner.py`, `test_app_launch.py`?**
  _High betweenness centrality (0.074) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `Hit` (e.g. with `_make_event()` and `_hit()`) actually correct?**
  _`Hit` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `Catalog` (e.g. with `StubRetriever` and `FakeRetriever`) actually correct?**
  _`Catalog` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 24 inferred relationships involving `Config` (e.g. with `StubRetriever` and `_AltStub`) actually correct?**
  _`Config` has 24 INFERRED edges - model-reasoned connections that need verification._
- **Are the 19 inferred relationships involving `Collection` (e.g. with `StubRetriever` and `FakeRetriever`) actually correct?**
  _`Collection` has 19 INFERRED edges - model-reasoned connections that need verification._