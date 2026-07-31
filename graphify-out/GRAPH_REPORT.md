# Graph Report - rag_learn  (2026-07-31)

## Corpus Check
- 81 files · ~35,462 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 825 nodes · 1962 edges · 35 communities (26 shown, 9 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 161 edges (avg confidence: 0.61)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `cc2c8452`
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

## God Nodes (most connected - your core abstractions)
1. `Hit` - 75 edges
2. `Catalog` - 46 edges
3. `answer_stream()` - 40 edges
4. `DeepSeekLLM` - 36 edges
5. `Config` - 35 edges
6. `RAGEvent` - 34 edges
7. `JSONLEmitter` - 32 edges
8. `Collection` - 30 edges
9. `StreamPerf` - 27 edges
10. `load_config()` - 25 edges

## Surprising Connections (you probably didn't know these)
- `FakeRetriever` --uses--> `Collection`  [INFERRED]
  tests/test_collections.py → src/rag_learn/collections.py
- `_AltStub` --uses--> `Collection`  [INFERRED]
  tests/test_e2e.py → src/rag_learn/collections.py
- `_FakeChoice` --uses--> `Collection`  [INFERRED]
  tests/test_e2e.py → src/rag_learn/collections.py
- `_FakeChunk` --uses--> `Collection`  [INFERRED]
  tests/test_e2e.py → src/rag_learn/collections.py
- `_FakeStream` --uses--> `Collection`  [INFERRED]
  tests/test_e2e.py → src/rag_learn/collections.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **rag-learn Retrieval + Streaming Pipeline** — _claude_claude_chroma_retriever, _claude_claude_milvus_lite_retriever, _claude_claude_deepseek_llm, _claude_claude_gradio_ui [INFERRED 0.75]
- **graphify Skill Pipeline & References** — _claude_skills_graphify_skill_graphify, _claude_skills_graphify_references_extraction_spec_spec, _claude_skills_graphify_references_query_query, _claude_skills_graphify_references_update_update [EXTRACTED 1.00]
- **RAG retrieval flow** — readme_rag_learn, readme_chroma, readme_persistentclient, readme_all_minilm_l6_v2, readme_gradio, readme_gr_chatbot, readme_deepseek, readme_deepseekllm_stream [EXTRACTED 1.00]
- **batch evaluation pipeline** — readme_rag_learn_eval_cli, readme_csv_template, readme_ratelimiter, readme_shanzhongshi_qa_csv, readme_2026_07_22_batch_evaluation_design_md [EXTRACTED 1.00]

## Communities (35 total, 9 thin omitted)

### Community 0 - "Eval Event Tracing"
Cohesion: 0.06
Nodes (54): PromptMode, ListEmitter, MetricsEmitter, Protocol, Performance timing data class shared across the package., StreamPerf, _answer_catalog_recall(), answer_stream() (+46 more)

### Community 1 - "Batch Loader & Dedup"
Cohesion: 0.11
Nodes (44): _aggregate(), _ground_truth_to_dict(), main(), Path, GroundTruth, JSONLEmitter, NullEmitter, Path (+36 more)

### Community 2 - "Gradio App Surface"
Cohesion: 0.05
Nodes (48): LogCaptureFixture, main(), CLI shim: `python main.py` → launch the Gradio RAG compare app., _drain_to_chatbot(), _format_chunks(), _format_routing(), launch(), Any (+40 more)

### Community 3 - "Collections Catalog"
Cohesion: 0.06
Nodes (64): KeyError, _migrate_legacy_chroma(), 一次性：把 data/chroma/ 根下的遗留文件搬到 data/chroma/rag_doc/。      触发条件：data/chroma/rag_doc, _build_builtin(), build_catalog(), Catalog, Collection, CollectionNotFoundError (+56 more)

### Community 4 - "Document Loader & Chunking"
Cohesion: 0.11
Nodes (20): iter_markdown(), load_documents(), Path, Read all *.md in docs_dir and return a flat list of chunks., Return list of (filename, raw_text) sorted by filename, deterministic., Index once for the vector store and once for the BM25 keyword index., Each H1's text must appear inside the resulting chunk so the embedding     can m, Multi-H1 fixture (bean-card layout): every chunk must mention only its     own b (+12 more)

### Community 5 - "Eval CLI Dispatch"
Cohesion: 0.14
Nodes (21): ArgumentParser, _build_parser(), main(), format_csv_row(), Return a row dict suitable for DictWriter, with empty optional fields., _load_events(), Path, Sample online RAG events into a CSV for manual labeling. (+13 more)

### Community 6 - "Batch Metric Computation"
Cohesion: 0.05
Nodes (64): _compute_supervised(), _compute_unsupervised(), _dedupe(), _load_events(), _make_judge_fn(), Any, Batch evaluation CLI for RAG events stored in JSONL., Run unsupervised judge metrics under a shared ``RateLimiter``.      All metrics (+56 more)

### Community 7 - "LLM Judge & DeepSeek Client"
Cohesion: 0.06
Nodes (45): Blocks, build_app(), _format_perf(), Construct the Gradio UI but do not launch it.      Args:         catalog: The co, DeepSeekLLM, Any, _AltStub, _FakeChoice (+37 more)

### Community 8 - "Milvus Retriever Adapter"
Cohesion: 0.10
Nodes (25): _load_collection_subprocess(), MilvusRetriever, Any, Path, Milvus Lite (embedded) adapter implementing BaseRetriever.  Uses pymilvus.model., Subprocess entry point: open MilvusClient and load the collection.      Runs in, Run target(*args) in an isolated subprocess.      Returns True iff the subproces, Isolated wrapper around MilvusClient.load_collection.      Returns True on succe (+17 more)

### Community 9 - "Rate Limiting Primitives"
Cohesion: 0.13
Nodes (20): Retriever contract shared by all adapter implementations., ChromaRetriever, Path, Chroma adapter implementing BaseRetriever via PersistentClient + default embedde, build_retriever(), Path, Factory that picks the right retriever implementation based on config., Build a retriever instance.      When ``hybrid_enabled`` is True, wraps ``Chroma (+12 more)

### Community 10 - "Batch Eval Runner"
Cohesion: 0.13
Nodes (29): _load_catalog(), _load_existing_keys(), _make_llm(), Any, Path, Read a Q&A CSV, run each question through RAG, emit events, and evaluate., Return (collection, question) pairs already emitted to ``events_file``.      Rea, run_qa_csv() (+21 more)

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
Cohesion: 0.17
Nodes (20): HybridRetriever, Run vector and keyword retrieval side-by-side; fuse with RRF.      Implements ``, Fetch top-k from each retriever and fuse with Reciprocal Rank Fusion., _FakeBM25, _FakeVectorRetriever, _hit(), Tests for HybridRetriever (vector + BM25, fused via RRF)., Implements BaseRetriever via duck-typing. (+12 more)

### Community 16 - "CLAUDE.md Gotchas"
Cohesion: 0.67
Nodes (4): Chroma Retriever, Gradio UI, Milvus Lite Retriever, rag-learn Project

### Community 26 - "test_app_launch.py"
Cohesion: 0.12
Nodes (27): _FakeReranker, _make_catalog(), _make_config(), _NoopRetriever, _PerQueryFakeRetriever, Path, Integration tests for intent-aware routing in pipeline.answer_stream., When intent_enabled=False, the LLM's classify intent is never invoked. (+19 more)

### Community 27 - "doc_beans.md"
Cohesion: 0.50
Nodes (3): 耶加 TOH亚军地块-中度烘焙, 苏帕摩-中度烘焙, 达摩-中浅烘焙

### Community 31 - "RateLimiter"
Cohesion: 0.10
Nodes (26): BaseException, RuntimeError, is_rate_limit_error(), Any, Limiter, RateLimiter, Thin wrapper composing pyrate-limiter + threading.Semaphore + tenacity.  Used by, Return True iff ``exc`` represents an HTTP 429 from any layer. (+18 more)

### Community 32 - "loader.py"
Cohesion: 0.17
Nodes (15): Chunk, _chunk_size(), _chunk_text(), Markdown discovery + chunking for ingestion., Top-level entry: chunk a single file's content., Split a markdown file by H1 headings.      Each pre-doc starts with its H1 line, Greedy paragraph-then-sentence packing up to CHUNK_SIZE with OVERLAP., split_into_chunks() (+7 more)

### Community 33 - "BM25Index"
Cohesion: 0.18
Nodes (13): BM25Index, In-memory BM25 keyword index., Return up to ``k`` hits sorted by descending BM25 score., Path, _chunks(), Tests for the BM25 keyword index., test_build_replaces_existing_index(), test_search_finds_keyword_match_and_orders_by_score() (+5 more)

### Community 34 - "conftest.py"
Cohesion: 0.67
Nodes (3): fixtures_dir(), Path, sample_hits()

## Knowledge Gaps
- **34 isolated node(s):** `rag-learn`, `苏帕摩-中度烘焙`, `耶加 TOH亚军地块-中度烘焙`, `达摩-中浅烘焙`, `Tiny` (+29 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Hit` connect `Eval Event Tracing` to `loader.py`, `BM25Index`, `Gradio App Surface`, `conftest.py`, `Batch Loader & Dedup`, `Eval CLI Dispatch`, `Batch Metric Computation`, `Collections Catalog`, `Milvus Retriever Adapter`, `Rate Limiting Primitives`, `Batch Eval Runner`, `LLM Judge & DeepSeek Client`, `HybridRetriever`, `test_app_launch.py`?**
  _High betweenness centrality (0.220) - this node is a cross-community bridge._
- **Why does `answer_stream()` connect `Eval Event Tracing` to `Gradio App Surface`, `Collections Catalog`, `Batch Metric Computation`, `LLM Judge & DeepSeek Client`, `E2E Test Stubs`, `test_app_launch.py`?**
  _High betweenness centrality (0.083) - this node is a cross-community bridge._
- **Why does `DeepSeekLLM` connect `LLM Judge & DeepSeek Client` to `Eval Event Tracing`, `Gradio App Surface`, `Batch Eval Runner`, `Batch Metric Computation`?**
  _High betweenness centrality (0.073) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `Hit` (e.g. with `_make_event()` and `_hit()`) actually correct?**
  _`Hit` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `Catalog` (e.g. with `StubRetriever` and `FakeRetriever`) actually correct?**
  _`Catalog` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `DeepSeekLLM` (e.g. with `_AltStub` and `_FakeChoice`) actually correct?**
  _`DeepSeekLLM` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `Config` (e.g. with `StubRetriever` and `_AltStub`) actually correct?**
  _`Config` has 15 INFERRED edges - model-reasoned connections that need verification._