# Graph Report - rag_learn  (2026-07-29)

## Corpus Check
- 78 files · ~32,412 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 728 nodes · 1721 edges · 31 communities (21 shown, 10 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 148 edges (avg confidence: 0.62)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `bfc21111`
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

## God Nodes (most connected - your core abstractions)
1. `Hit` - 60 edges
2. `DeepSeekLLM` - 35 edges
3. `Catalog` - 34 edges
4. `RAGEvent` - 33 edges
5. `JSONLEmitter` - 32 edges
6. `Config` - 29 edges
7. `answer_stream()` - 27 edges
8. `StreamPerf` - 26 edges
9. `Collection` - 24 edges
10. `main()` - 24 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `launch()`  [EXTRACTED]
  main.py → src/rag_learn/app.py
- `FakeRetriever` --uses--> `Collection`  [INFERRED]
  tests/test_collections.py → src/rag_learn/collections.py
- `_AltStub` --uses--> `Collection`  [INFERRED]
  tests/test_e2e.py → src/rag_learn/collections.py
- `FakeRetriever` --uses--> `CollectionNotFoundError`  [INFERRED]
  tests/test_collections.py → src/rag_learn/collections.py
- `FakeRetriever` --uses--> `Catalog`  [INFERRED]
  tests/test_collections.py → src/rag_learn/collections.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **rag-learn Retrieval + Streaming Pipeline** — _claude_claude_chroma_retriever, _claude_claude_milvus_lite_retriever, _claude_claude_deepseek_llm, _claude_claude_gradio_ui [INFERRED 0.75]
- **graphify Skill Pipeline & References** — _claude_skills_graphify_skill_graphify, _claude_skills_graphify_references_extraction_spec_spec, _claude_skills_graphify_references_query_query, _claude_skills_graphify_references_update_update [EXTRACTED 1.00]
- **RAG retrieval flow** — readme_rag_learn, readme_chroma, readme_persistentclient, readme_all_minilm_l6_v2, readme_gradio, readme_gr_chatbot, readme_deepseek, readme_deepseekllm_stream [EXTRACTED 1.00]
- **batch evaluation pipeline** — readme_rag_learn_eval_cli, readme_csv_template, readme_ratelimiter, readme_shanzhongshi_qa_csv, readme_2026_07_22_batch_evaluation_design_md [EXTRACTED 1.00]

## Communities (31 total, 10 thin omitted)

### Community 0 - "Eval Event Tracing"
Cohesion: 0.09
Nodes (25): _format_chunks(), build_prompt(), Return ``(system_msg, user_msg)`` with display-safe chunk lengths., Return a new list of hits sorted by descending relevance.          The returned, Hit, _FakeLLM, _FakeRetriever, _hits() (+17 more)

### Community 1 - "Batch Loader & Dedup"
Cohesion: 0.06
Nodes (72): _drain_to_chatbot(), _format_perf(), Gradio UI: collection dropdown + single answer panel with chunks and perf metric, Consume an answer_stream's iterator and return the joined text., _process_row(), Run a prepared Q&A CSV through the RAG pipeline and evaluate it.  Pacing is dele, event_from_dict(), _event_to_dict() (+64 more)

### Community 2 - "Gradio App Surface"
Cohesion: 0.08
Nodes (31): LogCaptureFixture, launch(), Production entry: load config, build catalog + LLM, migrate, ingest, serve., Config, ConfigError, load_config(), _parse_bool(), Process-wide configuration loaded once from environment variables. (+23 more)

### Community 3 - "Collections Catalog"
Cohesion: 0.07
Nodes (53): KeyError, _migrate_legacy_chroma(), 一次性：把 data/chroma/ 根下的遗留文件搬到 data/chroma/rag_doc/。      触发条件：data/chroma/rag_doc, Catalog, Collection, CollectionNotFoundError, Eager 触发每个 collection 的 retriever 懒加载。fail-open.          Returns list of (colle, 一个独立的知识库：slug + 显示元数据 + 文档目录 + retriever 工厂。      `retriever` 是懒加载属性：首次访问时由 `ret (+45 more)

### Community 4 - "Document Loader & Chunking"
Cohesion: 0.05
Nodes (53): Chunk, _chunk_size(), _chunk_text(), iter_markdown(), load_documents(), Path, Markdown discovery + chunking for ingestion., Top-level entry: chunk a single file's content. (+45 more)

### Community 5 - "Eval CLI Dispatch"
Cohesion: 0.07
Nodes (38): ArgumentParser, _build_parser(), main(), CLI entry point for batch RAG evaluation., format_csv_row(), parse_csv_row(), CSV row parsing and formatting for batch evaluation., Parse a CSV row into question, collection, and optional ground truth.      Retur (+30 more)

### Community 6 - "Batch Metric Computation"
Cohesion: 0.05
Nodes (70): BaseException, _aggregate(), _compute_supervised(), _compute_unsupervised(), _dedupe(), _ground_truth_to_dict(), _load_events(), main() (+62 more)

### Community 7 - "LLM Judge & DeepSeek Client"
Cohesion: 0.10
Nodes (23): RuntimeError, DeepSeekLLM, Any, test_is_rate_limit_error_rejects_other_errors(), test_e2e_full_pipeline_runs(), _FakeChat, _FakeChoice, _FakeChunk (+15 more)

### Community 8 - "Milvus Retriever Adapter"
Cohesion: 0.10
Nodes (24): _load_collection_subprocess(), MilvusRetriever, Any, Path, Subprocess entry point: open MilvusClient and load the collection.      Runs in, Run target(*args) in an isolated subprocess.      Returns True iff the subproces, Isolated wrapper around MilvusClient.load_collection.      Returns True on succe, _run_isolated() (+16 more)

### Community 9 - "Rate Limiting Primitives"
Cohesion: 0.07
Nodes (35): _build_builtin(), build_catalog(), _default_factory(), _make_factory(), Path, Collection domain object: a single knowledge base (name, docs, retriever)., Build a retriever factory that captures the hybrid config., Build the default catalog, optionally wiring hybrid retrieval. (+27 more)

### Community 10 - "Batch Eval Runner"
Cohesion: 0.15
Nodes (25): _load_catalog(), _load_existing_keys(), _make_llm(), Any, Path, Read a Q&A CSV, run each question through RAG, emit events, and evaluate., Return (collection, question) pairs already emitted to ``events_file``.      Rea, run_qa_csv() (+17 more)

### Community 11 - "E2E Test Stubs"
Cohesion: 0.16
Nodes (20): Blocks, build_app(), Any, Construct the Gradio UI but do not launch it.      Args:         catalog: The co, _AltStub, _make_config(), Any, Path (+12 more)

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
Cohesion: 0.16
Nodes (22): HybridRetriever, Path, Run vector and keyword retrieval side-by-side; fuse with RRF.      Implements ``, Index once for the vector store and once for the BM25 keyword index., Fetch top-k from each retriever and fuse with Reciprocal Rank Fusion., _FakeBM25, _FakeVectorRetriever, _hit() (+14 more)

### Community 16 - "CLAUDE.md Gotchas"
Cohesion: 0.67
Nodes (4): Chroma Retriever, Gradio UI, Milvus Lite Retriever, rag-learn Project

### Community 27 - "doc_beans.md"
Cohesion: 0.50
Nodes (3): 耶加 TOH亚军地块-中度烘焙, 苏帕摩-中度烘焙, 达摩-中浅烘焙

## Knowledge Gaps
- **34 isolated node(s):** `rag-learn`, `苏帕摩-中度烘焙`, `耶加 TOH亚军地块-中度烘焙`, `达摩-中浅烘焙`, `Tiny` (+29 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Hit` connect `Eval Event Tracing` to `Batch Loader & Dedup`, `Gradio App Surface`, `Collections Catalog`, `Document Loader & Chunking`, `Eval CLI Dispatch`, `Batch Metric Computation`, `Milvus Retriever Adapter`, `Rate Limiting Primitives`, `Batch Eval Runner`, `E2E Test Stubs`, `HybridRetriever`?**
  _High betweenness centrality (0.213) - this node is a cross-community bridge._
- **Why does `DeepSeekLLM` connect `LLM Judge & DeepSeek Client` to `Batch Loader & Dedup`, `Gradio App Surface`, `Collections Catalog`, `Batch Metric Computation`, `Batch Eval Runner`, `E2E Test Stubs`?**
  _High betweenness centrality (0.080) - this node is a cross-community bridge._
- **Why does `Catalog` connect `Collections Catalog` to `Batch Loader & Dedup`, `Gradio App Surface`, `Rate Limiting Primitives`, `Batch Eval Runner`, `E2E Test Stubs`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `Hit` (e.g. with `_make_event()` and `_hit()`) actually correct?**
  _`Hit` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `DeepSeekLLM` (e.g. with `_AltStub` and `_FakeChoice`) actually correct?**
  _`DeepSeekLLM` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `Catalog` (e.g. with `StubRetriever` and `FakeRetriever`) actually correct?**
  _`Catalog` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `RAGEvent` (e.g. with `StreamPerf` and `_FakeLLM`) actually correct?**
  _`RAGEvent` has 5 INFERRED edges - model-reasoned connections that need verification._