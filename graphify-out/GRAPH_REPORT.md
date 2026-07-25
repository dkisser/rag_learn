# Graph Report - .  (2026-07-24)

## Corpus Check
- 66 files · ~63,057 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 588 nodes · 1382 edges · 26 communities (18 shown, 8 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 127 edges (avg confidence: 0.61)
- Token cost: 0 input · 0 output

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
- Eval CLI Tests
- CLAUDE.md Gotchas
- Config Errors
- DeepSeek LLM Anchor
- Package Distribution
- README Anchor
- Coffee Sample Doc
- Tea Sample Doc
- Recipe Sample Doc

## God Nodes (most connected - your core abstractions)
1. `Hit` - 51 edges
2. `Catalog` - 36 edges
3. `DeepSeekLLM` - 35 edges
4. `RAGEvent` - 32 edges
5. `JSONLEmitter` - 31 edges
6. `Collection` - 26 edges
7. `StreamPerf` - 25 edges
8. `main()` - 23 edges
9. `BaseRetriever` - 22 edges
10. `answer_stream()` - 21 edges

## Surprising Connections (you probably didn't know these)
- `StubRetriever` --uses--> `Collection`  [INFERRED]
  tests/test_app_launch.py → src/rag_learn/collections.py
- `_AltStub` --uses--> `Collection`  [INFERRED]
  tests/test_e2e.py → src/rag_learn/collections.py
- `_FakeStream` --uses--> `Collection`  [INFERRED]
  tests/test_e2e.py → src/rag_learn/collections.py
- `StubRetriever` --uses--> `Catalog`  [INFERRED]
  tests/test_app_launch.py → src/rag_learn/collections.py
- `_AltStub` --uses--> `Catalog`  [INFERRED]
  tests/test_e2e.py → src/rag_learn/collections.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **rag-learn Retrieval + Streaming Pipeline** — _claude_claude_chroma_retriever, _claude_claude_milvus_lite_retriever, _claude_claude_deepseek_llm, _claude_claude_gradio_ui [INFERRED 0.75]
- **graphify Skill Pipeline & References** — _claude_skills_graphify_skill_graphify, _claude_skills_graphify_references_extraction_spec_spec, _claude_skills_graphify_references_query_query, _claude_skills_graphify_references_update_update [EXTRACTED 1.00]
- **RAG retrieval flow** — readme_rag_learn, readme_chroma, readme_persistentclient, readme_all_minilm_l6_v2, readme_gradio, readme_gr_chatbot, readme_deepseek, readme_deepseekllm_stream [EXTRACTED 1.00]
- **batch evaluation pipeline** — readme_rag_learn_eval_cli, readme_csv_template, readme_ratelimiter, readme_shanzhongshi_qa_csv, readme_2026_07_22_batch_evaluation_design_md [EXTRACTED 1.00]

## Communities (26 total, 8 thin omitted)

### Community 0 - "Eval Event Tracing"
Cohesion: 0.07
Nodes (40): MetricsEmitter, Protocol, RAG event model, emitters, and JSONL serialization., Performance timing data class shared across the package., StreamPerf, answer_stream(), build_prompt(), _make_perf() (+32 more)

### Community 1 - "Batch Loader & Dedup"
Cohesion: 0.10
Nodes (47): _aggregate(), _load_events(), main(), Path, Load events from either a single .jsonl file or a directory of them.      When `, event_from_dict(), _event_to_dict(), GroundTruth (+39 more)

### Community 2 - "Gradio App Surface"
Cohesion: 0.08
Nodes (45): Blocks, main(), CLI shim: `python main.py` → launch the Gradio RAG compare app., build_app(), _drain_to_chatbot(), _format_chunks(), _format_perf(), launch() (+37 more)

### Community 3 - "Collections Catalog"
Cohesion: 0.09
Nodes (35): KeyError, _build_builtin(), Catalog, Collection, CollectionNotFoundError, Collection domain object: a single knowledge base (name, docs, retriever)., Eager 触发每个 collection 的 retriever 懒加载。fail-open.          Returns list of (colle, 一个独立的知识库：slug + 显示元数据 + 文档目录 + retriever 工厂。      `retriever` 是懒加载属性：首次访问时由 `ret (+27 more)

### Community 4 - "Document Loader & Chunking"
Cohesion: 0.07
Nodes (36): _default_factory(), Path, Chunk, _chunk_size(), _chunk_text(), iter_markdown(), load_documents(), Path (+28 more)

### Community 5 - "Eval CLI Dispatch"
Cohesion: 0.07
Nodes (38): ArgumentParser, _build_parser(), main(), CLI entry point for batch RAG evaluation., format_csv_row(), parse_csv_row(), CSV row parsing and formatting for batch evaluation., Parse a CSV row into question, collection, and optional ground truth.      Retur (+30 more)

### Community 6 - "Batch Metric Computation"
Cohesion: 0.12
Nodes (40): _compute_supervised(), _compute_unsupervised(), _dedupe(), _ground_truth_to_dict(), Any, Batch evaluation CLI for RAG events stored in JSONL., Run unsupervised judge metrics under a shared ``RateLimiter``.      All metrics, _safe_judge() (+32 more)

### Community 7 - "LLM Judge & DeepSeek Client"
Cohesion: 0.10
Nodes (23): _make_judge_fn(), Build the default LLM-based judge. Tests monkeypatch this to inject failures., DeepSeekLLM, Any, DeepSeek LLM client; uses the OpenAI SDK with DeepSeek's base URL., _FakeChat, _FakeChoice, _FakeChunk (+15 more)

### Community 8 - "Milvus Retriever Adapter"
Cohesion: 0.10
Nodes (25): _load_collection_subprocess(), MilvusRetriever, Any, Path, Milvus Lite (embedded) adapter implementing BaseRetriever.  Uses pymilvus.model., Subprocess entry point: open MilvusClient and load the collection.      Runs in, Run target(*args) in an isolated subprocess.      Returns True iff the subproces, Isolated wrapper around MilvusClient.load_collection.      Returns True on succe (+17 more)

### Community 9 - "Rate Limiting Primitives"
Cohesion: 0.10
Nodes (26): BaseException, RuntimeError, is_rate_limit_error(), Any, Limiter, RateLimiter, Thin wrapper composing pyrate-limiter + threading.Semaphore + tenacity.  Used by, Return True iff ``exc`` represents an HTTP 429 from any layer. (+18 more)

### Community 10 - "Batch Eval Runner"
Cohesion: 0.15
Nodes (27): build_catalog(), _load_catalog(), _load_existing_keys(), _make_llm(), _process_row(), Any, Path, Run a prepared Q&A CSV through the RAG pipeline and evaluate it.  Pacing is dele (+19 more)

### Community 11 - "E2E Test Stubs"
Cohesion: 0.15
Nodes (19): _AltStub, _FakeStream, _make_config(), Any, Path, End-to-end smoke: multi-collection catalog flows through the pipeline with a moc, Selecting a different collection must drive retrieval to that side., Submitting an empty question returns empty outputs without raising. (+11 more)

### Community 12 - "README Concept References"
Cohesion: 0.10
Nodes (24): 2026-07-22-batch-evaluation-design.md, all-MiniLM-L6-v2, Chroma, evaluation CSV template, data/chroma/, DeepSeek, DeepSeekLLM.stream, gr.Chatbot (+16 more)

### Community 13 - "Logging Configuration"
Cohesion: 0.14
Nodes (21): Handler, create_handlers(), get_log_level(), Path, Process-wide logging configuration., Walk up from ``start_path`` until we find pyproject.toml.      Robust to worktre, Resolve a log-level name to a ``logging`` level integer.      Reads from the ``L, Create console and file handlers, ensuring the log directory exists. (+13 more)

### Community 14 - "Graphify Skill Docs"
Cohesion: 0.22
Nodes (9): Add URL & Watch Folder Reference, Exports & Benchmark Reference, Extraction Subagent Spec, GitHub Clone & Cross-Repo Merge Reference, Commit Hook & CLAUDE.md Integration Reference, Query / Path / Explain Reference, Video/Audio Transcribe Reference, Incremental Update Reference (+1 more)

### Community 16 - "CLAUDE.md Gotchas"
Cohesion: 0.67
Nodes (4): Chroma Retriever, Gradio UI, Milvus Lite Retriever, rag-learn Project

## Knowledge Gaps
- **28 isolated node(s):** `rag-learn`, `DeepSeek LLM`, `Gradio UI`, `config.load_config / ConfigError`, `DEEPSEEK_API_KEY env var` (+23 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Hit` connect `Eval Event Tracing` to `Batch Loader & Dedup`, `Gradio App Surface`, `Collections Catalog`, `Document Loader & Chunking`, `Eval CLI Dispatch`, `Batch Metric Computation`, `Milvus Retriever Adapter`, `E2E Test Stubs`?**
  _High betweenness centrality (0.199) - this node is a cross-community bridge._
- **Why does `DeepSeekLLM` connect `LLM Judge & DeepSeek Client` to `Eval Event Tracing`, `Gradio App Surface`, `Collections Catalog`, `Batch Metric Computation`, `Batch Eval Runner`, `E2E Test Stubs`?**
  _High betweenness centrality (0.098) - this node is a cross-community bridge._
- **Why does `Catalog` connect `Collections Catalog` to `Eval Event Tracing`, `Gradio App Surface`, `Document Loader & Chunking`, `Batch Eval Runner`, `E2E Test Stubs`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `Hit` (e.g. with `ChromaRetriever` and `MilvusRetriever`) actually correct?**
  _`Hit` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `Catalog` (e.g. with `BaseRetriever` and `ChromaRetriever`) actually correct?**
  _`Catalog` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `DeepSeekLLM` (e.g. with `_AltStub` and `_FakeChoice`) actually correct?**
  _`DeepSeekLLM` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `RAGEvent` (e.g. with `StreamPerf` and `_FakeLLM`) actually correct?**
  _`RAGEvent` has 4 INFERRED edges - model-reasoned connections that need verification._