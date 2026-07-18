# RAG Multi-Retriever Comparison (Chroma vs Milvus) — Design

- **Status:** Approved (post-brainstorming)
- **Date:** 2026-07-18
- **Owner:** dkisser

## 1. Purpose

Build a local RAG demo that ingests 25 RAG survey documents, indexes them
into both **Chroma** and **Milvus Lite**, and exposes a Gradio UI where the
user types one question and **sees both retrievers' answers side by side**,
streamed token-by-token, with each side's retrieved chunks visible in a
collapsible panel. The goal is hands-on comparison of retrieval behavior —
learning-oriented, not production.

## 2. In Scope / Out of Scope

**In scope**

- Single-question, single-session Gradio app
- Two retrievers: Chroma (PersistentClient) and Milvus Lite (embedded)
- One LLM: DeepSeek via OpenAI-compatible SDK
- Streaming on both sides
- Auto-ingestion on first launch (skip if already indexed)
- Collapsible per-side panel showing retrieved chunks + perf metrics
- 80%+ test coverage; TDD; ruff format/lint; ty type check

**Out of scope (YAGNI)**

- Multi-user / multi-session
- LangChain / LlamaIndex abstraction layers
- Query rewriting, HyDE, multi-query, re-ranking, agentic RAG
- Persistent conversation history
- Authentication / rate limiting
- Cloud deployment
- Benchmark harness (we observe informally via perf timers, no numeric SLO)

## 3. Confirmed Decisions

| # | Decision | Choice |
|---|----------|--------|
| 1 | Comparison mode | Same question runs both retrievers side by side |
| 2 | Embedding source | Each retriever's bundled default embedder (both effectively all-MiniLM-L6-v2 via ONNX, 384-dim, L2) |
| 3 | LLM model | `deepseek-v4-flash` via OpenAI SDK + `base_url=https://api.deepseek.com` |
| 4 | Repo layout | Single Python package, multi-adapter (per retriever) |
| 5 | Streaming | Both sides stream `True`; rendered side by side, synchronously |
| 6 | Retrieved-chunk UI | Collapsible panel under each answer |
| 7 | Ingestion trigger | Auto-detect empty collection on startup, ingest if empty |
| 8 | Formatter/Linter | `ruff` (format + lint) |
| 9 | Type checker | `ty` (Astral) |

## 4. Architecture

### 4.1 High-level component diagram

```
   ┌─────────────────────────────────────────────────┐
   │                Gradio UI (app.py)               │
   │  [input] [clear] [send]                         │
   │  ┌──────────────┬──────────────┐                │
   │  │ Chroma answer│ Milvus answer│  ← sync stream │
   │  │ ▼ chunks    │ ▼ chunks     │  ← collapsible │
   │  └──────────────┴──────────────┘                │
   └────────┬───────────────────┬────────────────────┘
            │                   │
    ┌───────▼────────┐  ┌───────▼────────┐
    │ ChromaRetriever│  │ MilvusRetriever│
    │ (adapter)      │  │ (adapter)      │
    └───────┬────────┘  └───────┬────────┘
            │                   │
    ┌───────▼───────────────────▼────────┐
    │ BaseRetriever (Protocol)           │
    │   search(q, k) -> list[Hit]        │
    │   ensure_indexed(docs_dir) -> None │
    └────────────────────────────────────┘
            │                   │
    ┌───────▼────────┐  ┌───────▼────────┐
    │ chromadb       │  │ pymilvus Lite  │
    │ PersistentClnt │  │ (embedded)     │
    └────────────────┘  └────────────────┘

    ┌────────────────────────────────────┐
    │ DeepSeekLLM (OpenAI SDK)          │
    │ base_url=https://api.deepseek.com │
    │ model=deepseek-v4-flash           │
    │ stream(system, user) -> gen[str]  │
    └────────────────────────────────────┘
```

### 4.2 Directory layout

```
rag_learn/
├── docs/
│   ├── rag_doc/                          # existing 25 markdown sources
│   └── superpowers/specs/                # this doc lives here
├── data/                                  # gitignored, runtime-generated
│   ├── chroma/                            # Chroma persistent dir
│   └── milvus.db                          # Milvus Lite single-file DB
├── src/rag_learn/
│   ├── __init__.py
│   ├── config.py                          # env vars, paths
│   ├── loader.py                          # markdown scan + chunking
│   ├── retriever/
│   │   ├── __init__.py
│   │   ├── base.py                        # BaseRetriever + Hit
│   │   ├── chroma_impl.py
│   │   └── milvus_impl.py
│   ├── llm.py                             # DeepSeekLLM
│   ├── pipeline.py                        # retrieve + format + generate
│   └── app.py                             # Gradio UI entry
├── tests/
│   ├── fixtures/sample_docs/              # 3 hand-crafted tiny markdowns
│   ├── test_loader.py
│   ├── test_chunks.py
│   ├── test_chroma_retriever.py
│   ├── test_milvus_retriever.py
│   ├── test_llm.py
│   ├── test_pipeline.py
│   ├── test_pipeline_parallel.py
│   ├── test_e2e.py
│   └── test_app_launch.py
├── pyproject.toml                         # deps + tooling
├── .env.example                           # DEEPSEEK_API_KEY, LLM_MODEL
├── .gitignore                             # add data/, .env, etc.
└── main.py                                # `python main.py` → launch app
```

### 4.3 Dependencies

Runtime:
- `chromadb>=0.5`
- `pymilvus>=2.4` (Milvus Lite)
- `openai>=1.40`
- `gradio>=5.0`
- `python-dotenv>=1.0`

Dev:
- `pytest>=8.0`
- `pytest-cov>=5.0`
- `ruff>=0.6` (formatter + linter)
- `ty` (Astral type checker)

Python: `>=3.12` (matches existing `.python-version`).

## 5. Data Flow

### 5.1 Runtime (single question)

```
User submits question
       │
       ├──────────────────────────────────┐
       ▼                                  ▼
ChromaRetriever.search()         MilvusRetriever.search()
   → embed query (default fn)       → embed query (default fn)
   → ANN top-k L2                  → ANN top-k L2
       │                                  │
       ▼                                  ▼
   list[Hit] (top-k)                list[Hit] (top-k)
       │                                  │
       └─────────────┬────────────────────┘
                     ▼
       pipeline.build_prompt(chunks, question)
       identical template for both sides; only context chunks differ
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
DeepSeekLLM.stream()(chroma)  DeepSeekLLM.stream()(milvus)
        │                         │
        ▼                         ▼
Gradio Chatbot (chroma)    Gradio Chatbot (milvus)
   updates token-by-token    updates token-by-token
                     │
                     ▼
       Each side's panel shows:
         • answer (streamed)
         • retrieved chunks (collapsible)
         • perf: [timestamp] retrieve / first-token / total
```

### 5.2 Synchronous-stream strategy

Use `gr.Blocks` (not `ChatInterface`) with two `gr.Chatbot` widgets side
by side. Each stream is wired to its own `gr.Chatbot.append()` calls fed
by a Python `threading.Thread` running the LLM generator. Both threads
are started by the same `submit` button handler in parallel; each
Chatbot updates independently. The user perceives the two sides as
"streaming in sync" because both threads start at the same instant and
their token rates are dominated by the same upstream LLM round-trip.

Each side's perf (`retrieve`, `first_token`, `total`) is captured in the
pipeline thread and pushed to a `gr.Markdown` line under that side's
Chatbot after the stream completes.

### 5.3 Prompt template (identical for both sides)

```
system: 你是一个 RAG 助手。仅基于下方提供的「上下文」回答用户问题。
        如果上下文不足以回答，直接说「未找到相关上下文」。
        不要使用先验知识或编造内容。

user:   上下文：
        [1] (来源: {source_file}) {chunk_text_1}
        [2] (来源: {source_file}) {chunk_text_2}
        ...
        [k] (来源: {source_file}) {chunk_text_k}

        问题：{user_query}
        回答：
```

The only difference between the two streams is the `{chunk_text_i}`
content. This isolates the comparison to retrieval + embedding differences.

### 5.4 Chunking strategy

1. Scan `docs/rag_doc/*.md`
2. Split by H1 (`^# ` headings) into "documents"
3. For each document, greedily fill chunks up to `CHUNK_SIZE` characters,
   splitting at paragraph boundaries (`\n\n`) or sentence punctuation
4. Preserve `CHUNK_OVERLAP` characters between adjacent chunks
5. Chunk metadata: `{source_file, chunk_index, char_start, char_end}`

Fallback rules:
- If a markdown file has no H1, treat the entire file as a single
  document and chunk it the same way.
- If a single section is shorter than `CHUNK_SIZE`, it becomes one
  chunk (no padding).

Tuning: `CHUNK_SIZE` defaults to `800`, `CHUNK_OVERLAP` defaults to
`50`. Both are read once at startup in `config.py` from environment
variables of the same name (with those defaults applied if unset).
Changing them requires deleting `data/` and re-running the app.

### 5.5 Startup / ingestion

`app.py` boot sequence:

```
1. Load .env via python-dotenv
2. Verify DEEPSEEK_API_KEY present → else raise ConfigError
3. Build DeepSeekLLM, ChromaRetriever, MilvusRetriever
4. For each retriever: ensure_indexed(docs_dir)
     └─ log "[<ts>] Chroma: indexed N chunks in T seconds" (or "skip — N chunks already")
5. Launch Gradio queue-enabled server on 127.0.0.1:7860
```

If a single retriever's `ensure_indexed` raises, log the error,
mark that retriever `degraded=True`, and let the app still launch with
a UI banner warning the user. The app is **fail-open** on per-retriever
ingestion failure but **fail-closed** on missing `DEEPSEEK_API_KEY`.

## 6. Component Contracts

### 6.1 `Hit` dataclass (`retriever/base.py`)

```python
@dataclass(frozen=True)
class Hit:
    text: str             # chunk content
    source_file: str      # e.g. "18-graphrag.md"
    chunk_index: int      # index within source file
    score: float          # distance (L2); lower = more similar
```

The same `Hit` shape is returned by both retrievers; consumers
(pipeline, UI) don't need to know which retriever produced it.

### 6.2 `BaseRetriever` Protocol

```python
@runtime_checkable
class BaseRetriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Hit]: ...
    def ensure_indexed(self, docs_dir: str) -> None: ...
```

Structural typing (Protocol + `@runtime_checkable`) — new retrievers
don't have to inherit.

### 6.3 `ChromaRetriever`

- Uses `chromadb.PersistentClient(path=data/chroma)`
- Collection `rag_doc`, `hnsw:space=l2`
- Default embedding function (`all-MiniLM-L6-v2`, 384-dim, via ONNX)
- `ensure_indexed`: short-circuits if `collection.count() > 0`
- `search`: returns `top-k` with `Hit.score = distance`

### 6.4 `MilvusRetriever`

- Uses `pymilvus.MilvusClient(uri=data/milvus.db)` (embedded mode)
- Collection `rag_doc`, dim 384, metric L2
- Default embedder via pymilvus (`all-MiniLM-L6-v2-via-onnx`)
- `ensure_indexed`: short-circuits if collection exists
- `search`: returns `top-k` with `Hit.score = distance`

**Embedding note:** both retrievers' defaults resolve to `all-MiniLM-L6-v2`.
Any observed retrieval difference comes from index structure, ANN behavior,
or distance computation path — not from a different model. This is the
intended comparison surface.

### 6.5 `DeepSeekLLM`

```python
class DeepSeekLLM:
    def __init__(self, api_key: str, model: str,
                 base_url: str = "https://api.deepseek.com") -> None: ...
    def stream(self, system: str, user: str) -> Iterator[str]: ...
```

Uses `openai.OpenAI(api_key, base_url)`. `stream()` yields each token
delta from the chat-completion stream. Only the streaming API is exposed;
non-streaming callers collect tokens into a buffer themselves.

### 6.6 `pipeline.answer_stream`

```python
def answer_stream(
    retrievers: dict[str, BaseRetriever],   # {"chroma": ..., "milvus": ...}
    llm: DeepSeekLLM,
    question: str,
    k: int = 5,
) -> dict[str, tuple[Iterator[str], list[Hit]]]:
    """Parallel-retrieve from both sides; return dict[name] = (token_stream, hits).
    `token_stream` is a generator consumed by the Gradio layer.
    """
```

### 6.7 `pipeline.build_prompt`

```python
def build_prompt(chunks: list[Hit], question: str) -> tuple[str, str]:
    """Returns (system_msg, user_msg)."""
```

Chunk text is truncated to `CHUNK_DISPLAY_CHARS = 600` per chunk in the
prompt to avoid blowing context. Source filename and chunk index are
always retained for traceability.

## 7. Error Handling

| Failure | Trigger | Behavior |
|---------|---------|----------|
| `ConfigError` | `DEEPSEEK_API_KEY` unset | raise; abort boot with clear message |
| `IngestError` (single retriever) | doc read / split / embed fails | log; mark retriever `degraded`; **continue boot** with banner |
| `RetrievalError` (single side) | `search()` raises | UI shows red `⚠ 检索失败: <msg>` on that side; other side continues |
| `RetrievalError` (both sides) | both raise | top banner "所有 retriever 都失败" |
| `LLMError` | DeepSeek auth/network | generator emits error text once and stops; UI shows error frame |
| `EmptyHit` | `search()` returns `[]` | special system prompt branch: "未找到相关上下文，请回答不知道" |

All exceptions are logged at module level via `logging.getLogger(__name__)`.

## 8. Observability

### 8.1 Server-side logs

- Format: `%(asctime)s %(levelname)s %(name)s %(message)s`
- Ingestion: `[<ts>] Chroma: indexed N chunks in T seconds`
- Per-stream perf printed **after stream completes**:
  ```
  [<ts>] chroma  retrieve=30ms  first_token=420ms  total=1200ms
  [<ts>] milvus  retrieve=28ms  first_token=405ms  total=1180ms
  ```
  `<ts>` is wall-clock `HH:MM:SS.mmm` so both sides can be eyeballed.

### 8.2 UI-side perf panel

Below each answer's chunks, a small `gr.Markdown` line:

```
检索 30ms · 首个 token 420ms · 总 1.2s · 完成于 14:23:45.123
```

## 9. Testing

- TDD mandatory: write failing test → minimal impl → refactor
- Coverage gate: `--cov-fail-under=80`

| Test | Type | Mocks |
|------|------|-------|
| `test_loader.py` | unit | — |
| `test_chunks.py` | unit (boundary) | — |
| `test_chroma_retriever.py` | integration | chromadb `EphemeralClient` |
| `test_milvus_retriever.py` | integration | pymilvus temp dir |
| `test_llm.py` | unit | fake OpenAI client |
| `test_pipeline.py` | unit | fake retriever |
| `test_pipeline_parallel.py` | integration | fake retriever + fake LLM |
| `test_e2e.py` | e2e (real 25 docs) | fake LLM |
| `test_app_launch.py` | smoke | — |

Fixtures:
- `tests/fixtures/sample_docs/` — 3 hand-written tiny markdowns (200-500 chars each)
- Real `docs/rag_doc/` used **only** by `test_e2e.py`; tests never mutate it

CI / local:
- `pyproject.toml` `[tool.pytest.ini_options]`:
  `testpaths=["tests"]`, `addopts="--cov=src/rag_learn --cov-report=term-missing --cov-fail-under=80"`
- `Makefile` targets: `test`, `lint`, `format`, `typecheck`, `all`

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Chromadb / pymilvus default embedder weight download needs network on first run | Log explicit progress; document in README |
| DeepSeek rate limits / key quota | Surface error to UI (see §7); document getting a new key |
| Milvus Lite lacks concurrency primitives | Use single-thread per retriever; pipeline parallelizes **across** retrievers, not within |
| Two simultaneous Gradio streams look janky | Use `gr.Blocks` not `ChatInterface`; show "ready" tick when each side finishes |
| 25 docs × 800-char chunks may still produce > n_context tokens on big questions | `top-k=5` + per-chunk 600-char truncation caps total context at ~3KB; well within limits |

## 11. Glossary

- **RAG** — Retrieval-Augmented Generation
- **Lite (Milvus)** — embedded mode, no separate server process
- **Default embedder** — bundled `all-MiniLM-L6-v2` ONNX model used implicitly by Chroma / Milvus when no other embedding function is configured
- **Adapter** — a class implementing `BaseRetriever` for a specific vector store

## 12. Open Questions

None — all design-time questions resolved during brainstorming on 2026-07-18.
