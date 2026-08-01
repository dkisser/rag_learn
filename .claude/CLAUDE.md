# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`rag-learn` is a side-by-side RAG retrieval comparison demo: the same user
question is answered by **Chroma** and **Milvus Lite** in parallel, each side
streaming tokens through DeepSeek's OpenAI-compatible API. The Gradio 5 UI
shows one streamed answer + a collapsible retrieved-chunks panel per side.

## Build, test, lint

All commands live in the `Makefile`. The typechecker is **Astral's `ty`**
(`make typecheck` runs `ty check src`) — *not* `mypy`.

- `make install` — `pip install -e ".[dev]"` (pip-based)
- `make uv-sync` — `uv sync --extra dev` (uv-based; uses the committed `uv.lock`)
- `make test` — `pytest`. The `pyproject.toml` `[tool.pytest.ini_options]` block
  enforces `--cov-fail-under=80` on `src/rag_learn`, so coverage below 80% fails
  the run.
- `make lint` — `ruff check src tests` (line-length 100, selects `E F I B UP`)
- `make format` — `ruff format src tests`
- `make all` — `lint + typecheck + test`. **Run this before committing.**

`make clean` also wipes `data/`; the next `python main.py` re-ingests.

Under uv, prefix test/lint/format/typecheck with `uv run …` (or activate
`.venv` via `source .venv/bin/activate`). The Makefile targets work in
both environments.

## Required env vars

`DEEPSEEK_API_KEY` is **mandatory** — `rag_learn.config.load_config()` raises
`ConfigError` and the app refuses to start without it. `LLM_MODEL`,
`DEEPSEEK_BASE_URL`, `RETRIEVE_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP` all have
sensible defaults (see `.env.example`). Tests never hit DeepSeek: `test_e2e.py`
monkey-patches a fake OpenAI client, and `test_app_launch.py` passes a dummy
client object.

## Known gotchas

- **milvus-lite deadlocks on the full 25-doc corpus.** `tests/test_e2e.py`
  exercises Chroma only with the mocked LLM. Milvus coverage lives in
  `tests/test_milvus_retriever.py` against the small fixture. If you need
  real-doc Milvus e2e, downgrade `milvus-lite` or run a Milvus standalone server.
  `MilvusRetriever.ensure_indexed` calls `load_collection` after `insert+flush`
  (and defensively when the collection already exists from a prior run) so
  search() does not hit the "Collection in state released" RPC error.
  - **macOS ARM caveat**: `load_collection` SIGSEGVs at the C layer on
    `milvus-lite` 2.6+ when called outside the original insert+flush sequence
    (e.g. on a reopened DB file). The retriever runs the defensive call in a
    subprocess (`_safe_load_collection` in `src/rag_learn/retriever/milvus_impl.py`)
    via `multiprocessing.get_context("spawn")` so a SIGSEGV only kills the
    subprocess; if the subprocess fails, `ensure_indexed` drops the collection
    and recreates it from the source docs. There is still a residual risk that
    the recreate path itself SIGSEGVs on the full 25-doc corpus — if so, the
    documented escape hatches are `make clean`, downgrade `milvus-lite`, or
    run a Milvus standalone server.
  - **Cross-session reload contract**: pymilvus 2.6+ `MilvusClient.search()`
    does **not** auto-load collections; without the defensive call every
    restart of the app against an existing `data/milvus.db` hits
    `Collection in state released` on the first `search()`. The subprocess
    wrapper handles this on darwin and runs directly on Linux/Windows.
- **Gradio analytics crash on macOS.** Gradio 5.50 launches its analytics
  daemon in background threads that call `uuid4()` → `os.urandom()`. Combined
  with milvus-lite's gRPC fork handlers and the FD-poll-list leftovers, those
  concurrent `os.urandom` calls SIGSEGV the interpreter on macOS ARM. `launch()`
  sets `os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"` (Gradio 5.50 has no
  launch() kwarg for this) so the analytics daemon never starts. Set
  `GRADIO_ANALYTICS_ENABLED=True` if you need to re-enable it locally.
- **Chunking changes need a fresh index.** Changing `CHUNK_SIZE` /
  `CHUNK_OVERLAP` (or any `loader.py` logic) requires `rm -rf data/` before the
  next launch — `ChromaRetriever.ensure_indexed` and
  `MilvusRetriever.ensure_indexed` both short-circuit when the store is non-empty.
- **Version pins in `pyproject.toml` are deliberate.** `chromadb<1`,
  `pymilvus<3`, `openai<2`, `gradio<6` — the inline comment lists the
  breaking-change in each newer major. Don't bump casually.
- **UI streaming is batched, not incremental.** `app.on_submit` drains each
  side's iterator into a single `gr.Chatbot` frame update via
  `_drain_to_chatbot`. A `TODO` in `src/rag_learn/app.py` (around line 125)
  marks the conversion point for true per-token streaming.
- **`answer_stream`'s `metadata` is READ-ONLY input.** It is shallow-copied
  before any routing field is written, because `eval/runner.py` shares one
  dict across rows processed with `max_concurrency=3` — in-place writes
  would bleed between questions. Callers that need the routing decisions
  (the UI's caption) pass `routing_sink=`, a callback invoked at most once
  per call with an immutable `routing.RoutingInfo`. Do NOT "fix" a missing
  routing field by reintroducing mutation.
- **Two different `k`s in the catalog branch.** `CATALOG_SUB_K` (default 8)
  is how many candidates *each sub-query* pulls from *each retriever*;
  `CATALOG_RECALL_K` (default 20) is how many survive the round-robin merge
  and reach the prompt. The merge cap is the binding constraint — raising
  only `CATALOG_SUB_K` will not put more chunks in the prompt.
- **Catalog fan-out only searches the selected collection.** `app.on_submit`
  passes `{slug: retriever}`, so the decomposer prompt is scoped via
  `_build_catalog_summary(catalog, only=retrievers.keys())` — otherwise the
  LLM invents sub-queries aimed at collections that will never be searched.
  Unknown keys (legacy `chroma`/`milvus` compare mode) fall back to the full
  catalog summary.
- **Lazy SDK imports.** `retriever/chroma_impl.py` and `retriever/milvus_impl.py`
  import `chromadb` / `pymilvus` inside `__init__` so unrelated tests don't pay
  the model-download / native-import cost. Keep this pattern when adding new
  adapters.
- **Repo-root discovery is robust to worktrees.** `config._repo_root()` walks
  up from `config.py` looking for `pyproject.toml`, so the same code works from
  nested `.claude/worktrees/*` checkouts.

## Workflow

- **Branch model:** standard Git Flow — feature branches off `main`, PRs back
  to `main`. No force pushes to `main`.
- **Pre-commit gate:** `make all` (lint + typecheck + test). The coverage
  floor catches accidental drops. Use TDD per the global testing rules:
  write the failing test first, then the implementation.
- **Touching a retriever adapter?** Add or extend the matching
  `tests/test_<name>_retriever.py` against `tests/fixtures/sample_docs/`.
- **Touching the pipeline or prompt?** Update `tests/test_pipeline.py`,
  `tests/test_pipeline_parallel.py`, and (if end-to-end behavior changes)
  `tests/test_e2e.py`.
- **Touching the UI?** `tests/test_app_launch.py` covers `build_app`; live
  smoke-testing requires a real `DEEPSEEK_API_KEY` and a `python main.py`
  launch against `http://127.0.0.1:7860`.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- **Slash command trigger**: when the user types `/graphify`, use the installed graphify skill (`.claude/skills/graphify/SKILL.md`) before doing anything else.