# rag-learn

A RAG retrieval demo: a user question is answered by **Chroma**, with a
collapsible retrieved-chunks panel and per-stream perf metrics.

## Quick start

```bash
# Option A — uv (recommended; uses the committed uv.lock)
uv sync --extra dev
cp .env.example .env
# edit .env to set DEEPSEEK_API_KEY
uv run python main.py
# open http://127.0.0.1:7860

# Option B — pip
pip install -e ".[dev]"
cp .env.example .env
# edit .env to set DEEPSEEK_API_KEY
python main.py
# open http://127.0.0.1:7860
```

## Repository tooling

This repository uses the `graphify` CLI (provided by the `graphifyy` package) as
a development dependency. The RAG application itself does **not** import
`graphify` at runtime, but contributors use it to navigate, query, and refresh
the codebase knowledge graph in `graphify-out/`.

```bash
# Install the graphify CLI (uv is recommended)
uv tool install graphifyy

# Build the knowledge graph from scratch
# (writes graphify-out/graph.json, graph.html, and GRAPH_REPORT.md)
graphify .

# Query an existing graph without rebuilding it
graphify query "How does the retrieval pipeline work?"

# Explore relationships between two concepts/files
graphify path "pipeline" "retriever"

# Get a focused explanation of a concept
graphify explain "RAGEvent"

# Incrementally refresh the graph after code changes (AST-only, no LLM cost)
graphify update .
```

When `graphify-out/graph.json` exists, prefer these commands over broad
source-file browsing; they return a scoped subgraph that is usually much
smaller than `GRAPH_REPORT.md` or raw grep output. See `CLAUDE.md` for the
project's graphify usage rules.

If `graphify` is not installed, the application can still be launched, but
repository graph navigation and graph regeneration are unavailable.

## How it works

1. On first launch, the app ingests `docs/rag_doc/*.md` into
   Chroma (PersistentClient at `data/chroma/`). The store uses its bundled
   default embedder (all-MiniLM-L6-v2 384-dim L2).
2. You type a question into the Gradio UI.
3. The retriever searches and returns up to `RETRIEVE_K=5` hits. The
   `DeepSeekLLM.stream` opens against `https://api.deepseek.com` with the
   configured `LLM_MODEL`.
4. A `gr.Chatbot` widget streams tokens live; below it, a collapsible
   accordion lists the retrieved chunks with file + chunk-index + L2 distance;
   a one-liner shows retrieve / first-token / total perf with a wall-clock
   timestamp.

## Env vars

| Var | Default | Notes |
|-----|---------|-------|
| `DEEPSEEK_API_KEY` | _(required)_ | App refuses to start without it. |
| `LLM_MODEL` | `deepseek-v4-flash` | Any model the DeepSeek API accepts. |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | Override for proxies. |
| `RETRIEVE_K` | `5` | Top-k for the retriever. |
| `CHUNK_SIZE` | `800` | Per-chunk char cap; changing needs `rm -rf data/`. |
| `CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks. |

## Tests

```bash
make all   # ruff lint + ty + pytest --cov-fail-under=80
# or, under uv:
uv run pytest
```

The retriever model downloads once on first ingest; cached after.

## Batch evaluation CLI

The `rag_learn.eval.cli` module ships three subcommands for offline evaluation,
all driven by the same CSV template (`question,answer,source_files,chunk_ids,collection`).
`source_files` and `chunk_ids` use `;` as a multi-value separator; either field
may be empty.

```bash
# 1) Sample online traffic (random sample per collection) → CSV for manual labeling
uv run python -m rag_learn.eval.cli sample data \
    --samples-per-collection 5 --output samples.csv

# 2a) Run a Q&A bank through RAG, emit events, write a report
uv run python -m rag_learn.eval.cli run qa.csv \
    --collection rag_doc \
    --output-events data/shanzhongshi_events.jsonl \
    --output-report data/shanzhongshi_report.json

# 2b) Re-evaluate events on disk without re-querying (e.g. after tweaking metrics).
#     `evaluate` accepts either a single .jsonl file OR a directory of daily files.
uv run python -m rag_learn.eval.cli evaluate data \
    --output data/report.json --dry-run
```

> Note: `--output-events` is now a literal `.jsonl` file path. The runner
> writes every event to that exact file, and crash-resume reads from it
> too. The `evaluate` subcommand still globs `data/rag_events_*.jsonl`
> by default — pass it a single file if you want to evaluate one.

### Rate limiting & crash-resume

DeepSeek's free tier is sensitive to bursty traffic. `run` and `evaluate`
pace every LLM call through a shared `RateLimiter` (token-bucket RPM +
in-flight concurrency cap + 429 retry with exponential backoff).
Defaults are conservative for the free tier; tune via flags:

| Flag | Default | Effect |
|------|---------|--------|
| `--max-concurrency` | `3` | Max simultaneous judge (or generation) calls. |
| `--rate` | `20.0` | Requests per minute ceiling. Lower for free tier. |
| `--max-retries` | `3` | Per-call retries on HTTP 429 before recording `None`. |
| `--no-resume` | off | `run` only: re-process CSV rows whose question already has an emitted event. |

`run` is crash-resume by default: if it is interrupted mid-CSV, re-running
the same command skips `(collection, question)` pairs already written to
the `--output-events` file. To force a full re-run after a schema change,
pass `--no-resume`. Free-tier example that survives bursty traffic:

```bash
uv run python -m rag_learn.eval.cli run docs/eval/shanzhongshi_qa.csv \
    --collection shanzhongshi \
    --output-events data/shanzhongshi_events.jsonl \
    --output-report data/shanzhongshi_report.json \
    --max-concurrency 1 --rate 5
```

The report aggregates supervised metrics (`retrieval_recall@k`,
`retrieval_precision@k`, `retrieval_mrr`, `retrieval_ndcg@k`, `answer_f1`)
when `ground_truth.source_files` is provided, plus unsupervised LLM-judge
metrics (`context_relevance`, `faithfulness`, `answer_relevance`,
`overall_usefulness`, `answer_llm_correctness`).

See `docs/superpowers/specs/2026-07-22-batch-evaluation-design.md` for full
design context.

## Known limitations

- **Incremental UI streaming is deferred.** Today's `gr.Chatbot` accumulates the
  full reply into a single frame update (one delta per click) rather than
  incrementally flushing as tokens arrive. A `TODO` in `src/rag_learn/app.py`
  marks the conversion point. See `progress.md` for context.