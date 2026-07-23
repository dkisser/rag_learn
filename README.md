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
    --output-events data/rag_events \
    --output-report data/report.json

# 2b) Re-evaluate events on disk without re-querying (e.g. after tweaking metrics)
uv run python -m rag_learn.eval.cli evaluate data \
    --output data/report.json --dry-run
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