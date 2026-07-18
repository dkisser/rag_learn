# rag-learn

A side-by-side RAG retrieval comparison demo: the same question answered
twice, once against **Chroma** and once against **Milvus Lite**, each with
its own collapsible retrieved-chunks panel and per-stream perf metrics.

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

1. On first launch, the app ingests `docs/rag_doc/*.md` into both
   Chroma (PersistentClient at `data/chroma/`) and Milvus Lite
   (`data/milvus.db`). Each store uses its own bundled default
   embedder (both effectively all-MiniLM-L6-v2 384-dim L2).
2. You type a question into the Gradio UI.
3. Both retrievers search in parallel; each returns up to `RETRIEVE_K=5`
   hits. Each side's `DeepSeekLLM.stream` opens against
   `https://api.deepseek.com` with the configured `LLM_MODEL`.
4. Two `gr.Chatbot` widgets stream tokens live; below each, a
   collapsible accordion lists the retrieved chunks with file +
   chunk-index + L2 distance; a one-liner shows retrieve /
   first-token / total perf with a wall-clock timestamp.

## Env vars

| Var | Default | Notes |
|-----|---------|-------|
| `DEEPSEEK_API_KEY` | _(required)_ | App refuses to start without it. |
| `LLM_MODEL` | `deepseek-v4-flash` | Any model the DeepSeek API accepts. |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | Override for proxies. |
| `RETRIEVE_K` | `5` | Top-k for both retrievers. |
| `CHUNK_SIZE` | `800` | Per-chunk char cap; changing needs `rm -rf data/`. |
| `CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks. |

## Tests

```bash
make all   # ruff lint + ty + pytest --cov-fail-under=80
# or, under uv:
uv run pytest
```

Per-retriever model downloads happen once on first ingest; cached after.

## Known limitations

- **Milvus Lite on the full 25-doc corpus:** `milvus-lite 3.1.0` + `pymilvus 2.6.17`
  can deadlock on the ~433-row insert (and subsequent searches hang too). The unit
  tests for `MilvusRetriever` cover the small fixture; the end-to-end test
  (`tests/test_e2e.py`) exercises Chroma against the real 25 docs. Workarounds:
  downgrade `milvus-lite`, or run a Milvus standalone server.
- **Incremental UI streaming is deferred.** Today's `gr.Chatbot` accumulates each
  side's full reply into a single frame update (one delta per side per click) rather
  than incrementally flushing as tokens arrive. A `TODO` in `src/rag_learn/app.py`
  marks the conversion point. See `progress.md` for context.