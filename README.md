# rag-learn

> 🌐 **Languages**: [English (current)](#) · [中文](./README.zh-CN.md)

> **Learn RAG by measuring, not by tricks** — a progressive Chroma learning project, where every optimization starts from a number on an eval report.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Gradio 5](https://img.shields.io/badge/gradio-5-orange.svg)](https://gradio.app/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![uv-ready](https://img.shields.io/badge/uv-ready-purple.svg)](https://docs.astral.sh/uv/)
[![Tests](https://img.shields.io/badge/tests-make%20all-brightgreen.svg)](#tests)
[![Coverage 80%+](https://img.shields.io/badge/coverage-80%25%2B-success.svg)](#tests)

```
Question ─▶ Catalog ─▶ Hybrid (BM25 + vector, RRF) ─▶ Reranker (cross-encoder)
                                                          │
                                                          ▼
Gradio UI ◀── DeepSeek stream ◀── top-k filter ◀── Chroma store
   │
   └─▶ JSONL eval event  (rag_events_YYYY-MM-DD.jsonl)
```

## Why this project

RAG "stops working" in production far more often than it "suddenly gets better" after you add hybrid / reranker / routing. This project is not another checklist of tricks — it's a learning project **built around an evaluation loop**:

- **RAG does not get smarter just because you swap the retriever.** It only improves when you have ground truth, can run metrics, and can see the delta.
- Every new capability (hybrid, rerank, routing, threshold filtering, …) ships with a matching set of evaluation metrics. **An optimization without numbers does not enter the trunk.**
- Beyond the demo UI, there's a CSV-driven batch evaluation CLI (`sample / run / evaluate`) with both supervised and LLM-judge metrics.

It is for people who:

- Want to learn RAG but feel drowned by the "hybrid + reranker + routing + …" trick checklist.
- Want to see how LLM-judge and supervised metrics actually behave inside a real loop.
- Want to start from Chroma, layer capabilities gradually, and back every change with a number.

## Features

- **Chroma + multi-collection catalog**: switch knowledge bases from a Gradio dropdown.
- **Optional hybrid retrieval**: BM25 (jieba for Chinese tokenization) fused with vectors via RRF (`HYBRID_ENABLED`).
- **Optional cross-encoder reranker**: BGE by default, configurable score threshold.
- **Intent-aware routing + sub-query decomposition**: automatic fan-out across collections (`INTENT_ENABLED` / `DECOMPOSE_ENABLED`).
- **Retrieval threshold filtering**: `CHROMA_MAX_DISTANCE` and `RERANK_MIN_SCORE`.
- **Full instrumentation**: every Q&A write is appended to `data/rag_events_YYYY-MM-DD.jsonl`.
- **CSV-driven batch evaluation CLI**: `sample / run / evaluate` subcommands.
- **Supervised metrics**: `retrieval_recall@k / precision@k / MRR / NDCG@k / answer_f1`.
- **LLM-judge metrics**: `context_relevance / faithfulness / answer_relevance / overall_usefulness / answer_llm_correctness`.
- **Token-bucket rate limiting + crash-resume** (DeepSeek free tier friendly).
- **DeepSeek streamed answers** through the OpenAI-compatible SDK.
- **80%+ unit test coverage** (`make all` enforces the floor).

## Table of contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Evolution: a progressive journey](#evolution-a-progressive-journey)
- [Evaluation methodology](#evaluation-methodology)
- [Env vars](#env-vars)
- [Tests](#tests)
- [Batch evaluation CLI](#batch-evaluation-cli)
- [Historical adapter: Milvus](#historical-adapter-milvus)
- [Roadmap & known limitations](#roadmap--known-limitations)
- [Acknowledgements](#acknowledgements)
- [License](#license)

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

On first launch, `docs/shanzhongshi/*.md` is ingested into `data/chroma/`, and the dropdown shows the "山中事咖啡" (Shanzhongshi Coffee) collection. The embedder model downloads once and is cached afterwards.

## How it works

1. **Ingest**: on first launch, `docs/shanzhongshi/*.md` is chunked and written into a Chroma `PersistentClient` (bundled `all-MiniLM-L6-v2` 384-dim cosine).
2. **Pick a collection**: the "知识库" (Knowledge base) dropdown at the top of the UI selects the active catalog entry; single-collection or fan-out is decided by routing.
3. **Retrieve (optional pipeline)**: catalog → hybrid (BM25 + vector, RRF) → reranker (cross-encoder) → threshold filter.
4. **Generate**: DeepSeek streams the answer; a collapsible panel shows the chunks (file + chunk-index + distance/score) and a perf line (retrieve / first-token / total).
5. **Instrument**: every Q&A write is appended to `data/rag_events_YYYY-MM-DD.jsonl` (question / hits / answer / perf / metadata), ready for offline evaluation to consume.

Every step's toggle lives in [Env vars](#env-vars); with everything off, you get the plainest "vector recall → DeepSeek".

## Evolution: a progressive journey

The project evolved on the principle "every new capability ships with an evaluation loop first". The timeline below is reconstructed from `git log` and `docs/superpowers/specs/` — it is not a folder layout invented after the fact:

| Stage | Theme | Key capabilities | Design spec |
|---|---|---|---|
| **v0.1** | Vanilla retrieval | Chroma single-collection, DeepSeek streaming, perf instrumentation | [`2026-07-18-rag-multiretriever-design`](docs/superpowers/specs/2026-07-18-rag-multiretriever-design.md) |
| **v0.2** | Evaluation loop | `RAGEvent` persistence, 5 supervised + 5 LLM-judge metrics, `batch` aggregation | [`2026-07-22-rag-metrics-design`](docs/superpowers/specs/2026-07-22-rag-metrics-design.md), [`2026-07-22-batch-evaluation-design`](docs/superpowers/specs/2026-07-22-batch-evaluation-design.md) |
| **v0.3** | Multi-collection catalog | `Collection` / `Catalog`, `shanzhongshi` collection, dropdown replaces the legacy dual-side demo | [`2026-07-21-multi-collection-catalog-design`](docs/superpowers/specs/2026-07-21-multi-collection-catalog-design.md) |
| **v0.5** | Hybrid retrieval | BM25 (jieba Chinese tokenization) + Chroma vectors, RRF fusion | — |
| **v0.6** | Reranker | Cross-encoder scoring, configurable `RERANK_MIN_SCORE` | — |
| **v0.7** | Intent routing | Intent classification + sub-query decomposition + catalog fan-out | — |
| **v0.8** | Threshold filtering + prompt polish | `CHROMA_MAX_DISTANCE`, per-collection sliced top-k, prompt tightening | — |

Principle: **every new capability ships with a matching set of evaluation metrics; an optimization without numbers does not enter the trunk.**

## Evaluation methodology

> "Without ground truth, you can only run LLM-judge metrics; with ground truth, supervised metrics can finally enter the picture — and supervised metrics are the only reproducible way to judge whether a trick is genuinely useful."
>
> — paraphrased from [`2026-07-22-batch-evaluation-design`](docs/superpowers/specs/2026-07-22-batch-evaluation-design.md) §1

The loop looks like this:

```
Live UI  ──▶  JSONL  (rag_events_YYYY-MM-DD.jsonl)
                       │
                       ▼
              sample   (sample unlabeled questions)
                       │
                       ▼
              label    (manually or semi-automatically label ground_truth)
                       │
                       ▼
              run      (batch the RAG pipeline, write events with ground_truth)
                       │
                       ▼
              evaluate (aggregates / by_collection / details)
```

`aggregates` exposes mean / median / p95; `by_collection` slices by collection; `details` is the per-row breakdown. **All metric scoring consumes the same event stream — no separate "online metric" vs "offline metric" drift.**

## Env vars

Read via `rag_learn.config.load_config()` at startup. Keys are case-sensitive. A missing `DEEPSEEK_API_KEY` raises `ConfigError` and refuses to launch.

| Var | Default | Notes |
|---|---|---|
| `DEEPSEEK_API_KEY` | _(required)_ | DeepSeek API key; the app refuses to start without it. |
| `LLM_MODEL` | `deepseek-v4-flash` | Any model the DeepSeek API accepts. |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | Override for proxies. |
| `RETRIEVE_K` | `5` | Top-k for the retriever. |
| `CHUNK_SIZE` | `800` | Per-chunk char cap; changes need `rm -rf data/`. |
| `CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks. |
| `LOG_LEVEL` | `INFO` | Global log level. |
| `CHROMA_MAX_DISTANCE` | _(unset)_ | Cosine distance ceiling; hits above this are dropped. |
| `HYBRID_ENABLED` | `false` | Turn on BM25 + vector RRF fusion. |
| `HYBRID_RRF_K` | `60` | The `k` constant in the RRF formula. |
| `RERANK_ENABLED` | `false` | Turn on cross-encoder rerank. |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | Any `sentence-transformers` CrossEncoder. |
| `RERANK_K` | _(unset)_ | Candidates kept before rerank (defaults to `RETRIEVE_K * RERANK_FACTOR`). |
| `RERANK_DEVICE` | `auto` | `cpu` / `cuda` / `mps` / `auto`. |
| `RERANK_MIN_SCORE` | _(unset)_ | Cross-encoder score floor; below it gets dropped. |
| `RERANK_FACTOR` | `4` | `RERANK_K = RETRIEVE_K * RERANK_FACTOR`. |
| `RERANK_BATCH_SIZE` | `8` | Batch size for scoring. |
| `INTENT_ENABLED` | `false` | Turn on intent classification. |
| `INTENT_TIMEOUT_S` | `8.0` | Timeout for the intent classification LLM call. |
| `DECOMPOSE_ENABLED` | `false` | Turn on sub-query decomposition (catalog fan-out). |
| `DECOMPOSE_TIMEOUT_S` | `15.0` | Timeout for the decomposition LLM call. |
| `DECOMPOSE_MAX` | `8` | Maximum number of generated subqueries. |
| `CATALOG_SUB_K` | `8` | Candidates each sub-query pulls from each retriever. |
| `CATALOG_RECALL_K` | `20` | Final cap on chunks that reach the prompt after the round-robin merge. |

`.env.example` ships placeholders for every key above; just keep unused ones commented.

## Tests

```bash
make all   # ruff lint + ty + pytest --cov-fail-under=80
# or, under uv:
uv run pytest
```

`pyproject.toml` enforces `--cov-fail-under=80`; coverage below the threshold makes `make all` red. Touching a module? Update the matching `tests/test_<module>.py` — `tests/test_*_retriever.py` and `tests/test_eval*` are the usual landing sites.

## Batch evaluation CLI

`rag_learn.eval.cli` exposes three subcommands over a single CSV template (`question, answer, source_files, chunk_ids, collection`). `source_files` and `chunk_ids` use `;` as a multi-value separator; either field may be empty.

```bash
# 1) Sample online traffic into a label-ready CSV
uv run python -m rag_learn.eval.cli sample data \
    --samples-per-collection 5 --output samples.csv

# 2a) Run a labeled Q&A bank through RAG, write events + report
uv run python -m rag_learn.eval.cli run qa.csv \
    --collection shanzhongshi \
    --output-events data/shanzhongshi_events.jsonl \
    --output-report data/shanzhongshi_report.json

# 2b) Re-evaluate events already on disk without re-querying (good for tweaking weights / thresholds)
uv run python -m rag_learn.eval.cli evaluate data \
    --output data/report.json --dry-run
```

`run` is **crash-resume** by default: `(collection, question)` pairs already written to `--output-events` are skipped on re-runs. Force a full re-run with `--no-resume`.

DeepSeek's free tier is sensitive to bursty traffic. `run` and `evaluate` share a `RateLimiter` (token-bucket RPM + in-flight cap + 429 exponential backoff):

| Flag | Default | Effect |
|---|---|---|
| `--max-concurrency` | `3` | Cap on simultaneous judge / generation calls. |
| `--rate` | `20.0` | Requests-per-minute ceiling; lower for the free tier. |
| `--max-retries` | `3` | Per-call retries on HTTP 429 before recording `None`. |
| `--no-resume` | off | `run`-only: re-process questions whose events already exist. |

Free-tier example that survives bursty traffic:

```bash
uv run python -m rag_learn.eval.cli run docs/eval/shanzhongshi_qa.csv \
    --collection shanzhongshi \
    --output-events data/shanzhongshi_events.jsonl \
    --output-report data/shanzhongshi_report.json \
    --max-concurrency 1 --rate 5
```

`report` aggregates supervised metrics (`retrieval_recall@k`, `retrieval_precision@k`, `retrieval_mrr`, `retrieval_ndcg@k`, `answer_f1`) when `ground_truth.source_files` is provided, plus unsupervised LLM-judge metrics (`context_relevance`, `faithfulness`, `answer_relevance`, `overall_usefulness`, `answer_llm_correctness`).

Full design in [`2026-07-22-batch-evaluation-design`](docs/superpowers/specs/2026-07-22-batch-evaluation-design.md).

## Historical adapter: Milvus

`src/rag_learn/retriever/milvus_impl.py` and `tests/test_milvus_retriever.py` are still in the repo as a historical adapter, **but the main path no longer instantiates it**: `factory.build_retriever` returns only Chroma / Hybrid, and `app.launch` no longer imports `MilvusRetriever`. If you want to compare Milvus Lite, you can import the adapter manually and own the SIGSEGV risk of `pymilvus` 2.6+ on macOS ARM (see `CLAUDE.md` Known gotchas). The project does not treat Milvus as a demo feature.

## Roadmap & known limitations

- **Incremental UI streaming is not done.** The current `gr.Chatbot` collapses the full reply into a single frame update (one delta per click) instead of flushing per token; a `TODO` in `src/rag_learn/app.py` marks the conversion point.
- **This project is not production-grade RAG**: no auth, no monitoring, no horizontal scaling — it is a learning project.
- **The Milvus adapter is retired but still in the tree**: see the previous section; there is no plan to bring it back into the main path.
- **Chunking changes need a fresh index**: change `CHUNK_SIZE` / `CHUNK_OVERLAP` then `rm -rf data/` before the next launch.

## Acknowledgements

Built on top of:

- [Chroma](https://www.trychroma.com/) — vector store
- [DeepSeek](https://api-docs.deepseek.com/) — streaming LLM (OpenAI-compatible)
- [Gradio 5](https://gradio.app/) — UI
- [sentence-transformers](https://www.sbert.net/) — embedder + CrossEncoder
- [BAAI / BGE](https://huggingface.co/BAAI) — `bge-reranker-base` and friends
- [jieba](https://github.com/fxsjy/jieba) — Chinese tokenization
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25
- [pyrate-limiter](https://pypi.org/project/pyrate-limiter/) / [tenacity](https://tenacity.readthedocs.io/) — rate limiting & retries

Chinese-language domain corpus: `docs/shanzhongshi/` (山中事咖啡 — bean origins, roast levels, brew guides, company info).

## License

MIT — see [LICENSE](./LICENSE).