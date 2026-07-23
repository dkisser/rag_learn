# Batch RAG Evaluation Design

**Date:** 2026-07-22  
**Scope:** Add a CLI-driven batch evaluation workflow for prepared Q&A datasets and sampled online questions.

---

## 1. Background

The project already records online user questions from the Gradio UI into `data/rag_events_YYYY-MM-DD.jsonl`. These events contain:

- `question`
- retrieved `hits`
- generated `answer`
- `perf` timings
- `metadata`
- `ground_truth: null`

Because `ground_truth` is always `null`, the existing `rag_learn.eval.batch` module can only compute unsupervised LLM-judge metrics (`context_relevance`, `faithfulness`, `answer_relevance`, `overall_usefulness`). Supervised metrics such as `retrieval_recall@k`, `retrieval_precision@k`, `retrieval_mrr`, `retrieval_ndcg@k`, and `answer_f1` require `ground_truth.source_files` and optionally `ground_truth.answer`.

This design introduces a batch evaluation CLI that supports two workflows:

1. **Sample online questions**, label them with ground truth, and evaluate.
2. **Upload a prepared Q&A CSV**, run the RAG pipeline over it, and evaluate.

---

## 2. Goals

- Provide a single CLI entry point for sampling, running, and evaluating batches of questions.
- Accept ground truth via CSV (`question`, optional `answer`, optional `source_files`, optional `chunk_ids`, optional `collection`).
- Generate answer events to a separate JSONL file (`batch_events_YYYY-MM-DD.jsonl`).
- Compute both supervised and unsupervised metrics from those events.
- Output a JSON aggregate report with per-question details including `question`, `answer`, and `ground_truth`.
- Keep the existing `batch.py` evaluation logic reusable.

---

## 3. CSV Format

The input CSV uses `;` as the multi-value separator for file lists.

| Column | Required | Description |
|--------|----------|-------------|
| `question` | Yes | The question text. |
| `answer` | No | Ground-truth answer. If present, `answer_f1` and `answer_llm_correctness` are computed. |
| `source_files` | No | Semicolon-separated list of relevant source files, e.g. `18-graphrag.md;README.md`. Required for retrieval supervised metrics. |
| `chunk_ids` | No | Semicolon-separated list of relevant chunk IDs, e.g. `18-graphrag.md#0;README.md#2`. Optional, preserved in the report. |
| `collection` | No | Collection slug to use for this question. Falls back to the CLI `--collection` argument. |

### Example

```csv
question,answer,source_files,chunk_ids,collection
GraphRAG 是什么,GraphRAG is a graph-based RAG approach.,18-graphrag.md,18-graphrag.md#0,rag_doc
什么是 RAG,RAG stands for retrieval-augmented generation.,02-rag-survey-2024.md;README.md,,rag_doc
```

### Rules

- `collection` resolution order: CSV row > CLI `--collection`. If neither is provided, the row is skipped with a warning.
- Empty `source_files` means no supervised retrieval metrics are computed.
- Empty `answer` means `answer_f1` and `answer_llm_correctness` are not computed.
- Empty `chunk_ids` is allowed.

### Unified CSV Template

The `sample` subcommand and the `run` subcommand use the **same CSV template**. This avoids column mismatch and lets you reuse a labeled sample file as a prepared question bank.

**Template:**

```csv
question,answer,source_files,chunk_ids,collection
```

**After `sample` (ready for labeling):**

```csv
question,answer,source_files,chunk_ids,collection
GraphRAG 是什么,,,,rag_doc
什么是 RAG,,,,rag_doc
```

**After manual labeling (ready for `run`):**

```csv
question,answer,source_files,chunk_ids,collection
GraphRAG 是什么,GraphRAG is a graph-based RAG approach.,18-graphrag.md,,rag_doc
什么是 RAG,RAG stands for retrieval-augmented generation.,02-rag-survey-2024.md;README.md,,rag_doc
```

A hand-written question bank uses the identical format, so you can always append more rows to a labeled sample file and rerun evaluation.

---

## 4. CLI Interface

A single module exposes three subcommands:

```bash
python -m rag_learn.eval.cli <subcommand> ...
```

### 4.1 `sample` — sample online events

```bash
python -m rag_learn.eval.cli sample <events_dir> \
  --samples-per-collection 5 \
  --output samples_to_label.csv
```

- Reads all `rag_events_*.jsonl` files under `<events_dir>`.
- Groups events by `collection`.
- Randomly selects up to `--samples-per-collection` events per collection (default: 5).
- Exports a CSV with columns: `question`, `answer`, `source_files`, `chunk_ids`, `collection`.
- The exported `answer` / `source_files` / `chunk_ids` columns are empty, ready for manual labeling.

### 4.2 `run` — run a prepared Q&A CSV through the RAG pipeline

```bash
python -m rag_learn.eval.cli run <qa_csv> \
  --collection rag_doc \
  --output-events data/batch_events_2026-07-22.jsonl \
  --output-report data/eval_report_2026-07-22.json \
  [--judge-model deepseek-v4-flash]
```

- Reads the Q&A CSV.
- For each row, resolves the collection and calls `answer_stream`.
- Builds `RAGEvent` with `ground_truth` populated from the CSV.
- Emits events to `--output-events` via `JSONLEmitter`.
- Automatically invokes the evaluation step and writes the report to `--output-report`.

### 4.3 `evaluate` — evaluate existing events

```bash
python -m rag_learn.eval.cli evaluate <events_dir> \
  --output data/eval_report_2026-07-22.json \
  [--judge-model deepseek-v4-flash] \
  [--dry-run]
```

- Reads all `rag_events_*.jsonl` or `batch_events_*.jsonl` under `<events_dir>`.
- Deduplicates by `trace_id`.
- Computes supervised metrics where `ground_truth` is present.
- Computes unsupervised LLM-judge metrics unless `--dry-run` is set.
- Writes the JSON report to `--output`.

---

## 5. Module Layout

```text
src/rag_learn/eval/
├── __init__.py
├── cli.py          # argparse entry point and subcommand dispatch
├── sampler.py      # sample online events and export CSV
├── runner.py       # read CSV, run answer_stream, emit RAGEvent JSONL
├── batch.py        # read JSONL events, compute metrics, write JSON report
├── metrics.py      # existing metric functions (unchanged)
└── tracing.py      # RAGEvent, GroundTruth, JSONLEmitter (unchanged)
```

### Responsibilities

| Module | Responsibility |
|--------|----------------|
| `cli.py` | Parse arguments and dispatch to `sampler`, `runner`, or `batch`. |
| `sampler.py` | Load events, group by collection, sample, write CSV. |
| `runner.py` | Parse CSV, resolve collection per row, call `answer_stream`, emit events, then call `batch.main` to produce the report. |
| `batch.py` | Load events from JSONL, compute metrics, aggregate, write JSON report. Extended to include `question`, `answer`, `ground_truth` in `details`. |
| `metrics.py` | Pure metric functions; unchanged. |
| `tracing.py` | Event model and emitters; unchanged. |

---

## 6. Data Flow

### Workflow A: Sample online questions

```text
Online events (data/rag_events_*.jsonl)
    │
    ▼
sampler ──▶ samples_to_label.csv
    │
    ▼ (manual labeling)
labeled_samples.csv
    │
    ▼
runner ──▶ data/batch_events_YYYY-MM-DD.jsonl
    │
    ▼
batch.py ──▶ data/eval_report_YYYY-MM-DD.json
```

### Workflow B: Prepared Q&A CSV

```text
qa.csv
    │
    ▼
runner ──▶ data/batch_events_YYYY-MM-DD.jsonl
    │
    ▼
batch.py ──▶ data/eval_report_YYYY-MM-DD.json
```

---

## 7. Ground Truth Construction

`runner.py` converts each CSV row into a `GroundTruth` instance:

```python
GroundTruth(
    answer=row.get("answer") or None,
    source_files=tuple(f.strip() for f in row["source_files"].split(";") if f.strip())
    if row.get("source_files") else (),
    chunk_ids=tuple(c.strip() for c in row["chunk_ids"].split(";") if c.strip())
    if row.get("chunk_ids") else (),
)
```

The `RAGEvent` emitted by `runner` uses this `GroundTruth` instead of `None`.

---

## 8. Metrics Computed

Given a populated `ground_truth`:

- If `source_files` is non-empty:
  - `retrieval_recall@k`
  - `retrieval_precision@k`
  - `retrieval_mrr`
  - `retrieval_ndcg@k`
- If `answer` is non-empty:
  - `answer_f1`
- If not `--dry-run`:
  - `context_relevance`
  - `faithfulness`
  - `answer_relevance`
  - `overall_usefulness`
- If `answer` is non-empty and not `--dry-run`:
  - `answer_llm_correctness`

The value of `k` comes from `config.retrieve_k` (or `RETRIEVE_K` in dry-run mode).

---

## 9. Report Format

The JSON report contains aggregate statistics and per-question details.

```json
{
  "generated_at": "2026-07-22T12:00:00+00:00",
  "total_events": 10,
  "with_ground_truth": 10,
  "without_ground_truth": 0,
  "skipped_corrupted_lines": 0,
  "aggregates": {
    "retrieval_recall@5": {"mean": 0.8, "median": 1.0, "p95": 1.0},
    "answer_f1": {"mean": 0.65, "median": 0.7, "p95": 0.9}
  },
  "by_collection": {
    "rag_doc": {
      "retrieval_recall@5": {"mean": 0.8, "median": 1.0, "p95": 1.0}
    }
  },
  "details": [
    {
      "trace_id": "...",
      "collection": "rag_doc",
      "question": "GraphRAG 是什么",
      "answer": "根据上下文，GraphRAG 是...",
      "ground_truth": {
        "answer": "GraphRAG is a graph-based RAG approach.",
        "source_files": ["18-graphrag.md"],
        "chunk_ids": ["18-graphrag.md#0"]
      },
      "metrics": {
        "retrieval_recall@5": 1.0,
        "answer_f1": 0.72
      }
    }
  ]
}
```

### Notes

- `details` now includes `question`, `answer`, and `ground_truth` for easy human verification.
- `aggregates` and `by_collection` remain unchanged in structure.

---

## 10. Error Handling

| Scenario | Behavior |
|----------|----------|
| CSV row missing `question` | Skip row, log warning. |
| CSV row `collection` unresolved | Skip row, log warning. |
| Collection slug not found in catalog | Skip row, log warning. |
| `answer_stream` fails for one question | Log error, skip that question, continue the batch. |
| LLM judge fails for one metric | Return `null` for that metric, log warning, continue. |
| Report file cannot be written | Raise exception and exit with non-zero code. |

The `run` subcommand must be fail-open per question so that one bad row does not abort the entire batch.

---

## 11. Testing Strategy

- **Unit tests for `sampler.py`**
  - Sampling respects `--samples-per-collection`.
  - Output CSV columns are correct.
  - Events without a collection are handled.

- **Unit tests for `runner.py`**
  - CSV parsing handles empty optional columns.
  - `GroundTruth` construction splits `;` correctly.
  - Collection resolution: CSV > CLI argument.
  - Failed rows are skipped and logged.
  - Mocked `answer_stream` produces expected JSONL output.

- **Unit tests for `batch.py`**
  - Report `details` include `question`, `answer`, `ground_truth`.
  - Supervised metrics are computed when ground truth is present.

- **Integration test**
  - Create a 2-question CSV.
  - Run `cli run`.
  - Verify `batch_events_*.jsonl` exists and has 2 events.
  - Verify `eval_report_*.json` has expected structure.

Coverage target: maintain `>= 80%` for `src/rag_learn`.

---

## 12. Example Usage

### Step 1 — Sample online questions

```bash
python -m rag_learn.eval.cli sample data/ \
  --samples-per-collection 5 \
  --output samples_to_label.csv
```

### Step 2 — Label the CSV

Open `samples_to_label.csv` and fill in `answer` and `source_files` for each row.

### Step 3 — Run evaluation

```bash
python -m rag_learn.eval.cli run labeled_samples.csv \
  --collection rag_doc \
  --output-events data/batch_events_2026-07-22.jsonl \
  --output-report data/eval_report_2026-07-22.json
```

### Prepared Q&A workflow

```bash
python -m rag_learn.eval.cli run my_qa.csv \
  --collection rag_doc \
  --output-events data/batch_events_2026-07-22.jsonl \
  --output-report data/eval_report_2026-07-22.json
```

### Evaluate existing events only

```bash
python -m rag_learn.eval.cli evaluate data/ \
  --output data/eval_report_2026-07-22.json \
  --dry-run
```

---

## 13. Future Extensions (out of scope)

- Gradio UI tab for uploading CSV and downloading reports.
- Async / queued execution for large batches.
- Support for additional input formats (JSONL, YAML).
- Real-time metrics display in the UI.
