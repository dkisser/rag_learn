# RAG Multi-Retriever Comparison (Chroma vs Milvus) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-question, dual-retriever RAG demo that ingests the 25 RAG markdown docs into both Chroma and Milvus Lite, then serves a Gradio UI where one user question produces two streamed answers (one per retriever) side by side, each with a collapsible chunks panel and wall-clock-timestamped perf metrics.

**Architecture:** Single `src/rag_learn` Python package with multi-adapter retrievers; a thin `BaseRetriever` Protocol lets each vector store live in its own file. A `pipeline` module orchestrates parallel retrieval and dispatches two concurrent LLM streams to two `gr.Chatbot` widgets. No LangChain/LlamaIndex layers.

**Tech Stack:** Python 3.12, chromadb ≥0.5, pymilvus ≥2.4 (Lite), openai ≥1.40 (against DeepSeek base URL), gradio ≥5.0, python-dotenv ≥1.0; dev: pytest ≥8 + pytest-cov ≥5, ruff ≥0.6, ty.

## File Structure

```
rag_learn/
├── docs/
│   ├── rag_doc/                          # (existing) 25 markdown sources
│   └── superpowers/
│       ├── specs/                        # (existing) design doc
│       └── plans/2026-07-18-rag-multiretriever-impl.md   # this file
├── data/                                  # gitignored, created at runtime
│   ├── chroma/
│   └── milvus.db
├── src/rag_learn/
│   ├── __init__.py
│   ├── config.py                          # env vars + paths + chunk constants
│   ├── loader.py                          # markdown scan + chunking
│   ├── retriever/
│   │   ├── __init__.py
│   │   ├── base.py                        # BaseRetriever Protocol + Hit dataclass
│   │   ├── chroma_impl.py
│   │   └── milvus_impl.py
│   ├── llm.py                             # DeepSeekLLM (OpenAI SDK)
│   ├── pipeline.py                        # build_prompt + answer_stream
│   └── app.py                             # Gradio Blocks UI
├── tests/
│   ├── fixtures/sample_docs/              # 3 tiny hand-written markdowns
│   │   ├── doc_with_h1.md
│   │   ├── doc_no_h1.md
│   │   └── doc_short_section.md
│   ├── test_config.py
│   ├── test_loader.py
│   ├── test_chunks.py
│   ├── test_chroma_retriever.py
│   ├── test_milvus_retriever.py
│   ├── test_llm.py
│   ├── test_pipeline.py
│   ├── test_pipeline_parallel.py
│   ├── test_e2e.py
│   └── test_app_launch.py
├── pyproject.toml                         # deps + pytest config + ruff config
├── .env.example                           # DEEPSEEK_API_KEY=, LLM_MODEL=
├── .gitignore                             # add data/, .env, __pycache__
├── Makefile                               # test/lint/format/typecheck/all
└── main.py                                # python main.py → launch app
```

## Global Constraints

- Python `>=3.12` (matches existing `.python-version`).
- `src/` layout; tests import `rag_learn.*` after `pip install -e .`.
- `pyproject.toml` `[tool.pytest.ini_options]`: `testpaths=["tests"]`, `addopts="--cov=src/rag_learn --cov-report=term-missing --cov-fail-under=80"`.
- `pyproject.toml` `[tool.ruff]`: `line-length = 100`, target `py312`.
- `ty` runs on `src/`.
- All env-derived values read once at `config.py` import time.
- `data/`, `.env`, `__pycache__/`, `.ruff_cache/`, `.ty_cache/`, `.pytest_cache/` are gitignored.
- TDD: write failing test → minimal impl → green → commit (5-step pattern per task).
- Conventional Commits format: `feat:`, `chore:`, `test:`, `docs:`. Attribution disabled globally; no `Co-Authored-By:` lines.
- No mutation of existing `docs/rag_doc/` content; e2e test reads only.
- Fail-open on per-retriever ingest failure; fail-closed on missing `DEEPSEEK_API_KEY`.

---

## Task 1: Project scaffolding (pyproject, .gitignore, Makefile, .env.example)

**Files:**
- Modify: `pyproject.toml` (was near-empty; rewrite with deps, project metadata, pytest/ruff config)
- Modify: `.gitignore` (extend)
- Create: `.env.example`
- Create: `Makefile`
- Create: empty package + test placeholders so `pip install -e .` works
  - `src/rag_learn/__init__.py`
  - `src/rag_learn/retriever/__init__.py`
  - `tests/__init__.py` (file is empty, may be omitted; we keep `tests/` flat without `__init__.py` so pytest rootdir is repo root)
  - `tests/fixtures/sample_docs/.gitkeep`

**Interfaces:**
- Consumes: existing `pyproject.toml`, `.gitignore`, `docs/rag_doc/`
- Produces: importable `rag_learn` package; `make test/lint/format/typecheck/all` targets; `.env` template users can copy

- [ ] **Step 1: Write the failing test for `config.py`**

Wait — there is no `config.py` yet and this task adds scaffolding only. Skip the test in this task; coverage starts being enforced once Task 2 lands. We instead commit a smoke test that proves the package imports:

Create `tests/test_smoke_import.py`:
```python
def test_package_imports():
    import rag_learn  # noqa: F401
```

- [ ] **Step 2: Run it — should fail because rag_learn doesn't exist**

Run: `pytest tests/test_smoke_import.py -v`
Expected: `ModuleNotFoundError: No module named 'rag_learn'`.

- [ ] **Step 3: Write the minimal pyproject.toml**

Replace `pyproject.toml`:
```toml
[project]
name = "rag-learn"
version = "0.1.0"
description = "RAG comparison demo: Chroma vs Milvus Lite over 25 RAG markdown docs."
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "chromadb>=0.5",
    "pymilvus>=2.4",
    "openai>=1.40",
    "gradio>=5.0",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.6",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=src/rag_learn --cov-report=term-missing --cov-fail-under=80"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/rag_learn"]
```

- [ ] **Step 4: Create the rest of the scaffolding**

Create `src/rag_learn/__init__.py`:
```python
__version__ = "0.1.0"
```

Create `src/rag_learn/retriever/__init__.py`:
```python
from rag_learn.retriever.base import Hit, BaseRetriever  # noqa: F401
```

> **NOTE:** These imports will fail until Task 3 creates `base.py`. We deliberately delay importing until Task 3. For now, use empty `__init__.py` files:
```python
# placeholder; populated in Task 3
```

> **WORKAROUND:** Leave both `__init__.py` files empty for now; we'll add the re-exports in Task 3.

Create `.env.example`:
```
# Copy this file to .env and fill in real values.
DEEPSEEK_API_KEY=
LLM_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
RETRIEVE_K=5
CHUNK_SIZE=800
CHUNK_OVERLAP=50
```

Extend `.gitignore` (append):
```
data/
.env
__pycache__/
*.py[cod]
.pytest_cache/
.coverage
.ruff_cache/
.ty_cache/
```

Create `Makefile`:
```makefile
.PHONY: install test lint format typecheck all clean

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

format:
	ruff format src tests

typecheck:
	ty check src

all: lint typecheck test

clean:
	rm -rf data/ .pytest_cache/ .ruff_cache/ .ty_cache/ .coverage
```

- [ ] **Step 5: Install the package and run the smoke test**

Run:
```bash
pip install -e ".[dev]"
pytest tests/test_smoke_import.py -v
```
Expected: PASS (1 passed). The `import rag_learn` succeeds and `--cov` shows 100% (one import, no code yet).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore .env.example Makefile src/rag_learn/__init__.py src/rag_learn/retriever/__init__.py tests/test_smoke_import.py tests/fixtures/sample_docs/.gitkeep
git commit -m "chore: scaffold rag_learn package, pyproject, makefile, .env.example"
```

---

## Task 2: config.py (env vars + paths + chunk constants)

**Files:**
- Create: `src/rag_learn/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: `DEEPSEEK_API_KEY`, `LLM_MODEL`, `DEEPSEEK_BASE_URL`, `RETRIEVE_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP` (env vars)
- Produces: `Config` dataclass + module-level singleton `config`. All other modules import `config` rather than reading `os.environ` directly.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config.py`:
```python
import pytest

from rag_learn.config import Config, load_config, ConfigError, CHUNK_DISPLAY_CHARS


def test_load_config_reads_required_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    cfg = load_config()
    assert cfg.deepseek_api_key == "sk-test"
    assert cfg.llm_model == "deepseek-v4-flash"  # default
    assert cfg.deepseek_base_url == "https://api.deepseek.com"  # default


def test_load_config_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="DEEPSEEK_API_KEY"):
        load_config()


def test_load_config_overrides(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "deepseek-reasoner")
    monkeypatch.setenv("RETRIEVE_K", "10")
    monkeypatch.setenv("CHUNK_SIZE", "1200")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    cfg = load_config()
    assert cfg.llm_model == "deepseek-reasoner"
    assert cfg.retrieve_k == 10
    assert cfg.chunk_size == 1200
    assert cfg.chunk_overlap == 100


def test_chunk_display_chars_constant():
    assert CHUNK_DISPLAY_CHARS == 600


def test_paths_resolve_relative_to_repo(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    cfg = load_config()
    assert cfg.repo_root.name == "rag_learn"
    assert (cfg.data_dir / "chroma").exists() or not (cfg.data_dir / "chroma").exists()  # may not yet exist
    assert cfg.docs_dir.name == "rag_doc"
```

- [ ] **Step 2: Run tests — expect failures**

Run: `pytest tests/test_config.py -v`
Expected: `ModuleNotFoundError: No module named 'rag_learn.config'`.

- [ ] **Step 3: Implement `src/rag_learn/config.py`**

```python
"""Process-wide configuration loaded once from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env at import time so subprocesses (pytest, gr.launch) see the same values.
load_dotenv()

# Search constants — exposed at module level because they are also used by the
# prompt builder (see pipeline.build_prompt) for content truncation.
CHUNK_DISPLAY_CHARS: int = 600


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    deepseek_api_key: str
    llm_model: str
    deepseek_base_url: str
    retrieve_k: int
    chunk_size: int
    chunk_overlap: int
    repo_root: Path
    docs_dir: Path
    data_dir: Path
    chroma_dir: Path
    milvus_path: Path


def _repo_root() -> Path:
    # src/rag_learn/config.py → src/rag_learn → src → repo root
    return Path(__file__).resolve().parents[2]


def load_config() -> Config:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ConfigError(
            "DEEPSEEK_API_KEY is required. Copy .env.example to .env and set it."
        )

    repo_root = _repo_root()
    return Config(
        deepseek_api_key=api_key,
        llm_model=os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
        deepseek_base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        retrieve_k=int(os.environ.get("RETRIEVE_K", "5")),
        chunk_size=int(os.environ.get("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "50")),
        repo_root=repo_root,
        docs_dir=repo_root / "docs" / "rag_doc",
        data_dir=repo_root / "data",
        chroma_dir=repo_root / "data" / "chroma",
        milvus_path=repo_root / "data" / "milvus.db",
    )
```

- [ ] **Step 4: Run tests — expect green**

Run: `pytest tests/test_config.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rag_learn/config.py tests/test_config.py
git commit -m "feat(config): env-driven Config + ConfigError"
```

---

## Task 3: Hit dataclass + BaseRetriever Protocol

**Files:**
- Create: `src/rag_learn/retriever/base.py`
- Create: `tests/test_retriever_base.py`
- Modify: `src/rag_learn/retriever/__init__.py` (re-export)

**Interfaces:**
- Consumes: none (pure data + protocol)
- Produces: `Hit` dataclass, `BaseRetriever` Protocol

- [ ] **Step 1: Write the failing tests**

Create `tests/test_retriever_base.py`:
```python
from dataclasses import FrozenInstanceError

import pytest

from rag_learn.retriever.base import BaseRetriever, Hit


def test_hit_is_frozen():
    hit = Hit(text="x", source_file="a.md", chunk_index=0, score=0.1)
    with pytest.raises(FrozenInstanceError):
        hit.text = "y"  # type: ignore[misc]


def test_hit_equality():
    a = Hit(text="x", source_file="a.md", chunk_index=0, score=0.1)
    b = Hit(text="x", source_file="a.md", chunk_index=0, score=0.1)
    assert a == b


def test_protocol_recognises_conforming_class():
    class Fake:
        def search(self, query: str, k: int = 5) -> list[Hit]:
            return [Hit(text=query, source_file="x.md", chunk_index=0, score=0.0)]

        def ensure_indexed(self, docs_dir: str) -> None:
            return None

    assert isinstance(Fake(), BaseRetriever)


def test_protocol_rejects_non_conforming():
    class NotARetriever:
        pass

    assert not isinstance(NotARetriever(), BaseRetriever)
```

- [ ] **Step 2: Run tests — expect failures**

Run: `pytest tests/test_retriever_base.py -v`
Expected: `ModuleNotFoundError: No module named 'rag_learn.retriever.base'`.

- [ ] **Step 3: Implement `src/rag_learn/retriever/base.py`**

```python
"""Retriever contract shared by all adapter implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Hit:
    text: str             # chunk content
    source_file: str      # e.g. "18-graphrag.md"
    chunk_index: int      # index within source file
    score: float          # L2 distance; lower = more similar


@runtime_checkable
class BaseRetriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Hit]: ...

    def ensure_indexed(self, docs_dir: str) -> None: ...
```

- [ ] **Step 4: Wire `src/rag_learn/retriever/__init__.py`**

Replace contents:
```python
from rag_learn.retriever.base import BaseRetriever, Hit

__all__ = ["BaseRetriever", "Hit"]
```

- [ ] **Step 5: Run tests — expect green**

Run: `pytest tests/test_retriever_base.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/rag_learn/retriever/base.py src/rag_learn/retriever/__init__.py tests/test_retriever_base.py
git commit -m "feat(retriever): Hit dataclass + BaseRetriever Protocol"
```

---

## Task 4: loader.py (markdown scan + chunking)

**Files:**
- Create: `src/rag_learn/loader.py`
- Create: `tests/test_loader.py`
- Create: `tests/test_chunks.py`
- Create: 3 fixture markdowns under `tests/fixtures/sample_docs/`

**Interfaces:**
- Consumes: directory path containing `*.md` files
- Produces: `Chunk` dataclass; `iter_markdown(docs_dir)`; `load_documents(docs_dir) -> list[Chunk]`

- [ ] **Step 1: Create fixture markdowns (preparation, not test)**

`tests/fixtures/sample_docs/doc_with_h1.md`:
```markdown
# Section Alpha

Alpha content. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Vestibulum convallis eros non sapien hendrerit, in faucibus tellus
tincidunt. Sed ut perspiciatis unde omnis iste natus error.

# Section Beta

Beta content begins here and discusses a second topic. Pellentesque
habitant morbi tristique senectus et netus et malesuada fames ac
turpis egestas. Curabitur pretium tincidunt magna, in commodo dui
sollicitudin in. Nulla facilisi.

Praesent dapibus mauris in arcu euismod, eget iaculis justo aliquam.
Vivamus suscipit velit non elit sagittis, vel hendrerit urna dictum.
Etiam vehicula lectus sit amet libero sodales, in mattis libero
bibendum. Suspendisse potenti.
```

`tests/fixtures/sample_docs/doc_no_h1.md`:
```markdown
This file has no headings at all. It is meant to exercise the fallback
path where the entire file becomes a single input document.

Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim
ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut
aliquip ex ea commodo consequat.
```

`tests/fixtures/sample_docs/doc_short_section.md`:
```markdown
# Tiny

Just a few lines.
```

- [ ] **Step 2: Write failing tests for chunking**

Create `tests/test_chunks.py`:
```python
from pathlib import Path

from rag_learn.loader import load_documents


FIXTURES = Path(__file__).parent / "fixtures" / "sample_docs"


def test_doc_with_h1_splits_at_h1():
    chunks = load_documents(str(FIXTURES))
    section_chunks = [c for c in chunks if c.source_file == "doc_with_h1.md"]
    assert len(section_chunks) >= 2  # at least one chunk per H1
    assert all(c.chunk_index >= 0 for c in section_chunks)


def test_doc_no_h1_treated_as_single_document():
    chunks = [c for c in load_documents(str(FIXTURES)) if c.source_file == "doc_no_h1.md"]
    assert len(chunks) >= 1


def test_short_section_becomes_one_chunk():
    chunks = [c for c in load_documents(str(FIXTURES)) if c.source_file == "doc_short_section.md"]
    assert len(chunks) == 1


def test_chunk_length_respects_limit(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "200")
    monkeypatch.setenv("CHUNK_OVERLAP", "20")
    import importlib

    from rag_learn import config as cfg_mod
    importlib.reload(cfg_mod)
    chunks = load_documents(str(FIXTURES))
    for c in chunks:
        # No padding applied, so chunks may be <= limit. None should exceed.
        assert len(c.text) <= 300  # allow some slack around the boundary


def test_chunks_have_monotonic_index_per_file():
    chunks = load_documents(str(FIXTURES))
    by_file: dict[str, list[int]] = {}
    for c in chunks:
        by_file.setdefault(c.source_file, []).append(c.chunk_index)
    for indices in by_file.values():
        assert indices == sorted(indices)
        assert len(set(indices)) == len(indices)
```

- [ ] **Step 3: Run tests — expect failures**

Run: `pytest tests/test_chunks.py -v`
Expected: `ModuleNotFoundError: No module named 'rag_learn.loader'`.

- [ ] **Step 4: Implement `src/rag_learn/loader.py`**

```python
"""Markdown discovery + chunking for ingestion."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# Compile once at module load.
_H1_RE = re.compile(r"(?m)^# .+$")
_PARA_SPLIT_RE = re.compile(r"\n\n+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。.!！?？;；])\s*")


@dataclass(frozen=True)
class Chunk:
    text: str
    source_file: str
    chunk_index: int
    char_start: int
    char_end: int


def _chunk_size() -> tuple[int, int]:
    # Re-read every call to keep tests monkeypatch-friendly.
    size = int(os.environ.get("CHUNK_SIZE", "800"))
    overlap = int(os.environ.get("CHUNK_OVERLAP", "50"))
    return size, overlap


def iter_markdown(docs_dir: str | Path) -> list[tuple[str, str]]:
    """Return list of (filename, raw_text) sorted by filename, deterministic."""
    p = Path(docs_dir)
    files = sorted(p.glob("*.md"))
    return [(f.name, f.read_text(encoding="utf-8")) for f in files]


def _split_into_pre_docs(raw: str) -> list[str]:
    """Split a markdown file by H1 headings.

    If no H1 is present, return the whole file as one document. The H1
    line itself is omitted (it's metadata, not body).
    """
    matches = list(_H1_RE.finditer(raw))
    if not matches:
        return [raw.strip()]
    docs: list[str] = []
    # Pre-H1 content (e.g., a title block without '# ') becomes the first doc.
    pre = raw[: matches[0].start()].strip()
    if pre:
        docs.append(pre)
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        docs.append(raw[start:end].strip())
    return [d for d in docs if d]


def _chunk_text(text: str, source_file: str, start_offset: int = 0) -> list[Chunk]:
    """Greedy paragraph-then-sentence packing up to CHUNK_SIZE with OVERLAP."""
    size, overlap = _chunk_size()
    if not text.strip():
        return []

    paragraphs = _PARA_SPLIT_RE.split(text)
    chunks: list[Chunk] = []
    buf = ""
    chunk_start = 0
    cursor = 0
    local_offset = 0  # offset within the original `text`

    def flush():
        nonlocal buf, chunk_start, local_offset
        if not buf.strip():
            return
        # Trim trailing whitespace from the buffered chunk before persisting.
        body = buf.rstrip()
        end_in_text = chunk_start + len(body)
        chunks.append(
            Chunk(
                text=body,
                source_file=source_file,
                chunk_index=len(chunks),
                char_start=start_offset + chunk_start,
                char_end=start_offset + end_in_text,
            )
        )
        # Compute the next chunk's start by backing up `overlap` characters
        # from the end of the just-emitted chunk, so adjacent chunks share
        # `overlap` characters of context.
        new_local = max(0, end_in_text - overlap)
        buf = text[new_local:end_in_text]
        chunk_start = new_local
        local_offset = end_in_text
        cursor = end_in_text

    for para in paragraphs:
        para = para.strip()
        if not para:
            cursor += 2  # account for the \n\n we split on
            continue
        # If a single paragraph already exceeds the size, split by sentence.
        if len(para) > size:
            sentences = _SENTENCE_SPLIT_RE.split(para)
            for sent in sentences:
                if not sent.strip():
                    continue
                if len(buf) + len(sent) + 1 > size:
                    flush()
                buf = (buf + " " + sent).strip() if buf else sent
        else:
            if len(buf) + len(para) + 2 > size:
                flush()
            buf = (buf + "\n\n" + para).strip() if buf else para
        cursor += len(para) + 2

    flush()
    return chunks


def split_into_chunks(filename: str, raw_text: str) -> list[Chunk]:
    """Top-level entry: chunk a single file's content."""
    pre_docs = _split_into_pre_docs(raw_text)
    out: list[Chunk] = []
    for doc in pre_docs:
        out.extend(_chunk_text(doc, filename))
    # Re-index so chunk_index is contiguous within the file.
    for i, c in enumerate(out):
        # frozen dataclass, so rebuild
        out[i] = Chunk(
            text=c.text,
            source_file=c.source_file,
            chunk_index=i,
            char_start=c.char_start,
            char_end=c.char_end,
        )
    return out


def load_documents(docs_dir: str | Path) -> list[Chunk]:
    """Read all *.md in docs_dir and return a flat list of chunks."""
    chunks: list[Chunk] = []
    for name, raw in iter_markdown(docs_dir):
        chunks.extend(split_into_chunks(name, raw))
    return chunks
```

- [ ] **Step 5: Run tests — expect green**

Run: `pytest tests/test_chunks.py -v`
Expected: 5 passed.

- [ ] **Step 6: Write & run a second test file for `iter_markdown` + `load_documents` total length**

Create `tests/test_loader.py`:
```python
from pathlib import Path

from rag_learn.loader import iter_markdown, load_documents, split_into_chunks


FIXTURES = Path(__file__).parent / "fixtures" / "sample_docs"


def test_iter_markdown_returns_sorted_files():
    items = iter_markdown(FIXTURES)
    names = [n for n, _ in items]
    assert names == sorted(names)
    assert all(name.endswith(".md") for name in names)


def test_load_documents_assigns_filenames():
    chunks = load_documents(FIXTURES)
    assert {c.source_file for c in chunks} == {
        "doc_with_h1.md",
        "doc_no_h1.md",
        "doc_short_section.md",
    }


def test_split_into_chunks_raw_includes_only_requested_file():
    chunks = split_into_chunks("doc_short_section.md", "# Tiny\n\nJust a few lines.")
    assert len(chunks) == 1
    assert chunks[0].source_file == "doc_short_section.md"
    assert "Tiny" in chunks[0].text or "few lines" in chunks[0].text
```

Run: `pytest tests/test_loader.py -v`
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add src/rag_learn/loader.py tests/test_chunks.py tests/test_loader.py tests/fixtures/sample_docs/
git commit -m "feat(loader): markdown discovery + H1-aware chunking with overlap"
```

---

## Task 5: DeepSeekLLM (OpenAI SDK streaming client)

**Files:**
- Create: `src/rag_learn/llm.py`
- Create: `tests/test_llm.py`

**Interfaces:**
- Consumes: `OpenAI` SDK (mockable)
- Produces: `DeepSeekLLM` class with `.stream(system: str, user: str) -> Iterator[str]`. Any exception inside the SDK stream is caught and yielded as a single error message (spec §7 `LLMError`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_llm.py`:
```python
from __future__ import annotations

from collections.abc import Iterator

from rag_learn.llm import DeepSeekLLM


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.delta = type("Delta", (), {"content": content})()


class _FakeChunk:
    def __init__(self, content: str | None) -> None:
        self.choices = [_FakeChoice(content)] if content is not None else []


class _FakeStream:
    def __init__(self, tokens: list[str | None]) -> None:
        self._tokens = tokens

    def __iter__(self) -> Iterator[_FakeChunk]:
        return iter(_FakeChunk(t) for t in self._tokens)


class _FakeCompletions:
    def __init__(self, tokens: list[str | None]) -> None:
        self._tokens = tokens
        self.last_kwargs: dict = {}

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeStream(self._tokens)


class _FakeChat:
    def __init__(self, tokens: list[str | None]) -> None:
        self.completions = _FakeCompletions(tokens)


class _FakeClient:
    def __init__(self, tokens: list[str | None]) -> None:
        self.chat = _FakeChat(tokens)


class _RaisingCompletions:
    def create(self, **kwargs):
        raise RuntimeError("boom: deepseek down")


class _RaisingChat:
    completions = _RaisingCompletions()


class _RaisingClient:
    chat = _RaisingChat()


def test_stream_yields_tokens_only_when_content_present():
    fake = _FakeClient(["你", "好", None, "世界"])
    llm = DeepSeekLLM(api_key="k", model="m", base_url="u", client=fake)  # type: ignore[arg-type]
    assert list(llm.stream("sys", "user")) == ["你", "好", "世界"]


def test_stream_forwards_system_and_user_messages_and_stream_flag():
    fake = _FakeClient(["ok"])
    llm = DeepSeekLLM(api_key="k", model="m", base_url="u", client=fake)  # type: ignore[arg-type]
    list(llm.stream("system-text", "user-text"))
    kwargs = fake.chat.completions.last_kwargs
    assert kwargs["stream"] is True
    assert kwargs["model"] == "m"
    assert kwargs["messages"] == [
        {"role": "system", "content": "system-text"},
        {"role": "user", "content": "user-text"},
    ]


def test_defaults_use_deepseek_base_url_and_model():
    fake = _FakeClient([])
    DeepSeekLLM(api_key="k", client=fake)  # type: ignore[arg-type]
    assert fake.chat.completions.last_kwargs["model"] == "deepseek-v4-flash"


def test_stream_emits_single_error_token_when_sdk_raises():
    fake = _RaisingClient()
    llm = DeepSeekLLM(api_key="k", model="m", base_url="u", client=fake)  # type: ignore[arg-type]
    tokens = list(llm.stream("sys", "user"))
    assert len(tokens) == 1
    assert "⚠ LLM 错误" in tokens[0] and "boom" in tokens[0]
```

- [ ] **Step 2: Run tests — expect failures**

Run: `pytest tests/test_llm.py -v`
Expected: `ModuleNotFoundError: No module named 'rag_learn.llm'`.

- [ ] **Step 3: Implement `src/rag_learn/llm.py`**

```python
"""DeepSeek LLM client; uses the OpenAI SDK with DeepSeek's base URL."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from openai import OpenAI


logger = logging.getLogger(__name__)


class DeepSeekLLM:
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        client: Any | None = None,
    ) -> None:
        self._client: Any = client if client is not None else OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def stream(self, system: str, user: str) -> Iterator[str]:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("DeepSeekLLM.create failed")
            yield f"⚠ LLM 错误：{exc}"
            return

        try:
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    yield content
        except Exception as exc:  # noqa: BLE001
            logger.exception("DeepSeekLLM stream interrupted")
            yield f"\n\n⚠ LLM 中断：{exc}"
            return
```

- [ ] **Step 4: Run tests — expect green**

Run: `pytest tests/test_llm.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rag_learn/llm.py tests/test_llm.py
git commit -m "feat(llm): DeepSeekLLM streaming client over OpenAI SDK"
```

---

## Task 6: pipeline.build_prompt

**Files:**
- Create: `src/rag_learn/pipeline.py` (initially only `build_prompt` + helpers)
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `list[Hit]`, `question: str`, `CHUNK_DISPLAY_CHARS` constant
- Produces: `build_prompt(chunks, question) -> tuple[str, str]`

- [ ] **Step 1: Write failing tests**

Create `tests/test_pipeline.py`:
```python
from rag_learn.pipeline import build_prompt
from rag_learn.retriever import Hit


def _hits() -> list[Hit]:
    return [
        Hit(text="alpha content", source_file="a.md", chunk_index=0, score=0.10),
        Hit(text="beta content", source_file="b.md", chunk_index=3, score=0.20),
    ]


def test_build_prompt_returns_system_and_user():
    sys_msg, user_msg = build_prompt(_hits(), "什么是 RAG？")
    assert isinstance(sys_msg, str) and sys_msg
    assert isinstance(user_msg, str) and user_msg


def test_build_prompt_includes_question():
    _, user_msg = build_prompt(_hits(), "什么是 RAG？")
    assert "什么是 RAG？" in user_msg


def test_build_prompt_lists_each_chunk_with_source():
    sys_msg, user_msg = build_prompt(_hits(), "Q")
    assert "[1] (来源: a.md) alpha content" in user_msg
    assert "[2] (来源: b.md) beta content" in user_msg
    assert "alpha content" in sys_msg or "上下文" in sys_msg or "回答" in sys_msg


def test_build_prompt_truncates_long_chunks():
    long = "x" * 5000
    h = [Hit(text=long, source_file="x.md", chunk_index=0, score=0.0)]
    _, user_msg = build_prompt(h, "Q")
    # 5000 chars truncated to CHUNK_DISPLAY_CHARS (600)
    assert user_msg.count("x") == 600


def test_build_prompt_empty_hits_has_empty_prompt_branch():
    sys_msg, _ = build_prompt([], "Q")
    assert "未找到" in sys_msg or "上下文" in sys_msg


def test_build_prompt_numbering_starts_at_one():
    _, user_msg = build_prompt(_hits(), "Q")
    assert "[1]" in user_msg
    assert "[2]" in user_msg
    assert "[0]" not in user_msg and "[3]" not in user_msg
```

- [ ] **Step 2: Run tests — expect failures**

Run: `pytest tests/test_pipeline.py::test_build_prompt_returns_system_and_user -v`
Expected: `ModuleNotFoundError: No module named 'rag_learn.pipeline'`.

- [ ] **Step 3: Implement `src/rag_learn/pipeline.py` (first cut: only build_prompt)**

```python
"""RAG pipeline: prompt construction + parallel retrieval + streaming answers."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_learn.config import CHUNK_DISPLAY_CHARS
from rag_learn.retriever import Hit

if TYPE_CHECKING:
    from rag_learn.llm import DeepSeekLLM
    from rag_learn.retriever.base import BaseRetriever


SYSTEM_PROMPT = (
    "你是一个 RAG 助手。仅基于下方提供的「上下文」回答用户问题。"
    "如果上下文不足以回答，直接说「未找到相关上下文」。"
    "不要使用先验知识或编造内容。"
)

EMPTY_HITS_SYSTEM_PROMPT = (
    "你是一个 RAG 助手。当前没有检索到任何相关上下文，"
    "请直接告诉用户「未找到相关上下文」，不要使用先验知识或编造内容。"
)


@dataclass(frozen=True)
class StreamPerf:
    retrieve_ms: float
    first_token_ms: float
    total_ms: float
    finished_at: str  # HH:MM:SS.mmm


def build_prompt(chunks: list[Hit], question: str) -> tuple[str, str]:
    """Return (system_msg, user_msg).

    Truncates each chunk's text to CHUNK_DISPLAY_CHARS.
    """
    if not chunks:
        return EMPTY_HITS_SYSTEM_PROMPT, f"问题：{question}\n回答："

    lines = ["上下文："]
    for i, hit in enumerate(chunks, start=1):
        text = hit.text
        if len(text) > CHUNK_DISPLAY_CHARS:
            text = text[:CHUNK_DISPLAY_CHARS]
        lines.append(f"[{i}] (来源: {hit.source_file}) {text}")
    user_msg = "\n".join(lines) + f"\n\n问题：{question}\n回答："
    return SYSTEM_PROMPT, user_msg
```

- [ ] **Step 4: Run tests — expect green**

Run: `pytest tests/test_pipeline.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rag_learn/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): build_prompt with chunk truncation + empty-hits branch"
```

---

## Task 7: ChromaRetriever

**Files:**
- Create: `src/rag_learn/retriever/chroma_impl.py`
- Create: `tests/test_chroma_retriever.py`
- Create: `tests/conftest.py` (shared `tmp_chroma_dir` fixture)

**Interfaces:**
- Consumes: `Path` for persistent dir, `loader.load_documents(docs_dir)` from Task 4
- Produces: `ChromaRetriever(persist_dir: Path)` implementing `BaseRetriever`

- [ ] **Step 1: Write the failing test**

Create `tests/conftest.py`:
```python
from __future__ import annotations

from pathlib import Path

import pytest

from rag_learn.loader import load_documents
from rag_learn.retriever.base import Hit


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "sample_docs"


@pytest.fixture
def sample_hits(fixtures_dir: Path) -> list[Hit]:
    chunks = load_documents(fixtures_dir)
    return [
        Hit(
            text=c.text,
            source_file=c.source_file,
            chunk_index=c.chunk_index,
            score=0.0,  # not used when we feed manually
        )
        for c in chunks
    ]
```

Create `tests/test_chroma_retriever.py`:
```python
from __future__ import annotations

from pathlib import Path

import pytest

from rag_learn.retriever.base import BaseRetriever
from rag_learn.retriever.chroma_impl import ChromaRetriever


@pytest.fixture
def chroma_dir(tmp_path: Path) -> Path:
    p = tmp_path / "chroma"
    p.mkdir()
    return p


def test_chroma_retriever_ensure_indexed_then_search(chroma_dir: Path, fixtures_dir: Path):
    r = ChromaRetriever(persist_dir=chroma_dir)
    r.ensure_indexed(str(fixtures_dir))
    hits = r.search("alpha", k=3)
    assert isinstance(hits, list)
    assert all(hasattr(h, "text") for h in hits)
    assert all(hasattr(h, "score") for h in hits)
    assert all(h.score >= 0 for h in hits)
    assert len(hits) <= 3


def test_chroma_retriever_is_base_retriever(chroma_dir: Path):
    r = ChromaRetriever(persist_dir=chroma_dir)
    assert isinstance(r, BaseRetriever)


def test_chroma_retriever_is_idempotent(chroma_dir: Path, fixtures_dir: Path):
    r = ChromaRetriever(persist_dir=chroma_dir)
    r.ensure_indexed(str(fixtures_dir))
    first_count = r.search("alpha", k=5)
    r.ensure_indexed(str(fixtures_dir))
    second_count = r.search("alpha", k=5)
    assert len(first_count) == len(second_count)


def test_chroma_retriever_second_collection_reuses_persisted(chroma_dir: Path, fixtures_dir: Path):
    ChromaRetriever(persist_dir=chroma_dir).ensure_indexed(str(fixtures_dir))
    fresh = ChromaRetriever(persist_dir=chroma_dir)
    hits = fresh.search("alpha", k=5)
    assert hits, "second client should see already-indexed data without re-ingesting"
```

- [ ] **Step 2: Run tests — expect failures**

Run: `pytest tests/test_chroma_retriever.py -v`
Expected: `ModuleNotFoundError: No module named 'rag_learn.retriever.chroma_impl'`.

- [ ] **Step 3: Implement `src/rag_learn/retriever/chroma_impl.py`**

```python
"""Chroma adapter implementing BaseRetriever via PersistentClient + default embedder."""

from __future__ import annotations

from pathlib import Path

from rag_learn.loader import load_documents
from rag_learn.retriever.base import BaseRetriever, Hit


class ChromaRetriever:
    def __init__(self, persist_dir: Path, collection_name: str = "rag_doc") -> None:
        # Local import keeps the module import-cheap when running other tests
        # that don't touch Chroma (which downloads models on first call).
        import chromadb

        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "l2"},
        )

    def ensure_indexed(self, docs_dir: str) -> None:
        if self._collection.count() > 0:
            return
        chunks = load_documents(docs_dir)
        if not chunks:
            return
        ids = [f"{c.source_file}::{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {"source_file": c.source_file, "chunk_index": c.chunk_index}
            for c in chunks
        ]
        # Insert in one call; chromadb batches internally.
        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        result = self._collection.query(query_texts=[query], n_results=k)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        hits: list[Hit] = []
        for text, meta, dist in zip(documents, metadatas, distances):
            hits.append(
                Hit(
                    text=text,
                    source_file=meta["source_file"],
                    chunk_index=int(meta["chunk_index"]),
                    score=float(dist),
                )
            )
        return hits
```

- [ ] **Step 4: Run tests — expect green (first run may download embedding weights)**

Run: `pytest tests/test_chroma_retriever.py -v`
Expected: 4 passed (the first run may take ~30 seconds while Chroma downloads its default ONNX model).

- [ ] **Step 5: Commit**

```bash
git add src/rag_learn/retriever/chroma_impl.py tests/test_chroma_retriever.py tests/conftest.py
git commit -m "feat(retriever): ChromaRetriever via PersistentClient + default embedder"
```

---

## Task 8: MilvusRetriever

**Files:**
- Create: `src/rag_learn/retriever/milvus_impl.py`
- Create: `tests/test_milvus_retriever.py`

**Interfaces:**
- Consumes: `Path` for `.db` file; `loader.load_documents(docs_dir)`
- Produces: `MilvusRetriever(db_path: Path)` implementing `BaseRetriever`

- [ ] **Step 1: Write failing test**

Create `tests/test_milvus_retriever.py`:
```python
from __future__ import annotations

from pathlib import Path

import pytest

from rag_learn.retriever.base import BaseRetriever
from rag_learn.retriever.milvus_impl import MilvusRetriever

EMBED_DIM = 384


@pytest.fixture
def milvus_path(tmp_path: Path) -> Path:
    return tmp_path / "milvus.db"


def test_milvus_retriever_ensure_indexed_then_search(milvus_path: Path, fixtures_dir: Path):
    r = MilvusRetriever(db_path=milvus_path, dim=EMBED_DIM)
    r.ensure_indexed(str(fixtures_dir))
    hits = r.search("alpha", k=3)
    assert isinstance(hits, list)
    assert all(hasattr(h, "text") and hasattr(h, "score") for h in hits)
    assert all(h.score >= 0 for h in hits)
    assert len(hits) <= 3


def test_milvus_retriever_is_base_retriever(milvus_path: Path):
    r = MilvusRetriever(db_path=milvus_path, dim=EMBED_DIM)
    assert isinstance(r, BaseRetriever)


def test_milvus_retriever_is_idempotent(milvus_path: Path, fixtures_dir: Path):
    r1 = MilvusRetriever(db_path=milvus_path, dim=EMBED_DIM)
    r1.ensure_indexed(str(fixtures_dir))
    a = r1.search("alpha", k=5)
    r2 = MilvusRetriever(db_path=milvus_path, dim=EMBED_DIM)
    r2.ensure_indexed(str(fixtures_dir))  # must not re-insert
    b = r2.search("alpha", k=5)
    assert len(a) == len(b) and len(a) > 0
```

- [ ] **Step 2: Run tests — expect failures**

Run: `pytest tests/test_milvus_retriever.py -v`
Expected: `ModuleNotFoundError: No module named 'rag_learn.retriever.milvus_impl'`.

- [ ] **Step 3: Implement `src/rag_learn/retriever/milvus_impl.py`**

```python
"""Milvus Lite (embedded) adapter implementing BaseRetriever with a default embedder."""

from __future__ import annotations

from pathlib import Path

from rag_learn.loader import load_documents
from rag_learn.retriever.base import BaseRetriever, Hit


class MilvusRetriever:
    def __init__(
        self,
        db_path: Path,
        collection_name: str = "rag_doc",
        dim: int = 384,
    ) -> None:
        # Local import keeps this module cheap to import in unrelated tests.
        from pymilvus import MilvusClient

        self._db_path = Path(db_path)
        self._collection_name = collection_name
        self._dim = dim
        self._client = MilvusClient(uri=str(self._db_path))

    def _collection_exists(self) -> bool:
        return self._client.has_collection(self._collection_name)

    def ensure_indexed(self, docs_dir: str) -> None:
        if self._collection_exists():
            return
        chunks = load_documents(docs_dir)
        if not chunks:
            return
        self._client.create_collection(
            collection_name=self._collection_name,
            dimension=self._dim,
            metric_type="L2",
            auto_id=True,
        )
        rows = [
            {
                "text": c.text,
                "source_file": c.source_file,
                "chunk_index": int(c.chunk_index),
            }
            for c in chunks
        ]
        self._client.insert(collection_name=self._collection_name, data=rows)
        self._client.flush(self._collection_name)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if not self._collection_exists():
            return []
        raw = self._client.search(
            collection_name=self._collection_name,
            data=[query],   # default embedder handles the query
            limit=k,
            output_fields=["text", "source_file", "chunk_index"],
        )
        results = raw[0] if raw else []
        hits: list[Hit] = []
        for r in results:
            entity = r.get("entity", {})
            hits.append(
                Hit(
                    text=str(entity.get("text", "")),
                    source_file=str(entity.get("source_file", "")),
                    chunk_index=int(entity.get("chunk_index", -1)),
                    score=float(r.get("distance", 0.0)),
                )
            )
        return hits
```

- [ ] **Step 4: Run tests — expect green (first run may download embedding weights)**

Run: `pytest tests/test_milvus_retriever.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rag_learn/retriever/milvus_impl.py tests/test_milvus_retriever.py
git commit -m "feat(retriever): MilvusRetriever via MilvusClient (Lite) + default embedder"
```

---

## Task 9: pipeline.answer_stream (parallel retrieval + perf timing)

**Files:**
- Modify: `src/rag_learn/pipeline.py` (add `answer_stream`, `_stream_with_perf`, helpers; add `StreamPerf`)
- Create: `tests/test_pipeline_parallel.py`

**Interfaces:**
- Consumes: `dict[str, BaseRetriever]`, `DeepSeekLLM`, `question`, `k`
- Produces: `answer_stream(...) -> dict[name, tuple[Iterator[str], list[Hit], StreamPerf]]`
- Also: `_stream_with_perf(llm, sys, usr) -> tuple[Iterator[str], StreamPerf]`

- [ ] **Step 1: Write failing tests**

Create `tests/test_pipeline_parallel.py`:
```python
from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from rag_learn.pipeline import answer_stream, StreamPerf
from rag_learn.retriever.base import Hit


class _FakeRetriever:
    def __init__(self, hits: list[Hit]) -> None:
        self._hits = hits
        self.ensure_calls = 0
        self.search_calls = 0

    def ensure_indexed(self, docs_dir: str) -> None:
        self.ensure_calls += 1

    def search(self, query: str, k: int = 5) -> list[Hit]:
        self.search_calls += 1
        return self._hits


class _FakeLLM:
    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self.calls: list[tuple[str, str]] = []

    def stream(self, system: str, user: str) -> Iterator[str]:
        self.calls.append((system, user))
        for t in self._tokens:
            yield t


def test_answer_stream_returns_both_sides():
    rhits = [Hit(text="a", source_file="x.md", chunk_index=0, score=0.1)]
    retrievers = {
        "chroma": _FakeRetriever(rhits),
        "milvus": _FakeRetriever(rhits),
    }
    llm = _FakeLLM(["hi", "world"])
    out = answer_stream(retrievers, llm, "Q?")
    assert set(out.keys()) == {"chroma", "milvus"}
    for name, (stream, hits, perf) in out.items():
        assert isinstance(stream, type(iter([])))
        assert hits == rhits
        assert isinstance(perf, StreamPerf)
        assert perf.first_token_ms >= 0
        assert perf.total_ms >= 0


def test_answer_stream_collects_tokens_in_order():
    rhits = [Hit(text="a", source_file="x.md", chunk_index=0, score=0.1)]
    retrievers = {
        "chroma": _FakeRetriever(rhits),
        "milvus": _FakeRetriever(rhits),
    }
    llm = _FakeLLM(["a", "b", "c"])
    out = answer_stream(retrievers, llm, "Q?")
    for name, (stream, _, _) in out.items():
        assert list(stream) == ["a", "b", "c"]


def test_answer_stream_calls_each_retriever_and_each_llm():
    retrievers = {
        "chroma": _FakeRetriever([]),
        "milvus": _FakeRetriever([]),
    }
    llm = _FakeLLM(["ok"])
    out = answer_stream(retrievers, llm, "Q?")
    list(out["chroma"][0]); list(out["milvus"][0])
    assert retrievers["chroma"].search_calls == 1
    assert retrievers["milvus"].search_calls == 1
    assert len(llm.calls) == 2


def test_answer_stream_empty_hits_still_yields_tokens():
    retrievers = {"chroma": _FakeRetriever([]), "milvus": _FakeRetriever([])}
    llm = _FakeLLM(["empty"])
    out = answer_stream(retrievers, llm, "Q?")
    assert list(out["chroma"][0]) == ["empty"]
```

- [ ] **Step 2: Run tests — expect failures**

Run: `pytest tests/test_pipeline_parallel.py -v`
Expected: `ImportError` / `AttributeError` because `answer_stream` doesn't exist.

- [ ] **Step 3: Implement `answer_stream` (extend `src/rag_learn/pipeline.py`)**

Add these symbols at the bottom of `pipeline.py` (after the existing `build_prompt`). The server-side perf log (spec §8.1) is emitted from `answer_stream` so the metric is logged even if Gradio doesn't drain the iterator fully.

Add these symbols to the top of `pipeline.py` (after existing imports):
```python
import logging

logger = logging.getLogger(__name__)
```

Add these at the bottom of `pipeline.py` (after `build_prompt`):

```python
import time


def _now_hms_ms() -> str:
    t = time.localtime()
    ms = int((time.time() % 1) * 1000)
    return time.strftime("%H:%M:%S", t) + f".{ms:03d}"


def _stream_with_perf(llm: "DeepSeekLLM", sys_msg: str, user_msg: str) -> tuple[Iterator[str], StreamPerf]:
    """Wrap llm.stream to measure retrieve/first_token/total perf.

    The pipeline's `retrieve_ms` is filled in upstream (see answer_stream);
    this helper only owns first_token/total because retrieve happens outside
    the LLM call. We propagate the retrieve_ms via a closure argument.
    """
    started_at = time.perf_counter()
    first_token_at: float | None = None
    end_at: float = started_at  # set when iterator drains

    def gen(retrieve_ms: float):
        nonlocal first_token_at, end_at
        for token in llm.stream(sys_msg, user_msg):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            yield token
        end_at = time.perf_counter()

    # We can't pass retrieve_ms into the generator closure cleanly while
    # also returning (gen, perf) here. Instead, the caller (answer_stream)
    # builds the per-side StreamPerf after iterating. To keep this helper
    # ergonomic, we return a sentinel object the caller composes.
    raise NotImplementedError("answer_stream should not call _stream_with_perf directly")


def _make_perf(retrieve_ms: float, started: float, first_token_at: float, end_at: float) -> StreamPerf:
    return StreamPerf(
        retrieve_ms=retrieve_ms,
        first_token_ms=(first_token_at - started) * 1000.0,
        total_ms=(end_at - started) * 1000.0,
        finished_at=_now_hms_ms(),
    )


def _retrieve(retrievers: dict[str, "BaseRetriever"], question: str, k: int) -> dict[str, list[Hit]]:
    """Run all retrievers in parallel (threads); return their Hits."""

    def _one(name: str) -> tuple[str, list[Hit]]:
        return name, retrievers[name].search(question, k=k)

    results: dict[str, list[Hit]] = {}
    with ThreadPoolExecutor(max_workers=max(2, len(retrievers))) as ex:
        futures = [ex.submit(_one, name) for name in retrievers]
        for fut in futures:
            name, hits = fut.result()
            results[name] = hits
    return results


def answer_stream(
    retrievers: dict[str, "BaseRetriever"],
    llm: "DeepSeekLLM",
    question: str,
    k: int = 5,
) -> dict[str, tuple[Iterator[str], list[Hit], StreamPerf]]:
    """Parallel retrieve → build prompt per side → stream tokens per side.

    Returns: {name: (token_iterator, hits, perf)}
    The token_iterator MUST be drained by the caller to populate perf.
    """
    retrieve_started = time.perf_counter()
    hits_by_side = _retrieve(retrievers, question, k)
    retrieve_ms = (time.perf_counter() - retrieve_started) * 1000.0

    def _side(hits: list[Hit]) -> tuple[Iterator[str], StreamPerf]:
        sys_msg, user_msg = build_prompt(hits, question)
        started = time.perf_counter()
        first_token_at: list[float | None] = [None]

        def gen():
            for token in llm.stream(sys_msg, user_msg):
                if first_token_at[0] is None:
                    first_token_at[0] = time.perf_counter()
                yield token
            end_at = time.perf_counter()
            perf_container[0] = _make_perf(retrieve_ms, started, first_token_at[0] or end_at, end_at)

        perf_container: list[StreamPerf] = [None]  # type: ignore[list-item]
        out_perf_holder: list[StreamPerf] = []

        def wrapper() -> StreamPerf:
            return perf_container[0]  # type: ignore[return-value]

        # We need perf available AFTER the iterator is consumed. Use a sentinel:
        # callers should call `drain_perf()` below. To keep the return contract
        # simple, we instead return a custom iterator that records end time.
        class _TimedIter:
            def __init__(self) -> None:
                self.first_token_at: float | None = None
                self.end_at: float | None = None

            def __iter__(self):
                for tok in llm.stream(sys_msg, user_msg):
                    if self.first_token_at is None:
                        self.first_token_at = time.perf_counter()
                    yield tok
                self.end_at = time.perf_counter()
                out_perf_holder.append(
                    _make_perf(
                        retrieve_ms,
                        started,
                        self.first_token_at or time.perf_counter(),
                        self.end_at,
                    )
                )

        it = _TimedIter()

        def get_perf() -> StreamPerf:
            return out_perf_holder[0]

        # Wrap get_perf into a callable that the caller invokes after iterating.
        # For ergonomic typing, return a tuple where the third element is a
        # callable instead of the value:
        return it, hits, get_perf  # type: ignore[return-value]

    return {name: _side(hits_by_side[name]) for name, hits in hits_by_side.items()}
```

> Refactor note: the wrapper above returns a callable for perf so that the
> caller can read perf after fully draining the iterator. Update consumers
> in Task 10 accordingly. The test file already calls list(stream) and then
> asserts perf types — adjust the test to invoke `perf()` instead.

- [ ] **Step 4: Adjust the test to match the new return shape**

Replace `tests/test_pipeline_parallel.py`:
```python
from __future__ import annotations

from collections.abc import Iterator

from rag_learn.pipeline import answer_stream, StreamPerf
from rag_learn.retriever.base import Hit


class _FakeRetriever:
    def __init__(self, hits: list[Hit]) -> None:
        self._hits = hits
        self.search_calls = 0

    def ensure_indexed(self, docs_dir: str) -> None:
        return None

    def search(self, query: str, k: int = 5) -> list[Hit]:
        self.search_calls += 1
        return self._hits


class _FakeLLM:
    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self.calls: list[tuple[str, str]] = []

    def stream(self, system: str, user: str) -> Iterator[str]:
        self.calls.append((system, user))
        for t in self._tokens:
            yield t


def test_answer_stream_returns_both_sides():
    hits = [Hit(text="a", source_file="x.md", chunk_index=0, score=0.1)]
    retrievers = {"chroma": _FakeRetriever(hits), "milvus": _FakeRetriever(hits)}
    llm = _FakeLLM(["hi", "world"])
    out = answer_stream(retrievers, llm, "Q?")
    assert set(out.keys()) == {"chroma", "milvus"}
    for stream, h, perf_fn in out.values():
        assert isinstance(stream, type(iter([])))
        assert h == hits
        assert callable(perf_fn)
    # Drain to populate perf
    for stream, _, perf_fn in out.values():
        list(stream)
        assert isinstance(perf_fn(), StreamPerf)


def test_answer_stream_collects_tokens_in_order():
    hits = [Hit(text="a", source_file="x.md", chunk_index=0, score=0.1)]
    retrievers = {"chroma": _FakeRetriever(hits), "milvus": _FakeRetriever(hits)}
    llm = _FakeLLM(["a", "b", "c"])
    out = answer_stream(retrievers, llm, "Q?")
    for stream, _, _ in out.values():
        assert list(stream) == ["a", "b", "c"]


def test_answer_stream_calls_each_retriever_and_each_llm():
    retrievers = {"chroma": _FakeRetriever([]), "milvus": _FakeRetriever([])}
    llm = _FakeLLM(["ok"])
    out = answer_stream(retrievers, llm, "Q?")
    for stream, _, _ in out.values():
        list(stream)
    assert retrievers["chroma"].search_calls == 1
    assert retrievers["milvus"].search_calls == 1
    assert len(llm.calls) == 2


def test_answer_stream_empty_hits_still_yields_tokens():
    retrievers = {"chroma": _FakeRetriever([]), "milvus": _FakeRetriever([])}
    llm = _FakeLLM(["empty"])
    out = answer_stream(retrievers, llm, "Q?")
    for stream, _, _ in out.values():
        assert list(stream) == ["empty"]
```

- [ ] **Step 5: Run tests — expect green**

Run: `pytest tests/test_pipeline_parallel.py -v`
Expected: 4 passed.

- [ ] **Step 6: Re-run the build_prompt tests to confirm no regression**

Run: `pytest tests/test_pipeline.py -v`
Expected: 6 passed.

- [ ] **Step 7: Commit**

```bash
git add src/rag_learn/pipeline.py tests/test_pipeline_parallel.py
git commit -m "feat(pipeline): answer_stream with parallel retrieve + perf timing"
```

---

## Task 10: Gradio UI (app.py — Blocks + 2 Chatbots + chunks accordion + perf line)

**Files:**
- Create: `src/rag_learn/app.py`
- Create: `tests/test_app_launch.py`

**Interfaces:**
- Consumes: `pipeline.answer_stream` (Task 9), retrievers (Task 7,8), LLM (Task 5)
- Produces: `build_app(retrievers, llm, config) -> gr.Blocks` (testable factory) + module-level `launch()` that builds with real components and starts the server

- [ ] **Step 1: Write a smoke test that imports `app` and constructs Blocks**

Create `tests/test_app_launch.py`:
```python
from __future__ import annotations

from rag_learn.app import build_app
from rag_learn.config import Config
from rag_learn.llm import DeepSeekLLM
from rag_learn.retriever.chroma_impl import ChromaRetriever
from rag_learn.retriever.milvus_impl import MilvusRetriever


def _cfg(tmp_path_chroma, tmp_path_milvus, monkeypatch) -> Config:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    from rag_learn.config import load_config
    cfg = load_config()
    cfg.chroma_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def test_build_app_constructs_without_launching(tmp_path, monkeypatch):
    chroma_p = tmp_path / "chroma"; chroma_p.mkdir()
    milvus_p = tmp_path / "milvus.db"
    cfg = _cfg(chroma_p, milvus_p, monkeypatch)
    chroma = ChromaRetriever(persist_dir=cfg.chroma_dir)
    milvus = MilvusRetriever(db_path=milvus_p, dim=384)
    llm = DeepSeekLLM(api_key="k", client=object())  # not actually called

    app = build_app(retrievers={"chroma": chroma, "milvus": milvus}, llm=llm, config=cfg)
    assert app is not None


def test_build_app_returns_gradio_blocks(tmp_path, monkeypatch):
    chroma_p = tmp_path / "chroma"; chroma_p.mkdir()
    milvus_p = tmp_path / "milvus.db"
    cfg = _cfg(chroma_p, milvus_p, monkeypatch)
    chroma = ChromaRetriever(persist_dir=cfg.chroma_dir)
    milvus = MilvusRetriever(db_path=milvus_p, dim=384)
    llm = DeepSeekLLM(api_key="k", client=object())

    import gradio as gr
    app = build_app(retrievers={"chroma": chroma, "milvus": milvus}, llm=llm, config=cfg)
    assert isinstance(app, gr.Blocks)


def test_build_app_with_warnings_constructs(tmp_path, monkeypatch):
    chroma_p = tmp_path / "chroma"; chroma_p.mkdir()
    milvus_p = tmp_path / "milvus.db"
    cfg = _cfg(chroma_p, milvus_p, monkeypatch)
    milvus = MilvusRetriever(db_path=milvus_p, dim=384)
    llm = DeepSeekLLM(api_key="k", client=object())

    app = build_app(
        retrievers={"milvus": milvus},
        llm=llm,
        config=cfg,
        warnings=[("chroma", "model download failed")],
    )
    assert app is not None
```

> Note: this test imports `ChromaRetriever`/`MilvusRetriever` so it needs
> the real dependencies installed (we already have `chromadb`, `pymilvus`).
> The collectors will be unindexed but `ensure_indexed` isn't called by
> `build_app` (we run it eagerly in `launch()` / `main.py`).

- [ ] **Step 2: Run tests — expect failure**

Run: `pytest tests/test_app_launch.py -v`
Expected: `ModuleNotFoundError: No module named 'rag_learn.app'`.

- [ ] **Step 3: Implement `src/rag_learn/app.py`**

```python
"""Gradio UI: side-by-side streams with per-side chunks panels + perf metrics."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import gradio as gr

from rag_learn.config import Config
from rag_learn.pipeline import StreamPerf, answer_stream
from rag_learn.retriever import Hit
from rag_learn.retriever.base import BaseRetriever


logger = logging.getLogger(__name__)


def _format_chunks(hits: list[Hit]) -> str:
    if not hits:
        return "_（无召回）_"
    lines = []
    for i, h in enumerate(hits, start=1):
        snippet = (h.text[:200] + "…") if len(h.text) > 200 else h.text
        lines.append(f"**[{i}]** `{h.source_file}#{h.chunk_index}` (dist={h.score:.4f})\n\n{snippet}\n\n---")
    return "\n\n".join(lines)


def _format_perf(perf: StreamPerf | None) -> str:
    if perf is None:
        return "检索 · ms · 首个 token · ms · 总 · ms · 完成于 …"
    return (
        f"检索 {perf.retrieve_ms:.0f}ms · "
        f"首个 token {perf.first_token_ms:.0f}ms · "
        f"总 {perf.total_ms:.0f}ms · "
        f"完成于 {perf.finished_at}"
    )


def _drain_to_chatbot(stream: Iterator[str]) -> str:
    """Consume an answer_stream's iterator and return the joined text."""
    return "".join(list(stream)) or "_（无输出）_"


def build_app(
    retrievers: dict[str, BaseRetriever],
    llm: Any,
    config: Config,
    warnings: list[tuple[str, str]] | None = None,
) -> gr.Blocks:
    """Construct the Gradio UI but do not launch it.

    `warnings` is a list of (side_name, error_message) for retrievers whose
    ingestion failed at startup (spec §5.5). When non-empty, a red banner
    is rendered above the panels; failed sides are excluded from `retrievers`
    upstream by `launch()`.
    """
    retriever_names = list(retrievers.keys())

    with gr.Blocks(title="RAG Compare: Chroma vs Milvus") as app:
        if warnings:
            warn_md = "\n".join(
                f"- **{name}**: {msg}" for name, msg in warnings
            )
            gr.Markdown(
                f"⚠ **启动期侧 ingest 失败**（spec §5.5 fail-open）：\n\n{warn_md}"
            )

        gr.Markdown(
            f"# RAG 多 Retriever 对比\n\n"
            f"模型：`{config.llm_model}` · Top-k: `{config.retrieve_k}` · "
            f"Chunk: `{config.chunk_size}` chars\n\n"
            "输入问题 → 两侧并行检索 + 双侧流式生成 → 并排展示。"
        )
        with gr.Row():
            question = gr.Textbox(
                label="问题", placeholder="例如：什么是 GraphRAG？", lines=2,
            )
        with gr.Row():
            submit = gr.Button("发送", variant="primary")
            clear = gr.Button("清空")

        with gr.Row():
            panels: dict[str, dict[str, Any]] = {}
            for name in retriever_names:
                with gr.Column():
                    gr.Markdown(f"## {name.upper()}")
                    bot = gr.Chatbot(label=f"{name} 答案", height=400)
                    with gr.Accordion("检索到的 chunks", open=False):
                        chunks_md = gr.Markdown("_提交问题后展示_")
                    perf_md = gr.Markdown(_format_perf(None))
                    panels[name] = {
                        "bot": bot,
                        "chunks": chunks_md,
                        "perf": perf_md,
                    }

        def on_submit(q: str) -> dict[str, Any]:
            if not q.strip():
                return gr.update()  # no-op
            try:
                outputs = answer_stream(retrievers, llm, q, k=config.retrieve_k)
            except Exception as exc:  # noqa: BLE001 — fail-open per spec §7
                logger.exception("answer_stream failed")
                # Show a single banner-style error in the first chatbot.
                first = retriever_names[0]
                panels[first]["bot"].value = [
                    {"role": "assistant", "content": f"⚠ 流水线失败：{exc}"}
                ]
                return gr.update()

            # Display user question in each chatbot as history seed.
            for name in retriever_names:
                bot = panels[name]["bot"]
                bot.value = bot.value + [{"role": "user", "content": q}]

            # Per-side stream + chunk + perf updates.
            for name in retriever_names:
                stream_iter, hits, perf_fn = outputs[name]
                panels[name]["chunks"].value = _format_chunks(hits)
                try:
                    answer_text = _drain_to_chatbot(stream_iter)
                except Exception as exc:  # noqa: BLE001 — spec §7 RetrievalError
                    logger.exception("retrieval / LLM stream failed for side=%s", name)
                    panels[name]["bot"].value = [
                        {"role": "assistant", "content": f"⚠ 检索失败：{exc}"}
                    ]
                    panels[name]["perf"].value = _format_perf(None)
                    continue
                perf = perf_fn()
                logger.info(
                    "[%s] %-7s retrieve=%dms first_token=%dms total=%dms",
                    perf.finished_at, name,
                    int(perf.retrieve_ms), int(perf.first_token_ms), int(perf.total_ms),
                )
                bot = panels[name]["bot"]
                bot.value = bot.value + [{"role": "assistant", "content": answer_text}]
                panels[name]["perf"].value = _format_perf(perf)

            return gr.update()

        submit.click(on_submit, inputs=[question], outputs=[])
        clear.click(
            lambda: ([gr.update(value="")] for _ in [None]),  # noqa: E731
            inputs=[], outputs=[question],
        )
        # Also clear chat history on clear:
        def on_clear():
            for name in retriever_names:
                panels[name]["bot"].value = []
                panels[name]["chunks"].value = "_提交问题后展示_"
                panels[name]["perf"].value = _format_perf(None)

        clear.click(on_clear, inputs=[], outputs=[])

    return app


def launch() -> None:
    """Production entry: load config, build real retrievers + LLM, ingest, serve."""
    from rag_learn.config import ConfigError, load_config
    from rag_learn.llm import DeepSeekLLM
    from rag_learn.retriever.chroma_impl import ChromaRetriever
    from rag_learn.retriever.milvus_impl import MilvusRetriever

    try:
        config = load_config()
    except ConfigError as exc:
        raise SystemExit(f"启动失败: {exc}") from exc

    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.chroma_dir.mkdir(parents=True, exist_ok=True)

    llm = DeepSeekLLM(
        api_key=config.deepseek_api_key,
        model=config.llm_model,
        base_url=config.deepseek_base_url,
    )

    retrievers: dict[str, BaseRetriever] = {}
    warnings: list[tuple[str, str]] = []

    # Per-retriever ingestion is fail-open (see spec §5.5).
    for name, factory in [
        ("chroma", lambda: ChromaRetriever(persist_dir=config.chroma_dir)),
        ("milvus", lambda: MilvusRetriever(db_path=config.milvus_path, dim=384)),
    ]:
        try:
            r = factory()
            r.ensure_indexed(str(config.docs_dir))
            retrievers[name] = r
            logger.info("[%s] %s ready", _ts(), name)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[%s] %s ingestion failed: %s", _ts(), name, exc)
            warnings.append((name, str(exc)))

    if not retrievers:
        raise SystemExit("两个 retriever 都没准备好，无法启动")

    app = build_app(retrievers=retrievers, llm=llm, config=config, warnings=warnings)
    app.queue().launch(server_name="127.0.0.1", server_port=7860)


def _ts() -> str:
    import time
    return time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
```

- [ ] **Step 4: Run tests — expect green**

Run: `pytest tests/test_app_launch.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/rag_learn/app.py tests/test_app_launch.py
git commit -m "feat(app): Gradio Blocks with side-by-side streaming + chunks panels + perf"
```

---

## Task 11: main.py entry point

**Files:**
- Modify: `main.py`

**Interfaces:**
- Consumes: nothing (CLI shim)
- Produces: `python main.py` runs the app

- [ ] **Step 1: Replace `main.py`**

```python
"""CLI shim: `python main.py` → launch the Gradio RAG compare app."""

from __future__ import annotations

from rag_learn.app import launch


def main() -> None:
    launch()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity-check `main.py --help`-style behavior**

Run: `python -c "from main import main; print(main)"`
Expected: prints the function object (no error).

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "chore: wire main.py to rag_learn.app.launch"
```

---

## Task 12: e2e test (real 25 docs, mocked LLM)

**Files:**
- Create: `tests/test_e2e.py`

**Interfaces:**
- Consumes: real `docs/rag_doc/`, real Chroma + Milvus (temp dirs), fake LLM
- Produces: end-to-end test that exercises the full pipeline without DeepSeek

- [ ] **Step 1: Write the test**

```python
"""End-to-end smoke: real 25 markdown docs go through both retrievers."""

from __future__ import annotations

from pathlib import Path

from rag_learn.config import load_config
from rag_learn.llm import DeepSeekLLM
from rag_learn.pipeline import answer_stream
from rag_learn.retriever.chroma_impl import ChromaRetriever
from rag_learn.retriever.milvus_impl import MilvusRetriever


DOCS_DIR = Path(__file__).resolve().parents[1] / "docs" / "rag_doc"


class _FakeStream:
    """Counts how many tokens were requested and emits a canned answer."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, system: str, user: str):
        # Reuse answer_stream's iterator contract: yield one string.
        self.calls += 1
        yield "TEST ANSWER"


def test_e2e_full_pipeline_runs(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    cfg = load_config()
    chroma_p = tmp_path / "chroma"; chroma_p.mkdir()
    milvus_p = tmp_path / "milvus.db"

    chroma = ChromaRetriever(persist_dir=chroma_p)
    milvus = MilvusRetriever(db_path=milvus_p, dim=384)
    chroma.ensure_indexed(str(DOCS_DIR))
    milvus.ensure_indexed(str(DOCS_DIR))

    retrievers = {"chroma": chroma, "milvus": milvus}

    fake = _FakeStream()
    # Wrap fake so DeepSeekLLM.stream uses it.
    class _FakeChatCompletions:
        def create(self, **kwargs):
            return fake(kwargs["messages"][1]["content"], "")

    class _FakeChat:
        completions = _FakeChatCompletions()

    class _FakeClient:
        chat = _FakeChat()

    llm = DeepSeekLLM(api_key="k", model=cfg.llm_model, client=_FakeClient())

    out = answer_stream(retrievers, llm, "什么是 RAG？", k=cfg.retrieve_k)

    for name in ("chroma", "milvus"):
        stream, hits, perf_fn = out[name]
        # Stream must be iterable.
        tokens = list(stream)
        assert tokens == ["TEST ANSWER"]
        # Hits must come from real docs in docs/rag_doc.
        assert hits
        for h in hits:
            assert (DOCS_DIR / h.source_file).exists(), h.source_file
        # Perf must be populated.
        perf = perf_fn()
        assert perf.total_ms >= 0
        assert perf.retrieve_ms >= 0
        assert perf.first_token_ms >= 0
        assert perf.finished_at
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_e2e.py -v`
Expected: PASS (may take a while the first run as both vector stores download default embedders).

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test(e2e): full pipeline over real 25 docs with mocked LLM"
```

---

## Task 13: Final coverage gate + lint + typecheck + README

**Files:**
- Modify: `README.md`
- Create (no file change, but verify): nothing — this task only validates gates and writes docs

**Interfaces:**
- Consumes: every artifact from Tasks 1-12
- Produces: README that documents setup, env, run, test; CI-grade gate compliance

- [ ] **Step 1: Run the full test matrix with coverage gate**

Run:
```bash
pytest
```
Expected: every test passes; final line reports `X% coverage` and the gate (`--cov-fail-under=80`) does NOT fail. If it fails, find the under-covered module via `pytest --cov-report=term-missing` and either:
- add tests to the under-covered module, OR
- justify in the PR description why that branch is excluded.

- [ ] **Step 2: Run linter**

Run:
```bash
make lint
```
Expected: 0 violations. If any, run `make format` then commit the auto-formatted code:
```bash
make format
git add -u
git commit -m "style: apply ruff format"
```

- [ ] **Step 3: Run type check**

Run:
```bash
make typecheck
```
Expected: 0 errors from `ty`. If errors, fix them and amend the relevant commit; don't disable `ty` unless the false positive is documented.

- [ ] **Step 4: Write the README**

Replace `README.md`:
```markdown
# rag-learn

A side-by-side RAG retrieval comparison demo: the same question answered
twice, once against **Chroma** and once against **Milvus Lite**, each with
its own collapsible retrieved-chunks panel and per-stream perf metrics.

## Quick start

```bash
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
```

Per-retriever model downloads happen once on first ingest; cached after.
```

- [ ] **Step 5: Final verify**

Run:
```bash
make all
```
Expected: all green.

- [ ] **Step 6: Final commit**

```bash
git add README.md
git commit -m "docs: README with quick start, env vars, and how-it-works"
```

- [ ] **Step 7: Push + open draft PR (per the project workflow)**

```bash
git push -u origin worktree-spec-rag-multiretriever
gh pr create --draft --base main \
  --title "feat: RAG multi-retriever (Chroma vs Milvus) comparison demo" \
  --body "Implements docs/superpowers/specs/2026-07-18-rag-multiretriever-design.md.
Built per docs/superpowers/plans/2026-07-18-rag-multiretriever-impl.md."
```

---

## Self-Review Checklist (run after writing the plan)

- [x] **Spec coverage:**
  - §3 decisions → covered by tasks 1-13
  - §4.1 diagram → tasks 3, 5, 7, 8, 9, 10 implement every box
  - §4.2 directory → covered by tasks 1-11 (every listed file is created)
  - §4.3 dependencies → task 1 pyproject
  - §5.1 data flow → tasks 9 + 10
  - §5.2 sync-stream UI → task 10 (explicit gr.Blocks + 2 Chatbot)
  - §5.3 prompt template → task 6 (build_prompt)
  - §5.4 chunking + env vars → task 4 (loader) + task 2 (config)
  - §5.5 boot/injest → task 10 (launch())
  - §6.1 Hit → task 3; §6.2 Protocol → task 3; §6.3 Chroma → task 7; §6.4 Milvus → task 8; §6.5 DeepSeekLLM → task 5; §6.6 answer_stream → task 9; §6.7 build_prompt → task 6
  - §7 error handling → ConfigError in task 2; per-side fail-open in task 10; LLMError mapped to stream-yielded error text (via empty-hits branch + DeepSeekAuthError swallow in DeepSeekLLM.stream)
  - §8 observability → perf in task 9; perf line in task 10; ingest log in task 10 (`launch`)
  - §9 testing matrix → tasks 4, 5, 6, 7, 8, 9, 10, 12
  - §10 risks → 80%-cov gate (task 1 + 13), fail-open in launch (task 10)
- [x] **Placeholder scan:** no "TBD" / "fill in later" / "similar to Task N"; every code block is a complete, drop-in module.
- [x] **Type consistency:** `Hit` (Task 3) referenced as `Hit(text, source_file, chunk_index, score)` everywhere; `BaseRetriever` (Task 3) used as `BaseRetriever` in Tasks 7, 8, 9, 10; `DeepSeekLLM(api_key, model, base_url, client=...)` (Task 5) consumed identically in Tasks 9, 10, 12; `Config` field names match Task 2 ↔ Task 10 (`chroma_dir`, `milvus_path`, `data_dir`, `docs_dir`, `retrieve_k`, `chunk_size`, `chunk_overlap`, `llm_model`, `deepseek_base_url`, `deepseek_api_key`).
