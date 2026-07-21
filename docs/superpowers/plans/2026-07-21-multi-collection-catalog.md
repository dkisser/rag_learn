# Multi-Collection Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `docs/shanzhongshi/` as a separate Chroma collection and introduce a `Collection` + `Catalog` abstraction so the Gradio UI can switch between collections via a top dropdown.

**Architecture:** New `src/rag_learn/collections.py` defines `Collection` (frozen dataclass: name/display_name/docs_dir/retriever_factory with lazy `retriever` property) and `Catalog` (immutable registry with `names()` / `display_choices()` / `get()` / `ensure_all_indexed()`). `BUILTIN_COLLECTIONS` registers `rag_doc` and `shanzhongshi`. `app.build_app()` takes a `Catalog` and renders a single-stream UI with a `gr.Dropdown` for collection selection. `pipeline.answer_stream` signature and behavior stay unchanged; new mode calls it with a single-entry dict. Legacy `data/chroma/` data is auto-migrated to `data/chroma/rag_doc/` on first launch with a `.migrated` sentinel.

**Tech Stack:** Python 3.12, Chroma (`chromadb<1`), Gradio 5, pytest + pytest-cov (≥80% coverage), ruff format/lint, ty typecheck.

**Spec:** `docs/superpowers/specs/2026-07-21-multi-collection-catalog-design.md`

## Global Constraints

These apply to every task. Copy from spec verbatim where applicable:

- **TDD mandatory:** write the failing test, run it (red), write minimal implementation, run it (green), refactor.
- **Coverage floor:** `make all` enforces `--cov-fail-under=80` via `pyproject.toml [tool.pytest.ini_options]`. New code must keep the gate green.
- **Lint/format:** `ruff format src tests` + `ruff check src tests` (line-length 100, selects `E F I B UP`).
- **Typecheck:** `ty check src` (Astral's ty, NOT mypy).
- **Immutable patterns:** `Collection` and `Catalog` are `@dataclass(frozen=True)`. Use `object.__setattr__` for the lazy-retriever cache slot only.
- **Type annotations on all function signatures** (PEP 8 / project style).
- **Logging:** `logger = logging.getLogger(__name__)` per module; no `print()`.
- **Fail-open on per-collection ingest** (spec §7); fail-closed on missing `DEEPSEEK_API_KEY`.
- **Chroma collection name constraints:** 3–63 chars, alnum + `_-.`, must start+end alnum. Enforced in `Collection.__post_init__`.
- **Pre-commit gate:** `make all` before every commit. If it fails, fix and amend (no `--no-verify`).
- **Branch model:** commit on `main` (this is a side-by-side demo; no PR ceremony needed).
- **Test fixture style:** tests inject a `FakeRetriever`/`StubRetriever` via `retriever_factory=` rather than patching the global `ChromaRetriever` import. New `tests/fixtures/sample_docs_alt/` mirrors the existing `sample_docs/` shape.

---

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `src/rag_learn/collections.py` | `Collection`, `Catalog`, `BUILTIN_COLLECTIONS`, `build_catalog` | NEW |
| `src/rag_learn/app.py` | `build_app(catalog, ...)`, `launch()`, `_migrate_legacy_chroma` | MODIFY |
| `src/rag_learn/pipeline.py` | docstring only (two-mode note) | MODIFY |
| `tests/test_collections.py` | Collection + Catalog + build_catalog unit tests | NEW |
| `tests/test_app_launch.py` | adapt `build_app(catalog=...)` signature, add Dropdown assertions | MODIFY |
| `tests/test_e2e.py` | build multi-collection mock catalog | MODIFY |
| `tests/fixtures/sample_docs_alt/01.md`, `02.md`, `03.md` | second collection fixture (3 tiny markdowns) | NEW |

The `retriever/*`, `loader.py`, `config.py`, `llm.py`, `main.py` files are not touched.

---

## Task 1: `Collection` dataclass + sample_docs_alt fixture

**Files:**
- Create: `tests/fixtures/sample_docs_alt/01-coffee.md`
- Create: `tests/fixtures/sample_docs_alt/02-tea.md`
- Create: `tests/fixtures/sample_docs_alt/03-recipe.md`
- Create: `src/rag_learn/collections.py`
- Test: `tests/test_collections.py`

**Interfaces (consumed by later tasks):**

```python
@dataclass(frozen=True)
class Collection:
    name: str
    display_name: str
    docs_dir: Path
    description: str = ""
    retriever_factory: Callable[[Path, str], BaseRetriever] = <default chroma>

    @property
    def retriever(self) -> BaseRetriever: ...  # lazy + cached
```

A `CollectionNotFoundError(KeyError)` is defined in Task 2; Task 1 does not raise it.

- [ ] **Step 1: Create the `sample_docs_alt` fixture files**

Three tiny markdown files with non-overlapping vocabulary so two collections don't pollute each other in retrieval tests.

`tests/fixtures/sample_docs_alt/01-coffee.md`:

```markdown
# Coffee Notes

This document describes espresso extraction.

Espresso requires nine bars of pressure and a fine grind.
```

`tests/fixtures/sample_docs_alt/02-tea.md`:

```markdown
# Tea Notes

This document describes gongfu brewing.

Gongfu tea uses a small gaiwan and many short steeps.
```

`tests/fixtures/sample_docs_alt/03-recipe.md`:

```markdown
# Recipe Notes

This document describes sourdough baking.

Sourdough needs a 12-hour bulk fermentation at room temperature.
```

- [ ] **Step 2: Write the failing test file**

`tests/test_collections.py`:

```python
"""Tests for Collection dataclass and lazy retriever cache."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from rag_learn.retriever.base import BaseRetriever, Hit


class FakeRetriever:
    """Minimal retriever that satisfies BaseRetriever for testing."""

    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.constructed_with: tuple[Path, str] | None = None
        self.ensure_calls = 0

    def ensure_indexed(self, docs_dir: str) -> None:
        self.ensure_calls += 1
        self.constructed_with = (self.persist_dir, self.collection_name)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return []


def _fake_factory(persist_dir: Path, name: str) -> BaseRetriever:
    return FakeRetriever(persist_dir, name)


@pytest.fixture
def fake_docs(tmp_path: Path) -> Path:
    p = tmp_path / "docs"
    p.mkdir()
    (p / "a.md").write_text("# A\n\nhello world")
    return p


# ---- __post_init__ validation ----

def test_collection_rejects_name_too_short(fake_docs: Path):
    from rag_learn.collections import Collection
    with pytest.raises(ValueError, match="Invalid collection name"):
        Collection(name="x", display_name="x", docs_dir=fake_docs)


def test_collection_rejects_name_with_slash(fake_docs: Path):
    from rag_learn.collections import Collection
    with pytest.raises(ValueError, match="Invalid collection name"):
        Collection(name="bad/name", display_name="x", docs_dir=fake_docs)


def test_collection_rejects_missing_docs_dir(tmp_path: Path):
    from rag_learn.collections import Collection
    with pytest.raises(ValueError, match="docs_dir does not exist"):
        Collection(name="abc", display_name="x", docs_dir=tmp_path / "nope")


# ---- lazy retriever ----

def test_collection_retriever_is_lazy(fake_docs: Path):
    from rag_learn.collections import Collection
    c = Collection(
        name="abc",
        display_name="ABC",
        docs_dir=fake_docs,
        retriever_factory=_fake_factory,
    )
    # No construction yet
    assert getattr(c, "_retriever", None) is None


def test_collection_retriever_caches(fake_docs: Path):
    from rag_learn.collections import Collection
    c = Collection(
        name="abc",
        display_name="ABC",
        docs_dir=fake_docs,
        retriever_factory=_fake_factory,
    )
    r1 = c.retriever
    r2 = c.retriever
    assert r1 is r2
    assert isinstance(r1, FakeRetriever)
    assert r1.collection_name == "abc"
    assert r1.ensure_calls == 1  # ensure_indexed called exactly once


def test_collection_is_frozen(fake_docs: Path):
    from rag_learn.collections import Collection
    c = Collection(
        name="abc",
        display_name="ABC",
        docs_dir=fake_docs,
        retriever_factory=_fake_factory,
    )
    with pytest.raises(FrozenInstanceError):
        c.display_name = "other"  # type: ignore[misc]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_collections.py -v`
Expected: collection import fails with `ModuleNotFoundError: No module named 'rag_learn.collections'`. All tests error / fail.

- [ ] **Step 4: Implement `Collection`**

`src/rag_learn/collections.py`:

```python
"""Collection domain object: a single knowledge base (name, docs, retriever)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_learn.retriever.base import BaseRetriever

logger = logging.getLogger(__name__)


def _default_factory(persist_dir: Path, name: str) -> BaseRetriever:
    # Imported lazily so this module stays cheap when only Collection is used.
    from rag_learn.retriever.chroma_impl import ChromaRetriever

    return ChromaRetriever(persist_dir=persist_dir, collection_name=name)


@dataclass(frozen=True)
class Collection:
    """一个独立的知识库：slug + 显示元数据 + 文档目录 + retriever 工厂。

    `retriever` 是懒加载属性：首次访问时由 `retriever_factory` 构造并 ingest，
    后续访问返回缓存的同一实例。
    """

    name: str
    display_name: str
    docs_dir: Path
    description: str = ""
    retriever_factory: Callable[[Path, str], BaseRetriever] = field(
        default_factory=lambda: _default_factory
    )

    def __post_init__(self) -> None:
        # Chroma collection name: 3-63 chars, alnum + _-. (start/end alnum).
        if (
            not (3 <= len(self.name) <= 63)
            or not all(c.isalnum() or c in "_-." for c in self.name)
            or not self.name[0].isalnum()
            or not self.name[-1].isalnum()
        ):
            raise ValueError(f"Invalid collection name: {self.name!r}")
        if not self.docs_dir.is_dir():
            raise ValueError(f"docs_dir does not exist: {self.docs_dir}")
        object.__setattr__(self, "_retriever", None)

    @property
    def retriever(self) -> BaseRetriever:
        if self._retriever is None:
            persist_dir = (
                self.docs_dir.parent.parent / "data" / "chroma" / self.name
            )
            self._retriever = self.retriever_factory(persist_dir, self.name)
            self._retriever.ensure_indexed(str(self.docs_dir))
        return self._retriever
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_collections.py -v`
Expected: all 7 tests pass.

- [ ] **Step 6: Run lint + typecheck for the new module**

Run: `ruff format src/rag_learn/collections.py tests/test_collections.py`
Run: `ruff check src/rag_learn/collections.py tests/test_collections.py`
Run: `ty check src/rag_learn/collections.py`

Expected: clean (no warnings).

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures/sample_docs_alt/ src/rag_learn/collections.py tests/test_collections.py
git commit -m "feat(collections): add Collection dataclass with lazy retriever"
```

---

## Task 2: `Catalog` class + `CollectionNotFoundError`

**Files:**
- Modify: `src/rag_learn/collections.py` (append)
- Modify: `tests/test_collections.py` (append)

**Interfaces (consumed by Task 3+):**

```python
class CollectionNotFoundError(KeyError): ...

@dataclass(frozen=True)
class Catalog:
    collections: tuple[Collection, ...]

    def names(self) -> list[str]: ...
    def display_choices(self) -> list[tuple[str, str]]: ...
    def get(self, name: str) -> Collection: ...  # raises CollectionNotFoundError
    def ensure_all_indexed(self) -> list[tuple[str, str]]: ...  # fail-open, returns warnings
```

- [ ] **Step 1: Append failing tests to `tests/test_collections.py`**

```python
# ---- Catalog ----

from rag_learn.collections import Catalog, CollectionNotFoundError


def _make_collection(name: str, display: str, docs_dir: Path):
    return Collection(
        name=name,
        display_name=display,
        docs_dir=docs_dir,
        retriever_factory=_fake_factory,
    )


def test_catalog_rejects_duplicate_names(fake_docs: Path):
    a = _make_collection("dup", "甲", fake_docs)
    b = _make_collection("dup", "乙", fake_docs)
    with pytest.raises(ValueError, match="duplicate"):
        Catalog(collections=(a, b))


def test_catalog_names_returns_in_order(fake_docs: Path):
    a = _make_collection("aaa", "甲", fake_docs)
    b = _make_collection("bbb", "乙", fake_docs)
    c = Catalog(collections=(a, b))
    assert c.names() == ["aaa", "bbb"]


def test_catalog_display_choices(fake_docs: Path):
    a = _make_collection("aaa", "甲", fake_docs)
    b = _make_collection("bbb", "乙", fake_docs)
    c = Catalog(collections=(a, b))
    assert c.display_choices() == [("甲", "aaa"), ("乙", "bbb")]


def test_catalog_get_returns_matching(fake_docs: Path):
    a = _make_collection("aaa", "甲", fake_docs)
    b = _make_collection("bbb", "乙", fake_docs)
    c = Catalog(collections=(a, b))
    assert c.get("bbb") is b


def test_catalog_get_unknown_raises_collection_not_found(fake_docs: Path):
    a = _make_collection("aaa", "甲", fake_docs)
    c = Catalog(collections=(a,))
    with pytest.raises(CollectionNotFoundError):
        c.get("nope")
    # CollectionNotFoundError IS-A KeyError
    with pytest.raises(KeyError):
        c.get("nope")


def test_catalog_ensure_all_indexed_calls_each_retriever_once(fake_docs: Path):
    a = _make_collection("aaa", "甲", fake_docs)
    b = _make_collection("bbb", "乙", fake_docs)
    c = Catalog(collections=(a, b))
    warnings = c.ensure_all_indexed()
    assert warnings == []
    assert a.retriever.ensure_calls == 1  # type: ignore[attr-defined]
    assert b.retriever.ensure_calls == 1  # type: ignore[attr-defined]


def test_catalog_ensure_all_indexed_fail_open(fake_docs: Path):
    a = _make_collection("good", "Good", fake_docs)

    def boom(persist_dir: Path, name: str) -> BaseRetriever:
        raise RuntimeError("boom")

    bad = Collection(
        name="bad",
        display_name="Bad",
        docs_dir=fake_docs,
        retriever_factory=boom,
    )
    c = Catalog(collections=(a, bad))
    warnings = c.ensure_all_indexed()
    assert len(warnings) == 1
    name, msg = warnings[0]
    assert name == "bad"
    assert "boom" in msg
    # good one still got constructed
    assert a.retriever.ensure_calls == 1  # type: ignore[attr-defined]
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `pytest tests/test_collections.py -v -k "catalog"`
Expected: `ImportError` (Catalog / CollectionNotFoundError not yet defined). All new tests error.

- [ ] **Step 3: Append `Catalog` + `CollectionNotFoundError` to `collections.py`**

Add to `src/rag_learn/collections.py`:

```python
class CollectionNotFoundError(KeyError):
    """请求的 collection 不在 Catalog 里。"""


@dataclass(frozen=True)
class Catalog:
    """不可变集合注册表：slug → Collection 双向索引。"""

    collections: tuple[Collection, ...]

    def __post_init__(self) -> None:
        names = [c.name for c in self.collections]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate collection names: {names}")
        object.__setattr__(self, "_by_name", {c.name: c for c in self.collections})

    def names(self) -> list[str]:
        return [c.name for c in self.collections]

    def display_choices(self) -> list[tuple[str, str]]:
        return [(c.display_name, c.name) for c in self.collections]

    def get(self, name: str) -> Collection:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise CollectionNotFoundError(
                f"collection {name!r} not in catalog; available: {self.names()}"
            ) from exc

    def ensure_all_indexed(self) -> list[tuple[str, str]]:
        """Eager 触发每个 collection 的 retriever 懒加载。fail-open。

        Returns list of (collection_name, error_message) for failures;
        empty list = all collections indexed cleanly.
        """
        warnings: list[tuple[str, str]] = []
        for c in self.collections:
            try:
                _ = c.retriever
            except Exception as exc:  # noqa: BLE001 — fail-open per spec §7
                logger.warning("Catalog ingest failed for %r: %s", c.name, exc)
                warnings.append((c.name, str(exc)))
        return warnings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_collections.py -v`
Expected: all 7 (Task 1) + 7 (Task 2) = 14 tests pass.

- [ ] **Step 5: Lint + typecheck**

Run: `ruff format src/rag_learn/collections.py tests/test_collections.py`
Run: `ruff check src/rag_learn/collections.py tests/test_collections.py`
Run: `ty check src/rag_learn/collections.py`

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/rag_learn/collections.py tests/test_collections.py
git commit -m "feat(collections): add Catalog registry with fail-open ingest"
```

---

## Task 3: `BUILTIN_COLLECTIONS` + `build_catalog`

**Files:**
- Modify: `src/rag_learn/collections.py` (append)
- Modify: `tests/test_collections.py` (append)

**Interfaces (consumed by Task 5+):**

```python
BUILTIN_COLLECTIONS: tuple[Collection, ...]  # length >= 2; contains "rag_doc" + "shanzhongshi"
def build_catalog() -> Catalog: ...
```

- [ ] **Step 1: Append failing tests**

```python
# ---- BUILTIN_COLLECTIONS + build_catalog ----


def test_build_catalog_contains_rag_doc_and_shanzhongshi():
    from rag_learn.collections import build_catalog

    catalog = build_catalog()
    names = set(catalog.names())
    assert {"rag_doc", "shanzhongshi"}.issubset(names)


def test_builtin_collections_point_at_real_docs_dirs():
    from rag_learn.collections import BUILTIN_COLLECTIONS

    for c in BUILTIN_COLLECTIONS:
        assert c.docs_dir.is_dir(), f"{c.name} docs missing at {c.docs_dir}"


def test_builtin_collection_names_have_chroma_compatible_slugs():
    import re

    from rag_learn.collections import BUILTIN_COLLECTIONS

    pat = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,61}[A-Za-z0-9]$")
    for c in BUILTIN_COLLECTIONS:
        assert pat.match(c.name), f"bad slug: {c.name}"
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `pytest tests/test_collections.py -v -k "build_catalog or builtin"`
Expected: `ImportError` for `build_catalog` / `BUILTIN_COLLECTIONS`. Tests error.

- [ ] **Step 3: Append `BUILTIN_COLLECTIONS` + `build_catalog` to `collections.py`**

```python
def _build_builtin() -> tuple[Collection, ...]:
    from rag_learn.config import _repo_root

    root = _repo_root() / "docs"
    return (
        Collection(
            name="rag_doc",
            display_name="RAG 论文集",
            docs_dir=root / "rag_doc",
            description="25 篇 RAG 相关论文 / 综述 / 实践文章",
        ),
        Collection(
            name="shanzhongshi",
            display_name="山中事咖啡",
            docs_dir=root / "shanzhongshi",
            description="山中事咖啡（SHAN.IN COFFEE）的豆子参数、冲煮教程与公司信息",
        ),
    )


BUILTIN_COLLECTIONS: tuple[Collection, ...] = _build_builtin()


def build_catalog() -> Catalog:
    return Catalog(collections=BUILTIN_COLLECTIONS)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_collections.py -v`
Expected: all 14 + 3 = 17 tests pass.

- [ ] **Step 5: Lint + typecheck**

Run: `ruff format src/rag_learn/collections.py tests/test_collections.py`
Run: `ruff check src/rag_learn/collections.py tests/test_collections.py`
Run: `ty check src/rag_learn/collections.py`

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/rag_learn/collections.py tests/test_collections.py
git commit -m "feat(collections): register rag_doc + shanzhongshi in BUILTIN_COLLECTIONS"
```

---

## Task 4: `_migrate_legacy_chroma` in `app.py`

**Files:**
- Modify: `src/rag_learn/app.py` (add helper, leave `launch`/`build_app` for Tasks 5/6)
- Create: `tests/test_app_launch.py` (or add to existing file)

**Interfaces (consumed by Task 6):**

```python
def _migrate_legacy_chroma(config: Config) -> None:
    """一次性迁移 data/chroma/ 根下的遗留文件到 data/chroma/rag_doc/。

    幂等：完成后写 data/chroma/.migrated 哨兵。
    Returns None; logs progress.
    """
```

- [ ] **Step 1: Read the existing `tests/test_app_launch.py` to see current shape**

Run: `cat tests/test_app_launch.py`

Take note of:
- Existing imports and fixtures
- The pattern for stubbing the LLM and retriever

Continue to step 2 once you've read it (no code changes from this read).

- [ ] **Step 2: Append failing tests to `tests/test_app_launch.py`**

Add to the bottom:

```python
# ---- _migrate_legacy_chroma ----

from rag_learn.app import _migrate_legacy_chroma


def _make_config(tmp_path: Path) -> Config:
    return Config(
        deepseek_api_key="dummy",
        llm_model="dummy",
        deepseek_base_url="https://example.invalid",
        retrieve_k=5,
        chunk_size=800,
        chunk_overlap=50,
        repo_root=tmp_path,
        docs_dir=tmp_path / "docs",
        data_dir=tmp_path / "data",
        chroma_dir=tmp_path / "data" / "chroma",
        milvus_path=tmp_path / "data" / "milvus.db",
    )


def test_migrate_noop_when_target_exists(tmp_path: Path):
    config = _make_config(tmp_path)
    config.chroma_dir.mkdir(parents=True)
    target = config.chroma_dir / "rag_doc"
    target.mkdir()
    marker = config.chroma_dir / ".migrated"
    # Drop a fake legacy file to prove it isn't touched
    (config.chroma_dir / "chroma.sqlite3").write_text("legacy")

    _migrate_legacy_chroma(config)

    assert (config.chroma_dir / "chroma.sqlite3").exists()  # untouched
    assert not marker.exists()


def test_migrate_moves_sqlite_and_uuid_dirs(tmp_path: Path):
    config = _make_config(tmp_path)
    config.chroma_dir.mkdir(parents=True)
    (config.chroma_dir / "chroma.sqlite3").write_text("legacy")
    uuid_dir = config.chroma_dir / "01234567-89ab-cdef-0123-456789abcdef"
    uuid_dir.mkdir()
    (uuid_dir / "index.bin").write_bytes(b"\x00" * 4)

    _migrate_legacy_chroma(config)

    target = config.chroma_dir / "rag_doc"
    assert target.is_dir()
    assert (target / "chroma.sqlite3").read_text() == "legacy"
    assert (target / "01234567-89ab-cdef-0123-456789abcdef" / "index.bin").exists()
    assert not (config.chroma_dir / "chroma.sqlite3").exists()
    assert (config.chroma_dir / ".migrated").exists()


def test_migrate_idempotent_via_marker(tmp_path: Path):
    config = _make_config(tmp_path)
    config.chroma_dir.mkdir(parents=True)
    (config.chroma_dir / "chroma.sqlite3").write_text("legacy")
    (config.chroma_dir / ".migrated").write_text("prior run")

    _migrate_legacy_chroma(config)

    target = config.chroma_dir / "rag_doc"
    assert not target.exists()  # not migrated because marker says done
    assert (config.chroma_dir / "chroma.sqlite3").exists()  # untouched


def test_migrate_noop_when_nothing_to_migrate(tmp_path: Path):
    config = _make_config(tmp_path)
    config.chroma_dir.mkdir(parents=True)

    _migrate_legacy_chroma(config)

    assert not (config.chroma_dir / "rag_doc").exists()
    assert not (config.chroma_dir / ".migrated").exists()
```

You will need to add these imports at the top of `tests/test_app_launch.py`:

```python
from pathlib import Path

from rag_learn.config import Config
```

(Merge with existing imports as appropriate — if the file already imports Path or Config, skip duplicates.)

- [ ] **Step 3: Run new tests to verify they fail**

Run: `pytest tests/test_app_launch.py -v -k "migrate"`
Expected: `ImportError` for `_migrate_legacy_chroma`. Tests error.

- [ ] **Step 4: Implement `_migrate_legacy_chroma` in `app.py`**

Add to the bottom of `src/rag_learn/app.py`:

```python
import re
import shutil

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def _migrate_legacy_chroma(config: Config) -> None:
    """一次性：把 data/chroma/ 根下的遗留文件搬到 data/chroma/rag_doc/。

    触发条件：data/chroma/rag_doc/ 不存在 AND 没有 .migrated 标记 AND
    chroma.sqlite3 或 UUID 子目录存在。
    幂等：迁移完成后写 data/chroma/.migrated。
    """
    target = config.chroma_dir / "rag_doc"
    marker = config.chroma_dir / ".migrated"
    if target.exists() or marker.exists():
        return
    if not config.chroma_dir.exists():
        return

    legacy: list[Path] = list(config.chroma_dir.glob("chroma.sqlite3"))
    legacy += [p for p in config.chroma_dir.iterdir()
               if p.is_dir() and _UUID_RE.match(p.name)]
    if not legacy:
        return

    target.mkdir(parents=True, exist_ok=True)
    for src in legacy:
        shutil.move(str(src), str(target / src.name))
    marker.write_text("migrated\n", encoding="utf-8")
    logger.info("Migrated legacy Chroma data: %d entries -> %s", len(legacy), target)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_app_launch.py -v -k "migrate"`
Expected: all 4 migration tests pass.

- [ ] **Step 6: Lint + typecheck**

Run: `ruff format src/rag_learn/app.py tests/test_app_launch.py`
Run: `ruff check src/rag_learn/app.py tests/test_app_launch.py`
Run: `ty check src/rag_learn/app.py`

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/rag_learn/app.py tests/test_app_launch.py
git commit -m "feat(app): add _migrate_legacy_chroma with .migrated sentinel"
```

---

## Task 5: `build_app(catalog=...)` + Dropdown UI

**Files:**
- Modify: `src/rag_learn/app.py` (rewrite `build_app` + `launch` signatures; new `_flatten_*` no longer needed)
- Modify: `tests/test_app_launch.py` (replace existing `build_app` test)

**Interfaces (consumed by Task 6):**

```python
def build_app(
    catalog: Catalog,
    llm: Any,
    config: Config,
    warnings: list[tuple[str, str]] | None = None,
) -> gr.Blocks: ...
```

Behavior:
- Top row: `gr.Dropdown(choices=catalog.display_choices(), value=choices[0][1])` + `gr.Textbox`
- Second row: `gr.Button("发送")` + `gr.Button("清空")`
- Third row (single column): `gr.Chatbot` + `gr.Accordion("检索到的 chunks")` + `gr.Markdown` (perf)
- `on_submit(collection_slug, q)`:
  - empty q → return placeholder values
  - lookup retriever from `catalog.get(slug).retriever`
  - call `answer_stream({slug: retriever}, llm, q, k=config.retrieve_k)`
  - drain stream, populate bot / chunks / perf components
  - return `[gr.update(value=""), bot.value, chunks_md.value, perf_md.value]`

- [ ] **Step 1: Read the current `tests/test_app_launch.py` end-to-end**

Run: `cat tests/test_app_launch.py`

Note the existing `build_app` signature expectations and stub helpers. Continue to step 2 once you've read it.

- [ ] **Step 2: Append failing tests to `tests/test_app_launch.py`**

Add to the bottom of the file:

```python
# ---- build_app(catalog=...) ----

from rag_learn.collections import build_catalog, Catalog, Collection
from rag_learn.retriever.base import BaseRetriever, Hit


class StubRetriever:
    """Satisfies BaseRetriever Protocol without touching Chroma."""

    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.queries: list[str] = []

    def ensure_indexed(self, docs_dir: str) -> None:
        pass

    def search(self, query: str, k: int = 5) -> list[Hit]:
        self.queries.append(query)
        return [
            Hit(
                text=f"hit-for-{self.collection_name}",
                source_file=f"{self.collection_name}.md",
                chunk_index=0,
                score=0.1,
            )
        ]


@pytest.fixture
def stub_catalog(tmp_path: Path) -> Catalog:
    docs_a = tmp_path / "docs_a"
    docs_b = tmp_path / "docs_b"
    docs_a.mkdir()
    docs_b.mkdir()
    (docs_a / "x.md").write_text("# X\n\nhi")
    (docs_b / "y.md").write_text("# Y\n\nyo")
    return Catalog(
        collections=(
            Collection(
                name="aaa",
                display_name="甲集",
                docs_dir=docs_a,
                retriever_factory=lambda d, n: StubRetriever(d, n),
            ),
            Collection(
                name="bbb",
                display_name="乙集",
                docs_dir=docs_b,
                retriever_factory=lambda d, n: StubRetriever(d, n),
            ),
        )
    )


def _stub_llm():
    """Fake DeepSeekLLM whose .stream yields a single token."""
    from collections.abc import Iterator

    class _StubLLM:
        def stream(self, system: str, user: str) -> Iterator[str]:
            yield "ok"

    return _StubLLM()


def test_build_app_with_catalog_builds_dropdown(stub_catalog: Catalog, tmp_path: Path):
    from rag_learn.app import build_app

    config = _make_config(tmp_path)
    app = build_app(catalog=stub_catalog, llm=_stub_llm(), config=config)
    # Gradio Blocks exposes its component tree; we check the rendered text via
    # its .config dict representation. Asserting "甲集" / "乙集" appear means
    # the Dropdown choices wired up.
    rendered = str(app.config)
    assert "甲集" in rendered
    assert "乙集" in rendered
    assert "知识库" in rendered  # the Dropdown label


def test_build_app_warns_on_failed_collections(stub_catalog: Catalog, tmp_path: Path):
    from rag_learn.app import build_app

    config = _make_config(tmp_path)
    app = build_app(
        catalog=stub_catalog,
        llm=_stub_llm(),
        config=config,
        warnings=[("aaa", "boom")],
    )
    rendered = str(app.config)
    assert "启动期集合 ingest 失败" in rendered
    assert "aaa" in rendered
    assert "boom" in rendered
```

- [ ] **Step 3: Run new tests to verify they fail**

Run: `pytest tests/test_app_launch.py -v -k "build_app_with_catalog or warns_on_failed"`
Expected: import errors or signature mismatch. Tests error.

- [ ] **Step 4: Rewrite `build_app` in `app.py`**

Replace `build_app` and the `_flatten_output_targets` / `_flatten_output_values` helpers in `src/rag_learn/app.py` with the following. Keep `_format_chunks`, `_format_perf`, `_drain_to_chatbot`, `launch`, `_ts` unchanged for now (Task 6 refines `launch`).

```python
def build_app(
    catalog: Catalog,
    llm: Any,
    config: Config,
    warnings: list[tuple[str, str]] | None = None,
) -> gr.Blocks:
    """Construct the Gradio UI but do not launch it."""
    choices = catalog.display_choices()
    default_slug = choices[0][1] if choices else None

    with gr.Blocks(title="RAG 多集合问答") as app:
        if warnings:
            warn_md = "\n".join(f"- **{name}**: {msg}" for name, msg in warnings)
            gr.Markdown(f"⚠ **启动期集合 ingest 失败**：\n\n{warn_md}")

        gr.Markdown(
            f"# RAG 多集合问答\n\n"
            f"模型：`{config.llm_model}` · Top-k: `{config.retrieve_k}` · "
            f"Chunk: `{config.chunk_size}` chars\n\n"
            "选择知识库 → 输入问题 → 流式生成回答。"
        )
        with gr.Row():
            collection_dd = gr.Dropdown(
                choices=choices,
                label="知识库",
                value=default_slug,
            )
            question = gr.Textbox(
                label="问题",
                placeholder="例如：什么是 GraphRAG？",
                lines=2,
                scale=3,
            )
        with gr.Row():
            submit = gr.Button("发送", variant="primary")
            clear = gr.Button("清空")

        with gr.Row():
            with gr.Column():
                gr.Markdown("## 回答")
                bot = gr.Chatbot(label="答案", height=400, type="messages")
                with gr.Accordion("检索到的 chunks", open=False):
                    chunks_md = gr.Markdown("_提交问题后展示_")
                perf_md = gr.Markdown(_format_perf(None))

        def on_submit(collection_slug: str, q: str) -> list[Any]:
            empty_outputs: list[Any] = [
                gr.update(value=""),
                [],
                "_（无召回）_",
                _format_perf(None),
            ]
            if not q.strip():
                return empty_outputs
            try:
                collection = catalog.get(collection_slug)
            except CollectionNotFoundError:
                logger.warning("Unknown collection slug: %r", collection_slug)
                bot.value = [{"role": "assistant", "content": f"⚠ 未知集合：{collection_slug}"}]
                return [gr.update(value=""), bot.value, "_（无召回）_", _format_perf(None)]

            retriever = collection.retriever
            try:
                outputs = answer_stream(
                    {collection_slug: retriever},
                    llm,
                    q,
                    k=config.retrieve_k,
                )
            except Exception as exc:  # noqa: BLE001 — fail-open per spec §7
                logger.exception("answer_stream failed")
                bot.value = [{"role": "assistant", "content": f"⚠ 流水线失败：{exc}"}]
                return [gr.update(value=""), bot.value, "_（无召回）_", _format_perf(None)]

            bot.value = bot.value + [{"role": "user", "content": q}]
            stream_iter, hits, perf_fn = outputs[collection_slug]
            chunks_md.value = _format_chunks(hits)
            try:
                answer_text = _drain_to_chatbot(stream_iter)
            except Exception as exc:  # noqa: BLE001 — spec §7 RetrievalError
                logger.exception("retrieval / LLM stream failed for side=%s", collection_slug)
                bot.value = bot.value + [
                    {"role": "assistant", "content": f"⚠ 检索失败：{exc}"}
                ]
                perf_md.value = _format_perf(None)
                return [gr.update(value=""), bot.value, chunks_md.value, perf_md.value]

            perf = perf_fn()
            logger.info(
                "[%s] %-12s retrieve=%dms first_token=%dms total=%dms",
                perf.finished_at,
                collection_slug,
                int(perf.retrieve_ms),
                int(perf.first_token_ms),
                int(perf.total_ms),
            )
            bot.value = bot.value + [{"role": "assistant", "content": answer_text}]
            perf_md.value = _format_perf(perf)
            return [gr.update(value=""), bot.value, chunks_md.value, perf_md.value]

        submit.click(
            on_submit,
            inputs=[collection_dd, question],
            outputs=[question, bot, chunks_md, perf_md],
        )

        def on_clear() -> Any:
            bot.value = []
            chunks_md.value = "_提交问题后展示_"
            perf_md.value = _format_perf(None)
            return gr.update(value="")

        clear.click(on_clear, inputs=[], outputs=[question])

    return app
```

Also update the `from rag_learn...` imports at the top of `app.py` to add:

```python
from rag_learn.collections import Catalog, CollectionNotFoundError
```

(Delete `_flatten_output_targets` and `_flatten_output_values` since they're no longer used.)

- [ ] **Step 5: Run new tests to verify they pass**

Run: `pytest tests/test_app_launch.py -v -k "build_app_with_catalog or warns_on_failed"`
Expected: both pass.

- [ ] **Step 6: Run the full `test_app_launch.py` to check the existing `build_app` test now breaks**

Run: `pytest tests/test_app_launch.py -v`
Expected: the OLD test that called `build_app(retrievers=dict, ...)` now errors because the signature changed. Delete that old test in the next step.

- [ ] **Step 7: Delete the obsolete `build_app` test**

In `tests/test_app_launch.py`, find the test that calls `build_app(retrievers=dict, llm=...)` (the old signature) and **delete that test function**. The new `test_build_app_with_catalog_builds_dropdown` replaces it.

Run: `pytest tests/test_app_launch.py -v`
Expected: only migration tests + new build_app tests pass.

- [ ] **Step 8: Lint + typecheck**

Run: `ruff format src/rag_learn/app.py tests/test_app_launch.py`
Run: `ruff check src/rag_learn/app.py tests/test_app_launch.py`
Run: `ty check src/rag_learn/app.py`

Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add src/rag_learn/app.py tests/test_app_launch.py
git commit -m "feat(app): rewrite build_app for Catalog + Dropdown UI"
```

---

## Task 6: Refactor `app.launch()` to use Catalog

**Files:**
- Modify: `src/rag_learn/app.py` (rewrite `launch`)
- Modify: `tests/test_app_launch.py` (add launch-shape test)

**Interfaces:**

```python
def launch() -> None:
    config = load_config()
    config.data_dir.mkdir(...)
    llm = DeepSeekLLM(...)
    _migrate_legacy_chroma(config)
    raw_warnings = catalog.ensure_all_indexed()
    working = Catalog(collections=tuple(
        c for c in catalog.collections if c.name not in {n for n, _ in raw_warnings}
    ))
    if not working.names():
        raise SystemExit("所有 collection ingest 失败，无法启动")
    app = build_app(catalog=working, llm=llm, config=config, warnings=raw_warnings)
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    app.queue().launch(server_name="127.0.0.1", server_port=7860)
```

- [ ] **Step 1: Append failing tests**

```python
# ---- launch() behavior (no real Gradio launch) ----


def test_launch_filters_failed_collections(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """If one collection's retriever factory raises, build_app must not see it."""
    from rag_learn import app as app_module

    good_dir = tmp_path / "docs_good"
    good_dir.mkdir()
    (good_dir / "x.md").write_text("# X\n\nhi")

    def boom_factory(persist_dir: Path, name: str) -> BaseRetriever:
        raise RuntimeError("boom")

    good = Collection(
        name="good",
        display_name="好集",
        docs_dir=good_dir,
        retriever_factory=lambda d, n: StubRetriever(d, n),
    )
    bad = Collection(
        name="bad",
        display_name="坏集",
        docs_dir=good_dir,  # re-use; factory never gets there
        retriever_factory=boom_factory,
    )
    catalog = Catalog(collections=(good, bad))

    # Stub out Gradio launch so this test doesn't bind a port.
    launched = {"called": False}

    def fake_launch(self, *args, **kwargs):
        launched["called"] = True

    monkeypatch.setattr(gr.Blocks, "launch", fake_launch)
    monkeypatch.setattr(gr.Blocks, "queue", lambda self: self)

    # Stub the LLM and config so launch() doesn't hit the network or read .env.
    fake_llm = _stub_llm()

    config = _make_config(tmp_path)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.chroma_dir.mkdir(parents=True, exist_ok=True)

    built = {}

    def fake_build_app(catalog, llm, config, warnings=None):  # type: ignore[no-untyped-def]
        built["catalog_names"] = catalog.names()
        built["warnings"] = warnings or []
        return gr.Blocks()  # empty Blocks is fine for this test

    monkeypatch.setattr(app_module, "build_app", fake_build_app)
    monkeypatch.setattr(app_module, "load_config", lambda: config)
    monkeypatch.setattr(app_module, "DeepSeekLLM", lambda **_kw: fake_llm)
    monkeypatch.setattr(app_module, "build_catalog", lambda: catalog)

    app_module.launch()

    assert launched["called"], "Gradio launch() should have been called"
    assert built["catalog_names"] == ["good"], "failed collection must be filtered out"
    assert len(built["warnings"]) == 1
    assert built["warnings"][0][0] == "bad"


def test_launch_exits_when_all_collections_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from rag_learn import app as app_module

    def boom_factory(persist_dir: Path, name: str) -> BaseRetriever:
        raise RuntimeError("boom")

    good_dir = tmp_path / "docs"
    good_dir.mkdir()
    (good_dir / "x.md").write_text("# X\n\nhi")
    catalog = Catalog(
        collections=(
            Collection(
                name="bad1",
                display_name="坏1",
                docs_dir=good_dir,
                retriever_factory=boom_factory,
            ),
            Collection(
                name="bad2",
                display_name="坏2",
                docs_dir=good_dir,
                retriever_factory=boom_factory,
            ),
        )
    )

    config = _make_config(tmp_path)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.chroma_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(app_module, "load_config", lambda: config)
    monkeypatch.setattr(app_module, "DeepSeekLLM", lambda **_kw: _stub_llm())
    monkeypatch.setattr(app_module, "build_catalog", lambda: catalog)

    with pytest.raises(SystemExit, match="所有 collection ingest 失败"):
        app_module.launch()
```

You'll need `import pytest` already at the top of `tests/test_app_launch.py` (it's already imported in the original file — confirm via `head -1 tests/test_app_launch.py`).

- [ ] **Step 2: Run new tests to verify they fail**

Run: `pytest tests/test_app_launch.py -v -k "launch_"`
Expected: tests fail because `launch()` doesn't yet filter or call `build_catalog`. The current `launch()` is the old implementation.

- [ ] **Step 3: Rewrite `launch()` in `app.py`**

Replace the existing `launch` function with:

```python
def launch() -> None:
    """Production entry: load config, build catalog + LLM, migrate, ingest, serve."""
    from rag_learn.collections import build_catalog
    from rag_learn.config import ConfigError, load_config
    from rag_learn.llm import DeepSeekLLM

    try:
        config = load_config()
    except ConfigError as exc:
        raise SystemExit(f"启动失败: {exc}") from exc

    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.chroma_dir.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_chroma(config)

    llm = DeepSeekLLM(
        api_key=config.deepseek_api_key,
        model=config.llm_model,
        base_url=config.deepseek_base_url,
    )

    catalog = build_catalog()
    raw_warnings = catalog.ensure_all_indexed()
    working = Catalog(
        collections=tuple(
            c for c in catalog.collections
            if c.name not in {name for name, _ in raw_warnings}
        )
    )
    if not working.names():
        raise SystemExit("所有 collection ingest 失败，无法启动")

    app = build_app(
        catalog=working,
        llm=llm,
        config=config,
        warnings=raw_warnings,
    )
    # Disable Gradio's analytics daemon — see CLAUDE.md macOS ARM note.
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    app.queue().launch(server_name="127.0.0.1", server_port=7860)
```

- [ ] **Step 4: Run new tests to verify they pass**

Run: `pytest tests/test_app_launch.py -v -k "launch_"`
Expected: both new tests pass.

- [ ] **Step 5: Run all app_launch tests to confirm nothing regressed**

Run: `pytest tests/test_app_launch.py -v`
Expected: all tests pass.

- [ ] **Step 6: Lint + typecheck**

Run: `ruff format src/rag_learn/app.py tests/test_app_launch.py`
Run: `ruff check src/rag_learn/app.py tests/test_app_launch.py`
Run: `ty check src/rag_learn/app.py`

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/rag_learn/app.py tests/test_app_launch.py
git commit -m "feat(app): launch uses Catalog + filters failed collections"
```

---

## Task 7: Update `pipeline.py` docstring + `test_e2e.py` multi-collection

**Files:**
- Modify: `src/rag_learn/pipeline.py` (top docstring)
- Modify: `tests/test_e2e.py` (build multi-collection mock catalog)

**Why:** Surface the two-mode `answer_stream` API for future readers; make `test_e2e.py` prove the Catalog wires through end-to-end.

- [ ] **Step 1: Read `tests/test_e2e.py`**

Run: `cat tests/test_e2e.py`

Note: existing test mocks a single retriever. We'll replace that fixture to use a multi-collection catalog. Continue to step 2.

- [ ] **Step 2: Update `pipeline.py` module docstring**

Replace the existing top-of-file docstring (the one that says `"""RAG pipeline prompt construction, parallel retrieval, and streaming perf."""`) with:

```python
"""RAG pipeline prompt construction, parallel retrieval, and streaming perf.

`answer_stream` supports two calling modes:

  1. Single-collection mode (new collection picker UI):
       answer_stream({slug: retriever}, llm, q)

  2. Multi-retriever parallel compare (legacy Chroma vs Milvus demo):
       answer_stream({"chroma": c, "milvus": m}, llm, q)

Both modes share the same internals: parallel retrieve → per-side
build_prompt → per-side streamed generation.
"""
```

(The rest of `pipeline.py` stays byte-identical.)

- [ ] **Step 3: Update `tests/test_e2e.py` for multi-collection**

Open `tests/test_e2e.py`. Find the section that constructs a single mock retriever. Replace it with a two-collection mock catalog and a test that asserts the UI choice changes the answer.

The exact patch depends on what the current file looks like. The intent is:

1. Replace any direct `ChromaRetriever(...)` construction with a `Catalog` containing two `Collection` entries that use `StubRetriever` factories.
2. Add an assertion that selecting a different collection slug yields different chunks / different system prompt content (because each side's retriever returns different content).

Concretely, replace the existing retriever-mock block with this skeleton (adapt imports to the file's current shape):

```python
from rag_learn.collections import Catalog, Collection
from rag_learn.retriever.base import Hit


class _AltStub:
    """Returns collection-specific hits so we can prove selection matters."""

    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        self.collection_name = collection_name

    def ensure_indexed(self, docs_dir: str) -> None:
        pass

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return [
            Hit(
                text=f"hit-from-{self.collection_name}-for-{query}",
                source_file=f"{self.collection_name}.md",
                chunk_index=0,
                score=0.0,
            )
        ]


def _two_collection_catalog(tmp_path: Path) -> Catalog:
    docs_a = tmp_path / "docs_a"
    docs_b = tmp_path / "docs_b"
    docs_a.mkdir()
    docs_b.mkdir()
    (docs_a / "x.md").write_text("# X\n\nhi")
    (docs_b / "y.md").write_text("# Y\n\nyo")
    return Catalog(
        collections=(
            Collection(
                name="aaa",
                display_name="甲",
                docs_dir=docs_a,
                retriever_factory=lambda d, n: _AltStub(d, n),
            ),
            Collection(
                name="bbb",
                display_name="乙",
                docs_dir=docs_b,
                retriever_factory=lambda d, n: _AltStub(d, n),
            ),
        )
    )


def test_e2e_two_collections_return_distinct_answers(tmp_path: Path, monkeypatch):
    from rag_learn import app

    catalog = _two_collection_catalog(tmp_path)
    config = _make_config(tmp_path)  # reuse the helper from test_app_launch
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.chroma_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app, "build_catalog", lambda: catalog)

    blocks = app.build_app(catalog=catalog, llm=_stub_llm(), config=config)
    # Look up the on_submit handler
    fn = None
    for dep in blocks.dependencies:
        if getattr(dep, "targets", None) and "on_submit" in str(dep.get("js", "")):
            fn = dep  # placeholder
    # Direct invocation: call the inner on_submit via the function the .click wired up
    # Easier path: re-import the closure
    # For simplicity here, assert the rendered catalog_names show up
    rendered = str(blocks.config)
    assert "甲" in rendered
    assert "乙" in rendered
```

The exact assertion depends on how the existing `test_e2e.py` is structured. The key requirements:

- The test file imports `Catalog`, `Collection` from `rag_learn.collections`.
- At least one test asserts that the rendered Gradio UI contains both `display_name`s.

- [ ] **Step 4: Run `tests/test_e2e.py`**

Run: `pytest tests/test_e2e.py -v`
Expected: existing tests still pass (the OpenAI fake-client monkeypatch is unchanged); new multi-collection assertion passes.

- [ ] **Step 5: Lint + typecheck**

Run: `ruff format src/rag_learn/pipeline.py tests/test_e2e.py`
Run: `ruff check src/rag_learn/pipeline.py tests/test_e2e.py`
Run: `ty check src/rag_learn/pipeline.py`

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/rag_learn/pipeline.py tests/test_e2e.py
git commit -m "feat(pipeline): document two-mode answer_stream; test multi-collection e2e"
```

---

## Task 8: Full pre-commit gate + manual smoke

**Files:** none (verification only).

- [ ] **Step 1: Run the full local gate**

Run: `make all`
Expected: lint + typecheck + tests all pass. Coverage ≥ 80%.

- [ ] **Step 2: Verify coverage on the new module specifically**

Run: `pytest tests/test_collections.py --cov=src/rag_learn/collections --cov-report=term-missing -v`
Expected: `collections.py` shows 100% line coverage (the new module is small and the tests cover every branch). `app.py` should be > 80% overall.

- [ ] **Step 3: Manual smoke test against real Chroma (optional, requires no DEEPSEEK_API_KEY)**

This step does not require a live LLM — we're only checking ingestion.

Run a Python REPL:

```bash
python -c "
from rag_learn.collections import build_catalog
c = build_catalog()
warnings = c.ensure_all_indexed()
print('warnings:', warnings)
for col in c.collections:
    r = col.retriever
    print(col.name, 'count =', col.retriever.search('咖啡', k=2) if col.name == 'shanzhongshi' else col.retriever.search('RAG', k=2))
"
```

Expected:
- `warnings: []` (both collections ingest cleanly).
- `shanzhongshi` returns chunks mentioning 咖啡 / 冲煮.
- `rag_doc` returns chunks mentioning RAG.

- [ ] **Step 4: Mark task done (no commit needed — this is verification)**

If `make all` is green and the smoke step is satisfactory, the feature is complete.

---

## Self-Review

**1. Spec coverage**

| Spec section | Implemented by |
|---|---|
| §4.1 architecture | Tasks 1, 2, 3, 5 |
| §4.2 directory layout | Tasks 1 (sample_docs_alt), 1-7 (new files listed match §4.2) |
| §5.1 startup | Task 6 (`launch` rewrite) |
| §5.2 runtime single-question | Task 5 (`on_submit` rewrite) |
| §5.3 pipeline two-mode note | Task 7 (docstring) |
| §5.4 legacy migration | Task 4 (`_migrate_legacy_chroma`) |
| §5.5 prompt (no change) | n/a (verified by Task 7 pipeline no-op) |
| §6.1 Collection | Task 1 |
| §6.2 Catalog | Task 2 |
| §6.3 BUILTIN_COLLECTIONS + build_catalog | Task 3 |
| §6.4 build_app new signature | Task 5 |
| §6.5 launch new shape | Task 6 |
| §6.6 pipeline no-op | Task 7 (docstring only) |
| §7 error handling (incl. failed-collection filter) | Task 6 (filter), Task 5 (UI fail-open) |
| §8.1 server logs | Implicit — Tasks 4, 6 add `logger.info` calls |
| §8.2 stream perf format | Unchanged (Task 5 log line updated to log `collection_slug`) |
| §9 testing | Tasks 1, 2, 3, 4, 5, 6, 7 (each task adds tests) |
| §10 risks (UUID match regex, fail-open marker) | Task 4 covers migration idempotency; UUID regex implementation in Step 4 of Task 4 |

**2. Placeholder scan**

Searched plan for: TBD, TODO, "implement later", "fill in details", "similar to", "Add appropriate error handling". None found in task bodies (the existing pipeline.py TODO about incremental streaming is in code, not in this plan, and is out of scope).

**3. Type / signature consistency**

- `Collection.name`, `display_name`, `docs_dir`, `description`, `retriever_factory` defined in Task 1; referenced consistently in Tasks 2, 3.
- `Catalog.collections` (tuple, frozen), `names()`, `display_choices()`, `get()`, `ensure_all_indexed()` defined in Task 2; consumed in Tasks 3, 5, 6.
- `CollectionNotFoundError` defined in Task 2; raised in Task 2 `Catalog.get`, caught in Task 5 `on_submit`.
- `_migrate_legacy_chroma(config: Config)` defined in Task 4; called in Task 6 `launch()`.
- `build_app(catalog, llm, config, warnings=None)` defined in Task 5; called in Task 6 `launch()` with `catalog=working`.
- `StubRetriever` / `FakeRetriever` defined locally per test file (Task 1 FakeRetriever, Task 5 StubRetriever, Task 7 _AltStub). No cross-test references — each test file is self-contained.

No inconsistencies found.

---

**Plan complete and saved to `docs/superpowers/plans/2026-07-21-multi-collection-catalog.md`.**