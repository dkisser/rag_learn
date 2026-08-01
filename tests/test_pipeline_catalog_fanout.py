"""Tests for catalog fan-out scoping (A) and the split k parameters (B).

A — the decomposer prompt must describe ONLY the collection(s) actually
being searched. ``app.on_submit`` passes a single selected collection, so
feeding the whole catalog summary made the LLM invent sub-queries aimed at
collections that would never be searched (e.g. coffee sub-queries while
the RAG-paper collection is selected), wasting the fan-out on noise.

B — ``catalog_recall_k`` used to be both "how many hits each sub-query
pulls from each retriever" and "how many hits survive the merge". Those
are different knobs: the first controls candidate breadth, the second
controls prompt size.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

from rag_learn.collections import Catalog, Collection
from rag_learn.config import Config
from rag_learn.pipeline import _build_catalog_summary, answer_stream
from rag_learn.retriever import Hit


class _RecordingRetriever:
    """Returns ``n_hits`` unique hits per query and records the k it got."""

    def __init__(self, n_hits: int = 50) -> None:
        self._n_hits = n_hits
        self.calls: list[tuple[str, int]] = []

    def ensure_indexed(self, docs_dir: str) -> None:
        return None

    def search(self, query: str, k: int = 5) -> list[Hit]:
        self.calls.append((query, k))
        return [
            Hit(text=f"{query}-{i}", source_file=f"{query}.md", chunk_index=i, score=0.0)
            for i in range(min(k, self._n_hits))
        ]


class _ScriptedRoutingLLM:
    def __init__(self, intent_reply: str, decompose_reply: str) -> None:
        self.intent_reply = intent_reply
        self.decompose_reply = decompose_reply
        self.systems: list[str] = []

    def stream(self, system: str, user: str) -> Iterator[str]:
        self.systems.append(system)
        if "small catalog" in system:
            return iter([self.decompose_reply])
        if "classify" in system or "EXACTLY ONE WORD" in system:
            return iter([self.intent_reply])
        return iter(["answer"])


class _NoopRetriever:
    def ensure_indexed(self, docs_dir: str) -> None:
        return None

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return []


def _two_collection_catalog(tmp_path: Path) -> Catalog:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A")
    return Catalog(
        collections=(
            Collection(
                name="rag_doc",
                display_name="RAG 论文集",
                docs_dir=docs,
                description="25 篇 RAG 论文",
                retriever_factory=lambda _p, _n: _NoopRetriever(),
            ),
            Collection(
                name="shanzhongshi",
                display_name="山中事咖啡",
                docs_dir=docs,
                description="豆子参数与冲煮教程",
                retriever_factory=lambda _p, _n: _NoopRetriever(),
            ),
        )
    )


# ---- A: fan-out / prompt scoping ----


def test_catalog_summary_scoped_to_selected_collection(tmp_path: Path):
    catalog = _two_collection_catalog(tmp_path)
    summary = _build_catalog_summary(catalog, only=("shanzhongshi",))
    assert "山中事咖啡" in summary
    assert "RAG 论文集" not in summary


def test_catalog_summary_falls_back_to_full_catalog_for_unknown_keys(tmp_path: Path):
    """Legacy compare mode keys ('chroma'/'milvus') are not catalog names."""
    catalog = _two_collection_catalog(tmp_path)
    summary = _build_catalog_summary(catalog, only=("chroma", "milvus"))
    assert "山中事咖啡" in summary
    assert "RAG 论文集" in summary


def test_decompose_prompt_only_describes_selected_collection(
    tmp_path: Path, make_routing_config: Callable[..., Config]
):
    llm = _ScriptedRoutingLLM(intent_reply="all", decompose_reply='["sub1"]')
    catalog = _two_collection_catalog(tmp_path)

    out = answer_stream(
        {"shanzhongshi": _RecordingRetriever()},
        llm,
        "推荐豆子",
        k=2,
        config=make_routing_config(),
        catalog=catalog,
    )
    _ = "".join(out["shanzhongshi"][0])

    decompose_prompts = [s for s in llm.systems if "small catalog" in s]
    assert len(decompose_prompts) == 1
    assert "山中事咖啡" in decompose_prompts[0]
    assert "RAG 论文集" not in decompose_prompts[0]


# ---- B: split candidate-k from merged-k ----


def test_sub_query_fetch_uses_catalog_sub_k(
    tmp_path: Path, make_routing_config: Callable[..., Config]
):
    llm = _ScriptedRoutingLLM(intent_reply="all", decompose_reply='["s1", "s2", "s3"]')
    retriever = _RecordingRetriever()
    catalog = _two_collection_catalog(tmp_path)

    out = answer_stream(
        {"shanzhongshi": retriever},
        llm,
        "推荐豆子",
        k=2,
        config=make_routing_config(catalog_sub_k=6, catalog_recall_k=10),
        catalog=catalog,
    )
    _, hits, _ = out["shanzhongshi"]
    _ = "".join(out["shanzhongshi"][0])

    # Each sub-query pulls catalog_sub_k (NOT catalog_recall_k) candidates.
    assert [k for _q, k in retriever.calls] == [6, 6, 6]
    # ...and the merge caps the prompt at catalog_recall_k.
    assert len(hits) == 10


def test_merged_hits_capped_by_catalog_recall_k(
    tmp_path: Path, make_routing_config: Callable[..., Config]
):
    llm = _ScriptedRoutingLLM(intent_reply="all", decompose_reply='["s1", "s2"]')
    catalog = _two_collection_catalog(tmp_path)

    out = answer_stream(
        {"shanzhongshi": _RecordingRetriever()},
        llm,
        "推荐豆子",
        k=2,
        config=make_routing_config(catalog_sub_k=20, catalog_recall_k=5),
        catalog=catalog,
    )
    _, hits, _ = out["shanzhongshi"]
    _ = "".join(out["shanzhongshi"][0])

    assert len(hits) == 5


def test_config_defaults_catalog_sub_k(monkeypatch):
    from rag_learn.config import load_config

    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.delenv("CATALOG_SUB_K", raising=False)
    assert load_config().catalog_sub_k == 8


def test_config_reads_catalog_sub_k_from_env(monkeypatch):
    from rag_learn.config import load_config

    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("CATALOG_SUB_K", "12")
    assert load_config().catalog_sub_k == 12
