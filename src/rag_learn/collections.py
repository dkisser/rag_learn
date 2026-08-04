"""Collection domain object: a single knowledge base (name, docs, retriever)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_learn.retriever.base import BaseRetriever

logger = logging.getLogger(__name__)

# docs_dir is expected to sit at <repo-root>/docs/<collection-name>.
# Chroma persist directories therefore live at <repo-root>/data/chroma/<collection-name>.
PERSIST_DIR_SEGMENTS = ("data", "chroma")


def _default_factory(persist_dir: Path, name: str) -> BaseRetriever:
    # Imported lazily so this module stays cheap when only Collection is used.
    from rag_learn.retriever.factory import build_retriever

    return build_retriever(persist_dir=persist_dir, collection_name=name)


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
    _retriever: BaseRetriever | None = field(default=None, init=False, repr=False)

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
                self.docs_dir.parent.parent
                / PERSIST_DIR_SEGMENTS[0]
                / PERSIST_DIR_SEGMENTS[1]
                / self.name
            )
            retriever = self.retriever_factory(persist_dir, self.name)
            retriever.ensure_indexed(str(self.docs_dir))
            object.__setattr__(self, "_retriever", retriever)
        assert self._retriever is not None
        return self._retriever


class CollectionNotFoundError(KeyError):
    """请求的 collection 不在 Catalog 里。"""


@dataclass(frozen=True)
class Catalog:
    """不可变集合注册表：slug → Collection 双向索引。"""

    collections: tuple[Collection, ...]
    _by_name: dict[str, Collection] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        names = [c.name for c in self.collections]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate collection names: {names}")
        object.__setattr__(self, "_by_name", {c.name: c for c in self.collections})

    def names(self) -> list[str]:
        return [c.name for c in self.collections]

    def iter_collections(self) -> Iterator[Collection]:
        """Yield each contained :class:`Collection` in declaration order."""
        return iter(self.collections)

    def display_choices(self) -> list[tuple[str, str]]:
        return [(c.display_name, c.name) for c in self.collections]

    def get(self, name: str) -> Collection:
        assert self._by_name is not None
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise CollectionNotFoundError(
                f"collection {name!r} not in catalog; available: {self.names()}"
            ) from exc

    def ensure_all_indexed(self) -> list[tuple[str, str]]:
        """Eager 触发每个 collection 的 retriever 懒加载。fail-open.

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


def _make_factory(
    *,
    hybrid_enabled: bool,
    hybrid_rrf_k: int,
    chroma_max_distance: float | None = None,
) -> Callable[[Path, str], BaseRetriever]:
    """Build a retriever factory that captures the hybrid config."""
    from rag_learn.retriever.factory import build_retriever

    def _factory(persist_dir: Path, name: str) -> BaseRetriever:
        return build_retriever(
            persist_dir=persist_dir,
            collection_name=name,
            hybrid_enabled=hybrid_enabled,
            hybrid_rrf_k=hybrid_rrf_k,
            chroma_max_distance=chroma_max_distance,
        )

    return _factory


def _build_builtin(
    *,
    hybrid_enabled: bool = False,
    hybrid_rrf_k: int = 60,
    chroma_max_distance: float | None = None,
) -> tuple[Collection, ...]:
    from rag_learn.config import _repo_root

    root = _repo_root() / "docs"
    factory = _make_factory(
        hybrid_enabled=hybrid_enabled,
        hybrid_rrf_k=hybrid_rrf_k,
        chroma_max_distance=chroma_max_distance,
    )
    return (
        Collection(
            name="rag_doc",
            display_name="RAG 论文集",
            docs_dir=root / "rag_doc",
            description="25 篇 RAG 相关论文 / 综述 / 实践文章",
            retriever_factory=factory,
        ),
        Collection(
            name="shanzhongshi",
            display_name="山中事咖啡",
            docs_dir=root / "shanzhongshi",
            description="山中事咖啡（SHAN.IN COFFEE）的豆子参数、冲煮教程与公司信息",
            retriever_factory=factory,
        ),
    )


BUILTIN_COLLECTIONS: tuple[Collection, ...] = _build_builtin()


def build_catalog(
    hybrid_enabled: bool = False,
    hybrid_rrf_k: int = 60,
    chroma_max_distance: float | None = None,
) -> Catalog:
    """Build the default catalog, optionally wiring hybrid retrieval and filtering."""
    return Catalog(
        collections=_build_builtin(
            hybrid_enabled=hybrid_enabled,
            hybrid_rrf_k=hybrid_rrf_k,
            chroma_max_distance=chroma_max_distance,
        )
    )
