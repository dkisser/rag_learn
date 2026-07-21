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
            persist_dir = self.docs_dir.parent.parent / "data" / "chroma" / self.name
            retriever = self.retriever_factory(persist_dir, self.name)
            retriever.ensure_indexed(str(self.docs_dir))
            object.__setattr__(self, "_retriever", retriever)
        assert self._retriever is not None
        return self._retriever
