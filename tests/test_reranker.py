"""Tests for the reranker module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_learn.config import Config, load_config
from rag_learn.retriever.base import Hit


class TestCrossEncoderReranker:
    """Unit tests for CrossEncoderReranker using mocked CrossEncoder."""

    @patch("sentence_transformers.CrossEncoder")
    def test_rank_sorts_by_descending_score(self, mock_cls: MagicMock) -> None:
        from rag_learn.reranker.cross_encoder_impl import CrossEncoderReranker

        model = MagicMock()
        model.predict.return_value = [0.2, 0.9, 0.5]
        mock_cls.return_value = model

        hits = [
            Hit(text="irrelevant", source_file="generic.md", chunk_index=0, score=0.1),
            Hit(text="relevant", source_file="target.md", chunk_index=1, score=0.5),
            Hit(text="somewhat", source_file="other.md", chunk_index=2, score=0.3),
        ]
        reranker = CrossEncoderReranker("dummy-model")
        ranked = reranker.rank("query", hits)

        assert [h.text for h in ranked] == ["relevant", "somewhat", "irrelevant"]
        assert [h.score for h in ranked] == [0.9, 0.5, 0.2]
        model.predict.assert_called_once()
        pairs = model.predict.call_args[0][0]
        assert pairs == [("query", h.text) for h in hits]

    @patch("sentence_transformers.CrossEncoder")
    def test_rank_returns_new_hit_objects(self, mock_cls: MagicMock) -> None:
        from rag_learn.reranker.cross_encoder_impl import CrossEncoderReranker

        model = MagicMock()
        model.predict.return_value = [0.5]
        mock_cls.return_value = model

        original = Hit(text="a", source_file="a.md", chunk_index=0, score=0.1)
        reranker = CrossEncoderReranker("dummy-model")
        ranked = reranker.rank("q", [original])

        assert len(ranked) == 1
        assert ranked[0] is not original
        assert ranked[0].text == original.text
        assert ranked[0].source_file == original.source_file
        assert ranked[0].chunk_index == original.chunk_index

    @patch("sentence_transformers.CrossEncoder")
    def test_rank_empty_input_returns_empty_list(self, mock_cls: MagicMock) -> None:
        from rag_learn.reranker.cross_encoder_impl import CrossEncoderReranker

        mock_cls.return_value = MagicMock()
        reranker = CrossEncoderReranker("dummy-model")
        assert reranker.rank("q", []) == []

    @patch("sentence_transformers.CrossEncoder")
    def test_constructor_passes_device_and_batch_size(self, mock_cls: MagicMock) -> None:
        from rag_learn.reranker.cross_encoder_impl import CrossEncoderReranker

        mock_cls.return_value = MagicMock()
        CrossEncoderReranker(
            "dummy-model",
            device="cpu",
            batch_size=16,
            max_seq_length=256,
        )
        mock_cls.assert_called_once_with(
            "dummy-model",
            device="cpu",
            max_length=256,
        )


class TestBuildReranker:
    """Tests for build_reranker factory."""

    @patch("rag_learn.reranker.factory.CrossEncoderReranker")
    def test_build_reranker_disabled_returns_none(self, mock_cls: MagicMock) -> None:
        from rag_learn.reranker.factory import build_reranker

        config = Config(
            deepseek_api_key="k",
            llm_model="m",
            deepseek_base_url="u",
            retrieve_k=5,
            chunk_size=800,
            chunk_overlap=50,
            repo_root=load_config().repo_root,
            docs_dir=load_config().docs_dir,
            data_dir=load_config().data_dir,
            chroma_dir=load_config().chroma_dir,
            milvus_path=load_config().milvus_path,
            rerank_enabled=False,
            rerank_model="BAAI/bge-reranker-base",
            rerank_factor=4,
            rerank_k=None,
            rerank_batch_size=8,
            rerank_device=None,
            hybrid_enabled=False,
            hybrid_rrf_k=60,
        )
        assert build_reranker(config) is None
        mock_cls.assert_not_called()

    @patch("rag_learn.reranker.factory.CrossEncoderReranker")
    def test_build_reranker_enabled_returns_instance(self, mock_cls: MagicMock) -> None:
        from rag_learn.reranker.factory import build_reranker

        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        config = Config(
            deepseek_api_key="k",
            llm_model="m",
            deepseek_base_url="u",
            retrieve_k=5,
            chunk_size=800,
            chunk_overlap=50,
            repo_root=load_config().repo_root,
            docs_dir=load_config().docs_dir,
            data_dir=load_config().data_dir,
            chroma_dir=load_config().chroma_dir,
            milvus_path=load_config().milvus_path,
            rerank_enabled=True,
            rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            rerank_factor=4,
            rerank_k=24,
            rerank_batch_size=16,
            rerank_device="cpu",
            hybrid_enabled=False,
            hybrid_rrf_k=60,
        )
        result = build_reranker(config)
        assert result is mock_instance
        mock_cls.assert_called_once_with(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu",
            batch_size=16,
        )

    @patch("rag_learn.reranker.factory.CrossEncoderReranker")
    def test_build_reranker_load_failure_returns_none_and_logs(
        self, mock_cls: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        from rag_learn.reranker.factory import build_reranker

        mock_cls.side_effect = RuntimeError("model download failed")
        config = Config(
            deepseek_api_key="k",
            llm_model="m",
            deepseek_base_url="u",
            retrieve_k=5,
            chunk_size=800,
            chunk_overlap=50,
            repo_root=load_config().repo_root,
            docs_dir=load_config().docs_dir,
            data_dir=load_config().data_dir,
            chroma_dir=load_config().chroma_dir,
            milvus_path=load_config().milvus_path,
            rerank_enabled=True,
            rerank_model="bad-model",
            rerank_factor=4,
            rerank_k=None,
            rerank_batch_size=8,
            rerank_device=None,
            hybrid_enabled=False,
            hybrid_rrf_k=60,
        )
        with caplog.at_level("WARNING"):
            assert build_reranker(config) is None
        assert "bad-model" in caplog.text
        assert "falling back" in caplog.text


class TestRerankerProtocol:
    """Structural tests for the Reranker Protocol."""

    def test_cross_encoder_reranker_satisfies_protocol(self) -> None:
        from rag_learn.reranker.base import Reranker
        from rag_learn.reranker.cross_encoder_impl import CrossEncoderReranker

        with patch("sentence_transformers.CrossEncoder") as mock_cls:
            mock_cls.return_value = MagicMock()
            instance = CrossEncoderReranker("dummy")
            assert isinstance(instance, Reranker)
