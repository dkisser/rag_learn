"""Tests for rag_learn.logging_config."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_learn.logging_config import _repo_root, create_handlers, get_log_level, setup_logging


def test_get_log_level_defaults_to_info():
    assert get_log_level() == logging.INFO


def test_get_log_level_reads_env(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    assert get_log_level() == logging.DEBUG


def test_get_log_level_accepts_explicit_name():
    assert get_log_level("WARNING") == logging.WARNING


def test_get_log_level_invalid_env_falls_back_to_info(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "NOT_A_LEVEL")
    assert get_log_level() == logging.INFO


def test_repo_root_finds_pyproject_toml(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    nested = tmp_path / "src" / "rag_learn" / "logging_config.py"
    assert _repo_root(nested) == tmp_path


def test_repo_root_raises_when_no_pyproject_toml(tmp_path):
    with pytest.raises(RuntimeError, match="could not locate repo root"):
        _repo_root(tmp_path / "logging_config.py")


def test_create_handlers_creates_log_dir_and_returns_two_handlers(tmp_path):
    log_dir = tmp_path / "logs"
    handlers = create_handlers(log_dir)

    assert log_dir.is_dir()
    assert len(handlers) == 2
    assert any(type(h) is logging.StreamHandler for h in handlers)
    assert any(type(h) is logging.FileHandler for h in handlers)


def test_setup_logging_calls_basic_config_with_expected_settings(monkeypatch):
    fake_handlers = [MagicMock(spec=logging.Handler), MagicMock(spec=logging.Handler)]
    monkeypatch.setattr("rag_learn.logging_config.create_handlers", lambda _log_dir: fake_handlers)
    monkeypatch.setattr("rag_learn.logging_config._repo_root", lambda: Path("/tmp/fake"))

    with patch("logging.basicConfig") as mock_basic_config:
        setup_logging()

    mock_basic_config.assert_called_once()
    call_kwargs = mock_basic_config.call_args.kwargs
    assert call_kwargs["level"] == logging.INFO
    assert call_kwargs["format"] == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    assert call_kwargs["handlers"] is fake_handlers


def test_setup_logging_uses_debug_level_from_env(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    fake_handlers = [MagicMock(spec=logging.Handler)]
    monkeypatch.setattr("rag_learn.logging_config.create_handlers", lambda _log_dir: fake_handlers)
    monkeypatch.setattr("rag_learn.logging_config._repo_root", lambda: Path("/tmp/fake"))

    with patch("logging.basicConfig") as mock_basic_config:
        setup_logging()

    assert mock_basic_config.call_args.kwargs["level"] == logging.DEBUG
