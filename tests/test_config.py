import pytest

from rag_learn.config import CHUNK_DISPLAY_CHARS, ConfigError, load_config


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


def test_paths_resolve_relative_to_repo(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    cfg = load_config()
    # Walked-up root must be a directory that actually contains pyproject.toml.
    # (Name varies between main checkout "rag_learn" and worktrees.)
    assert (cfg.repo_root / "pyproject.toml").is_file()
    assert cfg.docs_dir.name == "rag_doc"
    assert cfg.data_dir.name == "data"
