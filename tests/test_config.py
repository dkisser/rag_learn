import pytest

from rag_learn.config import CHUNK_DISPLAY_CHARS, ConfigError, load_config


def test_load_config_reads_required_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.delenv("RERANK_ENABLED", raising=False)
    monkeypatch.delenv("RETRIEVE_K", raising=False)
    cfg = load_config()
    assert cfg.deepseek_api_key == "sk-test"
    assert cfg.llm_model == "deepseek-v4-flash"  # default
    assert cfg.deepseek_base_url == "https://api.deepseek.com"  # default
    assert cfg.retrieve_k == 5  # default


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


def test_load_config_rerank_defaults(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.delenv("RERANK_ENABLED", raising=False)
    monkeypatch.delenv("RERANK_MODEL", raising=False)
    monkeypatch.delenv("RERANK_K", raising=False)
    monkeypatch.delenv("RERANK_FACTOR", raising=False)
    monkeypatch.delenv("RERANK_BATCH_SIZE", raising=False)
    monkeypatch.delenv("RERANK_DEVICE", raising=False)
    cfg = load_config()
    assert cfg.rerank_enabled is False
    assert cfg.rerank_model == "BAAI/bge-reranker-base"
    assert cfg.rerank_factor == 4
    assert cfg.rerank_k is None
    assert cfg.rerank_batch_size == 8
    assert cfg.rerank_device == "auto"


def test_load_config_rerank_overrides(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    monkeypatch.setenv("RERANK_K", "24")
    monkeypatch.setenv("RERANK_FACTOR", "6")
    monkeypatch.setenv("RERANK_BATCH_SIZE", "16")
    monkeypatch.setenv("RERANK_DEVICE", "cpu")
    cfg = load_config()
    assert cfg.rerank_enabled is True
    assert cfg.rerank_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert cfg.rerank_k == 24
    assert cfg.rerank_factor == 6
    assert cfg.rerank_batch_size == 16
    assert cfg.rerank_device == "cpu"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
        ("off", False),
        ("", False),
        (None, False),
    ],
)
def test_load_config_boolean_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    if value is not None:
        monkeypatch.setenv("RERANK_ENABLED", value)
    else:
        monkeypatch.delenv("RERANK_ENABLED", raising=False)
    cfg = load_config()
    assert cfg.rerank_enabled is expected


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
