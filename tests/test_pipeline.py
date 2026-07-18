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
    h = [Hit(text=long, source_file="long.md", chunk_index=0, score=0.0)]
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
