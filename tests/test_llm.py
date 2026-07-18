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
    llm = DeepSeekLLM(api_key="k", client=fake)  # type: ignore[arg-type]
    list(llm.stream("s", "u"))
    assert fake.chat.completions.last_kwargs["model"] == "deepseek-v4-flash"


def test_stream_emits_single_error_token_when_sdk_raises():
    fake = _RaisingClient()
    llm = DeepSeekLLM(api_key="k", model="m", base_url="u", client=fake)  # type: ignore[arg-type]
    tokens = list(llm.stream("sys", "user"))
    assert len(tokens) == 1
    assert "⚠ LLM 错误" in tokens[0] and "boom" in tokens[0]
