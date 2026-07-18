"""DeepSeek LLM client; uses the OpenAI SDK with DeepSeek's base URL."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)


class DeepSeekLLM:
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        client: Any | None = None,
    ) -> None:
        self._client: Any = (
            client if client is not None else OpenAI(api_key=api_key, base_url=base_url)
        )
        self._model = model

    def stream(self, system: str, user: str) -> Iterator[str]:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("DeepSeekLLM.create failed")
            yield f"⚠ LLM 错误：{exc}"
            return

        try:
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    yield content
        except Exception as exc:  # noqa: BLE001
            logger.exception("DeepSeekLLM stream interrupted")
            yield f"\n\n⚠ LLM 中断：{exc}"
            return
