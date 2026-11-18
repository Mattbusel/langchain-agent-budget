"""A scripted chat model that reports Anthropic usage, so prices come from llm-prices."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, PrivateAttr

# claude-sonnet-4-5 list price in llm-prices: $3 input, $15 output, $3.75 / $6 cache writes per 1M.
MODEL = "claude-sonnet-4-5"
IN_RATE = 3.0 / 1_000_000
OUT_RATE = 15.0 / 1_000_000


def reply(
    content: str = "done",
    *,
    input_tokens: int = 20,
    output_tokens: int = 500,
    tool_calls: list[dict[str, Any]] | None = None,
    usage: bool = True,
    served_model: str = MODEL + "-20250929",
) -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=tool_calls or [],
        usage_metadata=(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
            if usage
            else None
        ),
        response_metadata={"model_name": served_model, "model_provider": "anthropic"},
    )


class ScriptedChat(BaseChatModel):
    """Answers with `respond(messages)`; counts the calls it actually made."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    respond: Callable[[list[BaseMessage]], AIMessage]
    model_name: str = MODEL
    max_tokens: int | None = 500
    delay: float = 0.0
    _calls: int = PrivateAttr(default=0)
    _lock: Any = PrivateAttr(default_factory=threading.Lock)

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> Any:
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "anthropic"
        return params

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        with self._lock:
            self._calls += 1
        if self.delay:
            time.sleep(self.delay)
        return ChatResult(generations=[ChatGeneration(message=self.respond(messages))])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        return self.bind(**kwargs)


@pytest.fixture
def tool_loop() -> Callable[[int], Callable[[list[BaseMessage]], AIMessage]]:
    """A model that calls the `lookup` tool `n` times, then answers."""

    def make(n: int) -> Callable[[list[BaseMessage]], AIMessage]:
        def respond(messages: list[BaseMessage]) -> AIMessage:
            done = sum(1 for m in messages if m.type == "tool")
            if done < n:
                call = {"name": "lookup", "args": {"q": str(done)}, "id": f"call-{done}"}
                return reply("looking", tool_calls=[call])
            return reply("answer")

        return respond

    return make
