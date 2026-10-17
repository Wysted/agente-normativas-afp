from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from agintis.modules.agents.direct_chat import direct_chat_runner
from agintis.modules.agents_catalog.api import (
    AgentUnavailableError,
    RunCompleted,
    RunInput,
    Step,
    Token,
    TokenUsage,
)
from agintis.platform.llm import ChatModels, ChatRole, LlmConfig
from agintis.shared_kernel.ids import RunId, TenantId


class StreamingFake(BaseChatModel):
    """Streams scripted chunks; the last one carries finish metadata and usage."""

    pieces: list[str]
    finish_key: str = "finish_reason"

    @property
    def _llm_type(self) -> str:
        return "streaming-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = "".join(self.pieces)
        return ChatResult(generations=[ChatGeneration(message=AIMessageChunk(content=text))])

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for piece in self.pieces:
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))
        usage = UsageMetadata(input_tokens=4, output_tokens=2, total_tokens=6)
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                usage_metadata=usage,
                response_metadata={self.finish_key: "length"},
            )
        )


def _models(model: BaseChatModel) -> ChatModels:
    config = LlmConfig(provider="openai", models={role: "m" for role in ChatRole})
    return ChatModels(config, builder=lambda *_a, **_k: model)


def _input() -> RunInput:
    return RunInput(TenantId("t"), "hi", RunId("r"))


async def test_streams_tokens_then_usage_and_finish_reason() -> None:
    runner = direct_chat_runner(_models(StreamingFake(pieces=["Hel", "lo"])))

    events = [e async for e in runner.astream(_input()) if not isinstance(e, Step)]

    assert events == [Token("Hel"), Token("lo"), RunCompleted("length", TokenUsage(4, 2, 6))]


async def test_anthropic_style_stop_reason() -> None:
    runner = direct_chat_runner(_models(StreamingFake(pieces=["a"], finish_key="stop_reason")))

    events = [e async for e in runner.astream(_input()) if not isinstance(e, Step)]

    assert events[-1] == RunCompleted("length", TokenUsage(4, 2, 6))


def test_disabled_llm_is_unavailable_but_constructs() -> None:
    runner = direct_chat_runner(ChatModels(None))

    with pytest.raises(AgentUnavailableError) as info:
        runner.ensure_available()
    assert info.value.code == "llm_not_configured"


def test_enabled_llm_is_available_and_the_model_is_cached() -> None:
    models = _models(StreamingFake(pieces=[]))
    runner = direct_chat_runner(models)

    runner.ensure_available()

    assert models.get(ChatRole.SYNTHESIS) is models.get(ChatRole.SYNTHESIS)


def test_build_failure_is_llm_misconfigured_without_leaking_text() -> None:
    def failing(*_args: Any, **_kwargs: Any) -> BaseChatModel:
        raise RuntimeError("api key sk-secret rejected")

    config = LlmConfig(provider="openai", models={role: "m" for role in ChatRole})
    runner = direct_chat_runner(ChatModels(config, builder=failing))

    with pytest.raises(AgentUnavailableError) as info:
        runner.ensure_available()

    assert info.value.code == "llm_misconfigured"
    assert "sk-secret" not in str(info.value)
