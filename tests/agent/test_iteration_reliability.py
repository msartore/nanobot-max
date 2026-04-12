"""Tests for the new AI iteration reliability features:
- Task completion verification
- Context compaction
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from nanobot.agent.runner import AgentRunSpec, AgentRunner, AgentRunResult
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.utils.runtime import (
    build_completion_check_message,
    is_completion_confirmed,
    COMPLETION_CHECK_PROMPT,
    CONTEXT_COMPACTION_PROMPT,
)


# ---------------------------------------------------------------------------
# Completion verification helpers
# ---------------------------------------------------------------------------

class TestIsCompletionConfirmed:
    def test_done_prefix_with_answer(self):
        confirmed, answer = is_completion_confirmed("DONE: Task is complete")
        assert confirmed is True
        assert answer == "Task is complete"

    def test_done_prefix_without_answer(self):
        confirmed, answer = is_completion_confirmed("DONE:")
        assert confirmed is True
        assert answer == "DONE:"

    def test_done_without_colon(self):
        confirmed, answer = is_completion_confirmed("DONE Task is complete")
        assert confirmed is True
        assert answer == "Task is complete"

    def test_lowercase_done(self):
        confirmed, answer = is_completion_confirmed("done: finished")
        assert confirmed is True

    def test_not_confirmed(self):
        confirmed, answer = is_completion_confirmed("I still need to check the file")
        assert confirmed is False
        assert answer == "I still need to check the file"

    def test_none_content(self):
        confirmed, answer = is_completion_confirmed(None)
        assert confirmed is False
        assert answer is None

    def test_empty_content(self):
        confirmed, answer = is_completion_confirmed("")
        assert confirmed is False
        assert answer == ""

    def test_whitespace_only(self):
        confirmed, answer = is_completion_confirmed("   ")
        assert confirmed is False


class TestCompletionCheckMessage:
    def test_build_message_has_correct_role(self):
        msg = build_completion_check_message()
        assert msg["role"] == "user"

    def test_build_message_has_content(self):
        msg = build_completion_check_message()
        assert "Review the original task" in msg["content"]


# ---------------------------------------------------------------------------
# Completion verification in the runner
# ---------------------------------------------------------------------------

def _make_runner():
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    return AgentRunner(provider)


def _make_tools():
    tools = ToolRegistry()
    tools.register(_DummyTool())
    return tools


class _DummyTool:
    name = "dummy"
    description = "A dummy tool"
    parameters = {}
    concurrency_safe = True

    async def execute(self, **kwargs):
        return "ok"

    def to_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }


@pytest.mark.asyncio
async def test_completion_check_disabled_by_default(tmp_path):
    """When max_completion_checks=0, the loop breaks immediately on text response."""
    runner = _make_runner()
    call_count = {"n": 0}

    async def chat_with_retry(**kwargs):
        call_count["n"] += 1
        return LLMResponse(content="Final answer", tool_calls=[], usage={})

    runner.provider.chat_with_retry = chat_with_retry

    result = await runner.run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "Do something"}],
        tools=_make_tools(),
        model="test-model",
        max_iterations=10,
        max_tool_result_chars=16000,
        max_completion_checks=0,
    ))

    assert result.stop_reason == "completed"
    assert result.final_content == "Final answer"
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_completion_check_continues_when_not_confirmed(tmp_path):
    """When the LLM response doesn't start with DONE:, the loop continues."""
    runner = _make_runner()
    call_count = {"n": 0}

    async def chat_with_retry(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content="I think I'm done but let me verify...",
                tool_calls=[],
                usage={},
            )
        return LLMResponse(content="DONE: Now truly done", tool_calls=[], usage={})

    runner.provider.chat_with_retry = chat_with_retry

    result = await runner.run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "Do something complex"}],
        tools=_make_tools(),
        model="test-model",
        max_iterations=10,
        max_tool_result_chars=16000,
        max_completion_checks=2,
    ))

    assert result.stop_reason == "completed"
    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_completion_check_respects_max_rounds(tmp_path):
    """After max_completion_checks rounds, the loop breaks even if not confirmed."""
    runner = _make_runner()
    call_count = {"n": 0}

    async def chat_with_retry(**kwargs):
        call_count["n"] += 1
        return LLMResponse(content="Still working on it", tool_calls=[], usage={})

    runner.provider.chat_with_retry = chat_with_retry

    result = await runner.run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "Do something"}],
        tools=_make_tools(),
        model="test-model",
        max_iterations=10,
        max_tool_result_chars=16000,
        max_completion_checks=1,
    ))

    assert result.stop_reason == "completed"
    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_completion_check_allows_tool_calls_to_continue(tmp_path):
    """If the LLM calls tools during a completion check round, the loop continues normally."""
    runner = _make_runner()
    call_count = {"n": 0}

    async def chat_with_retry(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResponse(
                content="Let me check something",
                tool_calls=[],
                usage={},
            )
        if call_count["n"] == 2:
            return LLMResponse(
                content="Checking",
                tool_calls=[
                    ToolCallRequest(id="1", name="dummy", arguments={})
                ],
                usage={},
            )
        return LLMResponse(content="DONE: All done", tool_calls=[], usage={})

    runner.provider.chat_with_retry = chat_with_retry

    result = await runner.run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "Do something"}],
        tools=_make_tools(),
        model="test-model",
        max_iterations=10,
        max_tool_result_chars=16000,
        max_completion_checks=3,
    ))

    assert result.stop_reason == "completed"
    assert call_count["n"] == 3
    assert "dummy" in result.tools_used


# ---------------------------------------------------------------------------
# Context compaction
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compaction_not_triggered_below_threshold(tmp_path):
    """When below threshold, no compaction LLM call is made."""
    runner = _make_runner()
    compaction_called = {"n": 0}

    async def chat_with_retry(**kwargs):
        messages = kwargs.get("messages", [])
        for msg in messages:
            if isinstance(msg.get("content"), str) and "Summarize the following conversation" in msg["content"]:
                compaction_called["n"] += 1
                return LLMResponse(content="summary", tool_calls=[], usage={})
        return LLMResponse(content="DONE: Task complete", tool_calls=[], usage={})

    runner.provider.chat_with_retry = chat_with_retry

    result = await runner.run(AgentRunSpec(
        initial_messages=[{"role": "user", "content": "Hello"}],
        tools=_make_tools(),
        model="test-model",
        max_iterations=3,
        max_tool_result_chars=16000,
        context_compact_threshold_tokens=50000,
    ))

    assert compaction_called["n"] == 0


@pytest.mark.asyncio
async def test_compaction_triggered_above_threshold(tmp_path):
    """Context compaction is now handled by the memory layer, not the runner.
    
    The runner only does snipping for context window safety.
    Summary compaction happens in AgentLoop._update_context_summary after each turn.
    """
    runner = _make_runner()
    compaction_called = {"n": 0}
    iteration_count = {"n": 0}

    async def chat_with_retry(**kwargs):
        messages = kwargs.get("messages", [])
        for msg in messages:
            if isinstance(msg.get("content"), str) and "Summarize this conversation" in msg["content"]:
                compaction_called["n"] += 1
                return LLMResponse(content="Compacted summary", tool_calls=[], usage={})
        iteration_count["n"] += 1
        if iteration_count["n"] == 1:
            return LLMResponse(
                content="checking",
                tool_calls=[ToolCallRequest(id="1", name="dummy", arguments={})],
                usage={},
            )
        if iteration_count["n"] == 2:
            return LLMResponse(content="Still working", tool_calls=[], usage={})
        return LLMResponse(content="DONE: Task complete", tool_calls=[], usage={})

    runner.provider.chat_with_retry = chat_with_retry

    messages = [{"role": "user", "content": "Start"}]
    for i in range(10):
        messages.append({"role": "assistant", "content": f"Step {i} result"})
        messages.append({"role": "tool", "content": f"Tool result {i} with lots of data " * 100})

    token_counts = {"call": 0}

    def mock_estimate(*args, **kwargs):
        token_counts["call"] += 1
        if token_counts["call"] == 1:
            return (200, None)
        return (50, None)

    with patch("nanobot.agent.runner.estimate_prompt_tokens_chain", side_effect=mock_estimate):
        result = await runner.run(AgentRunSpec(
            initial_messages=messages,
            tools=_make_tools(),
            model="test-model",
            max_iterations=5,
            max_tool_result_chars=16000,
            context_compact_threshold_tokens=100,
        ))

    # Compaction is no longer in the runner; it's handled by AgentLoop._update_context_summary
    assert compaction_called["n"] == 0


@pytest.mark.asyncio
async def test_compaction_disabled_when_none(tmp_path):
    """When context_compact_threshold_tokens is None, no compaction."""
    runner = _make_runner()
    compaction_called = {"n": 0}

    async def chat_with_retry(**kwargs):
        messages = kwargs.get("messages", [])
        for msg in messages:
            if isinstance(msg.get("content"), str) and "Summarize the following conversation" in msg["content"]:
                compaction_called["n"] += 1
                return LLMResponse(content="summary", tool_calls=[], usage={})
        return LLMResponse(content="DONE: Done", tool_calls=[], usage={})

    runner.provider.chat_with_retry = chat_with_retry

    messages = [{"role": "user", "content": "Start"}]
    for i in range(10):
        messages.append({"role": "assistant", "content": f"Step {i}"})
        messages.append({"role": "tool", "content": f"Result {i} " * 100})

    result = await runner.run(AgentRunSpec(
        initial_messages=messages,
        tools=_make_tools(),
        model="test-model",
        max_iterations=3,
        max_tool_result_chars=16000,
        context_compact_threshold_tokens=None,
    ))

    assert compaction_called["n"] == 0


# ---------------------------------------------------------------------------
# Input validation error recovery (400 / context too large)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_input_validation_error_snips_context():
    """When LLM returns a 400 input validation error, the runner snips history
    rather than appending more messages, so context shrinks before retrying."""
    runner = _make_runner()
    call_count = {"n": 0}

    # Build a conversation with many messages so snipping has something to do
    initial_messages = [{"role": "user", "content": "Start"}]
    for i in range(20):
        initial_messages.append({"role": "assistant", "content": f"Step {i}"})
        initial_messages.append({"role": "user", "content": f"Result {i}"})

    message_counts = []

    async def chat_with_retry(**kwargs):
        call_count["n"] += 1
        message_counts.append(len(kwargs.get("messages", [])))
        if call_count["n"] == 1:
            return LLMResponse(
                content='Error: {"error":{"message":"Provider returned error","code":400,"metadata":{"raw":"Input validation error"}}}',
                finish_reason="error",
                tool_calls=[],
                usage={},
            )
        return LLMResponse(content="DONE: Recovered successfully", tool_calls=[], usage={})

    runner.provider.chat_with_retry = chat_with_retry

    result = await runner.run(AgentRunSpec(
        initial_messages=initial_messages,
        tools=_make_tools(),
        model="test-model",
        max_iterations=5,
        max_tool_result_chars=16000,
        context_window_tokens=8000,
        max_completion_checks=0,
    ))

    assert result.stop_reason == "completed"
    assert "Recovered successfully" in (result.final_content or "")
    # Second call must have fewer messages than the first (snipping or hard-truncation worked)
    assert message_counts[1] < message_counts[0], (
        f"Expected context to shrink after input validation error, "
        f"but got {message_counts[0]} -> {message_counts[1]} messages"
    )


@pytest.mark.asyncio
async def test_input_validation_error_hard_truncates_without_context_window():
    """When context_window_tokens is not set, input validation error triggers
    hard truncation to the last 8 non-system messages."""
    runner = _make_runner()
    call_count = {"n": 0}

    # Build a long conversation
    initial_messages = [{"role": "user", "content": "Start"}]
    for i in range(20):
        initial_messages.append({"role": "assistant", "content": f"Step {i}"})
        initial_messages.append({"role": "user", "content": f"Result {i}"})

    message_counts = []

    async def chat_with_retry(**kwargs):
        call_count["n"] += 1
        message_counts.append(len(kwargs.get("messages", [])))
        if call_count["n"] == 1:
            return LLMResponse(
                content='Error: {"error":{"message":"Provider returned error","code":400,"metadata":{"raw":"Input validation error"}}}',
                finish_reason="error",
                tool_calls=[],
                usage={},
            )
        return LLMResponse(content="DONE: Recovered", tool_calls=[], usage={})

    runner.provider.chat_with_retry = chat_with_retry

    result = await runner.run(AgentRunSpec(
        initial_messages=initial_messages,
        tools=_make_tools(),
        model="test-model",
        max_iterations=5,
        max_tool_result_chars=16000,
        # No context_window_tokens — forces hard truncation
        max_completion_checks=0,
    ))

    assert result.stop_reason == "completed"
    assert message_counts[1] <= 8, (
        f"Expected hard truncation to <=8 non-system messages, got {message_counts[1]}"
    )
