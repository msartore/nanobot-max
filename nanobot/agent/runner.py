"""Shared execution loop for tool-using agents."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.providers.base import LLMProvider, ToolCallRequest
from nanobot.utils.helpers import (
    build_assistant_message,
    estimate_message_tokens,
    estimate_prompt_tokens_chain,
    find_legal_message_start,
    maybe_persist_tool_result,
    truncate_text,
)
from nanobot.utils.prompt_templates import render_template
from nanobot.utils.runtime import (
    EMPTY_FINAL_RESPONSE_MESSAGE,
    build_completion_check_message,
    build_finalization_retry_message,
    ensure_nonempty_tool_result,
    is_blank_text,
    is_completion_confirmed,
    repeated_external_lookup_error,
)

_DEFAULT_ERROR_MESSAGE = "Sorry, I encountered an error calling the AI model."
_MAX_EMPTY_RETRIES = 2
_MAX_LENGTH_RECOVERIES = 3
_SNIP_SAFETY_BUFFER = 1024
_MICROCOMPACT_KEEP_RECENT = 10
_MICROCOMPACT_MIN_CHARS = 500
_COMPACTABLE_TOOLS = frozenset({
    "read_file", "exec", "grep", "glob",
    "web_search", "web_fetch", "list_dir",
})
_BACKFILL_CONTENT = "[Tool result unavailable — call was interrupted or lost]"
@dataclass(slots=True)
class AgentRunSpec:
    """Configuration for a single agent execution."""

    initial_messages: list[dict[str, Any]]
    tools: ToolRegistry
    model: str
    max_iterations: int
    max_tool_result_chars: int
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    hook: AgentHook | None = None
    error_message: str | None = _DEFAULT_ERROR_MESSAGE
    max_iterations_message: str | None = None
    concurrent_tools: bool = False
    fail_on_tool_error: bool = False
    workspace: Path | None = None
    session_key: str | None = None
    context_window_tokens: int | None = None
    context_block_limit: int | None = None
    context_compact_threshold_tokens: int | None = None
    provider_retry_mode: str = "standard"
    progress_callback: Any | None = None
    checkpoint_callback: Any | None = None
    max_completion_checks: int = 0


@dataclass(slots=True)
class AgentRunResult:
    """Outcome of a shared agent execution."""

    final_content: str | None
    messages: list[dict[str, Any]]
    tools_used: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "completed"
    error: str | None = None
    tool_events: list[dict[str, str]] = field(default_factory=list)


class AgentRunner:
    """Run a tool-capable LLM loop without product-layer concerns."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        hook = spec.hook or AgentHook()
        messages = list(spec.initial_messages)
        final_content: str | None = None
        last_real_response: str | None = None
        tools_used: list[str] = []
        usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        error: str | None = None
        stop_reason = "completed"
        tool_events: list[dict[str, str]] = []
        external_lookup_counts: dict[str, int] = {}
        completion_check_rounds = 0
        _suppress_streaming = False  # True when next call is a completion-check confirmation
        empty_content_retries = 0

        for iteration in range(spec.max_iterations):
            # Capture and reset the per-iteration streaming suppression flag.
            # Completion-check confirmation calls should not stream to the user.
            _streaming_this_iter = not _suppress_streaming
            _suppress_streaming = False
            try:
                messages = self._backfill_missing_tool_results(messages)
                messages = self._microcompact(messages)
                messages = self._apply_tool_result_budget(spec, messages)
                messages_for_model = self._snip_history(spec, messages)
            except Exception as exc:
                logger.warning(
                    "Context governance failed on turn {} for {}: {}; using raw messages",
                    iteration,
                    spec.session_key or "default",
                    exc,
                )
                messages_for_model = messages
            context = AgentHookContext(iteration=iteration, messages=messages)
            await hook.before_iteration(context)
            response = await self._request_model(
                spec, messages_for_model, hook, context,
                skip_stream=not _streaming_this_iter,
            )
            raw_usage = self._usage_dict(response.usage)
            context.response = response
            context.usage = dict(raw_usage)
            context.tool_calls = list(response.tool_calls)
            self._accumulate_usage(usage, raw_usage)

            if response.has_tool_calls:
                if hook.wants_streaming():
                    await hook.on_stream_end(context, resuming=True)

                assistant_message = build_assistant_message(
                    response.content or "",
                    tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                messages.append(assistant_message)
                tools_used.extend(tc.name for tc in response.tool_calls)
                await self._emit_checkpoint(
                    spec,
                    {
                        "phase": "awaiting_tools",
                        "iteration": iteration,
                        "model": spec.model,
                        "assistant_message": assistant_message,
                        "completed_tool_results": [],
                        "pending_tool_calls": [tc.to_openai_tool_call() for tc in response.tool_calls],
                    },
                )

                await hook.before_execute_tools(context)

                results, new_events, fatal_error = await self._execute_tools(
                    spec,
                    response.tool_calls,
                    external_lookup_counts,
                )
                tool_events.extend(new_events)
                context.tool_results = list(results)
                context.tool_events = list(new_events)
                if fatal_error is not None:
                    error = f"Error: {type(fatal_error).__name__}: {fatal_error}"
                    final_content = error
                    stop_reason = "tool_error"
                    self._append_final_message(messages, final_content)
                    context.final_content = final_content
                    context.error = error
                    context.stop_reason = stop_reason
                    await hook.after_iteration(context)
                    break
                completed_tool_results: list[dict[str, Any]] = []
                for tool_call, result in zip(response.tool_calls, results):
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": self._normalize_tool_result(
                            spec,
                            tool_call.id,
                            tool_call.name,
                            result,
                        ),
                    }
                    messages.append(tool_message)
                    completed_tool_results.append(tool_message)
                await self._emit_checkpoint(
                    spec,
                    {
                        "phase": "tools_completed",
                        "iteration": iteration,
                        "model": spec.model,
                        "assistant_message": assistant_message,
                        "completed_tool_results": completed_tool_results,
                        "pending_tool_calls": [],
                    },
                )
                empty_content_retries = 0
                await hook.after_iteration(context)
                continue

            clean = hook.finalize_content(context, response.content)

            # Detect XML function-call syntax (<function=name>...</function>) in text responses.
            # The model sometimes outputs this format when it wants to invoke a skill but skips
            # the required exec flow. Intercept it and redirect with a correction.
            xml_skill_name = self._detect_xml_skill_call(clean)
            if xml_skill_name is not None and response.finish_reason != "error":
                snippet = (clean or "")[:200].replace("\n", " ")
                logger.warning(
                    "XML skill call detected on turn {}/{} for {} | skill='{}' | snippet='{}'",
                    iteration,
                    spec.max_iterations,
                    spec.session_key or "default",
                    xml_skill_name,
                    snippet,
                )
                if hook.wants_streaming() and _streaming_this_iter:
                    await hook.on_stream_end(context, resuming=True)
                correction = (
                    f"`<function={xml_skill_name}>` syntax is not supported. "
                    f"Skills must be invoked via the `exec` tool. "
                    f"Use `read_file` on the skill's SKILL.md (see `<location>` in the skills listing), "
                    f"then call `exec` with the shell command shown there. "
                    f"The `<baseDir>` value in the skills listing gives you the path to substitute for {{baseDir}}."
                )
                messages.append(build_assistant_message(
                    clean or "",
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                ))
                messages.append({"role": "user", "content": correction})
                logger.debug(
                    "XML skill redirect correction appended for {} | total messages={}",
                    spec.session_key or "default",
                    len(messages),
                )
                await hook.after_iteration(context)
                continue

            if response.finish_reason != "error" and is_blank_text(clean):
                empty_content_retries += 1
                if empty_content_retries < _MAX_EMPTY_RETRIES:
                    logger.warning(
                        "Empty response on turn {} for {} ({}/{}); retrying",
                        iteration,
                        spec.session_key or "default",
                        empty_content_retries,
                        _MAX_EMPTY_RETRIES,
                    )
                    if hook.wants_streaming():
                        await hook.on_stream_end(context, resuming=False)
                    await hook.after_iteration(context)
                    continue
                logger.warning(
                    "Empty response on turn {} for {} after {} retries; attempting finalization",
                    iteration,
                    spec.session_key or "default",
                    empty_content_retries,
                )
                if hook.wants_streaming() and _streaming_this_iter:
                    await hook.on_stream_end(context, resuming=False)
                response = await self._request_finalization_retry(
                    spec, messages_for_model,
                    hook=hook if _streaming_this_iter else None,
                    context=context if _streaming_this_iter else None,
                )
                retry_usage = self._usage_dict(response.usage)
                self._accumulate_usage(usage, retry_usage)
                raw_usage = self._merge_usage(raw_usage, retry_usage)
                context.response = response
                context.usage = dict(raw_usage)
                context.tool_calls = list(response.tool_calls)
                clean = hook.finalize_content(context, response.content)

            # NOTE: on_stream_end is intentionally NOT called unconditionally here.
            # Each decision branch below calls it with the right resuming flag so that:
            # - real content is finalized before a completion-check continuation
            # - the confirmation-check call is run non-streaming (_suppress_streaming)

            if response.finish_reason == "error":
                error_detail = clean or spec.error_message or _DEFAULT_ERROR_MESSAGE
                if self._is_input_validation_error(error_detail):
                    before = len(messages)
                    logger.warning(
                        "LLM returned input validation error on turn {} for {}: {}",
                        iteration,
                        spec.session_key or "default",
                        error_detail,
                    )
                    snipped = self._snip_history(spec, messages)
                    if len(snipped) < before:
                        messages = snipped
                        logger.warning(
                            "Snipped {} -> {} messages and retrying",
                            before,
                            len(messages),
                        )
                    else:
                        # No context window configured or snip had no effect — hard-truncate to recent messages
                        system_msgs = [m for m in messages if m.get("role") == "system"]
                        non_system = [m for m in messages if m.get("role") != "system"]
                        keep = non_system[-8:] if len(non_system) > 8 else non_system
                        messages = system_msgs + keep
                        logger.warning(
                            "Hard-truncated {} -> {} messages and retrying",
                            before,
                            len(messages),
                        )
                else:
                    logger.warning(
                        "LLM returned error on turn {} for {}: {}",
                        iteration,
                        spec.session_key or "default",
                        error_detail,
                    )
                    messages.append(build_assistant_message(
                        "(LLM error occurred, retrying...)",
                        reasoning_content=response.reasoning_content,
                        thinking_blocks=response.thinking_blocks,
                    ))
                    messages.append({
                        "role": "user",
                        "content": f"The previous request failed with: {error_detail[:500]}. Please continue with your task using the conversation above."
                    })
                if hook.wants_streaming() and _streaming_this_iter:
                    await hook.on_stream_end(context, resuming=True)
                await hook.after_iteration(context)
                continue
            if is_blank_text(clean):
                if hook.wants_streaming() and _streaming_this_iter:
                    await hook.on_stream_end(context, resuming=False)
                final_content = EMPTY_FINAL_RESPONSE_MESSAGE
                stop_reason = "empty_final_response"
                error = final_content
                self._append_final_message(messages, final_content)
                context.final_content = final_content
                context.error = error
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                break

            # Track the last real response (with actual content, not meta-summaries)
            # A "real response" has tool calls or is not a completion confirmation
            has_tool_calls = bool(response.tool_calls)
            is_done_prefix = is_completion_confirmed(clean)[0]
            if has_tool_calls or not is_done_prefix:
                last_real_response = clean

            if spec.max_completion_checks > 0 and completion_check_rounds < spec.max_completion_checks:
                if not is_done_prefix:
                    # Finalize the real content stream now, then suppress streaming for
                    # the next (confirmation-check) call so it doesn't reach the user.
                    if hook.wants_streaming() and _streaming_this_iter:
                        await hook.on_stream_end(context, resuming=False)
                    _suppress_streaming = True
                    completion_check_rounds += 1
                    logger.info(
                        "Completion check round {} for {}; task not confirmed complete, continuing",
                        completion_check_rounds,
                        spec.session_key or "default",
                    )
                    messages.append(build_assistant_message(
                        clean,
                        reasoning_content=response.reasoning_content,
                        thinking_blocks=response.thinking_blocks,
                    ))
                    messages.append(build_completion_check_message())
                    await hook.after_iteration(context)
                    continue

                # Task is confirmed complete — use the last real response, not the meta-summary.
                # on_stream_end was already called in the previous (not-confirmed) iteration.
                if hook.wants_streaming() and _streaming_this_iter:
                    await hook.on_stream_end(context, resuming=False)
                final_content = last_real_response if last_real_response is not None else clean
                context.final_content = final_content
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                break

            if hook.wants_streaming() and _streaming_this_iter:
                await hook.on_stream_end(context, resuming=False)
            messages.append(build_assistant_message(
                clean,
                reasoning_content=response.reasoning_content,
                thinking_blocks=response.thinking_blocks,
            ))
            await self._emit_checkpoint(
                spec,
                {
                    "phase": "final_response",
                    "iteration": iteration,
                    "model": spec.model,
                    "assistant_message": messages[-1],
                    "completed_tool_results": [],
                    "pending_tool_calls": [],
                },
            )
            final_content = clean
            context.final_content = final_content
            context.stop_reason = stop_reason
            await hook.after_iteration(context)
            break
        else:
            stop_reason = "max_iterations"
            if spec.max_iterations_message:
                final_content = spec.max_iterations_message.format(
                    max_iterations=spec.max_iterations,
                )
            else:
                final_content = render_template(
                    "agent/max_iterations_message.md",
                    strip=True,
                    max_iterations=spec.max_iterations,
                )
            self._append_final_message(messages, final_content)

        return AgentRunResult(
            final_content=final_content,
            messages=messages,
            tools_used=tools_used,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
            tool_events=tool_events,
        )

    def _build_request_kwargs(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "tools": tools,
            "model": spec.model,
            "retry_mode": spec.provider_retry_mode,
            "on_retry_wait": spec.progress_callback,
        }
        if spec.temperature is not None:
            kwargs["temperature"] = spec.temperature
        if spec.max_tokens is not None:
            kwargs["max_tokens"] = spec.max_tokens
        if spec.reasoning_effort is not None:
            kwargs["reasoning_effort"] = spec.reasoning_effort
        return kwargs

    async def _request_model(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        hook: AgentHook,
        context: AgentHookContext,
        *,
        skip_stream: bool = False,
    ):
        kwargs = self._build_request_kwargs(
            spec,
            messages,
            tools=spec.tools.get_definitions(),
        )
        if hook.wants_streaming() and not skip_stream:
            async def _stream(delta: str) -> None:
                await hook.on_stream(context, delta)

            return await self.provider.chat_stream_with_retry(
                **kwargs,
                on_content_delta=_stream,
            )
        return await self.provider.chat_with_retry(**kwargs)

    async def _request_finalization_retry(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
        *,
        hook: AgentHook | None = None,
        context: AgentHookContext | None = None,
    ):
        retry_messages = list(messages)
        retry_messages.append(build_finalization_retry_message())
        kwargs = self._build_request_kwargs(spec, retry_messages, tools=None)
        if hook is not None and hook.wants_streaming() and context is not None:
            async def _stream(delta: str) -> None:
                await hook.on_stream(context, delta)
            return await self.provider.chat_stream_with_retry(**kwargs, on_content_delta=_stream)
        return await self.provider.chat_with_retry(**kwargs)

    _XML_SKILL_CALL_RE = re.compile(r"<function=([A-Za-z0-9_:.-]+)\s*>", re.IGNORECASE)

    @classmethod
    def _detect_xml_skill_call(cls, text: str | None) -> str | None:
        """Return the skill name if text contains a bare <function=name> XML invocation, else None."""
        if not text or "<function=" not in text:
            return None
        m = cls._XML_SKILL_CALL_RE.search(text)
        return m.group(1) if m else None

    _INPUT_VALIDATION_ERROR_MARKERS = (
        "input validation error",
        "\"code\":400",
        "'code': 400",
        "400",
    )

    @classmethod
    def _is_input_validation_error(cls, content: str | None) -> bool:
        """Return True if the error looks like a provider 400 / input validation error."""
        err = (content or "").lower()
        return "input validation" in err or (
            "400" in err and ("provider returned error" in err or "invalid" in err)
        )

    @staticmethod
    def _usage_dict(usage: dict[str, Any] | None) -> dict[str, int]:
        if not usage:
            return {}
        result: dict[str, int] = {}
        for key, value in usage.items():
            try:
                result[key] = int(value or 0)
            except (TypeError, ValueError):
                continue
        return result

    @staticmethod
    def _accumulate_usage(target: dict[str, int], addition: dict[str, int]) -> None:
        for key, value in addition.items():
            target[key] = target.get(key, 0) + value

    @staticmethod
    def _merge_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
        merged = dict(left)
        for key, value in right.items():
            merged[key] = merged.get(key, 0) + value
        return merged

    async def _execute_tools(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
        external_lookup_counts: dict[str, int],
    ) -> tuple[list[Any], list[dict[str, str]], BaseException | None]:
        batches = self._partition_tool_batches(spec, tool_calls)
        tool_results: list[tuple[Any, dict[str, str], BaseException | None]] = []
        for batch in batches:
            if spec.concurrent_tools and len(batch) > 1:
                tool_results.extend(await asyncio.gather(*(
                    self._run_tool(spec, tool_call, external_lookup_counts)
                    for tool_call in batch
                )))
            else:
                for tool_call in batch:
                    tool_results.append(await self._run_tool(spec, tool_call, external_lookup_counts))

        results: list[Any] = []
        events: list[dict[str, str]] = []
        fatal_error: BaseException | None = None
        for result, event, error in tool_results:
            results.append(result)
            events.append(event)
            if error is not None and fatal_error is None:
                fatal_error = error
        return results, events, fatal_error

    async def _run_tool(
        self,
        spec: AgentRunSpec,
        tool_call: ToolCallRequest,
        external_lookup_counts: dict[str, int],
    ) -> tuple[Any, dict[str, str], BaseException | None]:
        _hint = "\n\n[Analyze the error above and try a different approach.]"
        lookup_error = repeated_external_lookup_error(
            tool_call.name,
            tool_call.arguments,
            external_lookup_counts,
        )
        if lookup_error:
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": "repeated external lookup blocked",
            }
            if spec.fail_on_tool_error:
                return lookup_error + _hint, event, RuntimeError(lookup_error)
            return lookup_error + _hint, event, None
        prepare_call = getattr(spec.tools, "prepare_call", None)
        tool, params, prep_error = None, tool_call.arguments, None
        if callable(prepare_call):
            try:
                prepared = prepare_call(tool_call.name, tool_call.arguments)
                if isinstance(prepared, tuple) and len(prepared) == 3:
                    tool, params, prep_error = prepared
            except Exception:
                pass
        if prep_error:
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": prep_error.split(": ", 1)[-1][:120],
            }
            return prep_error + _hint, event, RuntimeError(prep_error) if spec.fail_on_tool_error else None
        try:
            if tool is not None:
                result = await tool.execute(**params)
            else:
                result = await spec.tools.execute(tool_call.name, params)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": str(exc),
            }
            if spec.fail_on_tool_error:
                return f"Error: {type(exc).__name__}: {exc}", event, exc
            return f"Error: {type(exc).__name__}: {exc}", event, None

        if isinstance(result, str) and result.startswith("Error"):
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": result.replace("\n", " ").strip()[:120],
            }
            if spec.fail_on_tool_error:
                return result + _hint, event, RuntimeError(result)
            return result + _hint, event, None

        detail = "" if result is None else str(result)
        detail = detail.replace("\n", " ").strip()
        if not detail:
            detail = "(empty)"
        elif len(detail) > 120:
            detail = detail[:120] + "..."
        return result, {"name": tool_call.name, "status": "ok", "detail": detail}, None

    async def _emit_checkpoint(
        self,
        spec: AgentRunSpec,
        payload: dict[str, Any],
    ) -> None:
        callback = spec.checkpoint_callback
        if callback is not None:
            await callback(payload)

    @staticmethod
    def _append_final_message(messages: list[dict[str, Any]], content: str | None) -> None:
        if not content:
            return
        if (
            messages
            and messages[-1].get("role") == "assistant"
            and not messages[-1].get("tool_calls")
        ):
            if messages[-1].get("content") == content:
                return
            messages[-1] = build_assistant_message(content)
            return
        messages.append(build_assistant_message(content))

    def _normalize_tool_result(
        self,
        spec: AgentRunSpec,
        tool_call_id: str,
        tool_name: str,
        result: Any,
    ) -> Any:
        result = ensure_nonempty_tool_result(tool_name, result)
        try:
            content = maybe_persist_tool_result(
                spec.workspace,
                spec.session_key,
                tool_call_id,
                result,
                max_chars=spec.max_tool_result_chars,
            )
        except Exception as exc:
            logger.warning(
                "Tool result persist failed for {} in {}: {}; using raw result",
                tool_call_id,
                spec.session_key or "default",
                exc,
            )
            content = result
        if isinstance(content, str) and len(content) > spec.max_tool_result_chars:
            return truncate_text(content, spec.max_tool_result_chars)
        return content

    @staticmethod
    def _backfill_missing_tool_results(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Insert synthetic error results for orphaned tool_use blocks."""
        declared: list[tuple[int, str, str]] = []  # (assistant_idx, call_id, name)
        fulfilled: set[str] = set()
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        name = ""
                        func = tc.get("function")
                        if isinstance(func, dict):
                            name = func.get("name", "")
                        declared.append((idx, str(tc["id"]), name))
            elif role == "tool":
                tid = msg.get("tool_call_id")
                if tid:
                    fulfilled.add(str(tid))

        missing = [(ai, cid, name) for ai, cid, name in declared if cid not in fulfilled]
        if not missing:
            return messages

        updated = list(messages)
        offset = 0
        for assistant_idx, call_id, name in missing:
            insert_at = assistant_idx + 1 + offset
            while insert_at < len(updated) and updated[insert_at].get("role") == "tool":
                insert_at += 1
            updated.insert(insert_at, {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": _BACKFILL_CONTENT,
            })
            offset += 1
        return updated

    @staticmethod
    def _microcompact(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Replace old compactable tool results with one-line summaries."""
        compactable_indices: list[int] = []
        for idx, msg in enumerate(messages):
            if msg.get("role") == "tool" and msg.get("name") in _COMPACTABLE_TOOLS:
                compactable_indices.append(idx)

        if len(compactable_indices) <= _MICROCOMPACT_KEEP_RECENT:
            return messages

        stale = compactable_indices[: len(compactable_indices) - _MICROCOMPACT_KEEP_RECENT]
        updated: list[dict[str, Any]] | None = None
        for idx in stale:
            msg = messages[idx]
            content = msg.get("content")
            if not isinstance(content, str) or len(content) < _MICROCOMPACT_MIN_CHARS:
                continue
            name = msg.get("name", "tool")
            summary = f"[{name} result omitted from context]"
            if updated is None:
                updated = [dict(m) for m in messages]
            updated[idx]["content"] = summary

        return updated if updated is not None else messages

    def _apply_tool_result_budget(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        updated = messages
        for idx, message in enumerate(messages):
            if message.get("role") != "tool":
                continue
            normalized = self._normalize_tool_result(
                spec,
                str(message.get("tool_call_id") or f"tool_{idx}"),
                str(message.get("name") or "tool"),
                message.get("content"),
            )
            if normalized != message.get("content"):
                if updated is messages:
                    updated = [dict(m) for m in messages]
                updated[idx]["content"] = normalized
        return updated

    def _snip_history(
        self,
        spec: AgentRunSpec,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not messages or not spec.context_window_tokens:
            return messages

        provider_max_tokens = getattr(getattr(self.provider, "generation", None), "max_tokens", 4096)
        max_output = spec.max_tokens if isinstance(spec.max_tokens, int) else (
            provider_max_tokens if isinstance(provider_max_tokens, int) else 4096
        )
        budget = spec.context_block_limit or (
            spec.context_window_tokens - max_output - _SNIP_SAFETY_BUFFER
        )
        if budget <= 0:
            return messages

        estimate, _ = estimate_prompt_tokens_chain(
            self.provider,
            spec.model,
            messages,
            spec.tools.get_definitions(),
        )
        if estimate <= budget:
            return messages

        system_messages = [dict(msg) for msg in messages if msg.get("role") == "system"]
        non_system = [dict(msg) for msg in messages if msg.get("role") != "system"]
        if not non_system:
            return messages

        system_tokens = sum(estimate_message_tokens(msg) for msg in system_messages)
        remaining_budget = max(128, budget - system_tokens)
        kept: list[dict[str, Any]] = []
        kept_tokens = 0
        for message in reversed(non_system):
            msg_tokens = estimate_message_tokens(message)
            if kept and kept_tokens + msg_tokens > remaining_budget:
                break
            kept.append(message)
            kept_tokens += msg_tokens
        kept.reverse()

        if kept:
            for i, message in enumerate(kept):
                if message.get("role") == "user":
                    kept = kept[i:]
                    break
            start = find_legal_message_start(kept)
            if start:
                kept = kept[start:]
        if not kept:
            kept = non_system[-min(len(non_system), 4) :]
            start = find_legal_message_start(kept)
            if start:
                kept = kept[start:]
        return system_messages + kept

    def _partition_tool_batches(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
    ) -> list[list[ToolCallRequest]]:
        if not spec.concurrent_tools:
            return [[tool_call] for tool_call in tool_calls]

        batches: list[list[ToolCallRequest]] = []
        current: list[ToolCallRequest] = []
        for tool_call in tool_calls:
            get_tool = getattr(spec.tools, "get", None)
            tool = get_tool(tool_call.name) if callable(get_tool) else None
            can_batch = bool(tool and tool.concurrency_safe)
            if can_batch:
                current.append(tool_call)
                continue
            if current:
                batches.append(current)
                current = []
            batches.append([tool_call])
        if current:
            batches.append(current)
        return batches

