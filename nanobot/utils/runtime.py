"""Runtime-specific helper functions and constants."""

from __future__ import annotations

from typing import Any

from loguru import logger

from nanobot.utils.helpers import stringify_text_blocks

_MAX_REPEAT_EXTERNAL_LOOKUPS = 2

EMPTY_FINAL_RESPONSE_MESSAGE = (
    "I completed the tool steps but couldn't produce a final answer. "
    "Please try again or narrow the task."
)

FINALIZATION_RETRY_PROMPT = (
    "Please provide your response to the user based on the conversation above."
)

LENGTH_RECOVERY_PROMPT = (
    "Output limit reached. Continue exactly where you left off "
    "— no recap, no apology. Break remaining work into smaller steps if needed."
)

COMPLETION_CHECK_PROMPT = (
    "Review the original task and all tool results above. "
    "Is the task FULLY complete? If not, briefly state what remains and call tools to finish it. "
    "If fully complete, start your response with DONE: followed by the final answer."
)

CONTEXT_COMPACTION_PROMPT = (
    "Summarize the following conversation history into a concise summary. "
    "Include: the original task, key decisions made, tool results obtained, "
    "and current progress state. Omit redundant details and verbose outputs. "
    "Keep all important facts, code snippets, and file paths. "
    "Format as a structured summary with clear sections."
)


def empty_tool_result_message(tool_name: str) -> str:
    """Short prompt-safe marker for tools that completed without visible output."""
    return f"({tool_name} completed with no output)"


def ensure_nonempty_tool_result(tool_name: str, content: Any) -> Any:
    """Replace semantically empty tool results with a short marker string."""
    if content is None:
        return empty_tool_result_message(tool_name)
    if isinstance(content, str) and not content.strip():
        return empty_tool_result_message(tool_name)
    if isinstance(content, list):
        if not content:
            return empty_tool_result_message(tool_name)
        text_payload = stringify_text_blocks(content)
        if text_payload is not None and not text_payload.strip():
            return empty_tool_result_message(tool_name)
    return content


def is_blank_text(content: str | None) -> bool:
    """True when *content* is missing or only whitespace."""
    return content is None or not content.strip()


def build_finalization_retry_message() -> dict[str, str]:
    """A short no-tools-allowed prompt for final answer recovery."""
    return {"role": "user", "content": FINALIZATION_RETRY_PROMPT}


def build_completion_check_message() -> dict[str, str]:
    """A prompt that asks the LLM to verify task completion."""
    return {"role": "user", "content": COMPLETION_CHECK_PROMPT}


def is_completion_confirmed(content: str | None) -> tuple[bool, str | None]:
    """Check if the LLM confirmed task completion with DONE: prefix.

    Returns (True, final_answer) if confirmed, (False, remaining_work) if not.
    """
    if content is None:
        return False, None
    stripped = content.strip()
    if stripped.upper().startswith("DONE:"):
        answer = stripped[5:].strip()
        return True, answer if answer else stripped
    if stripped.upper().startswith("DONE"):
        rest = stripped[4:].strip().lstrip(":").strip()
        return True, rest if rest else stripped
    return False, stripped


def external_lookup_signature(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Stable signature for repeated external lookups we want to throttle."""
    if tool_name == "web_fetch":
        url = str(arguments.get("url") or "").strip()
        if url:
            return f"web_fetch:{url.lower()}"
    if tool_name == "web_search":
        query = str(arguments.get("query") or arguments.get("search_term") or "").strip()
        if query:
            return f"web_search:{query.lower()}"
    return None


def repeated_external_lookup_error(
    tool_name: str,
    arguments: dict[str, Any],
    seen_counts: dict[str, int],
) -> str | None:
    """Block repeated external lookups after a small retry budget."""
    signature = external_lookup_signature(tool_name, arguments)
    if signature is None:
        return None
    count = seen_counts.get(signature, 0) + 1
    seen_counts[signature] = count
    if count <= _MAX_REPEAT_EXTERNAL_LOOKUPS:
        return None
    logger.warning(
        "Blocking repeated external lookup {} on attempt {}",
        signature[:160],
        count,
    )
    return (
        "Error: repeated external lookup blocked. "
        "Use the results you already have to answer, or try a meaningfully different source."
    )
