"""Tests for HtmlunitFetchTool parameter handling and extraction."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.tools.htmlunit import HtmlunitFetchTool
from nanobot.agent.tools.web import WebFetchTool


# ---------------------------------------------------------------------------
# HtmlunitFetchTool — camelCase param aliasing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_htmlunit_camel_case_max_chars_respected():
    """maxChars from LLM (camelCase) must limit output, not be silently ignored."""
    tool = HtmlunitFetchTool()
    html = "<html><body>" + ("x" * 10000) + "</body></html>"

    with patch.object(tool, "_run_scraper", return_value=html):
        result_json = await tool.execute(url="https://example.com", maxChars=500)

    data = json.loads(result_json)
    assert data["truncated"] is True
    assert len(data["text"]) <= 500 + 200  # banner overhead


@pytest.mark.asyncio
async def test_htmlunit_camel_case_js_wait_ms_forwarded():
    """jsWaitMs from LLM (camelCase) must be forwarded to the scraper."""
    tool = HtmlunitFetchTool()

    captured: list[int] = []

    def fake_scraper(url: str, js_wait_ms: int) -> str:
        captured.append(js_wait_ms)
        return "<html><body>done</body></html>"

    with patch.object(tool, "_run_scraper", side_effect=fake_scraper):
        await tool.execute(url="https://example.com", jsWaitMs=8000)

    assert captured == [8000]


@pytest.mark.asyncio
async def test_htmlunit_snake_case_still_works():
    """snake_case params (internal callers) must still work after the fix."""
    tool = HtmlunitFetchTool()

    captured: list[int] = []

    def fake_scraper(url: str, js_wait_ms: int) -> str:
        captured.append(js_wait_ms)
        return "<html><body>ok</body></html>"

    with patch.object(tool, "_run_scraper", side_effect=fake_scraper):
        await tool.execute(url="https://example.com", js_wait_ms=5000, max_chars=200)

    assert captured == [5000]


@pytest.mark.asyncio
async def test_htmlunit_java_not_found_returns_json_error():
    """FileNotFoundError (Java missing) must return a JSON error, not raise."""
    tool = HtmlunitFetchTool()

    with patch.object(tool, "_run_scraper", side_effect=FileNotFoundError("java")):
        result_json = await tool.execute(url="https://example.com")

    data = json.loads(result_json)
    assert "error" in data
    assert "Java" in data["error"] or "java" in data["error"].lower()


@pytest.mark.asyncio
async def test_htmlunit_invalid_url_blocked():
    """Private/invalid URLs must be rejected before hitting the scraper."""
    tool = HtmlunitFetchTool()
    called = False

    def fake_scraper(url, js_wait_ms):
        nonlocal called
        called = True
        return ""

    with patch.object(tool, "_run_scraper", side_effect=fake_scraper):
        result_json = await tool.execute(url="http://169.254.169.254/")

    data = json.loads(result_json)
    assert "error" in data
    assert called is False  # scraper must never be reached


# ---------------------------------------------------------------------------
# WebFetchTool — camelCase param aliasing
# ---------------------------------------------------------------------------

def _patch_web_fetch(tool):
    """Context manager that short-circuits the image-detection HTTP request."""
    import httpx
    from unittest.mock import MagicMock

    mock_response = MagicMock()
    mock_response.url = "https://example.com"
    mock_response.headers = {"content-type": "text/html"}
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    return patch("nanobot.agent.tools.web.httpx.AsyncClient", return_value=mock_client)


@pytest.mark.asyncio
async def test_web_fetch_camel_case_max_chars_respected():
    """maxChars from LLM (camelCase) must be forwarded to _fetch_readability, not ignored."""
    tool = WebFetchTool()

    captured: list[int] = []

    async def fake_readability(url, extract_mode, max_chars):
        captured.append(max_chars)
        return json.dumps({"url": url, "truncated": False, "length": 5, "text": "hello"})

    with _patch_web_fetch(tool):
        with patch.object(tool, "_fetch_readability", side_effect=fake_readability):
            with patch.object(tool, "_fetch_jina", new_callable=AsyncMock, return_value=None):
                await tool.execute(url="https://example.com", maxChars=500)

    assert captured == [500]


@pytest.mark.asyncio
async def test_web_fetch_camel_case_extract_mode_forwarded():
    """extractMode from LLM (camelCase) must override the default 'markdown'."""
    tool = WebFetchTool()

    captured_modes: list[str] = []

    async def fake_readability(url, extract_mode, max_chars):
        captured_modes.append(extract_mode)
        return "some content"

    with _patch_web_fetch(tool):
        with patch.object(tool, "_fetch_readability", side_effect=fake_readability):
            with patch.object(tool, "_fetch_jina", new_callable=AsyncMock, return_value=None):
                await tool.execute(url="https://example.com", extractMode="text")

    assert captured_modes == ["text"]
