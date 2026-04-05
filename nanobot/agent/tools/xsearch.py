"""X (Twitter) search tool via xAI Responses API."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import (
    ArraySchema,
    BooleanSchema,
    StringSchema,
    tool_parameters_schema,
)

if TYPE_CHECKING:
    from nanobot.config.schema import XSearchConfig

_API_URL = "https://api.x.ai/v1/responses"
_TIMEOUT = 120.0


@tool_parameters(
    tool_parameters_schema(
        query=StringSchema("What to search for on X (Twitter)."),
        handles=ArraySchema(
            StringSchema("X handle without @"),
            description="Limit results to these handles (max 10). Cannot be used with exclude_handles.",
            max_items=10,
            nullable=True,
        ),
        exclude_handles=ArraySchema(
            StringSchema("X handle without @"),
            description="Exclude posts from these handles (max 10). Cannot be used with handles.",
            max_items=10,
            nullable=True,
        ),
        from_date=StringSchema(
            "Only return posts on or after this date (YYYY-MM-DD).", nullable=True
        ),
        to_date=StringSchema(
            "Only return posts on or before this date (YYYY-MM-DD).", nullable=True
        ),
        images=BooleanSchema(description="Enable image understanding in posts."),
        video=BooleanSchema(description="Enable video understanding in posts."),
        required=["query"],
    )
)
class XSearchTool(Tool):
    """Search X (Twitter) posts via the xAI Grok API with real-time x_search access."""

    def __init__(self, config: XSearchConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "x_search"

    @property
    def description(self) -> str:
        return (
            "Search X (Twitter) posts in real time using xAI Grok. "
            "PREFER this over web_search whenever the query involves X/Twitter content, "
            "specific X handles (@user), tweets, market commentary, social media sentiment, "
            "or breaking news that is likely discussed on X."
        )

    @property
    def read_only(self) -> bool:
        return True

    def _api_key(self) -> str:
        return self._config.api_key or os.environ.get("XAI_API_KEY", "")

    def _build_tool_config(
        self,
        handles: list[str] | None,
        exclude_handles: list[str] | None,
        from_date: str | None,
        to_date: str | None,
        images: bool,
        video: bool,
    ) -> dict[str, Any]:
        t: dict[str, Any] = {"type": "x_search"}
        if handles:
            t["allowed_x_handles"] = [h.lstrip("@") for h in handles]
        if exclude_handles:
            t["excluded_x_handles"] = [h.lstrip("@") for h in exclude_handles]
        if from_date:
            t["from_date"] = from_date
        if to_date:
            t["to_date"] = to_date
        if images:
            t["enable_image_understanding"] = True
        if video:
            t["enable_video_understanding"] = True
        return t

    def _format_response(self, data: dict[str, Any], query: str) -> str:
        outputs = data.get("output") or []
        message = next(
            (o for o in outputs if isinstance(o, dict) and o.get("type") == "message"),
            None,
        )
        content_blocks = (message or {}).get("content") or []

        text = "\n\n".join(
            c["text"] for c in content_blocks if isinstance(c, dict) and c.get("text")
        )
        annotations = [
            a
            for c in content_blocks if isinstance(c, dict)
            for a in (c.get("annotations") or []) if isinstance(a, dict)
        ]
        citations = [
            f"- [{a.get('title', a['url'])}]({a['url']})"
            for a in annotations
            if a.get("type") == "url_citation" and a.get("url")
        ]

        status = data.get("status", "unknown")
        if status not in ("completed", "unknown"):
            error = (data.get("error") or {})
            err_msg = error.get("message", "") if isinstance(error, dict) else str(error)
            text = f"Search {status}" + (f": {err_msg}" if err_msg else "") + (f"\n\n{text}" if text else "")

        parts = [f"X search results for: {query}", "", text]
        if citations:
            parts += ["", "**Sources**", *citations]
        return "\n".join(parts).strip()

    async def execute(
        self,
        query: str,
        handles: list[str] | None = None,
        exclude_handles: list[str] | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        images: bool = False,
        video: bool = False,
        **kwargs: Any,
    ) -> str:
        api_key = self._api_key()
        if not api_key:
            return "Error: XAI_API_KEY is not configured. Set it in config (tools.xsearch.apiKey) or as an environment variable."

        if handles and exclude_handles:
            return "Error: handles and exclude_handles cannot be used together."

        tool_config = self._build_tool_config(
            handles, exclude_handles, from_date, to_date, images, video
        )
        body = {
            "model": self._config.model,
            "input": [{"role": "user", "content": query}],
            "tools": [tool_config],
        }

        try:
            import httpx

            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                r = await client.post(
                    _API_URL,
                    json=body,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "User-Agent": "nanobot/x-search",
                    },
                )
                if r.status_code != 200:
                    logger.warning("xAI API error {}: {}", r.status_code, r.text[:200])
                    return f"Error: xAI API returned {r.status_code}: {r.text[:200]}"
                data = r.json()
        except ImportError:
            return "Error: httpx is required for x_search. Install it with: pip install httpx"
        except Exception as exc:
            logger.warning("x_search request failed: {}", exc)
            return f"Error: x_search request failed: {exc}"

        return self._format_response(data, query)
