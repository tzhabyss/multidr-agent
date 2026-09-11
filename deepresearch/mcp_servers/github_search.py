"""SageResearch-compatible GitHub search MCP server."""

from __future__ import annotations

import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("deepresearch-github")
GITHUB_API = os.getenv("DEEPRESEARCH_GITHUB_API_URL", "https://api.github.com")


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "deepresearch-agent"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _limit(value: int) -> int:
    return max(1, min(int(value), 30))


@mcp.tool(name="search_repositories", structured_output=True)
async def search_repositories(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Search GitHub repositories by name, topic, or description."""

    query = query.strip()
    if not query:
        return []
    async with httpx.AsyncClient(timeout=20, trust_env=False, headers=_headers()) as client:
        response = await client.get(
            f"{GITHUB_API.rstrip('/')}/search/repositories",
            params={"q": query, "per_page": _limit(max_results), "sort": "stars", "order": "desc"},
        )
        response.raise_for_status()
        payload = response.json()
    return [
        {
            "title": str(item.get("full_name") or item.get("name") or ""),
            "url": str(item.get("html_url") or ""),
            "description": str(item.get("description") or ""),
            "stars": item.get("stargazers_count", 0),
            "language": item.get("language"),
            "provider": "github",
        }
        for item in payload.get("items", [])[: _limit(max_results)]
        if item.get("html_url")
    ]


@mcp.tool(name="search_code", structured_output=True)
async def search_code(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """Search code across GitHub repositories; a token may be required by GitHub."""

    query = query.strip()
    if not query:
        return []
    async with httpx.AsyncClient(timeout=20, trust_env=False, headers=_headers()) as client:
        response = await client.get(
            f"{GITHUB_API.rstrip('/')}/search/code",
            params={"q": query, "per_page": _limit(max_results)},
        )
        response.raise_for_status()
        payload = response.json()
    return [
        {
            "title": str(item.get("name") or ""),
            "url": str(item.get("html_url") or ""),
            "description": str(item.get("path") or ""),
            "repository": str((item.get("repository") or {}).get("full_name") or ""),
            "provider": "github",
        }
        for item in payload.get("items", [])[: _limit(max_results)]
        if item.get("html_url")
    ]


if __name__ == "__main__":
    mcp.run(transport="stdio")
