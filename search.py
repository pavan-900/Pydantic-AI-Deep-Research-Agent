"""DuckDuckGo search and page fetching utilities."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (compatible; PydanticDeepResearch/1.0; +https://ai.pydantic.dev)"
)
FETCH_TIMEOUT = 15.0
MAX_PAGE_CHARS = 6_000


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


def _ddg_search_sync(query: str, max_results: int) -> list[SearchResult]:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS  # noqa: F401

    with DDGS() as ddgs:
        rows = list(ddgs.text(query, max_results=max_results))
    results: list[SearchResult] = []
    for row in rows:
        title = (row.get("title") or "").strip()
        url = (row.get("href") or row.get("link") or "").strip()
        snippet = (row.get("body") or row.get("snippet") or "").strip()
        if title and url:
            results.append(SearchResult(title=title, url=url, snippet=snippet))
    return results


async def ddg_search(query: str, max_results: int = 8) -> list[SearchResult]:
    return await asyncio.to_thread(_ddg_search_sync, query, max_results)


async def fetch_page_text(url: str, max_chars: int = MAX_PAGE_CHARS) -> str | None:
    if not url.startswith(("http://", "https://")):
        return None
    if re.search(r"\.(pdf|zip|png|jpg|jpeg|gif)(\?|$)", url, re.I):
        return None

    headers = {"User-Agent": USER_AGENT}
    try:
        async with httpx.AsyncClient(
            timeout=FETCH_TIMEOUT,
            follow_redirects=True,
            headers=headers,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except Exception:
        return None

    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type and "text/plain" not in content_type:
        return None

    if "text/plain" in content_type:
        text = response.text
    else:
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)

    text = " ".join(text.split())
    return text[:max_chars] if text else None


async def fetch_pages(urls: list[str], limit: int = 3) -> dict[str, str | None]:
    sem = asyncio.Semaphore(4)

    async def _one(url: str) -> tuple[str, str | None]:
        async with sem:
            return url, await fetch_page_text(url)

    pairs = await asyncio.gather(*[_one(u) for u in urls[:limit]])
    return dict(pairs)


def format_results_for_llm(
    results: list[SearchResult],
    page_texts: dict[str, str | None] | None = None,
) -> str:
    blocks: list[str] = []
    for i, r in enumerate(results, 1):
        block = (
            f"[{i}] {r.title}\n"
            f"URL: {r.url}\n"
            f"Snippet: {r.snippet}"
        )
        if page_texts and r.url in page_texts and page_texts[r.url]:
            block += f"\nPage excerpt: {page_texts[r.url][:2500]}"
        blocks.append(block)
    return "\n\n".join(blocks) if blocks else "(no search results)"
