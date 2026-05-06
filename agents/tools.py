"""Search and fetch tools used by research agents.

Three independent functions, all returning a list of (title, url, snippet)
tuples. Agents synthesize over these snippets via LLM calls.

    ddg_search(query, k)        — DuckDuckGo web search (free, no key)
    arxiv_search(query, k)      — arxiv academic search (free, no key)
    browserbase_fetch(url)      — Browserbase rendered-page fetch
                                  (used by DeepRead for JS-heavy articles)

Each tool is fault-tolerant — returns [] on failure rather than raising,
so a flaky search doesn't break the entire graph.
"""

from __future__ import annotations

import os
from typing import TypedDict


class SearchResult(TypedDict):
    title: str
    url: str
    snippet: str


# ---------------------------------------------------------------------------
# DuckDuckGo
# ---------------------------------------------------------------------------

def ddg_search(query: str, k: int = 8) -> list[SearchResult]:
    """Web search via DuckDuckGo. Returns up to k results."""
    try:
        from ddgs import DDGS
    except ImportError:
        # duckduckgo-search package now imports as ddgs
        from duckduckgo_search import DDGS  # type: ignore

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=k))
        return [
            SearchResult(
                title=r.get("title", "")[:300],
                url=r.get("href", "") or r.get("url", ""),
                snippet=r.get("body", "")[:600],
            )
            for r in raw
            if (r.get("href") or r.get("url"))
        ]
    except Exception as e:
        print(f"[ddg_search] failed for {query!r}: {e}")
        return []


# ---------------------------------------------------------------------------
# arxiv
# ---------------------------------------------------------------------------

def arxiv_search(query: str, k: int = 5) -> list[SearchResult]:
    """Academic search via arxiv API. Returns up to k results."""
    import arxiv

    try:
        client = arxiv.Client(page_size=k, delay_seconds=2.0, num_retries=2)
        search = arxiv.Search(
            query=query,
            max_results=k,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        for paper in client.results(search):
            results.append(
                SearchResult(
                    title=paper.title.strip()[:300],
                    url=paper.entry_id,
                    snippet=(paper.summary or "").replace("\n", " ").strip()[:1200],
                )
            )
        return results
    except Exception as e:
        print(f"[arxiv_search] failed for {query!r}: {e}")
        return []


# ---------------------------------------------------------------------------
# Browserbase
# ---------------------------------------------------------------------------

async def browserbase_fetch(url: str, max_chars: int = 20000) -> str:
    """Fetch a fully-rendered page via Browserbase. Returns extracted text.

    Only used for high-signal URLs where DDG's snippet is too thin and the
    page is JS-heavy enough that requests+beautifulsoup wouldn't work.

    Async because the function is called from inside the LangGraph asyncio
    loop — sync_playwright() raises 'sync API inside asyncio loop' there.

    Sessions are always REQUEST_RELEASE'd in a finally block — Browserbase's
    free tier caps at 3 concurrent sessions, and orphaned sessions block
    subsequent runs. Releasing on every exit (success or failure) keeps the
    pool clean.
    """
    api_key = os.environ.get("BROWSERBASE_API_KEY")
    project_id = os.environ.get("BROWSERBASE_PROJECT_ID")
    if not (api_key and project_id):
        return ""

    from browserbase import Browserbase
    from playwright.async_api import async_playwright

    bb = Browserbase(api_key=api_key)
    session = None

    try:
        session = bb.sessions.create(project_id=project_id)

        async with async_playwright() as p:
            browser = await p.chromium.connect_over_cdp(session.connect_url)
            try:
                context = browser.contexts[0] if browser.contexts else await browser.new_context()
                page = context.pages[0] if context.pages else await context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                # Wait briefly for JS to populate; don't block on networkidle.
                try:
                    await page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    pass
                # Extract text content; trim aggressive whitespace.
                text = await page.evaluate("() => document.body.innerText") or ""
                # Collapse runs of whitespace
                lines = [ln.strip() for ln in text.splitlines()]
                lines = [ln for ln in lines if ln]
                joined = "\n".join(lines)
                return joined[:max_chars]
            finally:
                await browser.close()
    except Exception as e:
        print(f"[browserbase_fetch] failed for {url!r}: {e}")
        return ""
    finally:
        # Always release the Browserbase session so we don't hold a slot.
        if session is not None:
            try:
                bb.sessions.update(
                    session.id, status="REQUEST_RELEASE", project_id=project_id
                )
            except Exception:
                pass
