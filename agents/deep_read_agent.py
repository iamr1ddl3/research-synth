"""DeepReadAgent node — fetches and synthesizes long-form rendered articles.

Model: Gemini 2.5 Pro via OpenRouter (1M context — no chunking needed).

Strategy:
  1. After WebSearchAgent runs, look at its top-N highest-relevance findings
  2. Fetch the full rendered page via Browserbase for each
  3. Pass the long-form content to Gemini 2.5 Pro for deep synthesis
  4. Emit one Finding per deep-read URL

This is what justifies Browserbase's place in the stack: when a DDG snippet
is thin but the source is high-signal, we render the full page.
"""

from __future__ import annotations

import json
import re

from agents.models import acall_model
from agents.tools import browserbase_fetch
from graph.state import Finding, ResearchState


SYSTEM_PROMPT = """You are a long-form article analyst.

Given a sub-query and the full text of a rendered web article, produce a
detailed structured summary that captures the key claims, evidence, and
specific techniques mentioned in the article that are relevant to the
sub-query.

Output strictly this JSON (no markdown fences, no preamble):

{
  "relevance_score": 0.0-1.0,
  "snippet": "A detailed 3-5 sentence summary tied directly to the sub-query.",
  "key_claims": ["claim 1", "claim 2", "claim 3"]
}

Rules:
- relevance_score < 0.5 means we should drop this article
- snippet must reference specifics from the article (techniques, names, numbers, etc.)
- key_claims: 2-5 specific claims that someone could verify or quote
- No commentary outside the JSON object."""


# Limit how many URLs we deep-read per run. Browserbase free tier is 1000
# fetches/mo total; we keep this conservative so a smoke run doesn't burn
# the whole budget.
MAX_DEEP_READS_PER_RUN = 3


def _is_deep_read_friendly(url: str) -> bool:
    """Heuristic filter — skip URLs that won't yield useful long-form text.

    PDF URLs and arxiv landing pages (/abs/) have very thin Browserbase output
    even though they look authoritative. Prefer HTML article variants.
    """
    u = url.lower()
    if u.endswith(".pdf"):
        return False
    if "arxiv.org/abs/" in u:
        return False  # use the /html/ variant or skip
    return True


async def deep_read_agent(state: ResearchState) -> dict:
    """Pick top URLs from web_findings, fetch with Browserbase, synthesize."""
    web_findings = state.get("web_findings", [])
    print(f"[deep_read_agent] received {len(web_findings)} web_findings to triage")
    if not web_findings:
        return {"deep_read_findings": []}

    # Pick top URLs by relevance, dedup by URL
    sorted_web = sorted(
        web_findings,
        key=lambda f: f.get("relevance_score", 0.0),
        reverse=True,
    )
    seen_urls: set[str] = set()
    targets: list[Finding] = []
    for f in sorted_web:
        url = f.get("url", "")
        if not url or url in seen_urls:
            continue
        if not _is_deep_read_friendly(url):
            continue
        seen_urls.add(url)
        targets.append(f)
        if len(targets) >= MAX_DEEP_READS_PER_RUN:
            break

    if not targets:
        print(f"[deep_read_agent] no targets after dedup")
        return {"deep_read_findings": []}

    print(f"[deep_read_agent] deep-reading {len(targets)} URL(s):")
    for t in targets:
        print(f"  - {t.get('url', '')[:80]}")

    findings: list[Finding] = []
    for target in targets:
        deep = await _deep_read_one(target)
        if deep is not None:
            findings.append(deep)
            print(f"  ✓ deep-read OK: {target.get('url', '')[:80]} (score={deep.get('relevance_score', 0):.2f})")
        else:
            print(f"  ✗ deep-read FAILED or low-score: {target.get('url', '')[:80]}")

    print(f"[deep_read_agent] produced {len(findings)} findings")
    return {"deep_read_findings": findings}


async def _deep_read_one(target: Finding) -> Finding | None:
    url = target.get("url", "")
    sub_query = target.get("sub_query", "")
    if not url:
        return None

    page_text = await browserbase_fetch(url, max_chars=20000)
    if not page_text:
        print(f"  [deep_read] empty fetch: {url[:80]}")
        return None
    if len(page_text) < 200:
        print(f"  [deep_read] page too thin ({len(page_text)} chars): {url[:80]}")
        return None

    user_msg = (
        f"Sub-query:\n{sub_query}\n\n"
        f"Article URL: {url}\n"
        f"Article title: {target.get('title', '')}\n\n"
        f"Full article text:\n{page_text}"
    )

    try:
        result = await acall_model(
            node="deep_read_agent",
            system=SYSTEM_PROMPT,
            user=user_msg,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        parsed = _parse_synthesis(result["content"])
    except Exception as e:
        print(f"[deep_read_agent] LLM synthesis failed for {url!r}: {e}")
        return None

    if parsed is None:
        # Truncated or malformed JSON — usually a max_tokens cap on a long
        # snippet. Surfacing this loudly so it doesn't fail silently again.
        preview = (result.get("content") or "")[:120].replace("\n", " ")
        print(f"  [deep_read] JSON parse failed for {url[:70]}: {preview!r}")
        return None

    score = float(parsed.get("relevance_score", 0.0))
    if score < 0.5:
        print(f"  [deep_read] low score {score:.2f}: {url[:80]}")
        return None

    snippet = parsed.get("snippet", "")[:1500]
    key_claims = parsed.get("key_claims", [])
    if isinstance(key_claims, list) and key_claims:
        snippet += "\n\nKey claims:\n" + "\n".join(
            f"- {c}" for c in key_claims if isinstance(c, str)
        )

    return Finding(
        source_type="deep_read",
        source_agent="deep_read_agent",
        title=target.get("title", url),
        url=url,
        snippet=snippet[:2000],
        raw_content=page_text[:5000],
        relevance_score=score,
        sub_query=sub_query,
    )


def _parse_synthesis(content: str) -> dict | None:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None
