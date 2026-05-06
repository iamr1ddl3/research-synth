"""WebSearchAgent node — broad web search specialist.

Model: DeepSeek V3 via OpenRouter (cheap summarization at scale).

For each sub-query: DDG search → score relevance → emit structured Findings.
"""

from __future__ import annotations

import json
import re

from agents.models import acall_model
from agents.tools import ddg_search
from graph.state import Finding, ResearchState


SYSTEM_PROMPT = """You are a web research specialist.

Given a sub-query and a list of candidate web results (title + snippet + URL),
return ONLY the results that are directly relevant to the sub-query, with a
relevance score (0.0-1.0) and a 1-2 sentence snippet that explains why this
result matters for the sub-query.

Output strictly this JSON (no markdown fences, no preamble):

{
  "findings": [
    {
      "index": 0,
      "relevance_score": 0.85,
      "snippet": "Why this result matters for the sub-query, in 1-2 sentences."
    }
  ]
}

Rules:
- Only include results with relevance_score >= 0.5
- Prefer authoritative sources (research orgs, official docs, vendor blogs) over content farms
- Reference results by their `index` in the input list
- Empty findings list is valid (don't fabricate relevance)
- No commentary outside the JSON object."""


async def web_search_agent(state: ResearchState) -> dict:
    sub_queries = state.get("sub_queries", [])
    if not sub_queries:
        return {"web_findings": []}

    all_findings: list[Finding] = []
    for sub_q in sub_queries:
        findings = await _process_sub_query(sub_q)
        all_findings.extend(findings)

    return {"web_findings": all_findings}


async def _process_sub_query(sub_query: str) -> list[Finding]:
    candidates = ddg_search(sub_query, k=8)
    if not candidates:
        return []

    user_msg = (
        f"Sub-query:\n{sub_query}\n\n"
        f"Candidate results:\n"
        + "\n\n".join(
            f"[{i}] {c['title']}\nSnippet: {c['snippet']}\nURL: {c['url']}"
            for i, c in enumerate(candidates)
        )
    )

    try:
        result = await acall_model(
            node="web_search_agent",
            system=SYSTEM_PROMPT,
            user=user_msg,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        scored = _parse_findings(result["content"])
    except Exception as e:
        print(f"[web_search_agent] LLM scoring failed for {sub_query!r}: {e}")
        return []

    findings: list[Finding] = []
    for item in scored:
        idx = item.get("index")
        if not isinstance(idx, int) or idx >= len(candidates):
            continue
        c = candidates[idx]
        findings.append(
            Finding(
                source_type="web",
                source_agent="web_search_agent",
                title=c["title"],
                url=c["url"],
                snippet=item.get("snippet", "")[:1000],
                raw_content=c["snippet"],
                relevance_score=float(item.get("relevance_score", 0.0)),
                sub_query=sub_query,
            )
        )

    return findings


def _parse_findings(content: str) -> list[dict]:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    findings = payload.get("findings", [])
    return findings if isinstance(findings, list) else []
