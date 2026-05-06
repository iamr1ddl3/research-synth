"""ArxivAgent node — academic-paper specialist.

Model: GPT-4.1 (OpenAI direct).

For each sub-query: search arxiv → extract titles + abstracts → ask GPT-4.1
to score relevance and emit a structured Finding for each kept paper.
"""

from __future__ import annotations

import json
import re

from agents.models import acall_model
from agents.tools import arxiv_search
from graph.state import Finding, ResearchState


SYSTEM_PROMPT = """You are an arxiv research specialist.

Given a sub-query and a list of candidate arxiv papers (title + abstract),
return ONLY the papers that are directly relevant to the sub-query, with a
relevance score (0.0-1.0) and a 1-3 sentence snippet that explains why this
paper matters for the sub-query.

Output strictly this JSON (no markdown fences, no preamble):

{
  "findings": [
    {
      "index": 0,
      "relevance_score": 0.85,
      "snippet": "Why this paper matters for the sub-query, in 1-3 sentences."
    }
  ]
}

Rules:
- Only include papers with relevance_score >= 0.5
- Reference papers by their `index` in the input list
- Empty findings list is valid (better than fabricating relevance)
- No commentary outside the JSON object."""


async def arxiv_agent(state: ResearchState) -> dict:
    """Process all sub_queries against arxiv. Append findings to state."""
    sub_queries = state.get("sub_queries", [])
    if not sub_queries:
        return {"arxiv_findings": []}

    all_findings: list[Finding] = []
    for sub_q in sub_queries:
        findings = await _process_sub_query(sub_q)
        all_findings.extend(findings)

    return {"arxiv_findings": all_findings}


async def _process_sub_query(sub_query: str) -> list[Finding]:
    candidates = arxiv_search(sub_query, k=5)
    if not candidates:
        return []

    user_msg = (
        f"Sub-query:\n{sub_query}\n\n"
        f"Candidate papers:\n"
        + "\n\n".join(
            f"[{i}] {c['title']}\nAbstract: {c['snippet']}\nURL: {c['url']}"
            for i, c in enumerate(candidates)
        )
    )

    try:
        result = await acall_model(
            node="arxiv_agent",
            system=SYSTEM_PROMPT,
            user=user_msg,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        scored = _parse_findings(result["content"])
    except Exception as e:
        print(f"[arxiv_agent] LLM scoring failed for {sub_query!r}: {e}")
        return []

    findings: list[Finding] = []
    for item in scored:
        idx = item.get("index")
        if not isinstance(idx, int) or idx >= len(candidates):
            continue
        c = candidates[idx]
        findings.append(
            Finding(
                source_type="arxiv",
                source_agent="arxiv_agent",
                title=c["title"],
                url=c["url"],
                snippet=item.get("snippet", "")[:1000],
                raw_content=c["snippet"],   # arxiv abstract is already concise
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
