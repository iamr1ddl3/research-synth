"""Decomposer node — splits a research query into 3-5 focused sub-queries.

Model: Gemini 2.5 Flash via OpenRouter (cheapest in the lineup; mechanical
sub-query generation doesn't need a premium model).

Output contract: a JSON object with a "sub_queries" array of 3-5 strings.
Each sub-query should be focused enough that a single research agent can
make progress on it independently.
"""

from __future__ import annotations

import json
import re

from agents.models import acall_model
from graph.state import ResearchState


SYSTEM_PROMPT = """You are a research query decomposer.

Given a user's research question, split it into 3-5 focused sub-queries
that can be researched independently and in parallel. Each sub-query
should:

1. Be specific enough that a single targeted search will find relevant sources
2. Cover a DIFFERENT angle of the original question (no overlap)
3. Together fully cover the original question (no gaps)

Output strictly this JSON (no markdown fences, no preamble):

{
  "sub_queries": [
    "first sub-query string",
    "second sub-query string",
    "third sub-query string"
  ]
}

3 sub-queries minimum, 5 maximum. No commentary."""


async def decompose(state: ResearchState) -> dict:
    query = state["query"]
    user_msg = f"Research question:\n\n{query}\n\nDecompose."

    result = await acall_model(
        node="decomposer",
        system=SYSTEM_PROMPT,
        user=user_msg,
        max_tokens=512,
        response_format={"type": "json_object"},
    )

    sub_queries = _parse_sub_queries(result["content"])
    return {"sub_queries": sub_queries}


def _parse_sub_queries(content: str) -> list[str]:
    """Tolerant JSON parse — strip markdown fences if the model added them."""
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Decomposer returned non-JSON output: {content[:200]!r}"
        ) from e

    sub_queries = payload.get("sub_queries", [])
    if not isinstance(sub_queries, list) or not (3 <= len(sub_queries) <= 5):
        raise ValueError(
            f"Decomposer must return 3-5 sub_queries; got {len(sub_queries)}"
        )
    if not all(isinstance(q, str) and q.strip() for q in sub_queries):
        raise ValueError("All sub_queries must be non-empty strings")

    return [q.strip() for q in sub_queries]
