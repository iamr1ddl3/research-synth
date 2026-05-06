"""Synthesizer node — merges all findings into a structured research report.

Model: Claude Sonnet 4.6 (Anthropic direct).

Reads the accumulated arxiv + web + deep_read findings and produces a
markdown-formatted report with citations. Sonnet is used here because
long-form structured writing with citations is its strongest task.
"""

from __future__ import annotations

import re

from agents.models import acall_model
from graph.state import Finding, ResearchState


SYSTEM_PROMPT = """You are a research synthesizer.

Given a research question and a body of findings from multiple sources
(arxiv papers, web articles, deep-read articles), produce a structured
markdown report that:

1. Directly answers the research question
2. Organizes findings into 3-6 thematic sections (you choose section names
   based on what the findings actually cover)
3. Cites EVERY claim using the inline format [N] where N is the 1-indexed
   citation number
4. Includes a final "## Sources" section listing all citations in order:
       [1] Title — URL (source_type)
5. Does NOT introduce facts that aren't in the findings
6. Does NOT speculate beyond what sources support
7. Notes disagreements between sources where they exist

Format guide:
    # Research Report: <one-line restatement of the question>

    <2-3 sentence executive summary tied directly to the question>

    ## <Theme 1>
    Findings... [1][3]

    ## <Theme 2>
    Findings... [2]

    ## Sources
    [1] Title — URL (arxiv)
    [2] ...

Quality bar: every paragraph cites at least one source. Every source listed
in ## Sources must appear in at least one inline citation. No filler."""


def _format_findings(findings: list[Finding]) -> tuple[str, list[Finding]]:
    """Render findings as a numbered list for the prompt; return mapping."""
    # Dedup by URL while preserving order; combine snippets if same URL appears
    # multiple times across agents (rare but possible).
    seen: dict[str, Finding] = {}
    for f in findings:
        url = f.get("url", "")
        if not url:
            continue
        if url not in seen:
            seen[url] = dict(f)
        else:
            # Combine snippets; keep highest relevance score
            existing = seen[url]
            existing["snippet"] = (
                existing.get("snippet", "") + " | " + f.get("snippet", "")
            )[:2000]
            existing["relevance_score"] = max(
                existing.get("relevance_score", 0.0),
                f.get("relevance_score", 0.0),
            )

    # Sort by relevance descending so the strongest evidence is upfront
    ordered = sorted(seen.values(), key=lambda f: f.get("relevance_score", 0.0), reverse=True)

    rendered = []
    for i, f in enumerate(ordered, start=1):
        rendered.append(
            f"[{i}] {f.get('title', '')} ({f.get('source_type', '')})\n"
            f"    URL: {f.get('url', '')}\n"
            f"    Sub-query: {f.get('sub_query', '')}\n"
            f"    Relevance: {f.get('relevance_score', 0.0):.2f}\n"
            f"    Snippet: {f.get('snippet', '')}"
        )
    return "\n\n".join(rendered), ordered


async def synthesizer(state: ResearchState) -> dict:
    query = state["query"]
    arxiv = state.get("arxiv_findings", [])
    web = state.get("web_findings", [])
    deep = state.get("deep_read_findings", [])

    all_findings = arxiv + web + deep
    if not all_findings:
        return {
            "report": "# Research Report\n\n_No findings were retrieved. Search tools or LLM agents may have failed for this query._",
            "citations": [],
        }

    findings_text, ordered = _format_findings(all_findings)

    user_msg = (
        f"Research question:\n{query}\n\n"
        f"Findings (numbered for citation; sorted by relevance):\n\n{findings_text}\n\n"
        f"Synthesize a structured markdown report following the format guide."
    )

    # Retry-on-error path is simple — if the synthesizer fails, the gate
    # will mark this run as failed and the graph escalates.
    result = await acall_model(
        node="synthesizer",
        system=SYSTEM_PROMPT,
        user=user_msg,
        max_tokens=4096,
    )

    report = result["content"].strip()
    citations = [f.get("url", "") for f in ordered]
    return {"report": report, "citations": citations}
