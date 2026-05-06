"""QualityGate node — LLM-as-judge over the synthesized report.

Model: Gemini 2.5 Flash via OpenRouter (cheap; runs on every retry).

Scoring rubric (each axis 0.0-1.0, total = average):
    coverage           — does the report actually answer the question?
    citation_validity  — does every claim cite a source listed in Sources?
    hallucination_risk — does the report stick to the findings provided?
                         (1.0 = sticks tightly; 0.0 = clearly fabricates)

Verdict routing (mirrors CodeOrch's gate):
    score >= 0.75    → pass     (ship the report)
    0.50 <= s < 0.75 → retry    (re-synthesize, max once)
    score < 0.50     → escalate (human review)
"""

from __future__ import annotations

import json
import re

from agents.models import acall_model
from graph.state import ResearchState


SYSTEM_PROMPT = """You are a quality judge for research reports.

Given a research question, a list of findings (numbered 1..N with URLs),
and a synthesized report, score the report on three axes.

Output strictly this JSON (no markdown fences, no preamble):

{
  "coverage": 0.0-1.0,
  "citation_validity": 0.0-1.0,
  "hallucination_risk": 0.0-1.0,
  "notes": "2-4 sentences explaining the score, citing specific issues if any."
}

Scoring rubric:

1. COVERAGE — does the report answer the question?
   1.0  = answers fully, with multiple supporting sources
   0.7  = covers most of the question; some gaps
   0.4  = partial; major aspects unaddressed
   0.0  = doesn't answer the question

2. CITATION_VALIDITY — does every claim trace to a listed source?
   1.0  = every paragraph cites; every cited number appears in Sources
   0.7  = mostly cited; 1-2 minor uncited claims
   0.4  = many uncited claims; or citations point to wrong sources
   0.0  = effectively no citations or all wrong

3. HALLUCINATION_RISK — does the report stay within the findings?
   1.0  = report content is recoverable from findings; no extrapolation
   0.7  = mostly grounded; minor unsupported framing
   0.4  = several claims that aren't in any finding
   0.0  = report invents facts not present in findings

Be strict. Don't grade on a curve. The verdict matters for routing."""


async def quality_gate(state: ResearchState) -> dict:
    query = state["query"]
    report = state.get("report", "")
    findings = (
        state.get("arxiv_findings", [])
        + state.get("web_findings", [])
        + state.get("deep_read_findings", [])
    )

    if not report:
        return {
            "gate_verdict": "escalate",
            "gate_score": 0.0,
            "gate_notes": "No report was produced.",
        }

    findings_text = "\n".join(
        f"[{i}] {f.get('title', '')} — {f.get('url', '')}"
        for i, f in enumerate(findings, start=1)
    )

    user_msg = (
        f"Research question:\n{query}\n\n"
        f"Findings (1..N):\n{findings_text}\n\n"
        f"Synthesized report:\n{report}\n\n"
        f"Score the report."
    )

    result = await acall_model(
        node="quality_gate",
        system=SYSTEM_PROMPT,
        user=user_msg,
        max_tokens=512,
        response_format={"type": "json_object"},
    )

    parsed = _parse_score(result["content"])
    if parsed is None:
        return {
            "gate_verdict": "escalate",
            "gate_score": 0.0,
            "gate_notes": f"Gate output unparseable: {result['content'][:200]!r}",
        }

    coverage = _clamp(parsed.get("coverage", 0.0))
    citation = _clamp(parsed.get("citation_validity", 0.0))
    halluc = _clamp(parsed.get("hallucination_risk", 0.0))
    score = round((coverage + citation + halluc) / 3.0, 4)

    if score >= 0.75:
        verdict = "pass"
    elif score >= 0.50:
        verdict = "retry"
    else:
        verdict = "escalate"

    notes = (
        f"coverage={coverage:.2f} citation_validity={citation:.2f} "
        f"hallucination_risk={halluc:.2f} composite={score:.2f}\n"
        + parsed.get("notes", "")
    )

    return {
        "gate_verdict": verdict,
        "gate_score": score,
        "gate_notes": notes,
    }


def _parse_score(content: str) -> dict | None:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _clamp(x) -> float:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, f))
