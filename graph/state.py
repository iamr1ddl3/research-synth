"""LangGraph state schema for research-synth.

The state is a single TypedDict that flows through every node. LangGraph's
state-merge semantics: each node returns a partial dict; LangGraph merges
it into the running state. For lists we use Annotated[..., operator.add]
so parallel branches accumulate findings instead of overwriting.

Node-by-node, this state evolves like:

    Decomposer:        sub_queries = [..., ..., ...]
    ArxivAgent:        arxiv_findings = [Finding, Finding, ...]
    WebSearchAgent:    web_findings = [Finding, ...]
    DeepReadAgent:     deep_read_findings = [Finding, ...]
    Synthesizer:       report = "..."
    QualityGate:       gate_verdict = "pass" | "retry" | "escalate"
                       gate_score = 0.0-1.0
                       gate_notes = "..."
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict
from uuid import UUID


class Finding(TypedDict, total=False):
    """One piece of evidence retrieved by a research agent.

    Each Finding becomes a citation in the synthesized report. The agent
    fields tell us which agent produced it (for the Sources section), and
    `score` is the agent's self-rated relevance (0-1).
    """
    source_type: Literal["arxiv", "web", "deep_read"]
    source_agent: str          # one of: arxiv_agent, web_search_agent, deep_read_agent
    title: str
    url: str
    snippet: str               # 1-3 sentence summary
    raw_content: str           # full text the agent based the snippet on (may be long)
    relevance_score: float     # 0.0-1.0, agent's self-assessment
    sub_query: str             # which sub-query produced this


class ResearchState(TypedDict, total=False):
    """The single state object that flows through the LangGraph.

    Required-on-entry: run_id, query.
    Everything else is populated by nodes as they run.
    """
    # === Provided by caller ===
    run_id: str                # UUID4 string; used as Langfuse session_id
    query: str                 # user's research question

    # === Decomposer outputs ===
    sub_queries: list[str]     # 3-5 focused sub-queries

    # === Parallel research-agent outputs ===
    # Annotated with operator.add so parallel branches APPEND rather than
    # overwrite. Each agent returns a list of Findings.
    arxiv_findings: Annotated[list[Finding], operator.add]
    web_findings: Annotated[list[Finding], operator.add]
    deep_read_findings: Annotated[list[Finding], operator.add]

    # === Synthesizer output ===
    report: str                # markdown-formatted synthesized report
    citations: list[str]       # ordered list of URLs cited

    # === Quality Gate output ===
    gate_verdict: Literal["pass", "retry", "escalate"]
    gate_score: float          # 0.0-1.0
    gate_notes: str            # human-readable critique

    # === Loop control ===
    retry_count: int           # incremented on retry, capped at 1

    # === Bookkeeping ===
    error: str                 # set if any node failed catastrophically
