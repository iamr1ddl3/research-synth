"""LangGraph state machine for research-synth.

Graph topology:

    START
      ↓
    decomposer
      ↓
    [parallel branches via fan-out edges]
      ├── arxiv_agent
      ├── web_search_agent
      └──> deep_read_agent (runs AFTER web_search since it needs web_findings)
      ↓ (join)
    synthesizer
      ↓
    quality_gate
      ↓
    [conditional edge]
      ├── verdict == "pass"     → END
      ├── verdict == "retry"    → synthesizer (loop, retry_count++)
      └── verdict == "escalate" → END (with escalate state)

Why this shape (not the simpler "all three in pure parallel"):

DeepReadAgent needs web_search_agent's output to pick which URLs to fetch.
So we run arxiv_agent and web_search_agent truly in parallel, then run
deep_read_agent right after web_search_agent finishes. LangGraph models
this naturally via fan-out edges from decomposer → arxiv + web, and a
sequential edge web → deep_read.

The retry edge from quality_gate → synthesizer caps at retry_count <= 1
to avoid infinite loops. After one retry, a still-failing report goes to
escalate.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.arxiv_agent import arxiv_agent
from agents.decomposer import decompose
from agents.deep_read_agent import deep_read_agent
from agents.quality_gate import quality_gate
from agents.synthesizer import synthesizer
from agents.web_search_agent import web_search_agent
from graph.state import ResearchState


MAX_RETRIES = 1


def _route_after_gate(state: ResearchState) -> str:
    """Conditional edge from quality_gate.

    Returns the next node name, or END.
    """
    verdict = state.get("gate_verdict", "escalate")
    retries = state.get("retry_count", 0)

    if verdict == "pass":
        return "done"
    if verdict == "retry" and retries < MAX_RETRIES:
        return "retry"
    return "done"  # escalate or retries exhausted both terminate the graph


def _bump_retry(state: ResearchState) -> dict:
    """Synchronous helper node — increments retry_count between gate and re-synth."""
    return {"retry_count": state.get("retry_count", 0) + 1}


def build_graph():
    """Compile the LangGraph state machine. Returns the compiled app."""
    g = StateGraph(ResearchState)

    # === Nodes ===
    g.add_node("decomposer", decompose)
    g.add_node("arxiv_agent", arxiv_agent)
    g.add_node("web_search_agent", web_search_agent)
    g.add_node("deep_read_agent", deep_read_agent)
    g.add_node("synthesizer", synthesizer)
    g.add_node("quality_gate", quality_gate)
    g.add_node("retry_bump", _bump_retry)

    # === Edges ===
    g.add_edge(START, "decomposer")

    # Fan out from decomposer to arxiv (parallel) and web (sequential gateway
    # for deep_read). Both write to different state keys (arxiv_findings vs
    # web_findings) so concurrent updates don't collide.
    g.add_edge("decomposer", "arxiv_agent")
    g.add_edge("decomposer", "web_search_agent")

    # deep_read runs AFTER web_search (needs its findings to pick URLs).
    g.add_edge("web_search_agent", "deep_read_agent")

    # All three research agents converge on synthesizer. LangGraph fans in
    # automatically when all incoming edges have completed.
    g.add_edge("arxiv_agent", "synthesizer")
    g.add_edge("deep_read_agent", "synthesizer")

    g.add_edge("synthesizer", "quality_gate")

    # Conditional routing from gate
    g.add_conditional_edges(
        "quality_gate",
        _route_after_gate,
        {
            "retry": "retry_bump",
            "done": END,
        },
    )

    # Retry path: bump counter then back to synthesizer
    g.add_edge("retry_bump", "synthesizer")

    return g.compile()
