"""research-synth Streamlit report viewer.

Reads the run-state JSON files written by scripts/run_query.py and renders:
  - The synthesized report (markdown)
  - Per-vendor / per-node usage table (proves multi-vendor routing happened)
  - Findings breakdown by source (arxiv / web / deep_read)
  - Quality-gate verdict + score

Not connected to pgvector yet. Each completed run leaves a `reports/<name>.json`
file; this dashboard discovers them and lets you toggle between runs.

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Repo root on sys.path so we can import the routing table for the header
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


st.set_page_config(
    page_title="research-synth — report viewer",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 research-synth")
st.caption(
    "LangGraph multi-agent research synthesizer · multi-vendor routing "
    "(Anthropic + OpenAI + OpenRouter[Gemini/DeepSeek]) · Browserbase deep-read"
)


# ---------------------------------------------------------------------------
# Sidebar: pick a run
# ---------------------------------------------------------------------------

run_files = sorted(REPORTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
if not run_files:
    st.warning(
        "No runs found. Execute `python -m scripts.run_query \"<query>\"` "
        "from the project root to produce one."
    )
    st.stop()

run_choice = st.sidebar.selectbox(
    "Pick a run",
    run_files,
    format_func=lambda p: f"{p.stem} ({p.stat().st_mtime:.0f})",
)
state = json.loads(run_choice.read_text())


# ---------------------------------------------------------------------------
# Top: query + verdict + score
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)
verdict = state.get("gate_verdict", "?")
verdict_emoji = {"pass": "✅", "retry": "🟡", "escalate": "🔴"}.get(verdict, "❓")
col1.metric("Verdict", f"{verdict_emoji} {verdict}")
col2.metric("Gate score", f"{state.get('gate_score', 0):.2f}")
col3.metric("Sub-queries", len(state.get("sub_queries", [])))
col4.metric("Retries", state.get("retry_count", 0))

st.subheader("Query")
st.write(f"**{state.get('query', '')}**")

with st.expander("Sub-queries", expanded=False):
    for i, q in enumerate(state.get("sub_queries", []), start=1):
        st.write(f"{i}. {q}")


# ---------------------------------------------------------------------------
# Findings breakdown by source
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Findings breakdown")

findings_rows = []
for f in state.get("arxiv_findings", []):
    findings_rows.append({
        "source": "arxiv",
        "agent": f.get("source_agent", ""),
        "title": (f.get("title") or "")[:80],
        "url": f.get("url", ""),
        "score": f.get("relevance_score", 0),
        "sub_query": (f.get("sub_query") or "")[:60],
    })
for f in state.get("web_findings", []):
    findings_rows.append({
        "source": "web",
        "agent": f.get("source_agent", ""),
        "title": (f.get("title") or "")[:80],
        "url": f.get("url", ""),
        "score": f.get("relevance_score", 0),
        "sub_query": (f.get("sub_query") or "")[:60],
    })
for f in state.get("deep_read_findings", []):
    findings_rows.append({
        "source": "deep_read",
        "agent": f.get("source_agent", ""),
        "title": (f.get("title") or "")[:80],
        "url": f.get("url", ""),
        "score": f.get("relevance_score", 0),
        "sub_query": (f.get("sub_query") or "")[:60],
    })

if findings_rows:
    df = pd.DataFrame(findings_rows)
    counts_col, chart_col = st.columns([1, 2])
    with counts_col:
        counts = df["source"].value_counts().reset_index()
        counts.columns = ["source", "count"]
        st.dataframe(counts, hide_index=True, use_container_width=True)
    with chart_col:
        score_by_source = df.groupby("source")["score"].mean().reset_index()
        fig = px.bar(
            score_by_source,
            x="source",
            y="score",
            title="Mean relevance score by source",
            range_y=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("All findings", expanded=False):
        st.dataframe(df.sort_values("score", ascending=False), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Routing table — proves multi-vendor architecture
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Multi-vendor routing")
st.caption(
    "Each node uses a different model from a different vendor. "
    "The argument: vendor lock-in is a routing decision, not a default."
)
try:
    from agents.models import ROUTING
    routing_rows = [
        {"node": s.node_name, "model": s.litellm_model, "rationale": s.rationale}
        for s in ROUTING.values()
    ]
    st.dataframe(pd.DataFrame(routing_rows), hide_index=True, use_container_width=True)
except Exception as e:
    st.error(f"Could not load routing table: {e}")


# ---------------------------------------------------------------------------
# The synthesized report itself
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Synthesized report")
st.markdown(state.get("report", "_(no report)_"))


# ---------------------------------------------------------------------------
# Quality-gate notes (judge's reasoning)
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Quality-gate notes")
st.code(state.get("gate_notes", "(none)"), language="text")
