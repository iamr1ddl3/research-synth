"""CLI entry point for research-synth.

Usage:
    python -m scripts.run_query "What are the latest techniques for prompt injection defense?"
    python -m scripts.run_query --query "..." --user-id ravi --out reports/run.md

Runs the LangGraph pipeline end-to-end inside a Langfuse trace_run context
so every LLM call appears in the project's Langfuse session view.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

# Repo root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.builder import build_graph  # noqa: E402
from observability.langfuse import trace_run  # noqa: E402


async def run(query: str, user_id: str, out_path: Path | None) -> dict:
    run_id = str(uuid4())
    print(f"\n{'=' * 70}")
    print(f"research-synth run")
    print(f"  run_id : {run_id}")
    print(f"  query  : {query}")
    print(f"  user_id: {user_id}")
    print(f"{'=' * 70}\n")

    graph = build_graph()
    initial_state = {
        "run_id": run_id,
        "query": query,
        "retry_count": 0,
        "arxiv_findings": [],
        "web_findings": [],
        "deep_read_findings": [],
    }

    t0 = time.perf_counter()
    with trace_run(run_id=run_id, query=query, user_id=user_id):
        final_state = await graph.ainvoke(initial_state)
    elapsed = time.perf_counter() - t0

    # Summary printout
    arxiv_n = len(final_state.get("arxiv_findings", []))
    web_n = len(final_state.get("web_findings", []))
    deep_n = len(final_state.get("deep_read_findings", []))
    sub_q_n = len(final_state.get("sub_queries", []))
    verdict = final_state.get("gate_verdict", "?")
    score = final_state.get("gate_score", 0.0)

    print(f"\n{'=' * 70}")
    print(f"run complete in {elapsed:.1f}s")
    print(f"  sub_queries        : {sub_q_n}")
    print(f"  arxiv_findings     : {arxiv_n}")
    print(f"  web_findings       : {web_n}")
    print(f"  deep_read_findings : {deep_n}")
    print(f"  gate_verdict       : {verdict}")
    print(f"  gate_score         : {score:.3f}")
    print(f"  retry_count        : {final_state.get('retry_count', 0)}")
    print(f"{'=' * 70}\n")

    # Persist
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report = final_state.get("report", "")
        out_path.write_text(report, encoding="utf-8")
        print(f"Report written to: {out_path}")

        # Also persist full state as JSON for the dashboard / debugging
        json_path = out_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(final_state, indent=2, default=str), encoding="utf-8"
        )
        print(f"Full state written to: {json_path}")

    return final_state


def main() -> int:
    p = argparse.ArgumentParser(description="Run research-synth on a query")
    p.add_argument("query", nargs="?", help="Research question (positional)")
    p.add_argument("--query", "-q", dest="query_flag", help="Research question (flag)")
    p.add_argument("--user-id", default="anon")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("reports/latest_run.md"),
        help="Where to write the synthesized report (default: reports/latest_run.md)",
    )
    args = p.parse_args()

    query = args.query or args.query_flag
    if not query:
        p.error("Provide a query as positional arg or via --query")

    final_state = asyncio.run(run(query, args.user_id, args.out))
    verdict = final_state.get("gate_verdict", "escalate")
    return 0 if verdict in ("pass", "retry") else 1


if __name__ == "__main__":
    sys.exit(main())
