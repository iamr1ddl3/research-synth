"""Smoke-ping every provider + Langfuse before a full graph run.

Usage:
    python -m scripts.smoke_ping

Hits each model in the routing table with a 5-token "ping" prompt. Fails
loudly on the first 401/403/404 so we never debug cryptic LangGraph errors
that are really 'OpenRouter doesn't have access to this model on your tier'.

Also pings Langfuse to confirm the trace pipeline is healthy.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Make repo root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.models import ROUTING, acall_model  # noqa: E402
from observability.langfuse import safe_check_connection, trace_run  # noqa: E402


PING_SYSTEM = "You are a helpful assistant. Reply with exactly one word: pong."
PING_USER = "ping"


async def ping_node(node: str) -> tuple[str, bool, str]:
    """Returns (node, ok, message)."""
    try:
        result = await acall_model(
            node=node,
            system=PING_SYSTEM,
            user=PING_USER,
            max_tokens=10,
        )
        content = (result["content"] or "").strip()[:30]
        usage = result["usage"]
        return (
            node,
            True,
            f"OK — model={result['model']} latency={result['latency_ms']}ms "
            f"tokens={usage['total_tokens']} reply={content!r}",
        )
    except Exception as e:
        msg = str(e)
        # Truncate long stack-trace style messages
        if len(msg) > 200:
            msg = msg[:200] + "..."
        return (node, False, f"FAILED — {msg}")


async def main() -> int:
    print("=" * 70)
    print("research-synth — smoke ping")
    print("=" * 70)

    # 1. Provider keys — wrap in a trace_run so the LLM spans land in the
    #    'smoke-ping' Langfuse session (proves the full pipeline works).
    print("\n[1/3] LLM provider connections")
    print("-" * 70)
    with trace_run(run_id="smoke-ping", query="multi-vendor smoke ping", extra_tags=["smoke-ping"]):
        results = await asyncio.gather(*[ping_node(node) for node in ROUTING])
    failed = []
    for node, ok, msg in results:
        marker = "✓" if ok else "✗"
        print(f"  {marker} {node:20} {msg}")
        if not ok:
            failed.append(node)

    # 2. Langfuse
    print("\n[2/3] Langfuse")
    print("-" * 70)
    ok, msg = safe_check_connection()
    marker = "✓" if ok else "✗"
    print(f"  {marker} langfuse  {msg}")
    if not ok:
        failed.append("langfuse")

    # 3. Browserbase (cheap auth check — list sessions)
    print("\n[3/3] Browserbase")
    print("-" * 70)
    bb_ok, bb_msg = check_browserbase()
    marker = "✓" if bb_ok else "✗"
    print(f"  {marker} browserbase  {bb_msg}")
    if not bb_ok:
        failed.append("browserbase")

    print("\n" + "=" * 70)
    if failed:
        print(f"FAIL — {len(failed)} provider(s) failed: {', '.join(failed)}")
        print("Fix .env and re-run before building graph nodes.")
        return 1
    print(f"PASS — all {len(ROUTING) + 2} connections healthy")
    return 0


def check_browserbase() -> tuple[bool, str]:
    api_key = os.environ.get("BROWSERBASE_API_KEY")
    project_id = os.environ.get("BROWSERBASE_PROJECT_ID")
    if not (api_key and project_id):
        return False, "Missing BROWSERBASE_API_KEY / BROWSERBASE_PROJECT_ID"
    try:
        from browserbase import Browserbase
        bb = Browserbase(api_key=api_key)
        # List sessions is the cheapest authenticated call.
        # If creds are wrong this returns 401.
        list(bb.sessions.list(status="COMPLETED"))
        return True, f"OK — project={project_id[:8]}..."
    except Exception as e:
        msg = str(e)
        if len(msg) > 200:
            msg = msg[:200] + "..."
        return False, f"FAILED — {msg}"


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
