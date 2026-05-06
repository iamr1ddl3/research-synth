"""Langfuse v4 instrumentation for research-synth.

Same pattern as codeorch/observability/langfuse.py — explicit Langfuse v4
SDK spans via start_as_current_observation. We deliberately do NOT use
LiteLLM's bundled Langfuse callback because it is broken on Python 3.14 +
Langfuse v4 (calls the removed langfuse.version.__version__ attribute).

Trace structure:
    one query  = one trace = one outer span (research_synth.run)
    one node   = one child span (or generation, when wrapping an LLM call)

Trace attributes (set once at run start, propagated to every child span):
    session_id  = str(run_id) — groups all nodes of one run
    user_id     = caller-supplied (defaults to "anon")
    tags        = ["research-synth", ...]
    trace_name  = "research_synth.run"

Env-var compat: user provided LANGFUSE_BASE_URL but the SDK looks for
LANGFUSE_HOST. We accept either at lookup time.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID

from dotenv import load_dotenv

# Load .env at import time
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH, override=True)


def _host() -> str:
    """Accept either LANGFUSE_HOST (SDK convention) or LANGFUSE_BASE_URL
    (research-synth .env convention)."""
    return (
        os.environ.get("LANGFUSE_HOST")
        or os.environ.get("LANGFUSE_BASE_URL")
        or "https://cloud.langfuse.com"
    )


@lru_cache(maxsize=1)
def get_langfuse():
    """Return a singleton Langfuse v4 client. Reads env at first call."""
    from langfuse import Langfuse
    return Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=_host(),
    )


class _SpanProxy:
    """Adapter so callers can use a stable .update() shape regardless of
    whether the underlying observation is a span or a generation."""

    def __init__(self, span: Any):
        self._span = span

    def update(
        self,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        level: str | None = None,
        status_message: str | None = None,
        usage_details: dict[str, int] | None = None,
        model: str | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if output is not None:
            kwargs["output"] = output
        if metadata is not None:
            kwargs["metadata"] = metadata
        if level is not None:
            kwargs["level"] = level
        if status_message is not None:
            kwargs["status_message"] = status_message
        if usage_details is not None:
            kwargs["usage_details"] = usage_details
        if model is not None:
            kwargs["model"] = model
        if kwargs:
            self._span.update(**kwargs)


@contextmanager
def trace_run(
    run_id: UUID | str,
    query: str,
    user_id: str = "anon",
    extra_tags: list[str] | None = None,
) -> Iterator[_SpanProxy]:
    """Open the OUTER span for one research-synth run.

    Every node span opened inside this context becomes a child of this span
    and inherits session_id (= run_id) so the run shows up as a single
    session in the Langfuse Sessions view.
    """
    from langfuse import propagate_attributes
    lf = get_langfuse()
    tags = ["research-synth"] + (extra_tags or [])
    with lf.start_as_current_observation(
        as_type="span",
        name="research_synth.run",
        input={"query": query},
    ) as root:
        with propagate_attributes(
            session_id=str(run_id),
            user_id=user_id,
            tags=tags,
            trace_name="research_synth.run",
        ):
            yield _SpanProxy(root)
        # exit-time flush is the safety net for short scripts
        lf.flush()


@contextmanager
def trace_node(
    node_name: str,
    model: str,
    input_data: dict[str, Any] | None = None,
    as_type: str = "generation",
) -> Iterator[_SpanProxy]:
    """Open a CHILD span for one node call inside an active run trace.

    as_type='generation' is the default because every node in research-synth
    wraps an LLM call. Langfuse computes cost automatically from
    usage_details + model.
    """
    from langfuse import get_client
    lf = get_client()
    with lf.start_as_current_observation(
        as_type=as_type,
        name=f"node.{node_name}",
        input=input_data,
        metadata={"model": model},
    ) as span:
        yield _SpanProxy(span)


def safe_check_connection() -> tuple[bool, str]:
    """Verify Langfuse credentials work. Returns (ok, message)."""
    pub = os.environ.get("LANGFUSE_PUBLIC_KEY")
    sec = os.environ.get("LANGFUSE_SECRET_KEY")
    if not (pub and sec):
        return False, "Missing LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY in .env"
    try:
        lf = get_langfuse()
        # Round-trip a no-op span to verify the network path
        with lf.start_as_current_observation(
            as_type="span",
            name="research_synth.connection_check",
        ) as span:
            span.update(output={"status": "ok"})
        lf.flush()
        return True, f"Langfuse OK at {_host()}"
    except Exception as e:
        return False, f"Langfuse connection failed: {e}"
