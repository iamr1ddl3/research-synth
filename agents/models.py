"""Multi-vendor routing table for research-synth.

The architectural argument this module exists to make:

    CodeOrch (Days 3-7) routed within ONE vendor's tiers (Opus/Sonnet/Haiku).
    research-synth routes ACROSS vendors. Different model strengths, different
    cost profiles, different context-length capabilities — picked per node
    based on the actual task, not the default vendor.

LiteLLM is the unified gateway. One `acompletion()` call signature, four
backends:

    Direct Anthropic           — claude-sonnet-4-6 (Synthesizer)
    Direct OpenAI              — gpt-4.1 (ArxivAgent)
    OpenRouter                 — google/gemini-2.5-flash (Decomposer, Gate)
    OpenRouter                 — google/gemini-2.5-pro (DeepReadAgent)
    OpenRouter                 — deepseek/deepseek-chat (WebSearchAgent)

Routing rationale (defensible in interviews):

    Decomposer        → Gemini Flash via OpenRouter
                        Mechanical sub-query generation. Premium model is
                        overkill. Cheapest in the lineup; runs on every query.

    ArxivAgent        → GPT-4.1 direct
                        Academic comprehension. Different vendor than
                        CodeOrch's Anthropic stack — adds OpenAI fluency to
                        the portfolio.

    WebSearchAgent    → DeepSeek V3 via OpenRouter
                        Cheap, strong at summarization. Adds open-source-
                        routing signal. Heavily used (broad search returns
                        many snippets to summarize).

    DeepReadAgent     → Gemini 2.5 Pro via OpenRouter
                        1M context window. When Browserbase pulls a 50-page
                        article, no chunking needed. Genuine differentiator.

    Synthesizer       → Claude Sonnet 4.6 direct
                        Strongest long-form structured writer. The headline
                        output deserves the strongest writer. Keeping ONE
                        Anthropic node is engineering, not ideology.

    QualityGate       → Gemini Flash via OpenRouter
                        LLM-as-judge on coverage/citations. Cheap, runs every
                        retry. Same model as Decomposer because the task is
                        comparably mechanical.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load .env from this package's parent directory (research-synth/.env).
# override=True because the user's shell sometimes exports empty
# ANTHROPIC_API_KEY= which would mask the dotenv value otherwise.
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_PATH, override=True)


# ===========================================================================
# Routing table — single source of truth.
# ===========================================================================

@dataclass(frozen=True)
class ModelSpec:
    """Per-node model specification.

    Attributes:
        node_name: Logical role in the graph (decomposer, arxiv_agent, ...).
        litellm_model: The model id LiteLLM expects.
        rationale: One-line justification for this choice (used in tracing
            metadata + the README routing table).
    """
    node_name: str
    litellm_model: str
    rationale: str


ROUTING: dict[str, ModelSpec] = {
    "decomposer": ModelSpec(
        node_name="decomposer",
        litellm_model="openrouter/google/gemini-2.5-flash",
        rationale="Mechanical sub-query generation; cheapest in lineup.",
    ),
    "arxiv_agent": ModelSpec(
        node_name="arxiv_agent",
        litellm_model="openai/gpt-4.1",
        rationale="Academic comprehension; multi-vendor signal vs CodeOrch.",
    ),
    "web_search_agent": ModelSpec(
        node_name="web_search_agent",
        litellm_model="openrouter/deepseek/deepseek-chat",
        rationale="Cheap summarization at scale; open-source routing signal.",
    ),
    "deep_read_agent": ModelSpec(
        node_name="deep_read_agent",
        litellm_model="openrouter/google/gemini-2.5-pro",
        rationale="1M context — no chunking needed for long-form articles.",
    ),
    "synthesizer": ModelSpec(
        node_name="synthesizer",
        litellm_model="anthropic/claude-sonnet-4-6",
        rationale="Strongest long-form structured writer; headline output.",
    ),
    "quality_gate": ModelSpec(
        node_name="quality_gate",
        litellm_model="openrouter/google/gemini-2.5-flash",
        rationale="LLM-as-judge; cheap, runs every retry.",
    ),
}


# ===========================================================================
# LiteLLM-compatible env var setup. Called once at import.
# ===========================================================================

def _configure_litellm_env() -> None:
    """LiteLLM reads provider keys from env vars by name.

    - Anthropic   → ANTHROPIC_API_KEY (already set)
    - OpenAI      → OPENAI_API_KEY (already set)
    - OpenRouter  → OPENROUTER_API_KEY (already set, LiteLLM v1.55+ reads
                    this directly when model id starts with 'openrouter/')

    Nothing to mutate — this function exists only to fail loudly if a key is
    missing, instead of letting LiteLLM throw a confusing 401 mid-graph.
    """
    required = {
        "ANTHROPIC_API_KEY": "Synthesizer node (claude-sonnet)",
        "OPENAI_API_KEY": "ArxivAgent node (gpt-4.1)",
        "OPENROUTER_API_KEY": "Decomposer/WebSearch/DeepRead/Gate (Gemini, DeepSeek)",
    }
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        details = "\n".join(f"  {k} — needed by {required[k]}" for k in missing)
        raise RuntimeError(
            f"Missing required API keys in .env:\n{details}\n"
            f"Edit {_ENV_PATH} and re-run."
        )


_configure_litellm_env()


# ===========================================================================
# Unified call wrapper. Every node calls through this.
# ===========================================================================

async def acall_model(
    *,
    node: str,
    system: str,
    user: str,
    response_format: dict[str, Any] | None = None,
    max_tokens: int = 2048,
    temperature: float | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Unified async LLM call. Returns parsed response + usage metadata.

    Args:
        node: One of the ROUTING keys. Resolves to the model id.
        system: System prompt.
        user: User message content.
        response_format: Optional LiteLLM/OpenAI-style response_format hint
            (e.g. {"type": "json_object"}). Anthropic doesn't honor this
            directly but LiteLLM tries to translate.
        max_tokens: Output cap. 2048 default — the synthesis node bumps this.
        temperature: Optional. Anthropic's claude-opus-4-7 rejects temperature
            entirely; for safety we omit unless explicitly passed.
        extra_metadata: Per-call metadata to attach to the Langfuse span
            (e.g. sub_query_index, source_count). Merged with standard fields.

    Returns:
        {
            "content": str,            # the model's text output
            "usage": {                  # LiteLLM-normalized
                "prompt_tokens": int,
                "completion_tokens": int,
                "total_tokens": int,
            },
            "model": str,               # the resolved litellm_model id
            "node": str,                # the logical node name
            "latency_ms": int,
        }
    """
    import time
    import litellm

    spec = ROUTING[node]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    kwargs: dict[str, Any] = {
        "model": spec.litellm_model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if response_format is not None:
        kwargs["response_format"] = response_format

    # Langfuse via LiteLLM's built-in callback adds these fields to the trace.
    kwargs["metadata"] = {
        "node": node,
        "rationale": spec.rationale,
        **(extra_metadata or {}),
    }

    # Wrap the LLM call in a Langfuse generation span if a trace is active.
    # This replaces LiteLLM's bundled Langfuse callback (which is broken on
    # Python 3.14 + Langfuse v4) with the same explicit-span pattern that
    # CodeOrch uses.
    from observability.langfuse import trace_node

    t0 = time.perf_counter()
    with trace_node(
        node_name=node,
        model=spec.litellm_model,
        input_data={"system": system, "user": user},
        as_type="generation",
    ) as span:
        response = await litellm.acompletion(**kwargs)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        # Feed token usage into the Langfuse generation so cost auto-computes.
        span.update(
            output=content,
            usage_details={
                "input": usage["prompt_tokens"],
                "output": usage["completion_tokens"],
                "total": usage["total_tokens"],
            },
            model=spec.litellm_model,
            metadata={"latency_ms": latency_ms, "rationale": spec.rationale},
        )

    return {
        "content": content,
        "usage": usage,
        "model": spec.litellm_model,
        "node": node,
        "latency_ms": latency_ms,
    }


def model_for(node: str) -> str:
    """Convenience accessor — returns the LiteLLM model id for a node."""
    return ROUTING[node].litellm_model


def routing_table_markdown() -> str:
    """Render the routing table as markdown — used by README and dashboards."""
    rows = ["| Node | Model | Why |", "|---|---|---|"]
    for spec in ROUTING.values():
        rows.append(f"| {spec.node_name} | `{spec.litellm_model}` | {spec.rationale} |")
    return "\n".join(rows)
