"""Model tier registry.

Two tiers only: a free "standard" tier (cheap models to cap free-tier cost) and
a login-gated "frontier" tier (each provider's flagship). gerp.run(prompt,
provider=p, model=..., **kwargs) already forwards the model and any extra kwargs
to the provider's run(), so a tier is just a per-provider {model, kwargs} lookup.

Future-proofing for a Haiku->Sonnet->Opus "ladder": nothing here assumes exactly
one model per provider beyond the current registry shape. A ladder tier would
make a provider entry a list of models and shift the results key to
(provider, model); the resolution helpers below don't block that.
"""

from __future__ import annotations

# Deliberately the BASIC web-search tool variant, on every tier including
# Opus 4.8 (which also supports the newer web_search_20260209). The newer
# variant's "dynamic filtering" runs searches inside an internal
# code-execution wrapper and never emits the server_tool_use /
# web_search_tool_result blocks our parser reads — so issued queries and the
# considered-set pool vanish from the response. Observing exactly those is
# GERP's entire product, so transparency beats the newer variant's filtering.
_ANTHROPIC_WEB_SEARCH_TOOL = "web_search_20250305"

TIERS: dict[str, dict[str, dict]] = {
    "standard": {
        # Current (2026-08-10) cheap/entry model per provider — the free tier's
        # job is to cap cost, so each is the provider's current-generation budget
        # tier and stays a rung below the frontier flagship above.
        "anthropic": {"model": "claude-haiku-4-5"},   # still Anthropic's fastest/cheapest
        "gemini":    {"model": "gemini-3.5-flash-lite"},  # 3.x budget Flash-Lite; supports Search grounding
        "openai":    {"model": "gpt-5.6-luna"},           # GPT-5.6 cost-sensitive variant; supports web_search
    },
    "frontier": {
        "anthropic": {"model": "claude-opus-4-8",
                      "kwargs": {"tool_type": _ANTHROPIC_WEB_SEARCH_TOOL}},
        # gemini-3.5-flash is Google's current Gemini 3.5 model, documented as
        # its "most intelligent model for sustained frontier performance."
        # thinking_level defaults to "high" on the 3.x models, so no extra
        # kwargs are needed; see GeminiProvider for the passthrough seam.
        "gemini":    {"model": "gemini-3.5-flash"},
        # gpt-5.5 is the reasoning-capable frontier model; effort is set on the
        # Responses API reasoning object (passed through OpenAIProvider.run()).
        "openai":    {"model": "gpt-5.5",
                      "kwargs": {"reasoning": {"effort": "high"}}},
    },
}

DEFAULT_TIER = "standard"

# Short display labels for the results-page model chip. Falls back to the raw
# model id for anything unlisted.
MODEL_LABELS = {
    "claude-opus-4-8": "Opus 4.8",
    "claude-haiku-4-5": "Haiku 4.5",
    "gemini-3.5-flash": "Gemini 3.5",
    "gemini-3.5-flash-lite": "Gemini 3.5 Lite",
    "gemini-2.5-flash": "Gemini 2.5",
    "gpt-5.5": "GPT-5.5",
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "gpt-4.1": "GPT-4.1",
}


def resolve_tier(name: str | None) -> str:
    """Map an arbitrary (possibly stale or hostile) form value to a real tier.
    Unknown -> DEFAULT_TIER, so a stale 'frontier' form never errors."""
    return name if name in TIERS else DEFAULT_TIER


def tier_config(tier: str, provider: str) -> dict:
    """Per-provider {model, kwargs} for a tier. Unknown tier falls back to the
    default tier; unknown provider yields {} (provider uses its own default)."""
    return (TIERS.get(tier) or TIERS[DEFAULT_TIER]).get(provider, {})


def model_label(model: str | None) -> str:
    return MODEL_LABELS.get(model or "", model or "")


# Provider order for the tier model-summary line shown under the tier pills.
_SUMMARY_ORDER = ("openai", "gemini", "anthropic")


def tier_model_summary(tier: str) -> str:
    """A human-readable 'GPT-5.6 Luna · Gemini 3.5 Lite · Haiku 4.5' line for a
    tier, built from the registry so the UI can never drift from the models the
    tier actually runs. A model configured with reasoning effort gets a
    'Thinking' tag (e.g. the frontier OpenAI entry)."""
    cfg = TIERS.get(tier) or {}
    parts = []
    for provider in _SUMMARY_ORDER:
        entry = cfg.get(provider) or {}
        model = entry.get("model")
        if not model:
            continue
        label = model_label(model)
        if ((entry.get("kwargs") or {}).get("reasoning") or {}).get("effort"):
            label += " Thinking"
        parts.append(label)
    return " · ".join(parts)
