"""Registry + high-level runner."""

from __future__ import annotations

from typing import Optional

from .schema import GERP
from .providers.base import BaseProvider, ProviderError
from .providers.anthropic_provider import AnthropicProvider
from .providers.gemini_provider import GeminiProvider
from .providers.openai_provider import OpenAIProvider

PROVIDERS: dict[str, type[BaseProvider]] = {
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
}


def get_provider(name: str, **kwargs) -> BaseProvider:
    key = name.lower()
    if key not in PROVIDERS:
        raise ProviderError(
            f"Unknown provider '{name}'. Choices: {', '.join(PROVIDERS)}"
        )
    return PROVIDERS[key](**kwargs)


def run(prompt: str, provider: str, model: Optional[str] = None,
        api_key: Optional[str] = None, **kwargs) -> GERP:
    """Run one prompt against one provider; return a normalized GERP."""
    p = get_provider(provider, api_key=api_key, model=model)
    return p.run(prompt, **kwargs)
