"""
Base provider adapter.

Each LLM provider subclasses BaseProvider and implements run(). The job of an
adapter is: take a prompt -> call the provider's web-search/grounded endpoint
-> normalize the response into a GERP object (citations-only for now).

The considered-set step is intentionally NOT done here. It's a separate,
pluggable post-processing pass (see considered.py) so it can be added later
without touching adapters.
"""

from __future__ import annotations

import abc
import os
from urllib.parse import urlparse
from typing import Optional

from ..schema import GERP, Citation, SCHEMA_VERSION, ConsideredMethod


def domain_of(url: str) -> Optional[str]:
    try:
        netloc = urlparse(url).netloc
        return netloc[4:] if netloc.startswith("www.") else netloc or None
    except Exception:
        return None


class ProviderError(RuntimeError):
    pass


class BaseProvider(abc.ABC):
    name: str = "base"
    default_model: str = ""
    env_key: str = ""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get(self.env_key)
        self.model = model or self.default_model
        if not self.api_key:
            raise ProviderError(
                f"No API key for {self.name}. Set {self.env_key} or pass api_key."
            )

    @abc.abstractmethod
    def run(self, prompt: str, **kwargs) -> GERP:
        """Call the provider and return a normalized GERP (citations-only)."""
        raise NotImplementedError

    def _new_gerp(self, prompt: str, answer_text: str, **extra) -> GERP:
        return GERP(
            schema_version=SCHEMA_VERSION,
            provider=self.name,
            model=self.model,
            prompt=prompt,
            answer_text=answer_text,
            considered_method=ConsideredMethod.NONE,
            **extra,
        )

    @staticmethod
    def _mk_citation(url, title=None, cited_text=None, start=None, end=None) -> Citation:
        return Citation(
            url=url,
            title=title,
            domain=domain_of(url),
            cited_text=cited_text,
            start_index=start,
            end_index=end,
        )

    @staticmethod
    def _dedupe(citations: list[Citation]) -> list[Citation]:
        """Collapse duplicate URLs, keeping the first span seen."""
        seen: dict[str, Citation] = {}
        for c in citations:
            if c.url not in seen:
                seen[c.url] = c
        return list(seen.values())
