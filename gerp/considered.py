"""
Considered-but-not-cited layer (ADDITIVE — stubbed for now).

This is a separate post-processing pass over a GERP. In citations-only mode it
does nothing. When you turn it on later, each strategy fills
gerp.considered_not_cited and sets gerp.considered_method.

Strategies (per the comparison):
  - rerun_queries  : re-run gerp.issued_queries via a SearchBackend, diff vs citations
  - provider_delta : Perplexity-only, search_results minus referenced (needs raw_response)
  - reconstructed  : infer queries ourselves, then rerun

Only the SerpAPI backend wiring is sketched. enrich() is a no-op unless a
backend is supplied, so nothing calls SerpAPI until you flip it on.
"""

from __future__ import annotations

import abc
import os
from typing import Optional

from .schema import GERP, ConsideredDoc, ConsideredMethod


class SearchBackend(abc.ABC):
    @abc.abstractmethod
    def search(self, query: str, num: int = 10) -> list[dict]:
        """Return [{url, title, rank}, ...] for a query."""
        raise NotImplementedError


class SerpAPIBackend(SearchBackend):
    """SerpAPI-backed search. STUB — not wired to live calls yet."""

    env_key = "SERPAPI_API_KEY"

    def __init__(self, api_key: Optional[str] = None, engine: str = "google"):
        self.api_key = api_key or os.environ.get(self.env_key)
        self.engine = engine

    def search(self, query: str, num: int = 10) -> list[dict]:
        # TODO (considered-set phase): call SerpAPI and map organic_results
        # to [{url, title, rank}]. Left intentionally unimplemented so the
        # citations-only build never hits the network or spends credits.
        raise NotImplementedError(
            "SerpAPIBackend.search is stubbed. Enable in the considered-set phase."
        )


def _cited_urls(gerp: GERP) -> set[str]:
    return {c.url for c in gerp.citations}


def enrich(gerp: GERP, backend: Optional[SearchBackend] = None,
           num_per_query: int = 10) -> GERP:
    """Populate considered_not_cited. No-op without a backend (current default)."""
    if backend is None:
        gerp.considered_method = ConsideredMethod.NONE
        return gerp

    # Perplexity native delta needs no backend, handled separately if desired.
    if not gerp.issued_queries:
        gerp.considered_method = ConsideredMethod.NONE
        return gerp

    cited = _cited_urls(gerp)
    considered: list[ConsideredDoc] = []
    seen: set[str] = set()
    for q in gerp.issued_queries:
        for res in backend.search(q, num=num_per_query):
            url = res.get("url")
            if not url or url in cited or url in seen:
                continue
            seen.add(url)
            considered.append(ConsideredDoc(
                url=url, title=res.get("title"),
                domain=None, source_query=q, rank=res.get("rank"),
            ))
    gerp.considered_not_cited = considered
    gerp.considered_method = ConsideredMethod.RERUN_QUERIES
    return gerp
