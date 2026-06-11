"""
Considered-but-not-cited layer.

A separate post-processing pass over a GERP. Anthropic and OpenAI expose
their result pool natively (provider_delta, handled in their adapters);
Gemini does not, so its considered-set is built here by re-running the
engine's own fan-out queries through a SERP backend and diffing against
the citations (rerun_queries).

Matching note: Gemini citation URLs are vertexaisearch redirect URLs that
will never string-match a SERP result URL, so enrich() also excludes any
SERP result whose *domain* matches a cited domain. The considered-set is
therefore domain-level conservative: a different page on an already-cited
domain is treated as cited.
"""

from __future__ import annotations

import abc
import json
import os
import urllib.parse
import urllib.request
from typing import Optional

from .schema import GERP, ConsideredDoc, ConsideredMethod
from .providers.base import domain_of


class SearchBackend(abc.ABC):
    @abc.abstractmethod
    def search(self, query: str, num: int = 10) -> list[dict]:
        """Return [{url, title, rank}, ...] for a query."""
        raise NotImplementedError


class SerpAPIBackend(SearchBackend):
    """SerpAPI-backed Google search."""

    env_key = "SERPAPI_API_KEY"

    def __init__(self, api_key: Optional[str] = None, engine: str = "google"):
        self.api_key = api_key or os.environ.get(self.env_key)
        self.engine = engine

    def search(self, query: str, num: int = 10) -> list[dict]:
        if not self.api_key:
            raise RuntimeError(f"Set {self.env_key} or pass api_key.")
        params = urllib.parse.urlencode({
            "engine": self.engine,
            "q": query,
            "num": num,
            "api_key": self.api_key,
        })
        req = urllib.request.Request(
            f"https://serpapi.com/search.json?{params}",
            headers={"User-Agent": "gerp/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        return [
            {"url": o.get("link"), "title": o.get("title"),
             "rank": o.get("position", i + 1)}
            for i, o in enumerate(data.get("organic_results", []))
        ]


def enrich(gerp: GERP, backend: Optional[SearchBackend] = None,
           num_per_query: int = 20) -> GERP:
    """Populate considered_not_cited by re-running issued fan-out queries.

    No-op (and leaves considered_method untouched) without a backend or
    without issued queries, so provider_delta results are never clobbered.
    """
    if backend is None or not gerp.issued_queries:
        return gerp

    cited_urls = {c.url for c in gerp.citations}
    cited_domains = {c.domain for c in gerp.citations if c.domain}
    considered: list[ConsideredDoc] = []
    seen: set[str] = set()
    for q in gerp.issued_queries:
        for res in backend.search(q, num=num_per_query):
            url = res.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            domain = domain_of(url)
            if url in cited_urls or (domain and domain in cited_domains):
                continue
            considered.append(ConsideredDoc(
                url=url, title=res.get("title"), domain=domain,
                source_query=q, rank=res.get("rank"),
            ))
    gerp.considered_not_cited = considered
    gerp.considered_method = ConsideredMethod.RERUN_QUERIES
    return gerp
