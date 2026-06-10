"""
OpenAI adapter.

Uses the Responses API with the web_search tool. Citations come back as
url_citation annotations on the output_text content. Each web_search_call
item carries an `action` object whose `query` is the issued search query and
whose `sources` (when requested via include=) lists every URL consulted —
making the considered-set a native provider_delta.
"""

from __future__ import annotations

from typing import Optional

from .base import BaseProvider, ProviderError, domain_of
from ..schema import GERP, Citation, ConsideredDoc, ConsideredMethod


def _clean_url(url: str) -> str:
    """Strip the ?utm_source=openai tracking suffix OpenAI appends to URLs."""
    for sep in ("?", "&"):
        url = url.replace(f"{sep}utm_source=openai", "")
    return url


class OpenAIProvider(BaseProvider):
    name = "openai"
    default_model = "gpt-4.1"
    env_key = "OPENAI_API_KEY"

    def run(self, prompt: str, **kwargs) -> GERP:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ProviderError("pip install openai") from e

        client = OpenAI(api_key=self.api_key)
        resp = client.responses.create(
            model=self.model,
            input=prompt,
            tools=[{"type": "web_search"}],
            include=["web_search_call.action.sources"],
        )

        answer_parts: list[str] = []
        citations: list[Citation] = []
        issued_queries: list[str] = []
        source_docs: dict[str, ConsideredDoc] = {}  # url -> doc

        for item in (getattr(resp, "output", None) or []):
            itype = getattr(item, "type", None)
            if itype == "web_search_call":
                # Sources and the issued query live on item.action; older
                # SDK shapes exposed sources on the item itself, so check both.
                action = getattr(item, "action", None)
                queries = (getattr(action, "queries", None) if action else None) \
                    or ([getattr(action, "query", None)] if action else [])
                queries = [q for q in queries if q]
                issued_queries.extend(q for q in queries if q not in issued_queries)
                query = queries[0] if queries else None
                sources = (getattr(action, "sources", None) if action else None) \
                    or getattr(item, "sources", None) or []
                for rank, src in enumerate(sources, start=1):
                    url = getattr(src, "url", None) or (src if isinstance(src, str) else None)
                    if not url:
                        continue
                    url = _clean_url(url)
                    if url not in source_docs:
                        source_docs[url] = ConsideredDoc(
                            url=url, domain=domain_of(url),
                            source_query=query, rank=rank,
                        )
            elif itype == "message":
                for content in (getattr(item, "content", None) or []):
                    if getattr(content, "type", None) != "output_text":
                        continue
                    text = getattr(content, "text", "") or ""
                    answer_parts.append(text)
                    for ann in (getattr(content, "annotations", None) or []):
                        if getattr(ann, "type", None) != "url_citation":
                            continue
                        url = getattr(ann, "url", None)
                        if not url:
                            continue
                        url = _clean_url(url)
                        start = getattr(ann, "start_index", None)
                        end = getattr(ann, "end_index", None)
                        snippet = text[start:end] if (start is not None and end) else None
                        citations.append(self._mk_citation(
                            url=url,
                            title=getattr(ann, "title", None),
                            cited_text=snippet,
                            start=start,
                            end=end,
                        ))

        deduped = self._dedupe(citations)
        cited_urls = {c.url for c in deduped}
        considered = [d for u, d in source_docs.items() if u not in cited_urls]

        gerp = self._new_gerp(
            prompt=prompt,
            answer_text="".join(answer_parts).strip(),
            citations=deduped,
            issued_queries=issued_queries,
        )
        if considered:
            gerp.considered_not_cited = considered
            gerp.considered_method = ConsideredMethod.PROVIDER_DELTA
        return gerp
