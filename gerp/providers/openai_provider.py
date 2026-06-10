"""
OpenAI adapter.

Uses the Responses API with the web_search tool. Citations come back as
url_citation annotations on the output_text content. No issued-query list is
exposed, so considered-set falls back downstream.
"""

from __future__ import annotations

from typing import Optional

from .base import BaseProvider, ProviderError, domain_of
from ..schema import GERP, Citation, ConsideredDoc, ConsideredMethod


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
        all_source_urls: list[str] = []

        for item in (getattr(resp, "output", None) or []):
            itype = getattr(item, "type", None)
            if itype == "web_search_call":
                for src in (getattr(item, "sources", None) or []):
                    url = getattr(src, "url", None) or (src if isinstance(src, str) else None)
                    if url:
                        all_source_urls.append(url)
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
        considered = [
            ConsideredDoc(url=u, domain=domain_of(u))
            for u in dict.fromkeys(all_source_urls)
            if u not in cited_urls
        ]

        gerp = self._new_gerp(
            prompt=prompt,
            answer_text="".join(answer_parts).strip(),
            citations=deduped,
        )
        if considered:
            gerp.considered_not_cited = considered
            gerp.considered_method = ConsideredMethod.PROVIDER_DELTA
        return gerp
