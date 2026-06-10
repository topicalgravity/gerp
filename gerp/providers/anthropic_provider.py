"""
Anthropic adapter.

Uses the native web_search server tool. The response is a list of content
blocks; text blocks carry a `citations` array whose entries (type
web_search_result_location) expose url, title, and cited_text. We read
citations from those text blocks rather than from the web_search_tool_result
blocks, because the latter's content snippets are encrypted.

Issued queries are NOT exposed by Anthropic, so considered-set falls back to
reconstruct/none downstream.
"""

from __future__ import annotations

from typing import Optional

from .base import BaseProvider, ProviderError
from ..schema import GERP, Citation


class AnthropicProvider(BaseProvider):
    name = "anthropic"
    default_model = "claude-opus-4-6"
    env_key = "ANTHROPIC_API_KEY"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 max_uses: int = 5):
        super().__init__(api_key, model)
        self.max_uses = max_uses

    def run(self, prompt: str, **kwargs) -> GERP:
        try:
            import anthropic
        except ImportError as e:
            raise ProviderError("pip install anthropic") from e

        client = anthropic.Anthropic(api_key=self.api_key)
        resp = client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", 2048),
            messages=[{"role": "user", "content": prompt}],
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": self.max_uses,
            }],
        )

        # First pass: collect page_age from web_search_tool_result blocks.
        page_age_by_url: dict[str, str] = {}
        for block in resp.content:
            if getattr(block, "type", None) == "tool_result":
                for res in (getattr(block, "content", None) or []):
                    if getattr(res, "type", None) == "web_search_result":
                        url = getattr(res, "url", None)
                        age = getattr(res, "page_age", None)
                        if url and age:
                            page_age_by_url[url] = age

        answer_parts: list[str] = []
        citations: list[Citation] = []

        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text = getattr(block, "text", "") or ""
                answer_parts.append(text)
                for cit in (getattr(block, "citations", None) or []):
                    url = getattr(cit, "url", None)
                    if not url:
                        continue
                    c = self._mk_citation(
                        url=url,
                        title=getattr(cit, "title", None),
                        cited_text=getattr(cit, "cited_text", None) or text[:200],
                    )
                    if url in page_age_by_url:
                        c.metadata["page_age"] = page_age_by_url[url]
                    citations.append(c)

        return self._new_gerp(
            prompt=prompt,
            answer_text="".join(answer_parts).strip(),
            citations=self._dedupe(citations),
            raw_response=resp.model_dump() if hasattr(resp, "model_dump") else None,
        )
