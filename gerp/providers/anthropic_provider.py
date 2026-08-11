"""
Anthropic adapter.

Uses the native web_search server tool. The response is a list of content
blocks:
  - server_tool_use blocks (name == "web_search") carry the query the model
    issued in block.input -> issued_queries
  - web_search_tool_result blocks list every result returned for a query
    (url, title, page_age are plaintext; only the content snippet is
    encrypted) -> the candidate pool for the considered-set
  - text blocks carry a `citations` array (type web_search_result_location)
    exposing url, title, and cited_text -> citations

Because the full result pool is observable, the considered-set is a native
provider_delta (pool minus citations), same as OpenAI.
"""

from __future__ import annotations

from typing import Optional

from .base import BaseProvider, ProviderError, domain_of
from ..schema import GERP, Citation, ConsideredDoc, ConsideredMethod


class AnthropicProvider(BaseProvider):
    name = "anthropic"
    default_model = "claude-haiku-4-5"
    env_key = "ANTHROPIC_API_KEY"

    # Default web-search server-tool type. Every tier uses the basic variant
    # on purpose — the newer web_search_20260209 (dynamic filtering) hides the
    # server_tool_use / web_search_tool_result blocks _parse() reads, killing
    # the fan-out and considered-set. Settable on the constructor and per-call
    # for experiments, but transparency is the product here.
    default_tool_type = "web_search_20250305"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 max_uses: int = 5, tool_type: Optional[str] = None):
        super().__init__(api_key, model)
        self.max_uses = max_uses
        self.tool_type = tool_type or self.default_tool_type

    def run(self, prompt: str, **kwargs) -> GERP:
        try:
            import anthropic
        except ImportError as e:
            raise ProviderError("pip install anthropic") from e

        # A per-tier tool_type kwarg (forwarded by gerp.run) wins over the
        # instance default, so the frontier tier gets the dynamic-filtering tool.
        tool_type = kwargs.get("tool_type", self.tool_type)

        client = anthropic.Anthropic(api_key=self.api_key)
        resp = client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", 2048),
            messages=[{"role": "user", "content": prompt}],
            tools=[{
                "type": tool_type,
                "name": "web_search",
                "max_uses": self.max_uses,
            }],
        )
        return self._parse(prompt, resp)

    def _parse(self, prompt: str, resp) -> GERP:
        # First pass: issued queries (server_tool_use) and the full result
        # pool (web_search_tool_result), pairing results to queries via
        # tool_use_id so each pool doc records its source query and rank.
        issued_queries: list[str] = []
        query_by_tool_id: dict[str, str] = {}
        pool: dict[str, ConsideredDoc] = {}  # url -> doc, first sighting wins
        page_age_by_url: dict[str, str] = {}

        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "server_tool_use" and getattr(block, "name", None) == "web_search":
                query = (getattr(block, "input", None) or {}).get("query")
                if query:
                    issued_queries.append(query)
                    if getattr(block, "id", None):
                        query_by_tool_id[block.id] = query
            elif btype == "web_search_tool_result":
                source_query = query_by_tool_id.get(
                    getattr(block, "tool_use_id", None))
                content = getattr(block, "content", None)
                if not isinstance(content, list):  # error result block
                    continue
                for rank, res in enumerate(content, start=1):
                    if getattr(res, "type", None) != "web_search_result":
                        continue
                    url = getattr(res, "url", None)
                    if not url:
                        continue
                    age = getattr(res, "page_age", None)
                    if age:
                        page_age_by_url[url] = age
                    if url not in pool:
                        doc = ConsideredDoc(
                            url=url,
                            title=getattr(res, "title", None),
                            domain=domain_of(url),
                            source_query=source_query,
                            rank=rank,
                        )
                        if age:
                            doc.metadata["page_age"] = age
                        pool[url] = doc

        answer_parts: list[str] = []
        citations: list[Citation] = []

        for block in resp.content:
            if getattr(block, "type", None) == "text":
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

        deduped = self._dedupe(citations)
        cited_urls = {c.url for c in deduped}
        considered = [d for u, d in pool.items() if u not in cited_urls]

        gerp = self._new_gerp(
            prompt=prompt,
            answer_text="".join(answer_parts).strip(),
            citations=deduped,
            issued_queries=issued_queries,
            raw_response=resp.model_dump() if hasattr(resp, "model_dump") else None,
        )
        # The pool was observed, so the delta is provider-native even when
        # it's empty (i.e. every result was cited).
        if pool:
            gerp.considered_not_cited = considered
            gerp.considered_method = ConsideredMethod.PROVIDER_DELTA
        return gerp
