"""
Gemini adapter.

Uses Google Search grounding. The response carries groundingMetadata with:
  - groundingChunks[].web.{uri, title}   -> the sources
  - groundingSupports[]                   -> maps answer spans to chunk indices
  - webSearchQueries[]                    -> the queries Gemini actually issued

The issued queries are captured here so the considered-set step can re-run them
later (rerun_queries strategy) without another API call. Citations-only for now.
"""

from __future__ import annotations

from typing import Optional

from .base import BaseProvider, ProviderError
from ..schema import GERP, Citation


class GeminiProvider(BaseProvider):
    name = "gemini"
    default_model = "gemini-2.5-flash"
    env_key = "GEMINI_API_KEY"

    def run(self, prompt: str, **kwargs) -> GERP:
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ProviderError("pip install google-genai") from e

        cfg_kwargs = {"tools": [types.Tool(google_search=types.GoogleSearch())]}
        # Optional tier seam: the frontier tier can dial reasoning depth via a
        # thinking_level kwarg (Gemini 3.x). The registry leaves it unset today
        # (3.x defaults to a high thinking level), so this is inert unless a
        # tier opts in; guarded so it degrades gracefully on older SDKs.
        level = kwargs.get("thinking_level")
        if level and hasattr(types, "ThinkingConfig"):
            try:
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level)
            except (TypeError, ValueError):
                pass

        client = genai.Client(api_key=self.api_key)
        resp = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )
        return self._parse(prompt, resp)

    def _parse(self, prompt: str, resp) -> GERP:
        answer_text = getattr(resp, "text", "") or ""
        citations: list[Citation] = []
        issued_queries: list[str] = []

        cand = (resp.candidates or [None])[0]
        gm = getattr(cand, "grounding_metadata", None) if cand else None

        if gm:
            issued_queries = list(getattr(gm, "web_search_queries", None) or [])
            chunks = getattr(gm, "grounding_chunks", None) or []
            supports = getattr(gm, "grounding_supports", None) or []

            # Build chunk_index -> span text and max confidence score.
            span_by_chunk: dict[int, str] = {}
            confidence_by_chunk: dict[int, float] = {}
            for sup in supports:
                seg = getattr(sup, "segment", None)
                seg_text = getattr(seg, "text", None) if seg else None
                indices = getattr(sup, "grounding_chunk_indices", None) or []
                scores = getattr(sup, "confidence_scores", None) or []
                for idx in indices:
                    if seg_text and idx not in span_by_chunk:
                        span_by_chunk[idx] = seg_text
                for idx, score in zip(indices, scores):
                    if score is not None:
                        confidence_by_chunk[idx] = max(
                            confidence_by_chunk.get(idx, 0.0), float(score)
                        )

            for i, chunk in enumerate(chunks):
                web = getattr(chunk, "web", None)
                if not web or not getattr(web, "uri", None):
                    continue
                meta = {}
                if i in confidence_by_chunk:
                    meta["confidence"] = confidence_by_chunk[i]
                c = self._mk_citation(
                    url=web.uri,
                    title=getattr(web, "title", None),
                    cited_text=span_by_chunk.get(i),
                )
                # web.uri is a vertexaisearch.cloud.google.com redirect URL.
                # The real source domain lives in web.domain (Vertex) or, on
                # the Developer API, in web.title (which is the bare domain).
                title = getattr(web, "title", None)
                real_domain = getattr(web, "domain", None) or (
                    title if title and "." in title and " " not in title else None
                )
                if real_domain:
                    c.domain = real_domain
                    meta["redirect_url"] = True
                c.metadata = meta
                citations.append(c)

        gerp = self._new_gerp(
            prompt=prompt,
            answer_text=answer_text.strip(),
            citations=self._dedupe(citations),
            issued_queries=issued_queries,
        )
        return gerp
