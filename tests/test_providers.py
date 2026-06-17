"""
Offline adapter-parsing tests.

Each provider's run() is split into an API call + _parse(prompt, resp); these
tests feed _parse synthetic response objects (SimpleNamespace mirrors of the
real SDK shapes) so the parsing logic is covered with no network or API spend.

Run: python -m unittest discover tests
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace as NS

from gerp.providers.base import domain_of, BaseProvider
from gerp.providers.anthropic_provider import AnthropicProvider
from gerp.providers.openai_provider import OpenAIProvider, _clean_url
from gerp.providers.gemini_provider import GeminiProvider
from gerp.schema import ConsideredMethod, Citation


class TestBaseHelpers(unittest.TestCase):
    def test_domain_of_strips_www(self):
        self.assertEqual(domain_of("https://www.example.com/p"), "example.com")
        self.assertEqual(domain_of("https://sub.example.com/p"), "sub.example.com")
        self.assertIsNone(domain_of("not a url"))

    def test_dedupe_keeps_first_span_and_counts(self):
        cits = [Citation(url="https://a.com", cited_text="first"),
                Citation(url="https://a.com", cited_text="second"),
                Citation(url="https://b.com")]
        deduped = BaseProvider._dedupe(cits)
        self.assertEqual([c.url for c in deduped], ["https://a.com", "https://b.com"])
        self.assertEqual(deduped[0].cited_text, "first")
        self.assertEqual(deduped[0].count, 2)
        self.assertEqual(deduped[1].count, 1)


class TestAnthropicParse(unittest.TestCase):
    def setUp(self):
        self.p = AnthropicProvider(api_key="test")

    def _resp(self):
        return NS(content=[
            NS(type="server_tool_use", name="web_search", id="tu_1",
               input={"query": "best widgets 2026"}),
            NS(type="web_search_tool_result", tool_use_id="tu_1", content=[
                NS(type="web_search_result", url="https://a.com/x",
                   title="A", page_age="January 2, 2026"),
                NS(type="web_search_result", url="https://b.com/y",
                   title="B", page_age=None),
            ]),
            NS(type="text", text="answer text",
               citations=[NS(url="https://a.com/x", title="A",
                             cited_text="snippet")]),
        ])

    def test_issued_queries_from_server_tool_use(self):
        g = self.p._parse("prompt", self._resp())
        self.assertEqual(g.issued_queries, ["best widgets 2026"])

    def test_citations_carry_page_age(self):
        g = self.p._parse("prompt", self._resp())
        self.assertEqual(len(g.citations), 1)
        self.assertEqual(g.citations[0].url, "https://a.com/x")
        self.assertEqual(g.citations[0].metadata["page_age"], "January 2, 2026")

    def test_provider_delta_considered_set(self):
        g = self.p._parse("prompt", self._resp())
        self.assertEqual(g.considered_method, ConsideredMethod.PROVIDER_DELTA)
        self.assertEqual(len(g.considered_not_cited), 1)
        doc = g.considered_not_cited[0]
        self.assertEqual(doc.url, "https://b.com/y")
        self.assertEqual(doc.source_query, "best widgets 2026")
        self.assertEqual(doc.rank, 2)

    def test_error_result_block_is_skipped(self):
        resp = NS(content=[
            NS(type="web_search_tool_result", tool_use_id="tu_1",
               content=NS(type="web_search_tool_result_error",
                          error_code="unavailable")),
            NS(type="text", text="answer", citations=[]),
        ])
        g = self.p._parse("prompt", resp)
        self.assertEqual(g.considered_method, ConsideredMethod.NONE)
        self.assertEqual(g.answer_text, "answer")


class TestOpenAIParse(unittest.TestCase):
    def setUp(self):
        self.p = OpenAIProvider(api_key="test")

    def _resp(self):
        return NS(output=[
            NS(type="web_search_call",
               action=NS(type="search", queries=["q one", "q two"], query="q one",
                         sources=[
                             NS(type="url", url="https://a.com/?utm_source=openai"),
                             NS(type="url", url="https://b.com/p?x=1&utm_source=openai"),
                         ])),
            NS(type="message", content=[
                NS(type="output_text", text="hello world",
                   annotations=[NS(type="url_citation",
                                   url="https://a.com/?utm_source=openai",
                                   title="A", start_index=0, end_index=5)]),
            ]),
        ])

    def test_clean_url(self):
        self.assertEqual(_clean_url("https://a.com/?utm_source=openai"), "https://a.com/")
        self.assertEqual(_clean_url("https://a.com/p?x=1&utm_source=openai"), "https://a.com/p?x=1")
        self.assertEqual(_clean_url("https://a.com/p"), "https://a.com/p")

    def test_issued_queries_prefer_plural(self):
        g = self.p._parse("prompt", self._resp())
        self.assertEqual(g.issued_queries, ["q one", "q two"])

    def test_citation_span_and_clean_url(self):
        g = self.p._parse("prompt", self._resp())
        self.assertEqual(g.citations[0].url, "https://a.com/")
        self.assertEqual(g.citations[0].cited_text, "hello")

    def test_sources_minus_citations_is_provider_delta(self):
        g = self.p._parse("prompt", self._resp())
        self.assertEqual(g.considered_method, ConsideredMethod.PROVIDER_DELTA)
        self.assertEqual([d.url for d in g.considered_not_cited],
                         ["https://b.com/p?x=1"])
        self.assertEqual(g.considered_not_cited[0].source_query, "q one")


class TestGeminiParse(unittest.TestCase):
    def setUp(self):
        self.p = GeminiProvider(api_key="test")

    def _resp(self, domain=None, title="a.com"):
        gm = NS(
            web_search_queries=["q1"],
            grounding_chunks=[NS(web=NS(
                uri="https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc",
                title=title, domain=domain))],
            grounding_supports=[NS(segment=NS(text="span text"),
                                   grounding_chunk_indices=[0],
                                   confidence_scores=[0.9])],
        )
        return NS(text="answer", candidates=[NS(grounding_metadata=gm)])

    def test_real_domain_recovered_from_title(self):
        g = self.p._parse("prompt", self._resp())
        c = g.citations[0]
        self.assertEqual(c.domain, "a.com")
        self.assertTrue(c.metadata["redirect_url"])
        self.assertEqual(c.metadata["confidence"], 0.9)
        self.assertEqual(c.cited_text, "span text")

    def test_domain_field_wins_over_title(self):
        g = self.p._parse("prompt", self._resp(domain="real.com", title="Some Page"))
        self.assertEqual(g.citations[0].domain, "real.com")

    def test_non_domain_title_falls_back_to_uri_host(self):
        g = self.p._parse("prompt", self._resp(title="A Nice Page"))
        self.assertEqual(g.citations[0].domain, "vertexaisearch.cloud.google.com")
        self.assertNotIn("redirect_url", g.citations[0].metadata)

    def test_issued_queries(self):
        g = self.p._parse("prompt", self._resp())
        self.assertEqual(g.issued_queries, ["q1"])


class FakeBackend:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query

    def search(self, query, num=10):
        return self.results_by_query.get(query, [])


class TestEnrich(unittest.TestCase):
    def _gerp(self):
        from gerp.schema import GERP, SCHEMA_VERSION
        return GERP(
            schema_version=SCHEMA_VERSION, provider="gemini", model="m",
            prompt="p", answer_text="a",
            citations=[Citation(
                url="https://vertexaisearch.cloud.google.com/redir/1",
                domain="cited.com")],
            issued_queries=["q1"],
        )

    def test_excludes_cited_domains_not_just_urls(self):
        from gerp.considered import enrich
        backend = FakeBackend({"q1": [
            {"url": "https://cited.com/deep/page", "title": "Cited dom", "rank": 1},
            {"url": "https://fresh.com/x", "title": "Fresh", "rank": 2},
        ]})
        g = enrich(self._gerp(), backend=backend)
        self.assertEqual(g.considered_method, ConsideredMethod.RERUN_QUERIES)
        self.assertEqual([d.url for d in g.considered_not_cited],
                         ["https://fresh.com/x"])
        doc = g.considered_not_cited[0]
        self.assertEqual(doc.domain, "fresh.com")
        self.assertEqual(doc.source_query, "q1")
        self.assertEqual(doc.rank, 2)

    def test_noop_without_backend_preserves_method(self):
        from gerp.considered import enrich
        g = self._gerp()
        g.considered_method = ConsideredMethod.PROVIDER_DELTA
        g = enrich(g, backend=None)
        self.assertEqual(g.considered_method, ConsideredMethod.PROVIDER_DELTA)

    def test_dedupes_across_queries(self):
        from gerp.considered import enrich
        g = self._gerp()
        g.issued_queries = ["q1", "q2"]
        backend = FakeBackend({
            "q1": [{"url": "https://fresh.com/x", "title": "F", "rank": 1}],
            "q2": [{"url": "https://fresh.com/x", "title": "F", "rank": 1},
                   {"url": "https://other.com/y", "title": "O", "rank": 2}],
        })
        g = enrich(g, backend=backend)
        self.assertEqual([d.url for d in g.considered_not_cited],
                         ["https://fresh.com/x", "https://other.com/y"])


class TestResolveRedirects(unittest.TestCase):
    def _gerp(self):
        from gerp.schema import GERP, SCHEMA_VERSION, ConsideredDoc
        wrap = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc"
        return GERP(
            schema_version=SCHEMA_VERSION, provider="gemini", model="m",
            prompt="p", answer_text="a",
            citations=[
                Citation(url=wrap, domain="cited.com", metadata={"confidence": 0.9}),
                Citation(url="https://direct.com/page", domain="direct.com"),
            ],
            considered_not_cited=[
                ConsideredDoc(url=wrap + "2", domain="other.com"),
            ],
        )

    def test_unwraps_redirects_and_preserves_originals(self):
        from gerp import resolve
        mapping = {
            "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc":
                "https://www.realsource.com/blog/post",
            "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc2":
                "https://second.com/x",
        }
        orig = resolve._resolve_one
        resolve._resolve_one = lambda url, timeout: mapping.get(url)
        try:
            g = resolve.resolve_redirects(self._gerp())
        finally:
            resolve._resolve_one = orig

        cit = g.citations[0]
        self.assertEqual(cit.url, "https://www.realsource.com/blog/post")
        self.assertEqual(cit.domain, "realsource.com")  # www stripped, recomputed
        self.assertEqual(cit.metadata["confidence"], 0.9)  # existing meta kept
        self.assertTrue(cit.metadata["redirect_url"])
        self.assertEqual(
            cit.metadata["grounding_url"],
            "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc")
        # Non-redirect citation is untouched.
        self.assertEqual(g.citations[1].url, "https://direct.com/page")
        # Considered docs are unwrapped too.
        self.assertEqual(g.considered_not_cited[0].url, "https://second.com/x")

    def test_failed_lookup_keeps_wrapper(self):
        from gerp import resolve
        orig = resolve._resolve_one
        resolve._resolve_one = lambda url, timeout: None  # every lookup fails
        try:
            g = resolve.resolve_redirects(self._gerp())
        finally:
            resolve._resolve_one = orig
        self.assertTrue(g.citations[0].url.startswith(
            "https://vertexaisearch.cloud.google.com"))
        self.assertNotIn("grounding_url", g.citations[0].metadata)

    def test_is_redirect(self):
        from gerp.resolve import _is_redirect
        self.assertTrue(_is_redirect(
            "https://vertexaisearch.cloud.google.com/grounding-api-redirect/x"))
        self.assertFalse(_is_redirect("https://example.com/x"))
        self.assertFalse(_is_redirect("not a url"))


if __name__ == "__main__":
    unittest.main()
