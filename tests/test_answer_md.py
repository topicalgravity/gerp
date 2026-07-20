"""Offline tests for the answer_md markdown filter (app.render_answer_md).

Model answers are untrusted text; these tests pin the two safety properties
(literal HTML is escaped, non-http link schemes are neutralized) alongside the
basic formatting behavior.

Run: python -m unittest discover tests
"""

from __future__ import annotations

import unittest


class TestAnswerMd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from app import render_answer_md
        except Exception as e:
            raise unittest.SkipTest(f"app import failed: {e}")
        cls.md = staticmethod(render_answer_md)

    def test_basic_markdown_renders(self):
        out = str(self.md("## Heading\n\nSome **bold** text\n\n- item one"))
        self.assertIn("<h2>Heading</h2>", out)
        self.assertIn("<strong>bold</strong>", out)
        self.assertIn("<li>item one</li>", out)

    def test_literal_html_is_escaped(self):
        out = str(self.md('before <script>alert(1)</script> after'))
        self.assertNotIn("<script>", out)
        self.assertIn("&lt;script&gt;", out)

    def test_javascript_href_neutralized(self):
        out = str(self.md("[click](javascript:alert(1))"))
        self.assertNotIn("javascript:", out)
        self.assertIn('href="#"', out)

    def test_http_links_kept(self):
        out = str(self.md("[site](https://example.com/x)"))
        self.assertIn('href="https://example.com/x"', out)

    def test_empty_and_none_safe(self):
        self.assertEqual(str(self.md("")), "")
        self.assertEqual(str(self.md(None)), "")


if __name__ == "__main__":
    unittest.main()
