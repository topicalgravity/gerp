"""Offline tests for the model tier registry.

Run: python -m unittest discover tests
"""

from __future__ import annotations

import unittest

from gerp.tiers import (TIERS, DEFAULT_TIER, resolve_tier, tier_config,
                        model_label)


class TestResolveTier(unittest.TestCase):
    def test_known_tiers_pass_through(self):
        self.assertEqual(resolve_tier("standard"), "standard")
        self.assertEqual(resolve_tier("frontier"), "frontier")

    def test_unknown_falls_back_to_default(self):
        self.assertEqual(resolve_tier("premium"), DEFAULT_TIER)
        self.assertEqual(resolve_tier(""), DEFAULT_TIER)
        self.assertEqual(resolve_tier(None), DEFAULT_TIER)


class TestTierConfig(unittest.TestCase):
    def test_standard_models(self):
        self.assertEqual(tier_config("standard", "anthropic")["model"],
                         "claude-haiku-4-5")
        self.assertEqual(tier_config("standard", "openai")["model"], "gpt-5.6-luna")
        self.assertEqual(tier_config("standard", "gemini")["model"],
                         "gemini-3.5-flash-lite")

    def test_frontier_models_and_kwargs(self):
        anthropic = tier_config("frontier", "anthropic")
        self.assertEqual(anthropic["model"], "claude-opus-4-8")
        # Deliberately the basic variant even on Opus 4.8: the _20260209
        # dynamic-filtering variant hides the fan-out/pool blocks GERP reads.
        self.assertEqual(anthropic["kwargs"]["tool_type"], "web_search_20250305")
        openai = tier_config("frontier", "openai")
        self.assertEqual(openai["model"], "gpt-5.5")
        self.assertEqual(openai["kwargs"]["reasoning"], {"effort": "high"})
        self.assertEqual(tier_config("frontier", "gemini")["model"],
                         "gemini-3.5-flash")

    def test_unknown_tier_falls_back_to_default_config(self):
        self.assertEqual(tier_config("bogus", "anthropic"),
                         TIERS[DEFAULT_TIER]["anthropic"])

    def test_unknown_provider_yields_empty(self):
        self.assertEqual(tier_config("standard", "nope"), {})


class TestModelLabel(unittest.TestCase):
    def test_known_and_unknown(self):
        self.assertEqual(model_label("claude-opus-4-8"), "Opus 4.8")
        self.assertEqual(model_label("gpt-5.5"), "GPT-5.5")
        self.assertEqual(model_label("some-future-model"), "some-future-model")
        self.assertEqual(model_label(None), "")


class TestAnthropicToolType(unittest.TestCase):
    """The frontier tool_type kwarg must reach the tool definition."""

    def test_default_and_override(self):
        from gerp.providers.anthropic_provider import AnthropicProvider
        p = AnthropicProvider(api_key="test")
        self.assertEqual(p.tool_type, "web_search_20250305")
        p2 = AnthropicProvider(api_key="test", tool_type="web_search_20260209")
        self.assertEqual(p2.tool_type, "web_search_20260209")


if __name__ == "__main__":
    unittest.main()
