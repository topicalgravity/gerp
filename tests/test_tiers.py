"""Offline tests for the model tier registry.

Run: python -m unittest discover tests
"""

from __future__ import annotations

import unittest

from gerp.runner import PROVIDERS
from gerp.tiers import (TIERS, DEFAULT_TIER, MODEL_LABELS, resolve_tier,
                        tier_config, model_label)


class TestResolveTier(unittest.TestCase):
    def test_known_tiers_pass_through(self):
        self.assertEqual(resolve_tier("standard"), "standard")
        self.assertEqual(resolve_tier("frontier"), "frontier")

    def test_unknown_falls_back_to_default(self):
        self.assertEqual(resolve_tier("premium"), DEFAULT_TIER)
        self.assertEqual(resolve_tier(""), DEFAULT_TIER)
        self.assertEqual(resolve_tier(None), DEFAULT_TIER)


class TestTierConfig(unittest.TestCase):
    # These assert the registry's shape, not specific model ids, so a model
    # bump in tiers.py needs no test edits.
    def test_every_tier_has_a_labelled_model_per_provider(self):
        for tier, providers in TIERS.items():
            self.assertEqual(set(providers), set(PROVIDERS), tier)
            for provider, entry in providers.items():
                self.assertTrue(entry.get("model"), (tier, provider))
                self.assertIn(entry["model"], MODEL_LABELS, (tier, provider))

    def test_tiers_use_different_models(self):
        for provider in PROVIDERS:
            self.assertNotEqual(tier_config("standard", provider)["model"],
                                tier_config("frontier", provider)["model"])

    def test_frontier_kwargs(self):
        # The basic variant, pinned so the flat parser sees the fan-out/pool
        # blocks (see tiers.py for the 2026-09 note that dynamic filtering no
        # longer hides those blocks).
        anthropic = tier_config("frontier", "anthropic")
        self.assertEqual(anthropic["kwargs"]["tool_type"], "web_search_20250305")
        openai = tier_config("frontier", "openai")
        self.assertEqual(openai["kwargs"]["reasoning"], {"effort": "high"})

    def test_provider_default_is_the_default_tier_model(self):
        for name, cls in PROVIDERS.items():
            self.assertEqual(cls(api_key="test").model,
                             tier_config(DEFAULT_TIER, name)["model"])

    def test_unknown_tier_falls_back_to_default_config(self):
        self.assertEqual(tier_config("bogus", "anthropic"),
                         TIERS[DEFAULT_TIER]["anthropic"])

    def test_unknown_provider_yields_empty(self):
        self.assertEqual(tier_config("standard", "nope"), {})


class TestModelLabel(unittest.TestCase):
    def test_known_and_unknown(self):
        model, label = next(iter(MODEL_LABELS.items()))
        self.assertEqual(model_label(model), label)
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
