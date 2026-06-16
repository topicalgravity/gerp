"""Regenerate the committed demo runs shown on the GERP home page.

These are real searches captured once and checked into the repo so every
visitor sees the same curated examples — real user searches are never listed
(that would leak one visitor's queries to the next).

Usage (from the repo root, with provider keys in .env):

    .venv/bin/python demo/generate_demos.py

Edit DEMOS below to change which examples appear, then re-run. Each entry
spends real API credits across all three providers.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gerp as g
from gerp.considered import enrich, SerpAPIBackend
from gerp.schema import GERP, ConsideredMethod

DEMO_DIR = Path(__file__).resolve().parent

# (stable id used in the URL, prompt). Keep ids URL-safe (alnum + hyphen).
DEMOS = [
    ("demo-ai-search-optimization",
     "how do you optimize a website to be cited by AI search engines"),
    ("demo-project-management-tools",
     "best project management software for small remote teams"),
    ("demo-ev-home-charger",
     "what to look for when buying a home EV charger"),
]


def _run_one(prompt: str) -> dict:
    providers = sorted(g.PROVIDERS.keys())
    results: dict = {}
    for p in providers:
        try:
            results[p] = g.run(prompt, provider=p)
        except Exception as e:  # capture, mirroring the app's per-provider errors
            results[p] = {"error": str(e)}

    if os.environ.get(SerpAPIBackend.env_key):
        backend = SerpAPIBackend()
        for r in results.values():
            if (isinstance(r, GERP) and r.issued_queries
                    and r.considered_method == ConsideredMethod.NONE):
                try:
                    enrich(r, backend=backend)
                except Exception:
                    pass
    return {p: results[p] for p in providers if p in results}


def main() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    for run_id, prompt in DEMOS:
        print(f"Running '{prompt}' …", flush=True)
        results = _run_one(prompt)
        payload = {
            "id": run_id,
            "prompt": prompt,
            "created_at": "",  # demos are timeless; home page hides empty dates
            "results": {
                p: (r.to_dict() if isinstance(r, GERP) else r)
                for p, r in results.items()
            },
        }
        out = DEMO_DIR / f"{run_id}.json"
        out.write_text(json.dumps(payload, indent=2))
        ok = sum(1 for r in results.values() if "error" not in (r if isinstance(r, dict) else {}))
        print(f"  → wrote {out.name} ({ok}/{len(results)} providers OK)")


if __name__ == "__main__":
    main()
