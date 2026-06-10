#!/usr/bin/env python3
"""
CLI: run a prompt through a provider, emit a GERP as JSON.

Examples:
  python -m gerp.cli --provider gemini --prompt "best personal injury firms in Dallas"
  python -m gerp.cli -p anthropic --prompt "..." --model claude-opus-4-6 --raw
  echo "long prompt" | python -m gerp.cli -p openai --stdin
"""

from __future__ import annotations

import argparse
import json
import sys

from .runner import run, PROVIDERS
from .providers.base import ProviderError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build a GERP from an LLM API call.")
    ap.add_argument("-p", "--provider", required=True, choices=sorted(PROVIDERS))
    ap.add_argument("--prompt", help="Prompt text (or use --stdin)")
    ap.add_argument("--stdin", action="store_true", help="Read prompt from stdin")
    ap.add_argument("--model", default=None, help="Override default model")
    ap.add_argument("--api-key", default=None, help="Override env API key")
    ap.add_argument("--raw", action="store_true", help="Include raw_response")
    ap.add_argument("--indent", type=int, default=2)
    args = ap.parse_args(argv)

    prompt = sys.stdin.read().strip() if args.stdin else args.prompt
    if not prompt:
        ap.error("Provide --prompt or --stdin")

    try:
        gerp = run(prompt, provider=args.provider, model=args.model,
                   api_key=args.api_key)
    except ProviderError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 2

    print(json.dumps(gerp.to_dict(include_raw=args.raw), indent=args.indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
