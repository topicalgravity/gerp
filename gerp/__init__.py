from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
except ImportError:
    pass

from .runner import run, get_provider, PROVIDERS

__all__ = ["run", "get_provider", "PROVIDERS"]
