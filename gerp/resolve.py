"""
Redirect-unwrapping layer.

Gemini returns its grounding citations as vertexaisearch.cloud.google.com
redirect URLs (a 302 that points at the real publisher page). This pass swaps
those wrappers for the URL they redirect to, so results show the actual source.

It is deliberately lightweight: it reads the *Location* header of the first
redirect (no following through to the destination, no page fetch/render — a
crawler/Firecrawl would be overkill for unwrapping a 302). Every lookup is
bounded by a timeout and runs in parallel, and any failure simply keeps the
original wrapper URL, so this can never fail or meaningfully slow a search.
"""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from .schema import GERP
from .providers.base import domain_of

REDIRECT_HOST = "vertexaisearch.cloud.google.com"
_REDIRECT_CODES = {301, 302, 303, 307, 308}


def _is_redirect(url: str) -> bool:
    try:
        return urllib.parse.urlparse(url).netloc.endswith(REDIRECT_HOST)
    except Exception:
        return False


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Stop urllib from following the redirect so we never touch the
    destination server — we only want the first hop's Location header."""
    def redirect_request(self, *args, **kwargs):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _resolve_one(url: str, timeout: float) -> str | None:
    """Return the URL this redirect points to, or None on any failure."""
    req = urllib.request.Request(url, headers={"User-Agent": "gerp/1.0"})
    try:
        _OPENER.open(req, timeout=timeout)
        return None  # 2xx with no redirect — nothing to unwrap
    except urllib.error.HTTPError as e:
        if e.code in _REDIRECT_CODES:
            loc = e.headers.get("Location")
            return urllib.parse.urljoin(url, loc) if loc else None
        return None
    except Exception:
        return None


def resolve_redirects(gerp: GERP, max_workers: int = 8,
                      timeout: float = 5.0) -> GERP:
    """Unwrap any redirect URLs among the citations and considered docs,
    rewriting url + domain in place. Additive and fail-safe."""
    targets = [t for t in (*gerp.citations, *gerp.considered_not_cited)
               if _is_redirect(getattr(t, "url", "") or "")]
    if not targets:
        return gerp

    ex = ThreadPoolExecutor(max_workers=min(max_workers, len(targets)))
    futures = {ex.submit(_resolve_one, t.url, timeout): t for t in targets}
    try:
        for fut in as_completed(futures, timeout=timeout + 2):
            t = futures[fut]
            try:
                real = fut.result()
            except Exception:
                real = None
            if not real:
                continue
            t.metadata = dict(t.metadata or {})
            t.metadata["grounding_url"] = t.url   # keep the original wrapper
            t.metadata["redirect_url"] = True
            t.url = real
            dom = domain_of(real)
            if dom:
                t.domain = dom
    except TimeoutError:
        pass  # slow redirects just keep their wrapper URL
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    return gerp
