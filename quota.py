"""Rolling-window quota: a signed cookie plus an in-memory tally.

Extracted from app.py so the anonymous free-search quota and the signed-in
frontier quota share one implementation instead of copy-paste. Each Quota has
its own cookie name, signer salt, and in-memory tally keyed by an identity
string — the caller decides what "identity" means (client IP for the free
tier, email for the frontier tier).

The window is a rolling one anchored at its FIRST use: a visitor gets `limit`
searches, and the count resets `window` seconds after the window's first
search (not the latest), so `window` after someone starts they get a fresh
allowance. The in-memory tally is a best-effort backstop that catches a visitor
clearing the cookie within one window; it's per-worker and non-persistent by
design (persistent counting waits for the Stage-3 SQLite-on-a-disk work).
"""

from __future__ import annotations

import time

from itsdangerous import BadSignature, URLSafeSerializer


class Quota:
    def __init__(self, secret: str, salt: str, cookie: str, limit: int,
                 window: int):
        self.signer = URLSafeSerializer(secret, salt=salt)
        self.cookie = cookie
        self.limit = limit
        self.window = window
        # identity -> (count, window_start_epoch)
        self._tally: dict[str, tuple[int, float]] = {}

    def _window_active(self, ts: float) -> bool:
        return (time.time() - ts) < self.window

    def cookie_state(self, request) -> tuple[int, float]:
        """(used, window_start) from the signed cookie. A missing, tampered, or
        expired-window cookie reads as a fresh (0, now)."""
        raw = request.cookies.get(self.cookie)
        if raw:
            try:
                n, ts = self.signer.loads(raw)
                n, ts = int(n), float(ts)
                if n >= 0 and self._window_active(ts):
                    return n, ts
            except (BadSignature, ValueError, TypeError):
                pass
        return 0, time.time()

    def tally_state(self, identity: str) -> tuple[int, float]:
        rec = self._tally.get(identity)
        if rec and self._window_active(rec[1]):
            return rec
        return 0, time.time()

    def used(self, request, identity: str) -> int:
        """Searches spent this window: the larger of the signed cookie and the
        in-memory tally, so clearing the cookie doesn't reset the count."""
        return max(self.cookie_state(request)[0], self.tally_state(identity)[0])

    def remaining(self, request, identity: str) -> int:
        return max(0, self.limit - self.used(request, identity))

    def bump(self, resp, request, identity: str, prior_used: int) -> None:
        """Record one more search on both the cookie and the tally, preserving
        the window's start so the reset clock isn't pushed back."""
        cookie_n, cookie_ts = self.cookie_state(request)
        tally_n, tally_ts = self.tally_state(identity)
        # Anchor the window at the earliest active start we have; if neither is
        # active this is a fresh window starting now.
        ts = min(cookie_ts, tally_ts) if (cookie_n or tally_n) else time.time()
        new = prior_used + 1
        resp.set_cookie(self.cookie, self.signer.dumps([new, ts]),
                        max_age=self.window, httponly=True, samesite="Lax")
        self._tally[identity] = (max(tally_n, new), ts)
