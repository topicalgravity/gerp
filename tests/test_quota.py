"""Offline tests for the rolling-window Quota (cookie + in-memory tally).

Uses tiny fake request/response objects so no Flask app or network is needed,
matching the offline style of tests/test_providers.py.

Run: python -m unittest discover tests
"""

from __future__ import annotations

import time
import unittest

from quota import Quota

SECRET = "test-secret"


class FakeReq:
    def __init__(self, cookies=None):
        self.cookies = cookies or {}


class FakeResp:
    """Captures set_cookie calls into a plain dict of name -> value."""
    def __init__(self):
        self.cookies = {}

    def set_cookie(self, name, value, **kw):
        self.cookies[name] = value


def _quota(limit=3, window=3600):
    return Quota(SECRET, "test-salt", "test_cookie", limit, window)


class TestQuota(unittest.TestCase):
    def test_fresh_identity_is_zero(self):
        q = _quota()
        self.assertEqual(q.used(FakeReq(), "ip-1"), 0)
        self.assertEqual(q.remaining(FakeReq(), "ip-1"), 3)

    def test_bump_increments_cookie_and_tally(self):
        q = _quota()
        req, resp = FakeReq(), FakeResp()
        q.bump(resp, req, "ip-1", prior_used=0)
        # Cookie was written; re-reading it shows 1 used.
        req2 = FakeReq(cookies={"test_cookie": resp.cookies["test_cookie"]})
        self.assertEqual(q.used(req2, "ip-1"), 1)
        # In-memory tally also reflects it even without the cookie.
        self.assertEqual(q.used(FakeReq(), "ip-1"), 1)

    def test_used_is_max_of_cookie_and_tally(self):
        q = _quota()
        # Cleared cookie but tally remembers -> count doesn't reset.
        resp = FakeResp()
        q.bump(resp, FakeReq(), "ip-1", 0)
        q.bump(resp, FakeReq(), "ip-1", 1)
        self.assertEqual(q.used(FakeReq(), "ip-1"), 2)  # cookie cleared, tally wins

    def test_limit_reached(self):
        q = _quota(limit=3)
        resp = FakeResp()
        for i in range(3):
            q.bump(resp, FakeReq(), "ip-1", i)
        self.assertEqual(q.used(FakeReq(), "ip-1"), 3)
        self.assertEqual(q.remaining(FakeReq(), "ip-1"), 0)

    def test_window_expiry_resets(self):
        q = _quota(window=1)
        q.bump(FakeResp(), FakeReq(), "ip-1", 0)
        self.assertEqual(q.used(FakeReq(), "ip-1"), 1)
        # Force the tally's window start into the past.
        n, _ = q._tally["ip-1"]
        q._tally["ip-1"] = (n, time.time() - 5)
        self.assertEqual(q.used(FakeReq(), "ip-1"), 0)

    def test_identities_are_independent(self):
        q = _quota()
        q.bump(FakeResp(), FakeReq(), "ip-1", 0)
        self.assertEqual(q.used(FakeReq(), "ip-1"), 1)
        self.assertEqual(q.used(FakeReq(), "ip-2"), 0)

    def test_tampered_cookie_reads_as_fresh(self):
        q = _quota()
        req = FakeReq(cookies={"test_cookie": "not-a-valid-signed-value"})
        self.assertEqual(q.used(req, "ip-1"), 0)


if __name__ == "__main__":
    unittest.main()
