"""Offline tests for the SQLite account-request/approval store.

Each test points GERP_DATA_DIR at a fresh temp dir (the store resolves the
path at call time), so nothing touches a real database.

Run: python -m unittest discover tests
"""

from __future__ import annotations

import os
import tempfile
import unittest

import store


class TestStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = os.environ.get("GERP_DATA_DIR")
        os.environ["GERP_DATA_DIR"] = self._tmp.name

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("GERP_DATA_DIR", None)
        else:
            os.environ["GERP_DATA_DIR"] = self._saved
        self._tmp.cleanup()

    def test_request_then_approve_round_trip(self):
        store.record_request("Ada", "Lovelace", "Ada@Engines.example", "Engines")
        self.assertFalse(store.is_approved("ada@engines.example"))
        self.assertEqual(store.approve("ada@engines.example"), "approved")
        # Approval and lookups are case-insensitive (emails lowercased).
        self.assertTrue(store.is_approved("ADA@engines.example"))
        self.assertEqual(store.count_approved(), 1)

    def test_approve_is_idempotent(self):
        store.record_request("A", "B", "x@example.com", "C")
        self.assertEqual(store.approve("x@example.com"), "approved")
        self.assertEqual(store.approve("x@example.com"), "already")
        self.assertEqual(store.count_approved(), 1)

    def test_approve_unknown_email_still_approves(self):
        # The owner can approve an address that never used the form.
        self.assertEqual(store.approve("walkin@example.com"), "unknown")
        self.assertTrue(store.is_approved("walkin@example.com"))

    def test_rerequest_preserves_approval(self):
        store.record_request("A", "B", "x@example.com", "OldCo")
        store.approve("x@example.com")
        store.record_request("A", "B", "x@example.com", "NewCo")
        self.assertTrue(store.is_approved("x@example.com"))

    def test_fresh_db_counts_zero(self):
        self.assertEqual(store.count_approved(), 0)


if __name__ == "__main__":
    unittest.main()
