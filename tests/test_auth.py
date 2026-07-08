"""Offline tests for magic-link auth.

Token round-trip / expiry / tamper are tested against the exact serializer
contract auth.init_auth uses (URLSafeTimedSerializer, salt "gerp-magic-link").
Route-level tests import the Flask app and are skipped if Flask isn't installed,
keeping the core token tests runnable in the no-dependency offline style.

Run: python -m unittest discover tests
"""

from __future__ import annotations

import time
import unittest

from itsdangerous import (BadSignature, SignatureExpired,
                          URLSafeTimedSerializer)

import os
import re
import tempfile

from auth import (MAGIC_LINK_MAX_AGE, _email_rate_ok, _EMAIL_MAX_IN_WINDOW,
                  _is_approved)

SALT = "gerp-magic-link"
SECRET = "test-secret-key"

# Point the approval store at a throwaway dir for the whole module: _is_approved
# consults SQLite now, and these tests must never read or write a real DB.
_tmp_data = None
_saved_data_dir = None


def setUpModule():
    global _tmp_data, _saved_data_dir
    _tmp_data = tempfile.TemporaryDirectory()
    _saved_data_dir = os.environ.get("GERP_DATA_DIR")
    os.environ["GERP_DATA_DIR"] = _tmp_data.name


def tearDownModule():
    if _saved_data_dir is None:
        os.environ.pop("GERP_DATA_DIR", None)
    else:
        os.environ["GERP_DATA_DIR"] = _saved_data_dir
    _tmp_data.cleanup()


class TestMagicLinkToken(unittest.TestCase):
    def setUp(self):
        self.signer = URLSafeTimedSerializer(SECRET, salt=SALT)

    def test_round_trip(self):
        token = self.signer.dumps("user@example.com")
        self.assertEqual(self.signer.loads(token, max_age=MAGIC_LINK_MAX_AGE),
                         "user@example.com")

    def test_expired_token_raises(self):
        token = self.signer.dumps("user@example.com")
        # Any positive age exceeds max_age=-1, so this reads as expired.
        with self.assertRaises(SignatureExpired):
            self.signer.loads(token, max_age=-1)

    def test_tampered_token_raises(self):
        token = self.signer.dumps("user@example.com")
        tampered = token[:-2] + ("aa" if not token.endswith("aa") else "bb")
        with self.assertRaises(BadSignature):
            self.signer.loads(tampered, max_age=MAGIC_LINK_MAX_AGE)

    def test_wrong_salt_rejected(self):
        token = self.signer.dumps("user@example.com")
        other = URLSafeTimedSerializer(SECRET, salt="not-the-magic-salt")
        with self.assertRaises(BadSignature):
            other.loads(token, max_age=MAGIC_LINK_MAX_AGE)


class TestPerEmailRateLimit(unittest.TestCase):
    def test_caps_after_window_max(self):
        email = f"cap-{time.time()}@example.com"
        for _ in range(_EMAIL_MAX_IN_WINDOW):
            self.assertTrue(_email_rate_ok(email))
        # The next request in the same window is refused.
        self.assertFalse(_email_rate_ok(email))


class _EnvMixin:
    """Set/restore env vars around a test."""

    def set_env(self, **kv):
        self._saved = {k: os.environ.get(k) for k in kv}
        for k, v in kv.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestApproval(_EnvMixin, unittest.TestCase):
    def test_open_when_no_approval_config(self):
        self.set_env(GERP_APPROVED_EMAILS=None, GERP_FRONTIER_ALLOWLIST=None)
        self.assertTrue(_is_approved("anyone@example.com"))

    def test_only_listed_emails_approved(self):
        self.set_env(GERP_APPROVED_EMAILS="A@example.com, b@example.com",
                     GERP_FRONTIER_ALLOWLIST=None)
        self.assertTrue(_is_approved("a@example.com"))   # case-insensitive
        self.assertTrue(_is_approved("b@example.com"))
        self.assertFalse(_is_approved("intruder@example.com"))

    def test_frontier_allowlist_is_implicitly_approved(self):
        self.set_env(GERP_APPROVED_EMAILS=None,
                     GERP_FRONTIER_ALLOWLIST="tester@example.com")
        self.assertTrue(_is_approved("tester@example.com"))
        self.assertFalse(_is_approved("other@example.com"))


class TestAuthRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import app  # noqa: F401
        except Exception as e:  # flask not installed in this interpreter
            raise unittest.SkipTest(f"app import failed: {e}")
        cls.app = app.app
        cls.app.config["TESTING"] = True
        # Skip the Turnstile bot-check in tests. _verify_turnstile reads this
        # module global at call time, so clearing it makes verification a no-op.
        app.TURNSTILE_SECRET_KEY = None
        # Not testing rate limits here; the shared in-memory limiter would
        # otherwise 429 once the module's POSTs exceed 3/minute.
        app.limiter.enabled = False

    def test_request_shows_confirmation_and_logs_link_in_dev(self):
        import auth
        # Open-mode (no approval config) needs an EMPTY store: other classes
        # in this module may have approved emails in the shared module temp
        # DB, so this test gets its own fresh data dir.
        fresh = tempfile.TemporaryDirectory()
        self.addCleanup(fresh.cleanup)
        prior = os.environ.get("GERP_DATA_DIR")
        os.environ["GERP_DATA_DIR"] = fresh.name
        self.addCleanup(lambda: os.environ.__setitem__("GERP_DATA_DIR", prior))
        captured = {}

        def _capture(email, link):
            captured["email"] = email
            captured["link"] = link

        orig = auth._send_magic_link
        # Patch the module symbol the route closure calls.
        auth._send_magic_link = _capture
        try:
            c = self.app.test_client()
            r = c.post("/auth/request", data={"email": "Alice@Example.com"})
            self.assertEqual(r.status_code, 200)
            self.assertIn(b"Check your email", r.data)
            # Email is lowercased before signing/sending.
            self.assertEqual(captured.get("email"), "alice@example.com")
            self.assertIn("/auth/verify?token=", captured.get("link", ""))
        finally:
            auth._send_magic_link = orig

        # The captured token verifies and signs the user in.
        token = captured["link"].split("token=", 1)[1]
        c = self.app.test_client()
        r = c.get(f"/auth/verify?token={token}")
        self.assertEqual(r.status_code, 302)
        with c.session_transaction() as sess:
            self.assertEqual(sess.get("email"), "alice@example.com")

    def test_invalid_token_errors(self):
        c = self.app.test_client()
        r = c.get("/auth/verify?token=garbage")
        self.assertEqual(r.status_code, 400)
        self.assertIn(b"sign-in link didn", r.data)

    def test_logout_clears_session(self):
        c = self.app.test_client()
        with c.session_transaction() as sess:
            sess["email"] = "bob@example.com"
        r = c.get("/logout")
        self.assertEqual(r.status_code, 302)
        with c.session_transaction() as sess:
            self.assertIsNone(sess.get("email"))

    def test_unapproved_email_gets_no_link_but_same_page(self):
        import auth
        os.environ["GERP_APPROVED_EMAILS"] = "vip@example.com"
        self.addCleanup(os.environ.pop, "GERP_APPROVED_EMAILS", None)
        sent = []
        orig = auth._send_magic_link
        auth._send_magic_link = lambda email, link: sent.append(email)
        try:
            c = self.app.test_client()
            r = c.post("/auth/request", data={"email": "nobody@example.com"})
            # Same neutral confirmation — no enumeration — but nothing sent.
            self.assertEqual(r.status_code, 200)
            self.assertIn(b"Check your email", r.data)
            self.assertEqual(sent, [])

            r = c.post("/auth/request", data={"email": "vip@example.com"})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(sent, ["vip@example.com"])
        finally:
            auth._send_magic_link = orig


class TestRequestAccountRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import app  # noqa: F401
        except Exception as e:
            raise unittest.SkipTest(f"app import failed: {e}")
        cls.app = app.app
        cls.app.config["TESTING"] = True
        app.TURNSTILE_SECRET_KEY = None
        # Not testing rate limits here; the shared in-memory limiter would
        # otherwise 429 once the module's POSTs exceed 3/minute.
        app.limiter.enabled = False

    FIELDS = {"first_name": "Ada", "last_name": "Lovelace",
              "email": "ada@analytical.engines", "company": "Analytical Engines"}

    def test_get_renders_form(self):
        r = self.app.test_client().get("/auth/request-account")
        self.assertEqual(r.status_code, 200)
        for name in self.FIELDS:
            self.assertIn(f'name="{name}"'.encode(), r.data)

    def test_missing_field_rejected_and_nothing_sent(self):
        import auth
        sent = []
        orig = auth._send_account_request
        auth._send_account_request = lambda *a: sent.append(a)
        try:
            data = dict(self.FIELDS, company="")
            r = self.app.test_client().post("/auth/request-account", data=data)
            self.assertEqual(r.status_code, 400)
            self.assertIn(b"All fields are required", r.data)
            self.assertEqual(sent, [])
        finally:
            auth._send_account_request = orig

    def test_valid_request_notifies_owner_without_exposing_recipient(self):
        import auth
        captured = {}
        orig = auth._send_email
        auth._send_email = (lambda to, subject, body, reply_to=None:
                            captured.update(to=to, subject=subject,
                                            body=body, reply_to=reply_to))
        os.environ["GERP_ACCOUNT_REQUEST_TO"] = "owner@secret.example"
        self.addCleanup(os.environ.pop, "GERP_ACCOUNT_REQUEST_TO", None)
        try:
            r = self.app.test_client().post("/auth/request-account",
                                            data=self.FIELDS)
            self.assertEqual(r.status_code, 200)
            self.assertIn(b"Request received", r.data)
            # The notification goes to the env-configured owner address with
            # Reply-To the requester...
            self.assertEqual(captured["to"], "owner@secret.example")
            self.assertEqual(captured["reply_to"], self.FIELDS["email"])
            for v in self.FIELDS.values():
                self.assertIn(v, captured["subject"] + captured["body"])
            # ...and that address never appears in the rendered page.
            self.assertNotIn(b"owner@secret.example", r.data)
        finally:
            auth._send_email = orig


class TestApproveFlow(unittest.TestCase):
    """End-to-end: request → one-click approve → requester can sign in."""

    @classmethod
    def setUpClass(cls):
        try:
            import app  # noqa: F401
        except Exception as e:
            raise unittest.SkipTest(f"app import failed: {e}")
        cls.app = app.app
        cls.app.config["TESTING"] = True
        app.TURNSTILE_SECRET_KEY = None
        # Not testing rate limits here; the shared in-memory limiter would
        # otherwise 429 once the module's POSTs exceed 3/minute.
        app.limiter.enabled = False

    def test_full_approval_flow(self):
        import auth
        emails = []  # (to, subject, body)
        orig = auth._send_email
        auth._send_email = (lambda to, subject, body, reply_to=None:
                            emails.append((to, subject, body)))
        # Gate must be CLOSED (env list non-empty) so the requester starts
        # unapproved, and the owner address set so the notification "sends".
        os.environ["GERP_APPROVED_EMAILS"] = "someone-else@example.com"
        os.environ["GERP_ACCOUNT_REQUEST_TO"] = "owner@secret.example"
        self.addCleanup(os.environ.pop, "GERP_APPROVED_EMAILS", None)
        self.addCleanup(os.environ.pop, "GERP_ACCOUNT_REQUEST_TO", None)
        try:
            c = self.app.test_client()

            # 1. Request an account → owner email carries an approve link.
            r = c.post("/auth/request-account", data={
                "first_name": "Grace", "last_name": "Hopper",
                "email": "grace@navy.example", "company": "USN"})
            self.assertEqual(r.status_code, 200)
            to, _, body = emails[-1]
            self.assertEqual(to, "owner@secret.example")
            m = re.search(r'href="([^"]*?/auth/approve\?token=[^"]+)"', body)
            self.assertIsNotNone(m, body)
            approve_link = m.group(1)

            # 2. Before approval: sign-in request sends nothing.
            n = len(emails)
            r = c.post("/auth/request", data={"email": "grace@navy.example"})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(len(emails), n)

            # 3. One click approves and notifies the requester.
            r = c.get(approve_link)
            self.assertEqual(r.status_code, 200)
            self.assertIn(b"Approved", r.data)
            to, subject, _ = emails[-1]
            self.assertEqual(to, "grace@navy.example")
            self.assertIn("approved", subject.lower())

            # 4. Now the magic link sends.
            r = c.post("/auth/request", data={"email": "grace@navy.example"})
            self.assertEqual(r.status_code, 200)
            to, subject, _ = emails[-1]
            self.assertEqual(to, "grace@navy.example")
            self.assertIn("sign-in link", subject)

            # 5. Re-clicking the approve link is a harmless no-op.
            n = len(emails)
            r = c.get(approve_link)
            self.assertEqual(r.status_code, 200)
            self.assertIn(b"Already approved", r.data)
            self.assertEqual(len(emails), n)
        finally:
            auth._send_email = orig

    def test_garbage_approve_token_rejected(self):
        r = self.app.test_client().get("/auth/approve?token=garbage")
        self.assertEqual(r.status_code, 400)
        self.assertIn(b"approve link didn", r.data)


if __name__ == "__main__":
    unittest.main()
