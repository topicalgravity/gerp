"""GERP account: passwordless email magic-link sign-in.

Deliberately NOT Google Sign-In. The site mimics google.com, so the auth flow
is fully GERP-branded and never mentions Google — a GERP account keyed on the
user's email. Identity lives in Flask's signed session cookie (no database yet;
SQLite arrives with Stripe in Stage 3), so a signed-in user is simply a session
carrying {"email": ...}.

The magic link itself is a short-lived itsdangerous token (email + timestamp,
15-minute expiry). Email delivery goes through Resend's HTTP API via
urllib.request — no new pip dependency. When RESEND_API_KEY is unset the link is
logged to stdout instead of sent, mirroring app.py's Turnstile-skip pattern so
local dev needs no Resend account.

Accounts are request-based: sign-in links only go to approved emails —
approved in the SQLite store (store.py) or listed in the env bootstrap/override
(GERP_APPROVED_EMAILS ∪ GERP_FRONTIER_ALLOWLIST). Everyone else uses the
"Request an account" form, which records the request and emails it to the
address in GERP_ACCOUNT_REQUEST_TO — env-only, so the owner's inbox never
appears in templates, page source, or the repo. That email carries a one-click
Approve link (signed 30-day token); clicking it flips the store row and emails
the requester that they can sign in.
"""

from __future__ import annotations

import html
import json
import os
import time
import urllib.error
import urllib.request
from datetime import timedelta

from flask import (Blueprint, redirect, render_template, request, session,
                   url_for)
from itsdangerous import (BadSignature, SignatureExpired,
                          URLSafeTimedSerializer)

import store

# Magic links expire 15 minutes after being issued.
MAGIC_LINK_MAX_AGE = 15 * 60
# Approve links (in the owner's notification email) stay valid for 30 days.
APPROVE_LINK_MAX_AGE = 30 * 24 * 3600

RESEND_API_KEY = os.environ.get("RESEND_API_KEY")
RESEND_ENDPOINT = "https://api.resend.com/emails"
# From address for the sign-in email. Requires a one-time Resend domain
# verification (SPF/DKIM DNS records) before real sends succeed.
FROM_EMAIL = os.environ.get("GERP_FROM_EMAIL", "GERP <signin@topicalgravity.com>")

def _approved_emails() -> set[str]:
    """Emails allowed to receive a sign-in link. Read at call time (not import)
    so approvals can be updated with a config change + restart, and so tests can
    toggle the env. The frontier allowlist is implicitly approved — those are
    the owner's testers."""
    raw = (os.environ.get("GERP_APPROVED_EMAILS", "") + ","
           + os.environ.get("GERP_FRONTIER_ALLOWLIST", ""))
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def _is_approved(email: str) -> bool:
    """Accounts are request-based: only approved emails get sign-in links —
    approved in the store (one-click link in the owner's email) or listed in
    the env bootstrap/override. With NO approval configuration anywhere (env
    lists empty AND zero approvals in the DB), sign-in stays open — the
    fresh-local-checkout escape hatch, same pattern as the Turnstile/Resend
    skips. Set GERP_APPROVED_EMAILS (at least the owner's address) in prod so
    the gate is closed from first boot."""
    env_approved = _approved_emails()
    if email in env_approved or store.is_approved(email):
        return True
    return not env_approved and store.count_approved() == 0


# Per-email in-memory backstop on top of the per-IP flask-limiter rule: caps how
# many links one address can request in a window, so a scraped address can't be
# spammed. Best-effort (in-memory, one worker) — same design as the free-quota
# IP tally. Maps lowercased email -> list of request epochs in the window.
_EMAIL_WINDOW = 3600          # 1 hour
_EMAIL_MAX_IN_WINDOW = 5
_email_requests: dict[str, list[float]] = {}


def _email_rate_ok(email: str) -> bool:
    """True if this email is under its per-window cap; records the attempt."""
    now = time.time()
    recent = [t for t in _email_requests.get(email, []) if now - t < _EMAIL_WINDOW]
    if len(recent) >= _EMAIL_MAX_IN_WINDOW:
        _email_requests[email] = recent
        return False
    recent.append(now)
    _email_requests[email] = recent
    return True


def _send_email(to: str, subject: str, body_html: str,
                reply_to: str | None = None) -> None:
    """Send one email via Resend. If RESEND_API_KEY is unset (local dev), log
    to stdout instead of sending — same escape hatch as the Turnstile skip.
    Never raises: send failures are swallowed so callers can always show the
    same neutral confirmation (no account enumeration)."""
    if not RESEND_API_KEY:
        print(f"[gerp-auth] DEV email to {to} | {subject}\n{body_html}", flush=True)
        return
    fields = {"from": FROM_EMAIL, "to": [to], "subject": subject,
              "html": body_html}
    if reply_to:
        fields["reply_to"] = reply_to
    req = urllib.request.Request(
        RESEND_ENDPOINT, data=json.dumps(fields).encode(), method="POST",
        headers={"Authorization": f"Bearer {RESEND_API_KEY}",
                 "Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=10).read()
    except urllib.error.HTTPError as e:
        # Resend puts the actual reason (unverified domain, key scoped to a
        # different domain, bad from-address, ...) in the response body — log
        # it, or a bare "403 Forbidden" is undiagnosable.
        try:
            detail = e.read().decode(errors="replace")[:300]
        except Exception:  # noqa: BLE001
            detail = ""
        print(f"[gerp-auth] Resend send failed for {to}: {e} {detail}",
              flush=True)
    except Exception as e:  # noqa: BLE001 - never leak a send failure to the user
        print(f"[gerp-auth] Resend send failed for {to}: {e}", flush=True)


def _send_magic_link(email: str, link: str) -> None:
    _send_email(
        email, "Your GERP sign-in link",
        "<p>Here's your sign-in link for GERP. It expires in 15 minutes.</p>"
        f'<p><a href="{link}">Sign in to GERP</a></p>'
        "<p>No password needed. If you didn't request this, you can ignore "
        "this email.</p>",
    )


def _send_account_request(first: str, last: str, email: str, company: str,
                          approve_link: str) -> None:
    """Notify the owner of an account request. The recipient comes ONLY from
    GERP_ACCOUNT_REQUEST_TO (env) — never templates or page source. Reply-To is
    the requester. The Approve button is a signed 30-day one-click link that
    flips the store row and emails the requester — no Render dashboard visit.
    All fields are user-supplied: escape them before they land in HTML."""
    to = os.environ.get("GERP_ACCOUNT_REQUEST_TO")
    f, l, e, c = (html.escape(v) for v in (first, last, email, company))
    body = (
        f"<p>New GERP account request:</p>"
        f"<p><strong>{f} {l}</strong><br>{e}<br>{c}</p>"
        f'<p><a href="{approve_link}" style="display:inline-block;'
        f"background:#4285F4;color:#fff;padding:10px 22px;border-radius:4px;"
        f'text-decoration:none;">Approve {e}</a></p>'
        f"<p>Approving emails them that they can sign in. To decline, just "
        f"ignore this. The link works for 30 days.</p>"
    )
    if not to:
        print(f"[gerp-auth] DEV account request (GERP_ACCOUNT_REQUEST_TO "
              f"unset):\n{body}", flush=True)
        return
    _send_email(to, f"GERP account request: {first} {last} ({company})",
                body, reply_to=email)


def init_auth(app, limiter, verify_turnstile):
    """Register the magic-link routes, session config, and the user context
    processor on the app. Takes the app's limiter and _verify_turnstile so auth
    reuses the exact bot-check + rate-limit machinery without a circular import.

    Returns the blueprint (registered) for reference/testing."""
    signer = URLSafeTimedSerializer(app.secret_key, salt="gerp-magic-link")
    approve_signer = URLSafeTimedSerializer(app.secret_key, salt="gerp-approve")
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)

    bp = Blueprint("auth", __name__)

    @bp.route("/auth/request", methods=["POST"])
    @limiter.limit("3 per minute")
    @limiter.limit("10 per hour")
    def auth_request():
        email = request.form.get("email", "").strip().lower()

        # Human check reuses the search form's Turnstile machinery.
        if not verify_turnstile():
            return render_template(
                "error.html", title="Couldn't verify you're human",
                detail="Please head back and try requesting your sign-in link "
                       "again.", show_key_form=False), 400

        # Send only for a plausible, APPROVED address that's under its
        # per-email cap. We ALWAYS render the same confirmation regardless —
        # no account enumeration, no signal about whether a link was actually
        # sent. Unapproved visitors are pointed at the request-account form by
        # the confirmation copy.
        if (email and "@" in email and _is_approved(email)
                and _email_rate_ok(email)):
            token = signer.dumps(email)
            link = url_for("auth.auth_verify", token=token, _external=True)
            _send_magic_link(email, link)

        return render_template("auth_sent.html", email=email)

    @bp.route("/auth/request-account", methods=["GET", "POST"])
    @limiter.limit("3 per minute", methods=["POST"])
    @limiter.limit("10 per hour", methods=["POST"])
    def request_account():
        if request.method == "GET":
            return render_template("request_account.html", sent=False)

        if not verify_turnstile():
            return render_template(
                "request_account.html", sent=False,
                error="Couldn't verify you're human — please try again.",
                form=request.form), 400

        fields = {k: request.form.get(k, "").strip()
                  for k in ("first_name", "last_name", "email", "company")}
        if not all(fields.values()) or "@" not in fields["email"]:
            return render_template(
                "request_account.html", sent=False,
                error="All fields are required.", form=request.form), 400

        # Reuse the per-email cap so one address can't flood the owner's inbox
        # even from rotating IPs.
        if _email_rate_ok(fields["email"].lower()):
            store.record_request(fields["first_name"], fields["last_name"],
                                 fields["email"], fields["company"])
            approve_link = url_for(
                "auth.auth_approve",
                token=approve_signer.dumps(fields["email"].lower()),
                _external=True)
            _send_account_request(fields["first_name"], fields["last_name"],
                                  fields["email"], fields["company"],
                                  approve_link)

        return render_template("request_account.html", sent=True,
                               email=fields["email"])

    @bp.route("/auth/approve")
    def auth_approve():
        # The signed token IS the credential: it's unguessable and only ever
        # sent to the owner's inbox, so the owner can approve from any device
        # with no login. error.html doubles as the app's generic notice page.
        token = request.args.get("token", "")
        try:
            email = approve_signer.loads(token, max_age=APPROVE_LINK_MAX_AGE)
        except (SignatureExpired, BadSignature):
            return render_template(
                "error.html", title="This approve link didn't work",
                detail="It may be older than 30 days or damaged. Find the "
                       "original request email and try again, or have them "
                       "re-request an account.", show_key_form=False), 400

        status = store.approve(email)
        if status == "already":
            return render_template(
                "error.html", title="Already approved",
                detail=f"{email} was already approved — nothing to do. "
                       "They can sign in from the GERP home page.",
                show_key_form=False)

        _send_email(
            email, "Your GERP account is approved",
            "<p>Good news — your GERP account request was approved.</p>"
            '<p>Head to <a href="https://gerp.topicalgravity.com">'
            "gerp.topicalgravity.com</a>, click <strong>Sign in</strong>, and "
            "enter this email address. We'll send you a sign-in link — no "
            "password needed.</p>",
        )
        return render_template(
            "error.html", title="Approved",
            detail=f"{email} can now sign in, and we've emailed them to say "
                   "so. Nothing else to do.", show_key_form=False)

    @bp.route("/auth/verify")
    def auth_verify():
        token = request.args.get("token", "")
        try:
            email = signer.loads(token, max_age=MAGIC_LINK_MAX_AGE)
        except (SignatureExpired, BadSignature):
            return render_template(
                "error.html", title="This sign-in link didn't work",
                detail="It may have expired (links are good for 15 minutes) or "
                       "already been used. Head back to GERP and request a new "
                       "one.", show_key_form=False), 400

        session["email"] = email
        session.permanent = True
        return redirect(url_for("index"))

    @bp.route("/logout")
    def logout():
        session.pop("email", None)
        return redirect(url_for("index"))

    @app.context_processor
    def _inject_user():
        # Exposes the signed-in email to every template (header avatar, tier
        # selector, quota lines).
        return {"user_email": session.get("email")}

    app.register_blueprint(bp)
    return bp
