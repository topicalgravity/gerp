"""GERP Flask app."""

from __future__ import annotations

import json
import os
import secrets
import urllib.parse
import urllib.request
import datetime as _dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from flask import (Flask, abort, make_response, redirect, render_template,
                   request, session, url_for)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix

import gerp as g
from gerp.considered import enrich, SerpAPIBackend
from gerp.resolve import resolve_redirects
from gerp.schema import GERP, ConsideredMethod
from gerp.tiers import resolve_tier, tier_config, model_label
from quota import Quota

app = Flask(__name__)

# Friendly model label ("Opus 4.8") for the results-page chip.
app.jinja_env.filters["model_label"] = model_label

# Render LB sits in front of gunicorn and sets X-Forwarded-For to the
# connecting client IP (Cloudflare's edge IP when behind Cloudflare, or the
# real client IP for direct traffic). x_for=1 unwraps that single hop.
# CF-Connecting-IP (see _real_ip) is the authoritative source when Cloudflare
# is in the chain and is used directly instead of relying on XFF counting.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)


def _real_ip() -> str:
    """Real client IP: CF-Connecting-IP is set by Cloudflare and is always
    the actual visitor's IP. Fall back to the ProxyFix-adjusted remote addr
    for local dev or direct-to-Render traffic."""
    return request.headers.get("CF-Connecting-IP") or get_remote_address()


RUNS_DIR = Path(os.environ.get("GERP_RUNS_DIR", Path(__file__).parent / "runs"))

# Curated, committed example runs shown on the home page. Real user searches
# are NEVER listed — that would leak one visitor's queries to the next.
DEMO_DIR = Path(__file__).parent / "demo"

# Per-IP rate limit on /search as a backstop to Turnstile: caps how fast a
# single IP can burn API credits even if it gets past the bot check. In-memory
# storage is fine for one gunicorn worker; move to Redis if scaling out.
limiter = Limiter(_real_ip, app=app, storage_uri="memory://")

# Cloudflare Turnstile (invisible bot check). Verification is skipped when the
# secret key isn't set, so local dev works with no Cloudflare account.
TURNSTILE_SITE_KEY = os.environ.get("TURNSTILE_SITE_KEY")
TURNSTILE_SECRET_KEY = os.environ.get("TURNSTILE_SECRET_KEY")
TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


@app.context_processor
def _inject_turnstile():
    # Lets every template render the widget without threading the key through
    # each render_template() call.
    return {"turnstile_site_key": TURNSTILE_SITE_KEY}


# Absolute URL for the social-share (Open Graph / Twitter) image. Hosted on the
# WordPress media library so it's served from the topicalgravity.com CDN; set
# the env var (or edit the fallback) to the uploaded file's URL.
OG_IMAGE = os.environ.get(
    "GERP_OG_IMAGE",
    "https://topicalgravity.com/wp-content/uploads/2026/06/gerp_og.png",
)


@app.context_processor
def _inject_site_meta():
    return {"og_image": OG_IMAGE}


def _verify_turnstile() -> bool:
    """Validate the Turnstile token on the current request before spending API
    credits. Checking is disabled unless BOTH keys are set — if only one is
    configured the widget won't render and every submission would fail."""
    if not TURNSTILE_SECRET_KEY or not TURNSTILE_SITE_KEY:
        return True
    token = request.form.get("cf-turnstile-response", "")
    if not token:
        return False
    data = urllib.parse.urlencode({
        "secret": TURNSTILE_SECRET_KEY,
        "response": token,
        "remoteip": _real_ip(),
    }).encode()
    try:
        with urllib.request.urlopen(TURNSTILE_VERIFY_URL, data=data, timeout=10) as resp:
            result = json.loads(resp.read())
        if not result.get("success"):
            app.logger.warning("Turnstile rejected: %s", result.get("error-codes"))
        return bool(result.get("success"))
    except Exception:
        # Don't let a Cloudflare hiccup hard-fail the search; the rate limiter
        # still bounds the downside.
        return True

# Hard cap per search request, comfortably under gunicorn's worker timeout: a
# hung provider becomes a per-provider error in its tab instead of the whole
# request dying as an unstyled 502. Frontier (reasoning) models are much
# slower, so they get a longer cap — gunicorn's --timeout in render.yaml is
# raised to stay comfortably above FRONTIER_TIMEOUT.
PROVIDER_TIMEOUT = 150
FRONTIER_TIMEOUT = 240

# ── Owner bypass + free-search quota ─────────────────────────────────────
# The app is public. Anonymous visitors get FREE_LIMIT searches, counted in a
# tamper-proof signed cookie with an in-memory per-IP tally as a soft
# cross-check. GERP_PASSWORD is no longer a gate — it's the owner's unlimited
# bypass: visit once with ?key=<password> and you're exempt from the quota.
PASSWORD = os.environ.get("GERP_PASSWORD")
FREE_LIMIT = 3
# Signed-in users get a frontier allowance too — frontier searches run ~$1+
# each, so they're quota'd (not free-for-all) until Stripe lands in Stage 3.
FRONTIER_LIMIT = 3
# The quota is a rolling window: a visitor gets `limit` searches, and the count
# resets WINDOW after the window's *first* search (not the latest), so 7 days
# after someone starts using GERP they get a fresh allowance.
FREE_WINDOW = 7 * 24 * 3600

# Signing key for the quota cookies. Set GERP_SECRET_KEY in prod so the counts
# survive restarts; the generated fallback just means everyone's counts reset
# on each deploy (acceptable — the quota is soft identity by design).
_SECRET = os.environ.get("GERP_SECRET_KEY") or secrets.token_hex(32)
app.secret_key = _SECRET

# Two rolling-window quotas sharing one implementation (quota.py): the anonymous
# free tier keyed by client IP, and the signed-in frontier tier keyed by email.
_free_quota = Quota(_SECRET, "gerp-free-quota", "gerp_free", FREE_LIMIT, FREE_WINDOW)
_frontier_quota = Quota(_SECRET, "gerp-frontier-quota", "gerp_frontier",
                        FRONTIER_LIMIT, FREE_WINDOW)

# Emails with unlimited frontier access (Ryan + testers), comma-separated.
FRONTIER_ALLOWLIST = {
    e.strip().lower()
    for e in os.environ.get("GERP_FRONTIER_ALLOWLIST", "").split(",")
    if e.strip()
}


def _is_owner() -> bool:
    if not PASSWORD:
        return False
    supplied = request.args.get("key") or request.cookies.get("gerp_key")
    return secrets.compare_digest(supplied or "", PASSWORD)


def _current_email() -> str | None:
    return session.get("email")


def _frontier_entitled() -> bool:
    """Who may run the frontier tier at all: the owner, or any signed-in user."""
    return _is_owner() or bool(_current_email())


def _frontier_unlimited() -> bool:
    """Who bypasses the frontier quota: the owner and allowlisted emails."""
    return _is_owner() or (_current_email() or "").lower() in FRONTIER_ALLOWLIST


def _free_used() -> int:
    return _free_quota.used(request, _real_ip())


def _bump_free(resp, prior_used: int) -> None:
    _free_quota.bump(resp, request, _real_ip(), prior_used)


def _frontier_used() -> int:
    return _frontier_quota.used(request, (_current_email() or "").lower())


def _bump_frontier(resp, prior_used: int) -> None:
    _frontier_quota.bump(resp, request, (_current_email() or "").lower(), prior_used)


@app.after_request
def _set_password_cookie(resp):
    if PASSWORD and request.args.get("key") == PASSWORD:
        resp.set_cookie("gerp_key", PASSWORD, httponly=True)
    return resp


@app.context_processor
def _inject_quota():
    """Expose quota state to every template (home counter, results indicator,
    frontier remaining line, and whether the frontier tier is usable)."""
    owner = _is_owner()
    ctx = {"is_owner": owner,
           "frontier_entitled": _frontier_entitled(),
           "frontier_unlimited": _frontier_unlimited(),
           "frontier_limit": FRONTIER_LIMIT}
    if owner:
        ctx.update(free_used=0, free_remaining=None, free_limit=FREE_LIMIT)
    else:
        used = _free_used()
        ctx.update(free_used=used, free_remaining=max(0, FREE_LIMIT - used),
                   free_limit=FREE_LIMIT)
    # Frontier remaining line for signed-in, non-unlimited users.
    if _current_email() and not ctx["frontier_unlimited"]:
        f_used = _frontier_used()
        ctx.update(frontier_used=f_used,
                   frontier_remaining=max(0, FRONTIER_LIMIT - f_used))
    else:
        ctx.update(frontier_used=0, frontier_remaining=None)
    return ctx


@app.errorhandler(404)
def _not_found(e):
    return render_template(
        "error.html", title="Page not found",
        detail="That run doesn't exist (results are kept on this server's disk "
               "and may have been cleared by a redeploy) or the address is wrong.",
        show_key_form=False), 404


@app.errorhandler(429)
def _rate_limited(e):
    return render_template(
        "error.html", title="Too many searches",
        detail="You've run a lot of searches in a short window. Give it a "
               "minute and try again — this limit is just here to keep "
               "automated traffic from running up the bill.",
        show_key_form=False), 429


@app.errorhandler(500)
def _server_error(e):
    return render_template(
        "error.html", title="Something went wrong",
        detail="The search couldn't be completed. Your API credits for any "
               "finished providers may have been spent, but nothing was saved. "
               "Head back and try again — if it keeps happening, check the "
               "provider API keys and status pages.",
        show_key_form=False), 500


def _new_run_id() -> str:
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{secrets.token_hex(3)}"


def _save_run(run_id: str, prompt: str, results: dict, tier: str = "standard") -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": run_id,
        "prompt": prompt,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "tier": tier,
        "results": {
            p: (r.to_dict() if isinstance(r, GERP) else r)
            for p, r in results.items()
        },
    }
    (RUNS_DIR / f"{run_id}.json").write_text(json.dumps(payload, indent=2))


def _load_run(run_id: str) -> dict | None:
    # run_id comes from the URL; restrict to the generated id alphabet.
    if not all(ch.isalnum() or ch == "-" for ch in run_id):
        return None
    # A real saved run wins; fall back to the committed demo set so the
    # example links on the home page resolve even after a redeploy wipes
    # the ephemeral RUNS_DIR.
    for base in (RUNS_DIR, DEMO_DIR):
        path = base / f"{run_id}.json"
        if path.is_file():
            return json.loads(path.read_text())
    return None


def _demo_runs(limit: int = 10) -> list[dict]:
    """Curated example runs for the home page. Deliberately reads only the
    committed demo/ directory — never real user searches in RUNS_DIR."""
    if not DEMO_DIR.is_dir():
        return []
    runs = []
    for path in sorted(DEMO_DIR.glob("*.json")):
        try:
            d = json.loads(path.read_text())
            runs.append({"id": d["id"], "prompt": d["prompt"],
                         "created_at": d.get("created_at", ""),
                         "providers": sorted(d.get("results", {}))})
        except (json.JSONDecodeError, KeyError):
            continue
    return runs[:limit]


@app.route("/favicon.ico")
def favicon():
    return redirect(url_for("static", filename="favicon.ico"), 301)


@app.route("/")
def index():
    return render_template("search.html", providers=sorted(g.PROVIDERS.keys()),
                           recent=_demo_runs())


@app.route("/search", methods=["POST"])
@limiter.limit("8 per minute")
@limiter.limit("40 per hour")
def search():
    prompt = request.form.get("prompt", "").strip()
    selected = request.form.getlist("providers")

    if not _verify_turnstile():
        return render_template("search.html", providers=sorted(g.PROVIDERS.keys()),
                               recent=_demo_runs(),
                               error="Couldn't verify you're human — please try again.")

    owner = _is_owner()

    # Which model tier to run. A stale/hostile form value resolves to standard.
    # Frontier is entitled to the owner and any signed-in user; anyone else is
    # downgraded — and told so on the results page (downgraded flag), because a
    # silent downgrade reads as "frontier is broken" to a user who selected it.
    requested_tier = request.form.get("tier")
    tier = resolve_tier(requested_tier)
    downgraded = False
    if tier == "frontier" and not _frontier_entitled():
        tier = "standard"
        downgraded = True
    print(f"[gerp-search] tier={tier} requested={requested_tier!r} "
          f"signed_in={bool(_current_email())} owner={owner}", flush=True)

    # Quota check before any provider call, so an out-of-quota request never
    # spends credits. Frontier runs draw from the per-email frontier quota
    # (unless owner/allowlisted); standard runs draw from the anonymous free
    # quota (unless owner). Track the prior count so the post-run bump preserves
    # the window start.
    used = 0            # free-tier searches spent this window
    f_used = 0          # frontier searches spent this window
    if tier == "frontier":
        if not _frontier_unlimited():
            f_used = _frontier_used()
            if f_used >= FRONTIER_LIMIT:
                return render_template(
                    "error.html",
                    title=f"You've used your {FRONTIER_LIMIT} frontier searches",
                    detail="Frontier models (GPT-5.5 Thinking, Gemini 3.5, Opus "
                           "4.8) are metered while GERP is free. More searches "
                           "are coming soon — in the meantime you can still run "
                           "the Standard tier, and your earlier results stay "
                           "reachable by their links.",
                    show_key_form=False), 402
    elif not owner:
        used = _free_used()
        if used >= FREE_LIMIT:
            return render_template(
                "error.html", title=f"You've used your {FREE_LIMIT} free searches",
                detail="Thanks for trying GERP! Paid access for more searches is "
                       "coming soon. Check back shortly — and in the meantime your "
                       "earlier results are still reachable by their links.",
                show_key_form=False), 402

    if not prompt:
        return render_template("search.html", providers=sorted(g.PROVIDERS.keys()),
                               recent=_demo_runs(), error="Enter a prompt.")
    if not selected:
        return render_template("search.html", providers=sorted(g.PROVIDERS.keys()),
                               recent=_demo_runs(),
                               error="Select at least one provider.")

    # Frontier (reasoning) models take much longer, so they get a longer cap.
    timeout = FRONTIER_TIMEOUT if tier == "frontier" else PROVIDER_TIMEOUT

    def _submit(pool, provider):
        cfg = tier_config(tier, provider)
        return pool.submit(g.run, prompt, provider=provider,
                           model=cfg.get("model"), **cfg.get("kwargs", {}))

    results: dict = {}
    ex = ThreadPoolExecutor(max_workers=len(selected))
    futures = {_submit(ex, p): p for p in selected}
    try:
        for fut in as_completed(futures, timeout=timeout):
            p = futures[fut]
            try:
                results[p] = fut.result()
            except Exception as e:
                results[p] = {"error": str(e)}
    except TimeoutError:
        pass
    finally:
        # Don't wait for hung threads; their providers get a timeout error.
        ex.shutdown(wait=False, cancel_futures=True)
    for p in selected:
        if p not in results:
            results[p] = {"error": f"No response after {timeout}s "
                                   "— the provider may be slow or down. "
                                   "Try again, or deselect it."}

    # Providers that never exposed their result pool (Gemini) get a
    # considered-set by re-running their fan-out queries through SerpAPI.
    if os.environ.get(SerpAPIBackend.env_key):
        backend = SerpAPIBackend()
        for r in results.values():
            if (isinstance(r, GERP) and r.issued_queries
                    and r.considered_method == ConsideredMethod.NONE):
                try:
                    enrich(r, backend=backend)
                except Exception:
                    pass  # enrichment is additive; never fail the run

    # Unwrap Gemini's vertexaisearch redirect URLs to the real source links.
    # Additive and fail-safe; runs concurrently per result with a timeout.
    for r in results.values():
        if isinstance(r, GERP):
            try:
                resolve_redirects(r)
            except Exception:
                pass

    run_id = _new_run_id()
    try:
        _save_run(run_id, prompt, results, tier=tier)
        # Redirect-after-POST: refreshing the results page never re-spends
        # API credits, and the URL is shareable. The downgraded flag rides the
        # redirect (not the saved run) — it's about THIS request's entitlement,
        # not a property of the results.
        resp = redirect(url_for("show_run", run_id=run_id,
                                downgraded=1 if downgraded else None))
    except OSError:
        # Disk unavailable: still show the results we paid for, just not
        # as a shareable saved run.
        ordered = {p: results[p] for p in sorted(g.PROVIDERS.keys())
                   if p in results}
        resp = make_response(render_template(
            "results.html", prompt=prompt, results=ordered,
            providers=sorted(g.PROVIDERS.keys()), tier=tier,
            downgraded=downgraded))

    # Count this search against the right quota (after it succeeded, so failed
    # validations above never burn a search).
    if tier == "frontier":
        if not _frontier_unlimited():
            _bump_frontier(resp, f_used)
    elif not owner:
        _bump_free(resp, used)
    return resp


@app.route("/r/<run_id>")
def show_run(run_id: str):
    run = _load_run(run_id)
    if run is None:
        abort(404)

    ordered = {}
    for p in sorted(g.PROVIDERS.keys()):
        if p not in run["results"]:
            continue
        data = run["results"][p]
        ordered[p] = data if "error" in data else GERP.from_dict(data)

    return render_template("results.html", prompt=run["prompt"], results=ordered,
                           providers=sorted(g.PROVIDERS.keys()),
                           tier=run.get("tier", "standard"),
                           downgraded=bool(request.args.get("downgraded")))


# Register the GERP magic-link auth routes (POST /auth/request, /auth/verify,
# /logout), the 30-day session lifetime, and the user_email context processor.
# Passed the limiter + Turnstile check so auth reuses the same machinery.
from auth import init_auth  # noqa: E402 - imported here to avoid a circular import
import store  # noqa: E402

# Fail fast if the data dir (persistent disk in prod) isn't writable.
store.init_db()
init_auth(app, limiter, _verify_turnstile)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
