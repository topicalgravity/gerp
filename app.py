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

from flask import Flask, abort, redirect, render_template, request, url_for
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix

import gerp as g
from gerp.considered import enrich, SerpAPIBackend
from gerp.schema import GERP, ConsideredMethod

app = Flask(__name__)

# Render (and most PaaS) sit behind a proxy, so the client IP arrives in
# X-Forwarded-For; without this, get_remote_address() and the per-IP rate
# limiter would see the single proxy IP and lump every visitor together.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

RUNS_DIR = Path(os.environ.get("GERP_RUNS_DIR", Path(__file__).parent / "runs"))

# Curated, committed example runs shown on the home page. Real user searches
# are NEVER listed — that would leak one visitor's queries to the next.
DEMO_DIR = Path(__file__).parent / "demo"

# Per-IP rate limit on /search as a backstop to Turnstile: caps how fast a
# single IP can burn API credits even if it gets past the bot check. In-memory
# storage is fine for one gunicorn worker; move to Redis if scaling out.
limiter = Limiter(get_remote_address, app=app, storage_uri="memory://")

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


def _verify_turnstile() -> bool:
    """Validate the Turnstile token on the current request before spending API
    credits. No secret configured ⇒ checking is disabled (returns True)."""
    if not TURNSTILE_SECRET_KEY:
        return True
    token = request.form.get("cf-turnstile-response", "")
    if not token:
        return False
    data = urllib.parse.urlencode({
        "secret": TURNSTILE_SECRET_KEY,
        "response": token,
        "remoteip": get_remote_address(),
    }).encode()
    try:
        with urllib.request.urlopen(TURNSTILE_VERIFY_URL, data=data, timeout=10) as resp:
            return bool(json.loads(resp.read()).get("success"))
    except Exception:
        # Don't let a Cloudflare hiccup hard-fail the search; the rate limiter
        # still bounds the downside.
        return True

# Hard cap per search request, comfortably under gunicorn's 300s worker
# timeout: a hung provider becomes a per-provider error in its tab instead
# of the whole request dying as an unstyled 502.
PROVIDER_TIMEOUT = 150

# Optional shared-secret protection for deployed instances: set GERP_PASSWORD
# and every request must carry ?key=<password> (or a previously set cookie).
PASSWORD = os.environ.get("GERP_PASSWORD")


@app.before_request
def _check_password():
    if not PASSWORD:
        return None
    supplied = request.args.get("key") or request.cookies.get("gerp_key")
    if secrets.compare_digest(supplied or "", PASSWORD):
        return None
    abort(401)


@app.after_request
def _set_password_cookie(resp):
    if PASSWORD and request.args.get("key") == PASSWORD:
        resp.set_cookie("gerp_key", PASSWORD, httponly=True)
    return resp


@app.errorhandler(401)
def _unauthorized(e):
    return render_template(
        "error.html", title="Access key required",
        detail="This GERP instance is private. Enter the access key to continue "
               "(it only needs to be entered once per browser).",
        show_key_form=True), 401


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


def _save_run(run_id: str, prompt: str, results: dict) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": run_id,
        "prompt": prompt,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
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
    if not prompt:
        return render_template("search.html", providers=sorted(g.PROVIDERS.keys()),
                               recent=_demo_runs(), error="Enter a prompt.")
    if not selected:
        return render_template("search.html", providers=sorted(g.PROVIDERS.keys()),
                               recent=_demo_runs(),
                               error="Select at least one provider.")

    results: dict = {}
    ex = ThreadPoolExecutor(max_workers=len(selected))
    futures = {ex.submit(g.run, prompt, provider=p): p for p in selected}
    try:
        for fut in as_completed(futures, timeout=PROVIDER_TIMEOUT):
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
            results[p] = {"error": f"No response after {PROVIDER_TIMEOUT}s "
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

    run_id = _new_run_id()
    try:
        _save_run(run_id, prompt, results)
    except OSError:
        # Disk unavailable: still show the results we paid for, just not
        # as a shareable saved run.
        ordered = {p: results[p] for p in sorted(g.PROVIDERS.keys())
                   if p in results}
        return render_template("results.html", prompt=prompt, results=ordered,
                               providers=sorted(g.PROVIDERS.keys()))
    # Redirect-after-POST: refreshing the results page never re-spends
    # API credits, and the URL is shareable.
    return redirect(url_for("show_run", run_id=run_id))


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
                           providers=sorted(g.PROVIDERS.keys()))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
