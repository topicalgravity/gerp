"""GERP Flask app."""

from __future__ import annotations

import json
import os
import secrets
import datetime as _dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from flask import Flask, abort, redirect, render_template, request, url_for

import gerp as g
from gerp.considered import enrich, SerpAPIBackend
from gerp.schema import GERP, ConsideredMethod

app = Flask(__name__)

RUNS_DIR = Path(os.environ.get("GERP_RUNS_DIR", Path(__file__).parent / "runs"))

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
    path = RUNS_DIR / f"{run_id}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _recent_runs(limit: int = 10) -> list[dict]:
    if not RUNS_DIR.is_dir():
        return []
    runs = []
    for path in sorted(RUNS_DIR.glob("*.json"), reverse=True)[:limit]:
        try:
            d = json.loads(path.read_text())
            runs.append({"id": d["id"], "prompt": d["prompt"],
                         "created_at": d.get("created_at", ""),
                         "providers": sorted(d.get("results", {}))})
        except (json.JSONDecodeError, KeyError):
            continue
    return runs


@app.route("/")
def index():
    return render_template("search.html", providers=sorted(g.PROVIDERS.keys()),
                           recent=_recent_runs())


@app.route("/search", methods=["POST"])
def search():
    prompt = request.form.get("prompt", "").strip()
    selected = request.form.getlist("providers")

    if not prompt:
        return render_template("search.html", providers=sorted(g.PROVIDERS.keys()),
                               recent=_recent_runs(), error="Enter a prompt.")
    if not selected:
        return render_template("search.html", providers=sorted(g.PROVIDERS.keys()),
                               recent=_recent_runs(),
                               error="Select at least one provider.")

    results: dict = {}
    with ThreadPoolExecutor(max_workers=len(selected)) as ex:
        futures = {ex.submit(g.run, prompt, provider=p): p for p in selected}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                results[p] = fut.result(timeout=180)
            except Exception as e:
                results[p] = {"error": str(e)}

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
    _save_run(run_id, prompt, results)
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
