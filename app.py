"""GERP Flask app."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, render_template, request

import gerp as g

app = Flask(__name__)


@app.route("/")
def index():
    providers = sorted(g.PROVIDERS.keys())
    return render_template("search.html", providers=providers)


@app.route("/search", methods=["POST"])
def search():
    prompt = request.form.get("prompt", "").strip()
    selected = request.form.getlist("providers")

    if not prompt:
        return render_template("search.html", providers=sorted(g.PROVIDERS.keys()),
                               error="Enter a prompt.")
    if not selected:
        return render_template("search.html", providers=sorted(g.PROVIDERS.keys()),
                               error="Select at least one provider.")

    results: dict = {}
    with ThreadPoolExecutor(max_workers=len(selected)) as ex:
        futures = {ex.submit(g.run, prompt, provider=p): p for p in selected}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                results[p] = fut.result()
            except Exception as e:
                results[p] = {"error": str(e)}

    # Preserve a consistent display order
    ordered = {p: results[p] for p in sorted(g.PROVIDERS.keys()) if p in results}

    return render_template("results.html", prompt=prompt, results=ordered,
                           providers=sorted(g.PROVIDERS.keys()))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
