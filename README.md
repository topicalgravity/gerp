# gerp

Build a **Generative Engine Results Page** from an LLM API call: run a prompt
through a web-search/grounded model and get back a normalized JSON object of the
**citations the model actually used** — across Anthropic, Gemini, and OpenAI.

The **"considered but not cited"** set is scaffolded and additive. The current
build is **citations-only**.

## Install

```bash
pip install anthropic google-genai openai   # whichever providers you'll use
export ANTHROPIC_API_KEY=...   # or GEMINI_API_KEY / OPENAI_API_KEY
```

## CLI

```bash
python -m gerp.cli -p gemini --prompt "best personal injury firms in Dallas"
python -m gerp.cli -p anthropic --prompt "..." --model claude-opus-4-8 --raw
echo "long prompt" | python -m gerp.cli -p openai --stdin
```

## Library

```python
import gerp
g = gerp.run("best personal injury firms in Dallas", provider="gemini")
print(g.to_dict())
```

## Output schema (v1.0)

```jsonc
{
  "schema_version": "1.0",
  "provider": "gemini",
  "model": "gemini-3.5-flash-lite",
  "prompt": "...",
  "answer_text": "...",
  "citations": [
    { "url": "...", "title": "...", "domain": "...",
      "cited_text": "...", "start_index": null, "end_index": null,
      "metadata": {} }
  ],
  "considered_not_cited": [],            // empty in citations-only mode
  "considered_method": "none",           // provenance flag
  "issued_queries": ["..."],             // Gemini exposes these; others may not
  "created_at": "..."
}
```

## Per-provider capability

| Provider   | Citations | Issued queries | Native considered-set | Considered strategy |
|------------|-----------|----------------|-----------------------|---------------------|
| Gemini     | yes (groundingSupports) | yes (webSearchQueries) | no  | rerun_queries |
| Anthropic  | yes (richest spans)     | yes (server_tool_use input) | yes (web_search_tool_result pool) | provider_delta |
| OpenAI     | yes (annotations + sources) | yes (action.queries) | yes (action.sources) | provider_delta |

## Adding the considered-set later

It's a separate pass over a GERP — adapters don't change:

```python
from gerp.considered import enrich, SerpAPIBackend
g = gerp.run(prompt, provider="gemini")          # captures issued_queries
g = enrich(g, backend=SerpAPIBackend())          # re-runs them, diffs vs citations
# -> g.considered_not_cited populated, g.considered_method = "rerun_queries"
```

`SerpAPIBackend.search()` is currently a stub (raises NotImplementedError) so the
citations-only build never spends search credits. Implement it to turn the layer
on. `considered_method` always records how the set was derived, so downstream
(and any client report) knows whether the considered pool was observed or
inferred.

## Web app: tiers & GERP accounts

`app.py` serves the google.com-style front end. Two model tiers, defined in
`gerp/tiers.py`:

| Tier | Anthropic | Gemini | OpenAI | Access |
|------|-----------|--------|--------|--------|
| **Standard** (free) | `claude-haiku-4-5` | `gemini-3.5-flash-lite` | `gpt-5.6-luna` | Anonymous, 3 searches / 7 days (signed cookie + per-IP tally) |
| **Frontier** (login) | `claude-opus-4-8` | `gemini-3.5-flash` | `gpt-5.5` | Signed-in GERP account, 3 searches / 7 days (per-email); owner + allowlist unlimited |

A tier is just a per-provider `{model, kwargs}` lookup that `gerp.run()` forwards
to the adapters. The frontier Anthropic entry sets `tool_type=web_search_20260209`
(Opus 4.8's dynamic-filtering web search); the frontier OpenAI entry sets
`reasoning={"effort": "high"}`. Unknown/stale `tier` form values resolve to
Standard, and a frontier request from a non-entitled visitor silently downgrades.

**GERP accounts** are passwordless email magic links (`auth.py`) — a GERP account
keyed on your email, **not** Google Sign-In. Accounts are **request-based**: new
users submit `/auth/request-account` (first name, last name, email, company),
which records the request in SQLite (`store.py`, on the Render persistent disk)
and emails it to `GERP_ACCOUNT_REQUEST_TO` (env-only — the owner's address never
appears in templates or page source). That email carries a **one-click Approve
button** (signed 30-day token → `GET /auth/approve`): clicking it flips the
store row and emails the requester that they can sign in — no dashboard, no
restart. `POST /auth/request` then signs a 15-minute `itsdangerous` token for
approved emails only and sends the link via Resend (`urllib.request`, no SDK
dep); `GET /auth/verify` sets the Flask session. `GERP_APPROVED_EMAILS` remains
as a bootstrap/override list. When `RESEND_API_KEY` is unset all email is logged
to stdout for local dev, and when no approval config exists anywhere (empty env
lists AND an empty store), sign-in is open — the fresh-checkout escape hatch, so
always set `GERP_APPROVED_EMAILS` in prod.

### Web-app env vars

| Var | Purpose |
|-----|---------|
| `GERP_SECRET_KEY` | Signs the session + both quota cookies (set a stable value in prod) |
| `GERP_PASSWORD` | Owner bypass — visit `?key=<value>` for unlimited access to both tiers |
| `RESEND_API_KEY` | Resend HTTP API key for magic-link email (unset ⇒ log link to stdout) |
| `GERP_FROM_EMAIL` | From-address for the sign-in email; its domain must be verified in Resend (SPF/DKIM) |
| `GERP_ACCOUNT_REQUEST_TO` | Where account-request submissions are emailed (kept out of page source) |
| `GERP_APPROVED_EMAILS` | Bootstrap/override approvals; day-to-day approvals live in SQLite via the one-click email link |
| `GERP_DATA_DIR` | Dir for the SQLite store (`/var/data` on Render's persistent disk; defaults to `./data` locally) |
| `GERP_FRONTIER_ALLOWLIST` | Comma-separated emails with unlimited frontier access (implicitly approved) |
| `TURNSTILE_SITE_KEY` / `TURNSTILE_SECRET_KEY` | Cloudflare Turnstile bot check (search + sign-in) |

## Notes / limitations

- Anthropic and OpenAI both expose a directly observed considered-set
  (`provider_delta`): Anthropic via the `web_search_tool_result` pool (url,
  title, page_age are plaintext — only content snippets are encrypted),
  OpenAI via `web_search_call.action.sources`. Gemini exposes its issued
  `webSearchQueries` but not the result pool, so its considered-set requires
  the `rerun_queries` strategy.
- Gemini citation URLs are vertexaisearch.cloud.google.com redirects; the
  adapter recovers the real source domain from the grounding chunk's
  domain/title field and flags `redirect_url` in metadata.
- OpenAI appends `?utm_source=openai` to all URLs; the adapter strips it.
- Per-URL `metadata` is left empty here — that's where your Screaming Frog /
  GEO-readiness enrichment (schema type, publish date, internal links) plugs in.
```
