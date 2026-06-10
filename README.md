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
python -m gerp.cli -p anthropic --prompt "..." --model claude-opus-4-6 --raw
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
  "model": "gemini-2.5-flash",
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
