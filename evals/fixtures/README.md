# Recorded generations

`ARAA_LLM_BACKEND=fixture` replays `generations.jsonl` from this directory
instead of calling a model, which lets CI exercise the whole pipeline —
planning, tool use, critique, revision, citation validation — deterministically
and without a model server or an API key.

The fixture backend **raises on a cache miss** rather than inventing a
response. A prompt change therefore surfaces as a loud failure telling you to
re-record, not as a quietly different answer that makes every downstream eval
number a lie.

Record against a live backend:

```bash
ARAA_LLM_BACKEND=ollama python -m evals.record_fixtures --limit 5
```

The `.jsonl` is gitignored: it is keyed on a hash of (model, system, prompt), so
a committed one goes stale as soon as a prompt or the default model changes.
