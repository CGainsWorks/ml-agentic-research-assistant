# Tool inventory

Regenerate with `python -m araa.tools.builtin --inventory` (or `araa tools`).

**Every tool is read-only.** Nothing in this inventory writes, sends, or
executes. That is a direct consequence of the threat model: the corpus is
third-party PDFs, any of which may carry instructions aimed at this agent, so a
successful injection must not be able to reach anything with an effect. There is
no `send_email`, no `write_file`, no `http_get` — and the output guardrail flags
the model *claiming* to have performed such an action, because that means it has
adopted an injected persona.

## Protocol

Tools are invoked with a line protocol rather than JSON function-calling:

```
TOOL: search_corpus | query=hybrid retrieval reranking
TOOL: query_table | table=corpus_metadata, op=value_counts, column=primary_category
TOOL: calculate | expression=(88.5 - 84.2) / 84.2 * 100
```

The default backend is an 8B open-weights model, and small models emit malformed
JSON often enough that a JSON protocol turns "chose the wrong tool" into
"crashed". This grammar has one bracket to get wrong and degrades to *no tool
call* rather than to an exception. It works unchanged on a hosted model.

Argument splitting is comma-separated on `key=` boundaries only, so a value may
itself contain commas: `query=BLEU, ROUGE and METEOR` is one argument.

---

## `search_corpus`

Hybrid retrieval over the paper corpus. Returns passages with provenance.

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `query` | yes | — | What to search for. Use the corpus's own terminology. |
| `top_k` | no | `5` | Passages to return (clamped to 1–10). |

Results are folded back into the agent's citable passage set, so anything this
tool surfaces can be cited rather than dangling in a tool log.

## `lookup_document`

Title, authors, categories, licence, URL and abstract for one document.

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `doc_id` | yes | — | The arXiv id, e.g. `2401.12345`. |

Exists because "what else did the authors of [C3] do" is a retrieval query that
retrieval answers badly — it wants a specific record, not a similarity search.

## `query_table`

Compute over the corpus metadata table.

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `table` | no | `corpus_metadata` | Table name. |
| `op` | yes | — | `schema` \| `value_counts` \| `aggregate` \| `filter`. |
| `column` | no | — | Column to count, group by, or filter on. |
| `contains` | no | — | Substring to filter for (`op=filter`). |
| `limit` | no | `20` | Maximum rows (clamped to 1–100). |

**Why this exists.** A retrieval-only agent gets counting questions wrong in a
specific, recognisable way: it reads the handful of rows that happened to be
retrieved, counts *those*, and reports the number as if it described the whole
corpus. "How many papers are in cs.CL" is arithmetic over a table, not a
similarity search. The fix is a tool, not a longer prompt.

## `calculate`

Arithmetic. `+ - * / // % **` and parentheses.

| Parameter | Required | Default | Description |
| --- | --- | --- | --- |
| `expression` | yes | — | e.g. `(88.5 - 84.2) / 84.2 * 100` |

Evaluated by an AST walker, **not** by `eval` with a restricted namespace. That
pattern has been escaped too many times to defend, and the input here can
originate in a document an attacker wrote. Exponents are bounded to stop
`9**9**9` from becoming a denial of service.

---

## Deliberately absent

| Not provided | Why |
| --- | --- |
| Web search / fetch | The corpus is the ground truth. A web tool would let the agent answer from outside the evidence set, and the groundedness metric would silently stop meaning anything. |
| File write | Nothing the agent produces should persist without a human reading it. |
| Email / HTTP POST | The exfiltration sink an injection is trying to reach. Its absence is the mitigation. |
| Code execution | `calculate` covers the legitimate arithmetic need at a fraction of the blast radius. |
