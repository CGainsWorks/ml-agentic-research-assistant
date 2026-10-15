# Agent graph

Rendered by GitHub directly. A Mermaid source diagram rather than a committed
PNG on purpose: a PNG of a graph goes stale the moment a node is added and
nothing tells you, whereas this sits next to the code it describes and is
diffable.

```mermaid
flowchart TD
    START([question]) --> GI[guard_input<br/><i>scan the question</i>]

    GI -->|critical detection| FIN
    GI -->|clean| PLAN[plan<br/><i>question → 1..4 sub-queries</i>]

    PLAN --> RET[retrieve<br/><i>hybrid search per sub-query</i><br/><i>union, deduped by chunk id</i>]
    RET --> TOOLS[use_tools<br/><i>counts, aggregates, lookups</i><br/><i>retrieval cannot do</i>]
    TOOLS --> GC[guard_context<br/><i>scan + sanitise passages</i>]

    GC -->|critical injection| FIN
    GC -->|clean| DRAFT[draft<br/><i>answer with [C1] markers</i>]

    DRAFT --> CRIT{critique<br/><i>unsupported? miscited?</i><br/><i>uncited? missing? injected?</i>}

    CRIT -->|PASS| FIN
    CRIT -->|REVISE and rounds left| REV[revise<br/><i>fix the stated issues only</i>]
    CRIT -->|round cap reached| FIN
    REV --> CRIT

    FIN[finalise<br/><i>parse citations, validate output</i><br/><i>canary / sinks / claimed actions</i>] --> END([answer + trace + cost])

    classDef guard fill:#3d2b2b,stroke:#a55,color:#eee
    classDef llm fill:#2b333d,stroke:#57a,color:#eee
    classDef plain fill:#2d2d2d,stroke:#777,color:#eee
    class GI,GC,FIN guard
    class PLAN,DRAFT,CRIT,REV llm
    class RET,TOOLS plain
```

Red nodes are guardrails, blue nodes make an LLM call, grey nodes are
deterministic.

## Node reference

| Node | LLM call | What it does | Failure behaviour |
| --- | :---: | --- | --- |
| `guard_input` | no | Scans the question for injection. | — |
| `plan` | yes | Decomposes into standalone retrieval sub-queries. | Falls back to the question itself as a one-step plan. |
| `retrieve` | no | Runs hybrid retrieval per sub-query; unions and dedupes by chunk id, keeping the best score. | Empty result → the drafter is told the corpus has nothing. |
| `use_tools` | yes | Emits `TOOL:` lines for counts, aggregates, document lookups, arithmetic. | Parse failure or tool error → continues with retrieval alone; the error is recorded as evidence. |
| `guard_context` | no | Scans and sanitises the retrieved passages, then formats the numbered context. | Critical detection → routes to `finalise` blocked. |
| `draft` | yes | Writes the answer with `[Cn]` citation markers. | — |
| `critique` | yes | Judges the draft against the passages. Mechanical findings (uncited claims, fabricated markers) are computed first and handed to the critic. | Critic failure → accepts the draft. A `REVISE` with no stated issue is treated as `PASS`. |
| `revise` | yes | Fixes only the stated issues. | Reviser failure → keeps the previous draft. |
| `finalise` | no | Parses citations, runs output validation (canary, exfiltration sinks, claimed actions, fabricated markers). | Validation failure at HIGH+ → replaces the answer with a refusal that carries the findings. |

## Termination

Two independent bounds, because one is not enough:

1. `max_critique_rounds` (default 2) caps the revise/critique loop.
2. A `REVISE` verdict with an empty issue list is coerced to `PASS`. Without
   this, a critic that says "REVISE" without saying why spins the loop to the
   cap and burns a full generation on each pass, having given the reviser
   nothing to act on.

The LangGraph `recursion_limit` is set from `max_critique_rounds` as a backstop
against a routing bug — it is not the mechanism that terminates the loop.

## Cost shape

The agent makes **4–6 LLM calls** per question (plan, tools, draft, critique,
and 0–2 revisions) against the single-shot baseline's **1**. Whether that is
worth it is not a matter of opinion — `make eval-groundedness` and
`make eval-cost` run both paths over the same questions and report the
groundedness delta alongside the token multiple. See [results.md](results.md).
