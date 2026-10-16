"""Evaluation harness.

The evals are the deliverable, not an accessory to it. A RAG demo without a
groundedness number is a screenshot; the number is the evidence.

Ordering is deliberate and matches the build order:

    evals/retrieval      does retrieval find the right chunk?      no LLM needed
    evals/groundedness   does a citation support its claim?        LLM judge
    evals/adversarial    do the guardrails hold under attack?      deterministic
    evals/costs          what does one query actually cost?        measured

Everything writes an artefact to data/runs/ (gitignored).
`python -m evals.report --from-runs` condenses them into the committed
docs/results.json and renders docs/results.md from it; `--check` is the CI
guard that the two agree.
"""
