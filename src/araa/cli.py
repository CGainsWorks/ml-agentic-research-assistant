"""Command line entry point.

araa ask "question"     answer a question (single-shot RAG by default)
araa ask --mode agent   the LangGraph agent: plan, tools, critique, revise
araa status             what is indexed, what model, is it reachable
araa tools              the tool inventory
araa ingest             build the indexes from the corpus

The default is the baseline rather than the agent because the groundedness eval
measured the agent grounding worse at 8.4x the token cost. See docs/results.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from araa.config import get_settings


async def _ask(args) -> int:
    from araa.graph import ResearchAgent
    from araa.llm import CostLedger
    from araa.rag import SingleShotRAG

    s = get_settings()
    from araa.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever(s)
    if not retriever.is_ready():
        print(
            "No index. Build one:\n"
            "  python scripts/fetch_corpus.py\n"
            "  python -m araa.ingestion.pipeline",
            file=sys.stderr,
        )
        return 1

    system = (
        ResearchAgent(s, retriever=retriever)
        if args.mode == "agent"
        else SingleShotRAG(s, retriever=retriever)
    )
    ledger = CostLedger()
    try:
        result = await system.answer(args.question, ledger=ledger)
    finally:
        await system.aclose()

    if args.json:
        print(
            json.dumps(result.to_dict(include_chunk_text=args.include_text), indent=2, default=str)
        )
        return 0

    if result.plan:
        print("Plan:")
        for step in result.plan:
            print(f"  - {step}")
        print()
    if result.tool_calls:
        print("Tool calls:")
        for call in result.tool_calls:
            status = "ok" if call.get("ok") else f"error: {call.get('error')}"
            print(f"  - {call.get('call')} -> {status}")
        print()
    if result.revisions:
        print(f"Self-critique: {len(result.revisions)} revision round(s)")
        for rev in result.revisions:
            for issue in rev.issues[:3]:
                print(f"  [{rev.round}] {issue}")
        print()

    print(result.answer)
    print()

    if result.sources:
        print("Sources:")
        for src in result.sources:
            print(f"  - {src.get('title')} ({src.get('doc_id')}) [{src.get('licence')}]")
            if src.get("url"):
                print(f"    {src['url']}")
        print()

    cites = result.citations or {}
    if cites:
        print(
            f"Citations: {cites.get('cited_claims', 0)}/{cites.get('total_claims', 0)} "
            f"factual claims cited ({cites.get('citation_rate', 0):.0%})"
        )
    if fabricated := cites.get("fabricated_markers"):
        print(f"  FABRICATED markers: {', '.join(fabricated)}")

    cost = result.cost or {}
    print(
        f"Cost: {cost.get('total_tokens', 0)} tokens across {cost.get('llm_calls', 0)} calls, "
        f"${cost.get('cost_usd', 0):.4f}"
    )
    if result.warnings:
        print("Warnings:")
        for w in result.warnings:
            print(f"  ! {w}")
    return 0


async def _status(_args) -> int:
    from araa.retrieval.hybrid import HybridRetriever

    s = get_settings()
    retriever = HybridRetriever(s)

    print(f"backend:   {s.llm_backend} ({s.active_model_name})")
    if s.llm_backend in ("ollama", "openai_compat"):
        from araa.llm import build_llm

        base_url = s.ollama_base_url if s.llm_backend == "ollama" else s.openai_base_url
        llm = build_llm(s)
        try:
            ok = await llm.health()
            print(f"server:    {'reachable' if ok else 'UNREACHABLE'} at {base_url}")
            if ok:
                models = await llm.list_models()
                present = s.active_model_name in models
                print(f"model:     {'present' if present else 'NOT AVAILABLE'}")
                if not present:
                    print(f"           available: {', '.join(models[:8])}")
        finally:
            await llm.aclose()

    try:
        chunks = len(retriever.chunks)
        vectors = retriever.vector.count()
    except Exception as exc:  # noqa: BLE001
        chunks = vectors = 0
        print(f"index:     unreadable ({exc})")
    print(f"chunks:    {chunks}   vectors: {vectors}")
    print(f"chunking:  {s.chunk_strategy} (size {s.chunk_size}, overlap {s.chunk_overlap})")
    print(f"embedding: {s.embedding_model}")
    print(f"reranker:  {s.reranker_model if s.use_reranker else 'disabled'}")

    manifest = s.corpus_dir / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        print(f"corpus:    {data.get('document_count')} documents")
        for lic, count in (data.get("licence_counts") or {}).items():
            print(f"           {lic}: {count}")
    else:
        print("corpus:    none (run scripts/fetch_corpus.py)")
    return 0


async def _tools(_args) -> int:
    from araa.ingestion.tabular import TableStore
    from araa.retrieval.hybrid import HybridRetriever
    from araa.tools import build_registry

    s = get_settings()
    registry = build_registry(s, HybridRetriever(s), TableStore())
    print(registry.inventory())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="araa", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_ask = sub.add_parser("ask", help="answer a question")
    p_ask.add_argument("question")
    # Baseline by default. The agent grounds worse on this corpus at 8.4x the
    # token cost - see docs/results.md. Opt into it with --mode agent.
    p_ask.add_argument("--mode", choices=["agent", "single_shot"], default="single_shot")
    p_ask.add_argument("--json", action="store_true", help="emit the structured result")
    p_ask.add_argument("--include-text", action="store_true", help="include chunk text in --json")
    p_ask.set_defaults(fn=_ask)

    sub.add_parser("status", help="index, corpus and model status").set_defaults(fn=_status)
    sub.add_parser("tools", help="print the tool inventory").set_defaults(fn=_tools)

    p_ingest = sub.add_parser("ingest", help="build the indexes from the corpus")
    p_ingest.add_argument("--strategy", default=None)
    p_ingest.set_defaults(fn=None)

    args = ap.parse_args()

    if args.command == "ingest":
        from araa.ingestion.pipeline import run_ingest

        s = get_settings()
        if args.strategy:
            s = s.model_copy(update={"chunk_strategy": args.strategy})
        run_ingest(s)
        return 0

    return asyncio.run(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
