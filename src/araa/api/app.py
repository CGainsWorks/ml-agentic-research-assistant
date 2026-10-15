"""Streaming FastAPI service.

Endpoints:

    GET  /health        index, corpus, model and guardrail status
    GET  /corpus        document count and the licence breakdown
    GET  /tools         the tool inventory
    GET  /runs/latest   the most recent ingestion digest
    POST /ask           structured JSON: answer, citations, trace, cost
    POST /ask/stream    the same work, streamed as Server-Sent Events

A note on what `/ask/stream` streams, because it is a deliberate design choice
rather than an omission:

* `single_shot` streams **tokens**, because there is exactly one generation and
  every token that arrives is final.
* `agent` streams **stage events** — plan, retrieval, tool calls, critique
  verdicts — and then the finished answer in one piece. Token-streaming an
  agent draft would show the user text that the critique step is about to
  rewrite. Showing someone a sentence and then silently replacing it is worse
  than making them wait.

Heavy objects (embedder, reranker, Chroma client) are built once at startup and
shared. The per-request state is the cost ledger, so concurrent requests get
independent cost attribution.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from araa.api.schemas import AskRequest, AskResponse, HealthResponse, ToolOut
from araa.config import get_settings
from araa.graph import ResearchAgent
from araa.graph.state import initial_state
from araa.ingestion.tabular import TableStore
from araa.llm import CostLedger, build_llm
from araa.rag import SingleShotRAG
from araa.rag.citations import parse_citations, render_sources
from araa.retrieval.hybrid import HybridRetriever
from araa.tools import build_registry

log = logging.getLogger("araa.api")

STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    log.info("loading retrieval stack (embedder, reranker, chroma)")
    retriever = HybridRetriever(s)
    llm = build_llm(s)
    registry = build_registry(s, retriever, TableStore())

    STATE.update(
        settings=s,
        retriever=retriever,
        llm=llm,
        registry=registry,
        agent=ResearchAgent(s, retriever=retriever, llm=llm, registry=registry),
        rag=SingleShotRAG(s, retriever=retriever, llm=llm),
    )
    log.info(
        "ready: %d chunks, backend=%s model=%s",
        len(retriever.chunks) if retriever.is_ready() else 0,
        s.llm_backend,
        s.active_model_name,
    )
    try:
        yield
    finally:
        await llm.aclose()


app = FastAPI(
    title="araa - agentic research assistant",
    version="0.1.0",
    description=__doc__,
    lifespan=lifespan,
)


# ------------------------------------------------------------------- health
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    s = STATE["settings"]
    retriever: HybridRetriever = STATE["retriever"]
    detail: list[str] = []

    try:
        chunk_count = len(retriever.chunks)
        vector_count = retriever.vector.count()
    except Exception as exc:  # noqa: BLE001
        chunk_count = vector_count = 0
        detail.append(f"index unreadable: {exc}")

    if chunk_count == 0:
        detail.append("index is empty; run: python -m araa.ingestion.pipeline")
    if vector_count != chunk_count:
        detail.append(
            f"vector store ({vector_count}) and chunk store ({chunk_count}) disagree; re-ingest"
        )

    generation: dict[str, Any] = s.generation_signature()
    if s.llm_backend in ("ollama", "openai_compat"):
        llm = STATE["llm"]
        base_url = s.ollama_base_url if s.llm_backend == "ollama" else s.openai_base_url
        try:
            reachable = await llm.health()
            generation["reachable"] = reachable
            if reachable:
                models = await llm.list_models()
                generation["model_present"] = s.active_model_name in models
                if not generation["model_present"]:
                    detail.append(
                        f"model {s.active_model_name!r} not available; "
                        f"available: {', '.join(models[:6])}"
                    )
            else:
                detail.append(f"{s.llm_backend} unreachable at {base_url}")
        except Exception as exc:  # noqa: BLE001
            generation["reachable"] = False
            detail.append(f"{s.llm_backend} check failed: {exc}")

    manifest = s.corpus_dir / "manifest.json"
    corpus: dict[str, Any] = {"present": manifest.exists()}
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        corpus |= {
            "documents": data.get("document_count"),
            "licence_counts": data.get("licence_counts"),
        }
    else:
        detail.append("no corpus manifest; run: python scripts/fetch_corpus.py")

    return HealthResponse(
        status="ok" if not detail else "degraded",
        detail=detail,
        index={"chunks": chunk_count, "vector_records": vector_count},
        corpus=corpus,
        generation=generation,
        retrieval=s.retrieval_signature(),
        guardrails={
            "enabled": s.guardrails_enabled,
            "block_on_detect": s.guardrail_block_on_detect,
        },
    )


@app.get("/corpus")
async def corpus() -> dict:
    s = STATE["settings"]
    manifest = s.corpus_dir / "manifest.json"
    if not manifest.exists():
        raise HTTPException(404, "No corpus. Run: python scripts/fetch_corpus.py")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        "document_count": data.get("document_count"),
        "licence_counts": data.get("licence_counts"),
        "fetch_params": data.get("fetch_params"),
        "licences_file": "data/corpus/LICENCES.md",
        "documents": [
            {k: d.get(k) for k in ("arxiv_id", "title", "licence", "licence_url", "categories")}
            for d in data.get("documents", [])
        ],
    }


@app.get("/tools", response_model=list[ToolOut])
async def tools() -> list[ToolOut]:
    registry = STATE["registry"]
    return [
        ToolOut(
            name=t.name,
            description=t.description,
            parameters=[
                {
                    "name": p.name,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                }
                for p in t.params
            ],
        )
        for t in registry.tools.values()
    ]


@app.get("/runs/latest")
async def latest_run() -> dict:
    s = STATE["settings"]
    path = s.runs_dir / "ingest-latest.json"
    if not path.exists():
        raise HTTPException(404, "No ingestion run recorded yet.")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------- ask
@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    retriever: HybridRetriever = STATE["retriever"]
    if not retriever.is_ready():
        raise HTTPException(503, "Index is empty. Run: python -m araa.ingestion.pipeline")

    ledger = CostLedger()
    system = STATE["agent"] if req.mode == "agent" else STATE["rag"]
    try:
        result = await system.answer(req.question, ledger=ledger)
    except Exception as exc:  # noqa: BLE001
        log.exception("ask failed")
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc

    return AskResponse(
        **result.to_dict(
            include_chunk_text=req.include_chunk_text, include_drafts=req.include_drafts
        )
    )


def _sse(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data, default=str)}


@app.post("/ask/stream")
async def ask_stream(req: AskRequest) -> EventSourceResponse:
    retriever: HybridRetriever = STATE["retriever"]
    if not retriever.is_ready():
        raise HTTPException(503, "Index is empty. Run: python -m araa.ingestion.pipeline")

    if req.mode == "single_shot":
        return EventSourceResponse(_stream_single_shot(req))
    return EventSourceResponse(_stream_agent(req))


async def _stream_single_shot(req: AskRequest):
    """Token streaming. Every token that arrives is final."""
    s = STATE["settings"]
    rag: SingleShotRAG = STATE["rag"]
    ledger = CostLedger()

    from araa.guardrails import sanitise_retrieved, scan_question, scan_retrieved
    from araa.guardrails.injection import Severity
    from araa.rag.citations import format_context
    from araa.rag.prompts import ANSWER_SYSTEM, ANSWER_USER

    yield _sse("status", {"stage": "guardrails", "message": "scanning question"})
    if s.guardrails_enabled:
        q_scan = scan_question(req.question)
        if s.guardrail_block_on_detect and q_scan.should_block(Severity.CRITICAL):
            yield _sse("blocked", {"reason": q_scan.detections[0].category})
            return

    yield _sse("status", {"stage": "retrieval", "message": "hybrid search + rerank"})
    trace = rag.retriever.retrieve(req.question, top_k=req.top_k or s.final_top_k)
    if not trace.results:
        yield _sse("token", {"text": "The corpus does not contain anything relevant."})
        yield _sse("done", {"citations": {}, "sources": [], "cost": ledger.to_dict()})
        return
    yield _sse("retrieval", trace.to_dict(include_text=False))

    passages = trace.results
    if s.guardrails_enabled:
        r_scan = scan_retrieved([(r.chunk.chunk_id, r.chunk.text) for r in passages])
        if r_scan.triggered:
            yield _sse("guardrail", r_scan.to_dict())
        if s.guardrail_block_on_detect and r_scan.should_block(Severity.CRITICAL):
            yield _sse("blocked", {"reason": "critical injection in retrieved passages"})
            return
        for r in passages:
            r.chunk.text = sanitise_retrieved(r.chunk.text)

    context, id_map = format_context(passages)
    yield _sse("status", {"stage": "generation", "message": s.active_model_name})

    parts: list[str] = []
    llm = STATE["llm"]
    async for piece in llm.stream(
        ANSWER_USER.format(question=req.question, context=context), system=ANSWER_SYSTEM
    ):
        parts.append(piece)
        yield _sse("token", {"text": piece})

    answer = "".join(parts)
    if usage := getattr(llm, "last_stream_usage", None):
        ledger.record_usage("answer", llm.model_name, usage, price_per_mtok=llm.price_per_mtok)

    report = parse_citations(answer, id_map)
    yield _sse(
        "done",
        {
            "citations": report.to_dict(),
            "sources": render_sources(report.used_chunk_ids, id_map),
            "cost": ledger.to_dict(),
            "warnings": (
                [f"{report.uncited_claims}/{report.total_claims} claims uncited"]
                if report.uncited_claims
                else []
            ),
        },
    )


# Node -> what to tell the user. Kept out of the graph so the graph stays a
# description of the computation rather than of the UI.
_STAGE_MESSAGES = {
    "guard_input": "scanning the question",
    "planner": "planning sub-queries",
    "retrieve": "hybrid retrieval",
    "use_tools": "considering tools",
    "guard_context": "scanning retrieved passages",
    "drafter": "drafting an answer",
    "critique": "critiquing the draft",
    "revise": "revising",
    "finalise": "validating output",
}


async def _stream_agent(req: AskRequest):
    """Stage events, then the finished answer.

    Not token-by-token: the critique step rewrites drafts, and streaming a
    sentence that is about to be replaced is worse than a short wait.
    """
    agent: ResearchAgent = STATE["agent"]
    s = STATE["settings"]
    ledger = CostLedger()

    state = initial_state(req.question, agent.registry, ledger)
    final: dict[str, Any] = {}

    try:
        async for update in agent.graph.astream(
            state,
            config={"recursion_limit": 4 + 3 * (s.max_critique_rounds + 1)},
        ):
            for node, payload in update.items():
                final.update(payload or {})
                yield _sse(
                    "status",
                    {"stage": node, "message": _STAGE_MESSAGES.get(node, node)},
                )
                if node == "planner" and payload.get("plan"):
                    yield _sse("plan", {"sub_queries": payload["plan"]})
                if node == "retrieve":
                    yield _sse(
                        "retrieval",
                        {"passages_considered": len(final.get("passages", []))},
                    )
                if node == "use_tools" and payload.get("tool_calls"):
                    yield _sse("tools", {"calls": payload["tool_calls"]})
                if node == "critique":
                    yield _sse(
                        "critique",
                        {
                            "verdict": payload.get("verdict"),
                            "issues": payload.get("issues", []),
                            "round": final.get("critique_round", 0),
                        },
                    )
                if node == "revise":
                    yield _sse("revision", {"round": payload.get("critique_round")})
                await asyncio.sleep(0)  # let the event actually flush
    except Exception as exc:  # noqa: BLE001
        log.exception("agent stream failed")
        yield _sse("error", {"error": f"{type(exc).__name__}: {exc}"})
        return

    answer = final.get("draft", "")
    id_map = final.get("id_map", {})
    report = parse_citations(answer, id_map)

    yield _sse("answer", {"text": answer})
    yield _sse(
        "done",
        {
            "blocked": bool(final.get("blocked")),
            "block_reason": final.get("block_reason"),
            "plan": final.get("plan", []),
            "citations": report.to_dict(),
            "sources": render_sources(report.used_chunk_ids, id_map),
            "guardrails": final.get("guardrails", {}),
            "cost": ledger.to_dict(),
            "warnings": final.get("warnings", []),
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(_request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})
