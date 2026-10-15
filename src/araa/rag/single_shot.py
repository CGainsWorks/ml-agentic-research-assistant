"""Single-shot RAG: retrieve once, answer once, cite.

This layer exists for two reasons beyond being a step on the way to the agent:

* It is the **baseline the agent must beat**. An agent that costs five LLM calls
  and grounds no better than one call is a worse system, and without this
  baseline in the same eval you would never find that out.
* It is the **cheap path**. Not every question needs planning and critique. The
  API exposes both and reports the cost of each.

The pipeline is the same at every layer, and the order matters:

    scan question -> retrieve -> scan passages -> sanitise -> generate
                  -> parse citations -> validate output

Guardrails run on the passages *before* they reach the model, and on the answer
*after* it leaves — because the input filter only catches attacks it recognises,
and the output check catches the ones that worked.
"""

from __future__ import annotations

import time

from araa.config import Settings, get_settings
from araa.guardrails import (
    Severity,
    refusal_message,
    sanitise_retrieved,
    scan_question,
    scan_retrieved,
    validate_output,
)
from araa.llm import LLM, CostLedger, build_llm
from araa.rag.citations import format_context, parse_citations, render_sources
from araa.rag.prompts import ANSWER_SYSTEM, ANSWER_USER, NO_CONTEXT_ANSWER
from araa.rag.result import AnswerResult
from araa.retrieval.hybrid import HybridRetriever, Retrieved


class SingleShotRAG:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        retriever: HybridRetriever | None = None,
        llm: LLM | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or HybridRetriever(self.settings)
        self._llm = llm
        self._owns_llm = llm is None

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = build_llm(self.settings)
        return self._llm

    async def aclose(self) -> None:
        if self._owns_llm and self._llm is not None:
            await self._llm.aclose()

    async def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        ledger: CostLedger | None = None,
        node: str = "answer",
    ) -> AnswerResult:
        s = self.settings
        ledger = ledger or CostLedger()
        result = AnswerResult(question=question, mode="single_shot")
        timings: dict[str, float] = {}

        # ------------------------------------------------- 1. question scan
        guard: dict = {"enabled": s.guardrails_enabled}
        if s.guardrails_enabled:
            q_scan = scan_question(question)
            guard["question_scan"] = q_scan.to_dict()
            if s.guardrail_block_on_detect and q_scan.should_block(Severity.CRITICAL):
                result.blocked = True
                result.block_reason = (
                    f"The question itself matched {q_scan.detections[0].category}."
                )
                result.answer = refusal_message(result.block_reason)
                result.guardrails = guard
                result.cost = ledger.to_dict()
                return result

        # ------------------------------------------------------ 2. retrieve
        t0 = time.perf_counter()
        trace = self.retriever.retrieve(question, top_k=top_k or s.final_top_k)
        timings["retrieval"] = (time.perf_counter() - t0) * 1000

        if not trace.results:
            result.answer = NO_CONTEXT_ANSWER
            result.retrieval_trace = trace.to_dict()
            result.guardrails = guard
            result.cost = ledger.to_dict()
            result.timings_ms = timings
            result.warnings.append("retrieval returned no passages")
            return result

        # --------------------------------- 3. scan + sanitise the passages
        passages: list[Retrieved] = trace.results
        if s.guardrails_enabled:
            r_scan = scan_retrieved([(r.chunk.chunk_id, r.chunk.text) for r in passages])
            guard["retrieved_scan"] = r_scan.to_dict()

            if s.guardrail_block_on_detect and r_scan.should_block(Severity.CRITICAL):
                offenders = sorted(
                    {d.source for d in r_scan.detections if d.severity >= Severity.CRITICAL}
                )
                result.blocked = True
                result.block_reason = (
                    f"{len(offenders)} retrieved passage(s) contained embedded instructions "
                    f"at critical severity: {', '.join(offenders[:3])}"
                    + (" ..." if len(offenders) > 3 else "")
                )
                result.answer = refusal_message(result.block_reason)
                result.retrieval_trace = trace.to_dict()
                result.guardrails = guard
                result.cost = ledger.to_dict()
                result.timings_ms = timings
                return result

            for r in passages:
                r.chunk.text = sanitise_retrieved(r.chunk.text)

        # ------------------------------------------------------ 4. generate
        context, id_map = format_context(passages)
        t0 = time.perf_counter()
        gen = ledger.record(
            node,
            await self.llm.generate(
                ANSWER_USER.format(question=question, context=context),
                system=ANSWER_SYSTEM,
            ),
        )
        timings["generation"] = (time.perf_counter() - t0) * 1000

        # ---------------------------------------- 5. citations + validation
        report = parse_citations(gen.text, id_map)
        allowed_urls = {
            str(c.metadata.get("url")) for c in id_map.values() if c.metadata.get("url")
        }
        validation = validate_output(
            gen.text,
            allowed_urls=allowed_urls,
            fabricated_markers=report.fabricated_markers,
        )
        guard["output_validation"] = validation.to_dict()

        answer_text = validation.redacted_answer or gen.text
        if s.guardrails_enabled and s.guardrail_block_on_detect and not validation.passed:
            result.blocked = True
            result.block_reason = "; ".join(
                i.detail for i in validation.issues if i.severity >= Severity.HIGH
            )
            answer_text = refusal_message(result.block_reason)

        result.answer = answer_text
        result.citations = report.to_dict()
        result.sources = render_sources(report.used_chunk_ids, id_map)
        result.retrieval_trace = trace.to_dict()
        result.guardrails = guard
        result.cost = ledger.to_dict()
        result.timings_ms = timings

        if report.uncited_claims:
            result.warnings.append(
                f"{report.uncited_claims} of {report.total_claims} factual claims carry no citation"
            )
        if report.fabricated_markers:
            result.warnings.append(
                f"fabricated citation markers: {', '.join(report.fabricated_markers)}"
            )
        return result
