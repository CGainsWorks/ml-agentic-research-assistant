"""Graph nodes.

Each node is a coroutine `(state) -> partial state`. Everything an operator
needs to audit an answer — the plan, every tool call, the evidence, each
critique round, the guardrail findings — is written into state as it happens
rather than reconstructed afterwards.

Failure policy: a node that cannot do its job degrades to the useful default
rather than raising. If planning fails, the plan is the question itself. If a
tool errors, its error is passed to the drafter as evidence that the tool
failed. An agent whose planner is having a bad day should still answer the
question.
"""

from __future__ import annotations

import re
import time

from araa.config import Settings
from araa.graph.state import AgentState
from araa.guardrails import (
    Severity,
    refusal_message,
    sanitise_retrieved,
    scan_question,
    scan_retrieved,
    validate_output,
)
from araa.llm import LLM
from araa.rag.citations import format_context, parse_citations
from araa.rag.prompts import (
    ANSWER_SYSTEM,
    ANSWER_USER,
    CRITIC_SYSTEM,
    CRITIC_USER,
    NO_CONTEXT_ANSWER,
    PLANNER_SYSTEM,
    PLANNER_USER,
    REVISER_SYSTEM,
    REVISER_USER,
    TOOL_SELECTOR_SYSTEM,
    TOOL_SELECTOR_USER,
)
from araa.rag.result import Revision
from araa.retrieval.hybrid import HybridRetriever
from araa.tools import parse_tool_calls

_VERDICT_RE = re.compile(r"VERDICT\s*:\s*(PASS|REVISE)", re.IGNORECASE)
_ISSUE_RE = re.compile(r"^\s*[-*]\s+(.{8,})$")


class Nodes:
    def __init__(self, settings: Settings, retriever: HybridRetriever, llm: LLM) -> None:
        self.s = settings
        self.retriever = retriever
        self.llm = llm

    # ------------------------------------------------------- 1. guard input
    async def guard_input(self, state: AgentState) -> AgentState:
        if not self.s.guardrails_enabled:
            return {"guardrails": {"enabled": False}}

        scan = scan_question(state["question"])
        guard = {"enabled": True, "question_scan": scan.to_dict()}
        if self.s.guardrail_block_on_detect and scan.should_block(Severity.CRITICAL):
            reason = f"The question matched {scan.detections[0].category}."
            return {
                "guardrails": guard,
                "blocked": True,
                "block_reason": reason,
                "draft": refusal_message(reason),
            }
        return {"guardrails": guard}

    # ------------------------------------------------------------- 2. plan
    async def plan(self, state: AgentState) -> AgentState:
        question = state["question"]
        t0 = time.perf_counter()

        # The tool inventory is deliberately NOT shown here. When it was, the
        # planner emitted `search_corpus query="..." top_k=10` as its sub-queries
        # and those went into the retriever as literal search text.
        try:
            gen = state["ledger"].record(
                "plan",
                await self.llm.generate(
                    PLANNER_USER.format(question=question),
                    system=PLANNER_SYSTEM.format(max_steps=self.s.max_plan_steps),
                    max_tokens=300,
                ),
            )
            steps = _parse_plan(gen.text, self.s.max_plan_steps)
        except Exception as exc:  # noqa: BLE001
            return {
                "plan": [question],
                "warnings": [
                    f"planner failed ({type(exc).__name__}); using the question as the plan"
                ],
                "timings_ms": {
                    **state.get("timings_ms", {}),
                    "plan": (time.perf_counter() - t0) * 1000,
                },
            }

        if not steps:
            steps = [question]
        return {
            "plan": steps,
            "timings_ms": {
                **state.get("timings_ms", {}),
                "plan": (time.perf_counter() - t0) * 1000,
            },
        }

    # --------------------------------------------------------- 3. retrieve
    async def retrieve(self, state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        collected = []
        # Each sub-query gets a slice of the context budget; the dedupe reducer
        # then unions them, so overlapping sub-queries do not waste the budget.
        per_step = max(2, self.s.final_top_k // max(1, len(state["plan"])) + 1)
        for step in state["plan"]:
            trace = self.retriever.retrieve(step, top_k=per_step)
            collected.extend(trace.results)

        warnings = [] if collected else ["retrieval returned no passages for any sub-query"]
        return {
            "passages": collected,
            "warnings": warnings,
            "timings_ms": {
                **state.get("timings_ms", {}),
                "retrieve": (time.perf_counter() - t0) * 1000,
            },
        }

    # ------------------------------------------------------------ 4. tools
    async def use_tools(self, state: AgentState) -> AgentState:
        registry = state["registry"]
        question = state["question"]
        t0 = time.perf_counter()

        try:
            gen = state["ledger"].record(
                "tools",
                await self.llm.generate(
                    TOOL_SELECTOR_USER.format(
                        question=question,
                        queries="\n".join(f"- {p}" for p in state["plan"]),
                        tools=registry.inventory(),
                        max_calls=self.s.max_tool_calls,
                    ),
                    system=TOOL_SELECTOR_SYSTEM,
                    max_tokens=250,
                ),
            )
            calls = parse_tool_calls(gen.text, max_calls=self.s.max_tool_calls)
        except Exception as exc:  # noqa: BLE001
            return {
                "warnings": [
                    f"tool selection failed ({type(exc).__name__}); continuing with retrieval only"
                ],
                "timings_ms": {
                    **state.get("timings_ms", {}),
                    "tools": (time.perf_counter() - t0) * 1000,
                },
            }

        records: list[dict] = []
        extra_passages = []
        for call in calls:
            result = registry.dispatch(call)
            records.append({"call": call.raw, **result.to_dict()})
            # search_corpus results are evidence; fold them into the passage set
            # so they are citable rather than dangling in a tool log.
            if result.ok and call.name == "search_corpus" and result.data:
                trace = self.retriever.retrieve(call.args.get("query", question), top_k=3)
                extra_passages.extend(trace.results)

        return {
            "tool_calls": records,
            "passages": extra_passages,
            "timings_ms": {
                **state.get("timings_ms", {}),
                "tools": (time.perf_counter() - t0) * 1000,
            },
        }

    # -------------------------------------------------- 5. guard the context
    async def guard_context(self, state: AgentState) -> AgentState:
        passages = state["passages"][: self.s.final_top_k]
        guard = dict(state.get("guardrails", {}))

        if self.s.guardrails_enabled and passages:
            scan = scan_retrieved([(r.chunk.chunk_id, r.chunk.text) for r in passages])
            guard["retrieved_scan"] = scan.to_dict()
            if self.s.guardrail_block_on_detect and scan.should_block(Severity.CRITICAL):
                offenders = sorted(
                    {d.source for d in scan.detections if d.severity >= Severity.CRITICAL}
                )
                reason = (
                    f"{len(offenders)} retrieved passage(s) carried embedded instructions "
                    f"at critical severity: {', '.join(offenders[:3])}"
                )
                return {
                    "guardrails": guard,
                    "blocked": True,
                    "block_reason": reason,
                    "draft": refusal_message(reason),
                }
            for r in passages:
                r.chunk.text = sanitise_retrieved(r.chunk.text)

        context, id_map = format_context(passages)
        return {"guardrails": guard, "context": context, "id_map": id_map}

    # ------------------------------------------------------------ 6. draft
    async def draft(self, state: AgentState) -> AgentState:
        if not state.get("context"):
            return {"draft": NO_CONTEXT_ANSWER, "verdict": "PASS"}

        t0 = time.perf_counter()
        gen = state["ledger"].record(
            "draft",
            await self.llm.generate(
                ANSWER_USER.format(question=state["question"], context=state["context"]),
                system=ANSWER_SYSTEM,
            ),
        )
        return {
            "draft": clean_answer(gen.text),
            "timings_ms": {
                **state.get("timings_ms", {}),
                "draft": (time.perf_counter() - t0) * 1000,
            },
        }

    # --------------------------------------------------------- 7. critique
    async def critique(self, state: AgentState) -> AgentState:
        if state.get("blocked") or not state.get("context"):
            return {"verdict": "PASS", "issues": []}

        # Mechanical findings are handed to the critic rather than left for it to
        # rediscover. A small model is much better at judging "is this claim
        # supported" than at counting missing brackets.
        report = parse_citations(state["draft"], state["id_map"])
        mechanical_lines = [
            f"- factual claims: {report.total_claims}, of which uncited: {report.uncited_claims}",
        ]
        if report.fabricated_markers:
            mechanical_lines.append(
                f"- FABRICATED markers pointing at passages that do not exist: "
                f"{', '.join(report.fabricated_markers)}"
            )
        uncited = [c.text for c in report.claims if c.factual and not c.is_cited][:5]
        mechanical_lines += [f'- uncited: "{t[:120]}"' for t in uncited]

        t0 = time.perf_counter()
        try:
            gen = state["ledger"].record(
                "critique",
                await self.llm.generate(
                    CRITIC_USER.format(
                        question=state["question"],
                        context=state["context"],
                        draft=state["draft"],
                        mechanical="\n".join(mechanical_lines),
                    ),
                    system=CRITIC_SYSTEM,
                    max_tokens=600,
                ),
            )
            verdict, issues = _parse_critique(gen.text)
        except Exception as exc:  # noqa: BLE001
            return {
                "verdict": "PASS",
                "issues": [],
                "warnings": [f"critic failed ({type(exc).__name__}); accepting the draft as-is"],
            }

        # The mechanical findings are not the critic's to overrule.
        if report.fabricated_markers or report.uncited_claims:
            verdict = "REVISE"
            issues = list(dict.fromkeys(issues + mechanical_lines[1:]))

        return {
            "verdict": verdict,
            "issues": issues,
            "timings_ms": {
                **state.get("timings_ms", {}),
                "critique": (time.perf_counter() - t0) * 1000,
            },
        }

    # ----------------------------------------------------------- 8. revise
    async def revise(self, state: AgentState) -> AgentState:
        round_no = state.get("critique_round", 0) + 1
        before = state["draft"]
        t0 = time.perf_counter()

        try:
            gen = state["ledger"].record(
                "revise",
                await self.llm.generate(
                    REVISER_USER.format(
                        question=state["question"],
                        context=state["context"],
                        draft=before,
                        issues="\n".join(f"- {i}" for i in state.get("issues", []))
                        or "- (none stated)",
                    ),
                    system=REVISER_SYSTEM,
                ),
            )
            after = clean_answer(gen.text) or before
        except Exception as exc:  # noqa: BLE001
            return {
                "critique_round": round_no,
                "warnings": [f"reviser failed ({type(exc).__name__}); keeping the previous draft"],
            }

        return {
            "draft": after,
            "critique_round": round_no,
            "revisions": [
                Revision(
                    round=round_no,
                    verdict=state.get("verdict", ""),
                    issues=list(state.get("issues", [])),
                    draft_before=before,
                    draft_after=after,
                )
            ],
            "timings_ms": {
                **state.get("timings_ms", {}),
                f"revise_{round_no}": (time.perf_counter() - t0) * 1000,
            },
        }

    # --------------------------------------------------------- 9. finalise
    async def finalise(self, state: AgentState) -> AgentState:
        if state.get("blocked"):
            return {}

        id_map = state.get("id_map", {})
        report = parse_citations(state["draft"], id_map)
        allowed_urls = {
            str(c.metadata.get("url")) for c in id_map.values() if c.metadata.get("url")
        }
        validation = validate_output(
            state["draft"],
            allowed_urls=allowed_urls,
            fabricated_markers=report.fabricated_markers,
        )

        guard = dict(state.get("guardrails", {}))
        guard["output_validation"] = validation.to_dict()

        out: AgentState = {"guardrails": guard}
        if validation.redacted_answer:
            out["draft"] = validation.redacted_answer
        if self.s.guardrails_enabled and self.s.guardrail_block_on_detect and not validation.passed:
            reason = "; ".join(i.detail for i in validation.issues if i.severity >= Severity.HIGH)
            out |= {"blocked": True, "block_reason": reason, "draft": refusal_message(reason)}
        return out


# ------------------------------------------------------------------ parsing

# A plan line that looks like a function call is not a search query. Observed in
# a real run: the planner emitted `search_corpus query="metrics" top_k=10`, which
# went into the retriever as literal text and pulled every passage from a single
# paper. The prompt now forbids this; this is the belt to that braces, so a
# future prompt regression degrades to "one plain query" rather than to silently
# terrible retrieval.
_TOOL_SYNTAX = re.compile(
    r"^\s*(?:TOOL\s*:)"  # explicit tool line
    r"|^\s*[a-z_][a-z0-9_]*\s*\("  # func(
    r"|^\s*[a-z_][a-z0-9_]*\s+\w+\s*="  # name arg=
    r"|\b(?:top_k|doc_id|max_calls|input_schema)\s*=",
    re.IGNORECASE,
)
_KNOWN_TOOL_NAMES = ("search_corpus", "lookup_document", "query_table", "calculate")


def _strip_tool_syntax(line: str) -> str:
    """Salvage the search intent from a line that wandered into tool syntax."""
    # `search_corpus query="measurement methods" top_k=10` -> `measurement methods`
    if quoted := re.findall(r'"([^"]{4,})"|\'([^\']{4,})\'', line):
        return next((a or b for a, b in quoted if (a or b)), "").strip()
    stripped = re.sub(r"\b\w+\s*=\s*\S+", " ", line)
    for name in _KNOWN_TOOL_NAMES:
        stripped = stripped.replace(name, " ")
    return re.sub(r"[^\w\s-]", " ", stripped).strip()


def _parse_plan(text: str, max_steps: int) -> list[str]:
    steps: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", line).strip().strip("\"'")
        if not cleaned:
            continue

        if _TOOL_SYNTAX.search(cleaned) or any(n in cleaned for n in _KNOWN_TOOL_NAMES):
            cleaned = _strip_tool_syntax(cleaned)

        if len(cleaned) < 8:
            continue
        low = cleaned.lower()
        if low.startswith(
            ("sub-quer", "subquer", "search quer", "plan:", "here are", "note:", "step")
        ):
            continue
        if cleaned in steps:
            continue
        steps.append(cleaned)
        if len(steps) >= max_steps:
            break
    return steps


_WRAPPING_QUOTES = re.compile(r'^\s*["“‘\']([\s\S]+)["”’\']\s*$')


def clean_answer(text: str) -> str:
    """Strip quotes the model wrapped around the whole answer.

    Small models sometimes return the answer as a quoted string. Left in, the
    leading quote attaches to the first sentence and the citation parser treats
    the closing quote as part of the last claim.
    """
    text = text.strip()
    if match := _WRAPPING_QUOTES.match(text):
        inner = match.group(1).strip()
        # Only unwrap when the quotes really do wrap the whole thing, rather
        # than the answer beginning and ending with legitimate quotations.
        if '"' not in inner[1:-1] or inner.count('"') % 2 == 0:
            text = inner
    return text.strip()


def _parse_critique(text: str) -> tuple[str, list[str]]:
    match = _VERDICT_RE.search(text)
    verdict = match.group(1).upper() if match else "PASS"
    issues: list[str] = []
    in_issues = False
    for line in text.splitlines():
        if re.match(r"^\s*ISSUES\s*:", line, re.IGNORECASE):
            in_issues = True
            continue
        if not in_issues:
            continue
        if issue := _ISSUE_RE.match(line):
            issues.append(issue.group(1).strip())
    if verdict == "REVISE" and not issues:
        # A REVISE with no stated issue gives the reviser nothing to act on.
        verdict = "PASS"
    return verdict, issues
