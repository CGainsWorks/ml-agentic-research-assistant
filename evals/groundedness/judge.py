"""Citation-support judging.

The question this answers is narrow and deliberately so: **for each claim that
carries a citation, does the cited passage actually support that claim?** Not
"is the answer good", not "is it helpful" — those are not measurable here and
pretending otherwise is how RAG demos end up with a number that means nothing.

Two judges, run together:

* `LLMJudge` — asks a model, one claim/passage pair at a time. Accurate on
  paraphrase, which lexical methods cannot handle at all.
* `LexicalJudge` — deterministic content-word overlap between claim and
  passage. Weak on paraphrase, but it costs nothing, cannot have a bad day, and
  runs in CI.

Reporting both is the point. If they disagree wildly, the LLM judge is
suspect; if they agree, the number is worth more than either alone. Their
agreement rate is reported as `judge_agreement`.

**Stated limitation, repeated in the README:** by default the judge is the same
local model that wrote the answer. A model grading its own output is a
sympathetic judge. Pass `--judge-backend openai_compat` to `run_groundedness`,
with `ARAA_OPENAI_MODEL` naming a different model, to grade independently; the
artefact records which judge produced the number either way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from araa.ingestion.chunking import split_sentences
from araa.llm import LLM

_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "to",
    "in",
    "for",
    "on",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "with",
    "that",
    "this",
    "these",
    "those",
    "as",
    "at",
    "by",
    "from",
    "it",
    "its",
    "we",
    "our",
    "can",
    "which",
    "their",
    "they",
    "has",
    "have",
    "had",
    "not",
    "but",
    "than",
    "when",
    "while",
    "also",
    "such",
    "may",
    "more",
    "most",
}
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-_.]*")


class Verdict(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    NOT_SUPPORTED = "not_supported"
    ERROR = "error"


@dataclass(slots=True)
class Judgement:
    verdict: Verdict
    reason: str = ""
    judge: str = ""
    score: float | None = None

    @property
    def counts_as_supported(self) -> bool:
        """PARTIAL does not count. A citation that half-supports a claim is a
        citation a reader would be misled by, and the metric exists to catch
        exactly that."""
        return self.verdict is Verdict.SUPPORTED


def content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


class LexicalJudge:
    """Deterministic overlap baseline.

    A claim is counted as supported when most of its content words appear in the
    cited passage. This over-credits copied phrasing and under-credits
    paraphrase, and it is reported as a floor rather than as the answer.
    """

    name = "lexical_overlap"

    def __init__(self, supported_at: float = 0.6, partial_at: float = 0.35) -> None:
        self.supported_at = supported_at
        self.partial_at = partial_at

    def judge(self, claim: str, passage: str) -> Judgement:
        claim_words = content_words(claim)
        if not claim_words:
            return Judgement(Verdict.NOT_SUPPORTED, "no content words in claim", self.name, 0.0)
        overlap = len(claim_words & content_words(passage)) / len(claim_words)
        if overlap >= self.supported_at:
            verdict = Verdict.SUPPORTED
        elif overlap >= self.partial_at:
            verdict = Verdict.PARTIAL
        else:
            verdict = Verdict.NOT_SUPPORTED
        return Judgement(
            verdict, f"content-word overlap {overlap:.2f}", self.name, round(overlap, 4)
        )


JUDGE_SYSTEM = """You are grading whether a passage supports a claim. You are not
grading whether the claim is true, well written, or interesting.

Answer with exactly one word on the first line:

SUPPORTED      - the passage states the claim, or states something the claim is
                 a faithful paraphrase of.
PARTIAL        - the passage is about the same subject and supports part of the
                 claim, but the claim asserts more than the passage does, or
                 changes a number, a condition, or a magnitude.
NOT_SUPPORTED  - the passage does not support the claim, or contradicts it.

Then a second line beginning "REASON: " with one short sentence.

Be strict. If the claim adds a number, a comparison, or a causal link that the
passage does not contain, that is PARTIAL at best. Do not use knowledge from
outside the passage."""

JUDGE_USER = """Claim:
\"\"\"
{claim}
\"\"\"

Passage:
\"\"\"
{passage}
\"\"\"

Verdict:"""

_VERDICT_RE = re.compile(r"\b(SUPPORTED|PARTIAL|NOT[_ ]SUPPORTED|UNSUPPORTED)\b", re.IGNORECASE)


class LLMJudge:
    def __init__(self, llm: LLM, name: str | None = None) -> None:
        self.llm = llm
        self.name = name or f"llm:{llm.model_name}"

    async def judge(self, claim: str, passage: str) -> Judgement:
        try:
            gen = await self.llm.generate(
                JUDGE_USER.format(claim=claim[:1200], passage=passage[:3000]),
                system=JUDGE_SYSTEM,
                max_tokens=120,
            )
        except Exception as exc:  # noqa: BLE001
            return Judgement(Verdict.ERROR, f"{type(exc).__name__}: {exc}", self.name)

        match = _VERDICT_RE.search(gen.text)
        if not match:
            # An unparseable verdict is an error, not a pass. Defaulting to
            # SUPPORTED here would inflate every number in the report.
            return Judgement(Verdict.ERROR, f"unparseable verdict: {gen.text[:120]!r}", self.name)

        token = match.group(1).upper().replace(" ", "_")
        verdict = {
            "SUPPORTED": Verdict.SUPPORTED,
            "PARTIAL": Verdict.PARTIAL,
            "NOT_SUPPORTED": Verdict.NOT_SUPPORTED,
            "UNSUPPORTED": Verdict.NOT_SUPPORTED,
        }[token]

        reason = ""
        if m := re.search(r"REASON\s*:\s*(.+)", gen.text, re.IGNORECASE):
            reason = m.group(1).strip()[:240]
        return Judgement(verdict, reason, self.name)


def claim_sentences(text: str) -> list[str]:
    return [s for s in split_sentences(text) if len(s.strip()) > 25]
