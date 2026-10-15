"""Record LLM generations so CI can replay the whole pipeline without a model.

Runs the real system against a live backend and writes every (system, prompt)
-> response pair to `evals/fixtures/generations.jsonl`. `ARAA_LLM_BACKEND=fixture`
then replays them, which lets CI exercise planning, tool use, critique,
revision and citation validation deterministically and in seconds.

The fixture backend raises on a cache miss rather than inventing a response, so
a prompt change surfaces as a loud failure telling you to re-record, not as a
quietly different answer.

Run:  python -m evals.record_fixtures --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from araa.config import get_settings
from araa.llm import build_llm
from araa.llm.base import LLM, Generation
from araa.llm.fixture import fixture_key
from evals.harness import load_question_set, require_index

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "generations.jsonl"


class RecordingLLM(LLM):
    """Wraps a real backend and records everything that passes through."""

    def __init__(self, inner: LLM) -> None:
        super().__init__(inner.settings)
        self.inner = inner
        self.records: dict[str, dict] = {}

    @property
    def model_name(self) -> str:
        return self.inner.model_name

    async def generate(
        self, prompt: str, *, system=None, temperature=None, max_tokens=None
    ) -> Generation:
        gen = await self.inner.generate(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )
        key = fixture_key(self.model_name, system, prompt)
        self.records[key] = {
            "key": key,
            "model": gen.model,
            "text": gen.text,
            "input_tokens": gen.usage.input_tokens,
            "output_tokens": gen.usage.output_tokens,
        }
        return gen

    async def aclose(self) -> None:
        await self.inner.aclose()


async def _amain() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=5, help="questions to record")
    ap.add_argument("--set", dest="question_set", default="research_questions")
    ap.add_argument("--modes", nargs="+", default=["single_shot", "agent"])
    args = ap.parse_args()

    s = get_settings()
    if s.llm_backend == "fixture":
        raise SystemExit("Set ARAA_LLM_BACKEND to a live backend before recording.")
    require_index(s)

    questions = load_question_set(args.question_set)[: args.limit]
    if not questions:
        raise SystemExit("No questions. Run: python -m evals.questions.build_question_set")

    from araa.graph import ResearchAgent
    from araa.rag import SingleShotRAG

    recorder = RecordingLLM(build_llm(s))
    existing: dict[str, dict] = {}
    if FIXTURES.exists():
        for line in FIXTURES.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                existing[rec["key"]] = rec

    try:
        for mode in args.modes:
            system = (
                ResearchAgent(s, llm=recorder)
                if mode == "agent"
                else SingleShotRAG(s, llm=recorder)
            )
            for i, q in enumerate(questions, 1):
                print(f"  [{i}/{len(questions)}] {mode}: {q['question'][:70]}", file=sys.stderr)
                try:
                    await system.answer(q["question"])
                except Exception as exc:  # noqa: BLE001
                    print(f"    failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        await recorder.aclose()

    merged = {**existing, **recorder.records}
    FIXTURES.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURES.open("w", encoding="utf-8") as fh:
        for key in sorted(merged):
            fh.write(json.dumps(merged[key], ensure_ascii=False) + "\n")

    print(
        f"\nrecorded {len(recorder.records)} generations "
        f"({len(merged)} total in the fixture set) -> {FIXTURES}"
    )
    return 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    raise SystemExit(main())
