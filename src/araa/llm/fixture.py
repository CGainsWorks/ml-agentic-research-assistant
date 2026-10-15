"""Deterministic replay backend.

This is NOT a model. It looks up a previously recorded generation by a hash of
(model, system, prompt) and replays it verbatim. Its only purpose is to let CI
exercise the entire graph — planning, tool use, critique, revision, citation
validation — without a model server or an API key, and to make agent-level
regressions detectable as diffs rather than as noise.

A cache miss raises. It never invents an answer, because a fixture backend that
silently fabricates output would turn every eval number downstream into a lie.

Record fixtures with:  python -m evals.record_fixtures
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from araa.config import Settings
from araa.llm.base import LLM, Generation, Usage


class FixtureMiss(KeyError):
    """Raised when a prompt has no recorded generation."""


def fixture_key(model: str, system: str | None, prompt: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\x00")
    h.update((system or "").encode())
    h.update(b"\x00")
    h.update(prompt.encode())
    return h.hexdigest()[:32]


class FixtureLLM(LLM):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._path = Path(settings.fixture_path)
        self._store: dict[str, dict] = {}
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._store[rec["key"]] = rec

    @property
    def model_name(self) -> str:
        return "fixture"

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Generation:
        # Fixtures are keyed on the model that recorded them so a fixture set is
        # never silently reused across a model change.
        key = fixture_key(self._recorded_model, system, prompt)
        rec = self._store.get(key)
        if rec is None:
            raise FixtureMiss(
                f"No recorded generation for key {key}. "
                f"Fixture file: {self._path}. "
                "Re-record with `python -m evals.record_fixtures` against a live backend."
            )
        return Generation(
            text=rec["text"],
            usage=Usage(
                input_tokens=rec.get("input_tokens", 0),
                output_tokens=rec.get("output_tokens", 0),
            ),
            model=rec.get("model", "fixture"),
            latency_s=0.0,
            raw={"replayed": True},
        )

    @property
    def _recorded_model(self) -> str:
        """Fixtures record under the model that produced them; the default
        matches what `record_fixtures` writes with the default env."""
        return self.settings.ollama_model

    def __len__(self) -> int:
        return len(self._store)
