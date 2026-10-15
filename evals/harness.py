"""Shared eval machinery: metrics, artefact IO, reproducibility stamping.

Design rule for everything in `evals/`: an eval writes a JSON artefact into
`data/runs/` containing the metric, the configuration that produced it, and the
per-item detail. `evals/report.py` renders `docs/results.md` from those
artefacts, and the README quotes `docs/results.md`. So a number in the README is
always traceable to a specific run of a specific script against a specific
config — never typed in by hand.
"""

from __future__ import annotations

import json
import math
import platform
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from araa.config import Settings, get_settings

__version__ = "1"


# ------------------------------------------------------------------ metrics
def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Fraction of gold items appearing in the top k."""
    if not gold:
        return 0.0
    top = set(retrieved[:k])
    return len(top & gold) / len(gold)


def hit_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """1.0 if any gold item is in the top k. The metric that matters when there
    is exactly one right answer and the reader only reads the first page."""
    return 1.0 if set(retrieved[:k]) & gold else 0.0


def mrr(retrieved: list[str], gold: set[str]) -> float:
    for i, item in enumerate(retrieved, start=1):
        if item in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Binary-relevance nDCG. Rewards putting the right thing higher, not just
    somewhere in the list — which is what actually matters when only the top
    `final_top_k` chunks reach the model's context."""
    if not gold:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 1) for i, item in enumerate(retrieved[:k], start=1) if item in gold
    )
    ideal_n = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
    return ordered[idx]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% CI for a proportion.

    Reported alongside every rate in this repo. A groundedness number from 40
    questions has an interval roughly +/-15 points wide, and quoting it as a
    bare percentage invites the reader to over-read it.
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ---------------------------------------------------------------- artefacts
@dataclass
class EvalArtefact:
    name: str
    metrics: dict[str, Any]
    detail: list[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    harness_version: str = __version__

    def write(self, settings: Settings | None = None) -> Path:
        s = settings or get_settings()
        s.runs_dir.mkdir(parents=True, exist_ok=True)
        self.finished_at = self.finished_at or datetime.now(UTC).isoformat()
        path = s.runs_dir / f"eval-{self.name}.json"
        path.write_text(json.dumps(asdict(self), indent=2, default=_json_default), encoding="utf-8")
        return path

    @staticmethod
    def load(name: str, settings: Settings | None = None) -> dict | None:
        s = settings or get_settings()
        path = s.runs_dir / f"eval-{name}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


def _json_default(o: Any) -> Any:
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def environment_stamp(settings: Settings) -> dict:
    """Recorded in every artefact. Retrieval numbers move with the index, and
    the index moves with the corpus — so the corpus fingerprint is part of the
    result, not context you are expected to remember."""
    manifest = settings.corpus_dir / "manifest.json"
    corpus: dict[str, Any] = {"present": manifest.exists()}
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            corpus |= {
                "documents": data.get("document_count"),
                "licence_counts": data.get("licence_counts"),
                "fetch_params": data.get("fetch_params"),
            }
        except json.JSONDecodeError:
            corpus["error"] = "unreadable manifest"

    ingest = settings.runs_dir / "ingest-latest.json"
    index: dict[str, Any] = {"present": ingest.exists()}
    if ingest.exists():
        try:
            data = json.loads(ingest.read_text(encoding="utf-8"))
            index |= {
                "chunks": data.get("chunks"),
                "documents_ingested": data.get("documents_ingested"),
                "chunk_chars_mean": data.get("chunk_chars_mean"),
                "config": data.get("config"),
            }
        except json.JSONDecodeError:
            index["error"] = "unreadable ingest log"

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "corpus": corpus,
        "index": index,
    }


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


# ------------------------------------------------------------- question sets
QUESTIONS_DIR = Path(__file__).resolve().parent / "questions"


def load_question_set(name: str) -> list[dict]:
    path = QUESTIONS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Question set {name!r} not found at {path}.\n"
            "Build it first:  python -m evals.questions.build_question_set"
        )
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def save_question_set(name: str, questions: list[dict], meta: dict) -> Path:
    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = QUESTIONS_DIR / f"{name}.json"
    path.write_text(json.dumps({"meta": meta, "questions": questions}, indent=2), encoding="utf-8")
    return path


def require_index(settings: Settings) -> None:
    """Fail loudly and usefully rather than reporting a metric of 0.0."""
    from araa.retrieval.hybrid import ChunkStore

    if not (settings.index_dir / "chunks.jsonl").exists():
        raise SystemExit(
            "No index found. Build one first:\n"
            "  python scripts/fetch_corpus.py\n"
            "  python -m araa.ingestion.pipeline"
        )
    if len(ChunkStore(settings)) == 0:
        raise SystemExit("Index is empty. Re-run: python -m araa.ingestion.pipeline")


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def fmt_ci(successes: int, total: int) -> str:
    lo, hi = wilson_interval(successes, total)
    return f"{fmt_pct(successes / total if total else 0)} (95% CI {fmt_pct(lo)}–{fmt_pct(hi)}, n={total})"
