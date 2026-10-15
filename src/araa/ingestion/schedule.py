"""Scheduled ingestion / re-embedding worker.

Runs the same `run_ingest` the manual path uses, on a cron schedule, and writes
a digest report after each run. The digest is the point: a pipeline that
silently re-indexes is indistinguishable from a pipeline that silently stopped
re-indexing, and you only find out when an answer cites something that no longer
exists.

Locally this is an APScheduler process (`docker compose --profile scheduler up`).
The same entry point is what `infra/lambda/handler.py` calls, so the scheduled
path is one implementation with two triggers rather than two implementations.

Run:  python -m araa.ingestion.schedule
"""

from __future__ import annotations

import json
import logging
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path

from araa.config import Settings, get_settings
from araa.ingestion.pipeline import IngestReport, run_ingest

log = logging.getLogger("araa.schedule")


def write_digest(report: IngestReport, settings: Settings, previous: dict | None) -> Path:
    """A short human-readable summary, plus the deltas against the last run."""
    delta = {}
    if previous:
        for key in ("documents_ingested", "chunks", "vector_records"):
            delta[key] = report.__dict__[key] - previous.get(key, 0)

    lines = [
        "# Ingestion digest",
        "",
        f"- Run finished: {report.finished_at}",
        f"- Duration: {report.duration_s}s",
        f"- Documents ingested: {report.documents_ingested} of {report.documents_seen} seen",
        f"- Chunks: {report.chunks} (mean {report.chunk_chars_mean} chars, p95 {report.chunk_chars_p95})",
        f"- Vector records: {report.vector_records}",
        f"- BM25 records: {report.bm25_records}",
        f"- Skipped, no licence: {report.documents_skipped_no_licence}",
        f"- Failed: {report.documents_failed}",
        "",
        "## Chunking / embedding configuration",
        "",
        "```json",
        json.dumps(report.config, indent=2),
        "```",
    ]

    if delta:
        lines += [
            "",
            "## Change since previous run",
            "",
            "| Metric | Delta |",
            "| --- | ---: |",
            *[f"| {k} | {v:+d} |" for k, v in delta.items()],
        ]

    if report.failures:
        lines += ["", "## Failures", ""]
        lines += [f"- `{f.get('doc')}`: {f.get('error')}" for f in report.failures[:50]]
        if len(report.failures) > 50:
            lines.append(f"- ... and {len(report.failures) - 50} more")

    if report.licences:
        lines += ["", "## Licences represented", "", "| Licence | Documents |", "| --- | ---: |"]
        lines += [
            f"| {k} | {v} |" for k, v in sorted(report.licences.items(), key=lambda kv: -kv[1])
        ]

    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = settings.runs_dir / f"digest-{stamp}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (settings.runs_dir / "digest-latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def scheduled_ingest(settings: Settings | None = None) -> IngestReport:
    s = settings or get_settings()
    previous = None
    latest = s.runs_dir / "ingest-latest.json"
    if latest.exists():
        try:
            previous = json.loads(latest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = None

    log.info("scheduled ingest starting")
    report = run_ingest(s, reset=True)
    digest = write_digest(report, s, previous)
    log.info("scheduled ingest complete; digest at %s", digest)
    return report


def _parse_cron(expr: str) -> dict[str, str]:
    parts = expr.split()
    if len(parts) != 5:
        raise ValueError(f"cron expression must have 5 fields, got {expr!r}")
    minute, hour, day, month, day_of_week = parts
    return {
        "minute": minute,
        "hour": hour,
        "day": day,
        "month": month,
        "day_of_week": day_of_week,
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    s = get_settings()

    if not s.ingest_enabled:
        log.warning(
            "ARAA_INGEST_ENABLED is false; the scheduler will idle. "
            "Set it to true to run on the %s schedule.",
            s.ingest_cron,
        )

    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler(timezone="UTC")
    if s.ingest_enabled:
        scheduler.add_job(
            scheduled_ingest,
            CronTrigger(**_parse_cron(s.ingest_cron), timezone="UTC"),
            id="reingest",
            max_instances=1,
            coalesce=True,  # a missed window runs once, not N times
            misfire_grace_time=3600,
        )
        log.info("scheduled re-ingestion on cron '%s' (UTC)", s.ingest_cron)

    def _shutdown(signum, _frame):  # pragma: no cover - signal path
        log.info("signal %s received, shutting down", signum)
        scheduler.shutdown(wait=False)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):  # pragma: no cover
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
