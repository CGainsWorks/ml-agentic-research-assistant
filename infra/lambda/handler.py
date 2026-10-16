"""AWS Lambda entry point for scheduled ingestion.

Deliberately thin. It syncs the corpus down from S3, calls the same
`scheduled_ingest()` the local APScheduler worker calls, and syncs the rebuilt
index and run log back up. All the actual logic lives in
`src/araa/ingestion/`, so the scheduled path has one implementation and two
triggers rather than two implementations that drift.

NOT DEPLOYED. The Terraform that would wire this up has never been applied; see
`infra/terraform/main.tf`. This file is written to be deployable, and is
imported by the test suite so it cannot rot into something that would not
import, but it has not run in Lambda.

Lambda specifics that shaped this:

* Only `/tmp` is writable, so every ARAA_* path points there.
* `/tmp` does not persist between invocations reliably, so the corpus is
  synced down rather than assumed present. Re-downloading 200 PDFs from arXiv
  on every run would be both slow and rude, which is why the S3 copy is the
  source of truth and arXiv is only consulted for what is missing.
* Ingestion is embeddings-only. It never calls a generative model, so this
  function needs no LLM credentials at all.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger()
log.setLevel(logging.INFO)

BUCKET = os.environ.get("ARAA_CORPUS_BUCKET", "")
CORPUS_PREFIX = "corpus/"
INDEX_PREFIX = "index/"
RUNS_PREFIX = "runs/"


def _s3():
    import boto3  # provided by the Lambda runtime

    return boto3.client("s3")


def _download_prefix(client, bucket: str, prefix: str, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            target = dest / key[len(prefix) :]
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(target))
            count += 1
    return count


def _upload_dir(client, bucket: str, src: Path, prefix: str) -> int:
    if not src.exists():
        return 0
    count = 0
    for path in src.rglob("*"):
        if path.is_file():
            client.upload_file(
                str(path), bucket, prefix + str(path.relative_to(src)).replace(os.sep, "/")
            )
            count += 1
    return count


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:  # noqa: ARG001
    from araa.config import get_settings
    from araa.ingestion.schedule import scheduled_ingest

    if not BUCKET:
        raise RuntimeError("ARAA_CORPUS_BUCKET is not set")

    settings = get_settings()
    settings.ensure_dirs()
    client = _s3()

    downloaded = _download_prefix(client, BUCKET, CORPUS_PREFIX, settings.corpus_dir)
    log.info("synced %d corpus objects from s3://%s/%s", downloaded, BUCKET, CORPUS_PREFIX)

    if not (settings.corpus_dir / "manifest.json").exists():
        # First run, or the bucket was emptied. Fetch from arXiv directly.
        log.info("no manifest in S3; fetching a fresh corpus")
        import subprocess
        import sys

        subprocess.run(
            [
                sys.executable,
                "/app/scripts/fetch_corpus.py",
                "--limit",
                str(settings.corpus_target_docs),
                "--out",
                str(settings.corpus_dir),
            ],
            check=True,
        )

    report = scheduled_ingest(settings)

    uploaded = 0
    uploaded += _upload_dir(client, BUCKET, settings.corpus_dir, CORPUS_PREFIX)
    uploaded += _upload_dir(client, BUCKET, settings.index_dir, INDEX_PREFIX)
    uploaded += _upload_dir(client, BUCKET, settings.chroma_dir, INDEX_PREFIX + "chroma/")
    uploaded += _upload_dir(client, BUCKET, settings.runs_dir, RUNS_PREFIX)
    log.info("uploaded %d objects to s3://%s", uploaded, BUCKET)

    body = {
        "documents_ingested": report.documents_ingested,
        "documents_failed": report.documents_failed,
        "chunks": report.chunks,
        "duration_s": report.duration_s,
        "objects_uploaded": uploaded,
        "digest": f"s3://{BUCKET}/{RUNS_PREFIX}digest-latest.md",
    }
    log.info("ingest complete: %s", json.dumps(body))

    # A non-zero failure count is not an exception - the run succeeded, some
    # documents did not. Surfacing it in the response keeps the CloudWatch
    # Errors alarm meaning "the job broke" rather than "a PDF was malformed".
    return {"statusCode": 200, "body": json.dumps(body)}
