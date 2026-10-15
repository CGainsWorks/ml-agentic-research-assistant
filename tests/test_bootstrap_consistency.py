"""The compose bootstrap must agree with the package configuration.

`scripts/bootstrap.sh` pulls the model the API will later ask for. When it
carried its own hardcoded default it drifted: `config.py` moved to
`llama3.1:8b` while the script still said `qwen2.5:7b-instruct`. A cold
`docker compose up` therefore downloaded 4.7 GB of the wrong model and then
served an API permanently reporting `degraded / model not pulled`.

Nothing caught it. Unit tests mock the model, and the mismatch only appears on
a genuinely cold start — which is exactly why the end-to-end test exists, and
why these cheap assertions exist alongside it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from araa.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap.sh"
COMPOSE = REPO_ROOT / "docker-compose.yml"


def _bootstrap_text() -> str:
    if not BOOTSTRAP.exists():
        pytest.skip("scripts/ not present in this build context")
    return BOOTSTRAP.read_text(encoding="utf-8")


def test_bootstrap_does_not_hardcode_a_model_name():
    """A literal `model:tag` default here is the drift waiting to happen."""
    text = _bootstrap_text()
    assignments = re.findall(r"^OLLAMA_MODEL=(.*)$", text, re.MULTILINE)
    assert assignments, "expected bootstrap.sh to set OLLAMA_MODEL"
    for value in assignments:
        # `${ARAA_OLLAMA_MODEL:-something-literal}` is the shape that drifted.
        literal_default = re.search(r":-\s*([A-Za-z0-9._-]+:[A-Za-z0-9._-]+)\s*\}", value)
        assert literal_default is None, (
            f"bootstrap.sh hardcodes the model {literal_default.group(1)!r}. "
            "Derive it from araa.config instead — that default will drift."
        )


def test_bootstrap_derives_the_model_from_the_package():
    text = _bootstrap_text()
    assert "araa.config" in text, (
        "bootstrap.sh should read the model name from araa.config so it cannot "
        "disagree with what the API will request."
    )


def test_bootstrap_fails_loudly_if_it_cannot_determine_the_model():
    """Silently pulling nothing is worse than refusing to start."""
    text = _bootstrap_text()
    assert re.search(r"could not determine the model", text), (
        "bootstrap.sh should exit non-zero with a message when the model cannot "
        "be resolved, rather than continuing with an empty name."
    )


def test_compose_does_not_pin_a_conflicting_model():
    """If compose ever sets ARAA_OLLAMA_MODEL it must not contradict the default
    silently — pinning is fine, but it should be a deliberate, visible choice."""
    if not COMPOSE.exists():
        pytest.skip("docker-compose.yml not present in this build context")
    text = COMPOSE.read_text(encoding="utf-8")
    pinned = re.findall(r"ARAA_OLLAMA_MODEL:\s*(\S+)", text)
    for value in pinned:
        assert value == Settings.model_fields["ollama_model"].default, (
            f"docker-compose.yml pins ARAA_OLLAMA_MODEL={value!r} which differs "
            f"from the package default "
            f"{Settings.model_fields['ollama_model'].default!r}."
        )


def test_bootstrap_is_idempotent_by_construction():
    """Every step must be skippable, or a second `docker compose up` re-fetches
    the corpus and re-embeds the whole index."""
    text = _bootstrap_text()
    assert "already present" in text, "model pull should be skipped when present"
    assert "skipping fetch" in text, "corpus fetch should be skipped when present"
    assert "skipping ingest" in text, "ingest should be skipped when the index exists"
