"""`.env.example` must describe the settings that actually exist.

Documentation drift is silent and cheap to prevent. This file had
`ARAA_OLLAMA_MODEL=qwen2.5:7b-instruct` and `ARAA_MAX_OUTPUT_TOKENS=1536` long
after the defaults had moved, so anyone copying it to `.env` would have pinned a
model they had not pulled and a token budget nobody had measured.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from araa.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = REPO_ROOT / ".env.example"

_ASSIGNMENT = re.compile(r"^\s*(ARAA_[A-Z0-9_]+)\s*=\s*(.*?)\s*$")


def _documented() -> dict[str, str]:
    # The Docker image does not copy .env.example, so these tests are a no-op
    # when run inside it and meaningful when run from the repo, which is what
    # CI does.
    if not ENV_EXAMPLE.exists():
        return {}
    out: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        if match := _ASSIGNMENT.match(line):
            out[match.group(1)] = match.group(2)
    return out


def _requires_env_example() -> None:
    if not ENV_EXAMPLE.exists():
        pytest.skip(".env.example not present in this build context")


@pytest.mark.parametrize("key", sorted(_documented()))
def test_every_documented_key_is_a_real_setting(key):
    field = key.removeprefix("ARAA_").lower()
    assert field in Settings.model_fields, (
        f"{key} is documented in .env.example but is not a Settings field. "
        "Either it was renamed or removed."
    )


@pytest.mark.parametrize("key,value", sorted(_documented().items()))
def test_documented_values_match_the_actual_defaults(key, value):
    """A documented value that differs from the default is worse than no
    documentation: it looks authoritative and is wrong.

    Compared after coercing to the field's own type, so `600` and `600.0`
    agree — the point is the value, not its spelling.
    """
    field = key.removeprefix("ARAA_").lower()
    default = Settings.model_fields[field].default

    if isinstance(default, bool):
        documented = value.strip().lower() in ("true", "1", "yes", "on")
    elif isinstance(default, float):
        documented = float(value)
    elif isinstance(default, int):
        documented = int(value)
    else:
        documented = value

    assert documented == default, f"{key}: documented={value!r} actual default={default!r}"


def test_the_settings_that_change_results_are_documented():
    """Not every field needs documenting, but the ones a reader would tune do."""
    _requires_env_example()
    must_document = {
        "ARAA_LLM_BACKEND",
        "ARAA_OLLAMA_MODEL",
        "ARAA_CHUNK_STRATEGY",
        "ARAA_CHUNK_SIZE",
        "ARAA_FINAL_TOP_K",
        "ARAA_USE_RERANKER",
        "ARAA_GUARDRAILS_ENABLED",
        "ARAA_INGEST_CRON",
    }
    missing = must_document - set(_documented())
    assert not missing, f"undocumented in .env.example: {sorted(missing)}"
