"""Scripts invoked as `./script.sh` must be executable *in git*.

Git stores the executable bit, and on Windows it is not inferred from the
filesystem — every script in this repository was committed as 100644. Locally
nothing noticed, because everything ran through `bash script.sh` or a container.
On a Linux CI runner `./scripts/e2e_test.sh` is "Permission denied", so the
brand-new e2e workflow would have failed on its first scheduled run, and
`./scripts/setup.sh` from the README would have failed for anyone who cloned it.

Checking the mode in the index is the only reliable test: a fresh clone gets
whatever git recorded, regardless of what the developer's working tree says.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Files a human or a workflow invokes directly, rather than via `bash x.sh`
# or `python x.py`.
MUST_BE_EXECUTABLE = [
    "scripts/e2e_test.sh",
    "scripts/setup.sh",
    "scripts/run_demo.sh",
    "scripts/bootstrap.sh",
    "scripts/fetch_corpus.py",
]


def _git_modes() -> dict[str, str]:
    """path -> mode, as recorded in the git index."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-s", "scripts/"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pytest.skip("git not available or not a repository (e.g. inside the image)")

    modes: dict[str, str] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        meta, _, path = line.partition("\t")
        modes[path.strip()] = meta.split()[0]
    if not modes:
        pytest.skip("scripts/ not tracked in this checkout")
    return modes


@pytest.mark.parametrize("script", MUST_BE_EXECUTABLE)
def test_script_is_executable_in_git(script):
    modes = _git_modes()
    if script not in modes:
        pytest.skip(f"{script} not tracked here")
    assert modes[script] == "100755", (
        f"{script} is committed as {modes[script]}, not 100755. "
        f"A fresh clone on Linux cannot run it. Fix with:\n"
        f"    git update-index --chmod=+x {script}"
    )


def test_every_shell_script_with_a_shebang_is_executable():
    """Catches scripts added later that this list has not been updated for."""
    modes = _git_modes()
    for path, mode in sorted(modes.items()):
        full = REPO_ROOT / path
        if not full.exists():
            continue
        first = full.read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]
        if first.startswith("#!"):
            assert mode == "100755", (
                f"{path} starts with a shebang ({first!r}) but is committed as "
                f"{mode}. Either drop the shebang or run:\n"
                f"    git update-index --chmod=+x {path}"
            )
