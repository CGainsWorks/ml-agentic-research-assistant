"""Every module in the repository must import.

This exists because a syntax error shipped undetected. `run_groundedness.py`
contained an f-string that reused the same quote character inside its
expression — legal from Python 3.12, a SyntaxError on the 3.11 this project
targets. `ruff check` did not flag it, `ruff format` was happy with it, and no
test imported the module, so it surfaced only when a long eval run reached step
two and died.

Import-checking is cheap and catches an entire class of failure that unit tests
of *behaviour* structurally cannot: the eval scripts and the Lambda handler are
entry points nothing else imports, so without this they are only ever
syntax-checked by being run.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _package_modules(package_name: str) -> list[str]:
    package = importlib.import_module(package_name)
    found = [package_name]
    for _finder, name, _ispkg in pkgutil.walk_packages(package.__path__, f"{package_name}."):
        found.append(name)
    return sorted(found)


@pytest.mark.parametrize("module", _package_modules("araa"))
def test_araa_module_imports(module):
    importlib.import_module(module)


@pytest.mark.parametrize("module", _package_modules("evals"))
def test_eval_module_imports(module):
    """The eval entry points nothing else imports."""
    importlib.import_module(module)


def test_standalone_scripts_compile():
    """scripts/ holds files run directly, never imported."""
    import py_compile

    scripts = sorted((REPO_ROOT / "scripts").glob("*.py"))
    assert scripts, "expected at least one script"
    for path in scripts:
        py_compile.compile(str(path), doraise=True)


def test_lambda_handler_compiles():
    """`infra/lambda/` is not an importable package name (`lambda` is a keyword),
    so it is compiled rather than imported. The Terraform points the function at
    this file, so it must at least be valid Python."""
    import py_compile

    handler = REPO_ROOT / "infra" / "lambda" / "handler.py"
    if not handler.exists():
        pytest.skip("infra/ not present in this build context")
    py_compile.compile(str(handler), doraise=True)
