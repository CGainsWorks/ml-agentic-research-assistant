"""Output-side guardrails.

Input filtering catches attacks it has seen. Output validation catches attacks
that *worked*, which is the more useful signal and the one that does not depend
on having anticipated the attack string.

Four checks, in order of how much they tell you:

1. **Canary leak.** The system prompt carries a random token generated per
   process. If it appears in the output, the system prompt has been echoed —
   a definitive, zero-false-positive detection of a successful exfiltration.
2. **Fabricated citations.** A marker referring to a passage that was never
   retrieved. Mechanically checkable, no judge required.
3. **Exfiltration sinks.** URLs and markdown images in the answer that did not
   come from the retrieved documents' own metadata. A research answer has no
   business inventing a link.
4. **Claimed actions.** The model asserting it sent, posted, emailed or
   executed something. It has no such tools; if it says it did, either it is
   hallucinating capability or it has adopted an injected persona.
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field

from araa.guardrails.injection import Severity

# One canary per process. Not persisted: a canary in a config file is a canary
# an attacker can read out of the repo and instruct the model to avoid.
CANARY = f"araa-canary-{secrets.token_hex(8)}"

_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_CLAIMED_ACTION_RE = re.compile(
    r"\b(i (have |'ve )?(sent|posted|uploaded|emailed|forwarded|transmitted|deleted|executed|ran)"
    r"|i (will|'ll) (send|post|upload|email|forward|execute|run)"
    r"|(message|email|request) (has been|was) (sent|delivered|posted))\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class OutputIssue:
    category: str
    severity: Severity
    detail: str

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "severity": self.severity.name.lower(),
            "detail": self.detail,
        }


@dataclass
class OutputValidation:
    issues: list[OutputIssue] = field(default_factory=list)
    redacted_answer: str | None = None

    @property
    def max_severity(self) -> Severity:
        return max((i.severity for i in self.issues), default=Severity.NONE)

    @property
    def passed(self) -> bool:
        return self.max_severity < Severity.HIGH

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "max_severity": self.max_severity.name.lower(),
            "issues": [i.to_dict() for i in self.issues],
        }


def validate_output(
    answer: str,
    *,
    allowed_urls: set[str] | None = None,
    fabricated_markers: list[str] | None = None,
    canary: str = CANARY,
) -> OutputValidation:
    result = OutputValidation()
    allowed_urls = allowed_urls or set()

    if canary and canary in answer:
        result.issues.append(
            OutputIssue(
                "system_prompt_leak",
                Severity.CRITICAL,
                "The system-prompt canary token appeared in the answer.",
            )
        )
        result.redacted_answer = answer.replace(canary, "[REDACTED]")

    for marker in fabricated_markers or []:
        result.issues.append(
            OutputIssue(
                "fabricated_citation",
                Severity.HIGH,
                f"Citation {marker} refers to a passage that was never retrieved.",
            )
        )

    allowed_prefixes = tuple(u.rstrip("/") for u in allowed_urls if u)
    for url in _URL_RE.findall(answer):
        clean = url.rstrip(".,);")
        if allowed_prefixes and clean.rstrip("/").startswith(allowed_prefixes):
            continue
        if clean.startswith(
            (
                "https://arxiv.org/",
                "http://arxiv.org/",
                "https://creativecommons.org/",
                "http://creativecommons.org/",
            )
        ):
            continue  # provenance links this system generates itself
        result.issues.append(
            OutputIssue(
                "unexpected_url",
                Severity.MEDIUM,
                f"Answer contains a URL not present in the retrieved sources: {clean}",
            )
        )

    for image in _MD_IMAGE_RE.findall(answer):
        result.issues.append(
            OutputIssue(
                "markdown_image",
                Severity.HIGH,
                f"Answer contains a markdown image, a classic exfiltration beacon: {image[:120]}",
            )
        )

    if match := _CLAIMED_ACTION_RE.search(answer):
        result.issues.append(
            OutputIssue(
                "claimed_action",
                Severity.HIGH,
                f"Answer claims to have performed an action it has no tool for: {match.group(0)!r}",
            )
        )

    return result


def refusal_message(reason: str) -> str:
    return (
        "I stopped before answering.\n\n"
        f"Reason: {reason}\n\n"
        "The retrieved passages contained text attempting to redirect this assistant "
        "rather than to inform it. Treating that text as evidence would produce an "
        "answer grounded in an attack rather than in the corpus. The detections are "
        "included in the `guardrails` field of this response so the affected documents "
        "can be inspected."
    )
