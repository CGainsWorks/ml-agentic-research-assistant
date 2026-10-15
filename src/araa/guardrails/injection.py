"""Prompt-injection detection.

The threat model this repo actually faces: the corpus is third-party PDFs. Any
one of them can contain text addressed to whatever LLM later reads it. That text
arrives in the model's context wearing the same clothes as legitimate evidence.

Detection is the *second* line of defence, not the first. The first is
structural, and lives in the prompt (see `rag/prompts.py`): retrieved passages
are fenced, labelled untrusted, and the system prompt states that instructions
found inside them are data to be reported, never commands to be followed. A
regex list will always be incomplete; an instruction hierarchy degrades more
gracefully when it is.

What this module adds on top:

1. **Normalisation before matching.** Attacks hide in encoding — zero-width
   joiners inside "ignore", full-width homoglyphs, HTML entities, base64. We
   normalise first, so `ig\\u200bnore previous instructions` is caught by the
   same rule as the plain text version.
2. **A categorised rule set** with severities, so the caller can block on
   `critical` and merely annotate on `low`.
3. **A structural heuristic**: research papers describe, they do not address
   the reader's assistant in the imperative. Second-person imperatives inside
   retrieved content are suspicious regardless of whether they match a known
   attack string.

Every detection is reported in the API response and in the retrieval trace.
Silently dropping a poisoned passage would hide the fact that the corpus is
compromised, which is information the operator needs more than the answer.
"""

from __future__ import annotations

import base64
import binascii
import html
import re
import unicodedata
from dataclasses import dataclass, field
from enum import IntEnum


class Severity(IntEnum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(slots=True)
class Detection:
    category: str
    severity: Severity
    pattern: str
    excerpt: str
    source: str = "unknown"  # "question" | chunk_id
    normalised: bool = False  # matched only after decoding/normalising

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "severity": self.severity.name.lower(),
            "pattern": self.pattern,
            "excerpt": self.excerpt,
            "source": self.source,
            "found_after_normalisation": self.normalised,
        }


@dataclass
class ScanResult:
    detections: list[Detection] = field(default_factory=list)

    @property
    def max_severity(self) -> Severity:
        return max((d.severity for d in self.detections), default=Severity.NONE)

    @property
    def triggered(self) -> bool:
        return bool(self.detections)

    def should_block(self, threshold: Severity = Severity.HIGH) -> bool:
        return self.max_severity >= threshold

    def by_source(self) -> dict[str, list[Detection]]:
        out: dict[str, list[Detection]] = {}
        for d in self.detections:
            out.setdefault(d.source, []).append(d)
        return out

    def to_dict(self) -> dict:
        return {
            "triggered": self.triggered,
            "max_severity": self.max_severity.name.lower(),
            "count": len(self.detections),
            "detections": [d.to_dict() for d in self.detections],
        }

    def extend(self, other: ScanResult) -> ScanResult:
        self.detections.extend(other.detections)
        return self


# (name, severity, regex)
RULES: list[tuple[str, Severity, re.Pattern[str]]] = [
    (
        "instruction_override",
        Severity.CRITICAL,
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass|discard)\b[^.\n]{0,40}?"
            r"\b(previous|prior|earlier|above|all|any|system|initial|original)\b"
            r"[^.\n]{0,30}?\b(instruction|prompt|rule|direction|guideline|context|message)s?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "persona_hijack",
        Severity.HIGH,
        re.compile(
            r"\b(you are now|from now on,? you|act as (if|though|a)|pretend (to be|you)|"
            r"new (instructions?|persona|role|system prompt)\s*:|roleplay as|"
            r"switch (to|into) .{0,20}mode)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "system_prompt_exfiltration",
        Severity.CRITICAL,
        re.compile(
            r"\b(reveal|repeat|print|output|show|display|disclose|dump|echo)\b[^.\n]{0,40}?"
            r"\b(system prompt|initial (prompt|instructions?)|your (instructions?|rules|prompt|"
            r"guidelines|configuration)|everything above|verbatim)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "data_exfiltration",
        Severity.CRITICAL,
        re.compile(
            r"\b(send|post|upload|transmit|exfiltrate|forward|email|leak)\b[^.\n]{0,50}?"
            r"(https?://|www\.|@[\w.-]+\.\w{2,}|\bto\s+(the\s+)?(following|this)\s+(url|address|endpoint|server))",
            re.IGNORECASE,
        ),
    ),
    (
        "markdown_exfiltration",
        Severity.HIGH,
        # ![](https://evil/?q=<data>) - the classic zero-click image beacon
        re.compile(r"!\[[^\]]*\]\(\s*https?://[^)\s]*[?&][^)\s]*\)", re.IGNORECASE),
    ),
    (
        "fake_authority",
        Severity.HIGH,
        re.compile(
            r"(\b(system|admin(istrator)?|developer|operator|vendor|provider|openai)\s*(message|note|"
            r"override|instruction)\s*:)|(</?(system|im_start|im_end)\b)|(\[/?INST\])|"
            r"(<\|(im_start|im_end|system|endoftext)\|>)",
            re.IGNORECASE,
        ),
    ),
    (
        "guardrail_disable",
        Severity.HIGH,
        re.compile(
            r"\b(do not|don'?t|never|stop)\b[^.\n]{0,30}?"
            r"\b(cite|citation|refuse|filter|safety|guardrail|verify|check|mention)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "tool_abuse",
        Severity.HIGH,
        re.compile(
            r"\b(call|invoke|execute|run|use)\b[^.\n]{0,30}?"
            r"\b(tool|function|command|shell|bash|python|os\.system|subprocess|eval)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "urgency_pressure",
        Severity.MEDIUM,
        re.compile(
            r"\b(urgent|immediately|critical|you must|it is (essential|imperative|vital)|"
            r"failure to comply|this is a test of)\b[^.\n]{0,40}?"
            r"\b(comply|obey|follow|instruction|respond|answer)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "hidden_channel",
        Severity.MEDIUM,
        re.compile(
            r"<!--.{0,400}?-->|<script\b|style\s*=\s*[\"'][^\"']*(display\s*:\s*none|font-size\s*:\s*0)",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "citation_forgery",
        Severity.HIGH,
        # A retrieved passage containing our own citation markers is trying to
        # mint provenance for a passage that was never retrieved. Sanitisation
        # defangs the marker; this rule makes sure the attempt is also *reported*
        # rather than silently rewritten.
        re.compile(r"\[C\d{1,3}\]"),
    ),
]

# --------------------------------------------------------------------------
# Descriptive vs imperative.
#
# This corpus is computer-science papers, so it genuinely contains papers about
# prompt injection. "Prior work shows that instructions embedded in retrieved
# documents can override a model's system prompt" is a sentence from a paper,
# not an attack — and the first version of these rules blocked it, which would
# make a security corpus unusable. The benign controls in the adversarial suite
# exist to catch precisely that, and they did.
#
# The discriminator is grammatical mood, not vocabulary. An attack issues a
# command; a paper describes one. So a match is downgraded when it sits in
# academic framing AND is not in imperative position. An attacker who prefixes
# "Prior work shows that" still gets caught if the verb is imperative or
# addressed at "you", which is what an actual instruction has to be to work.
_ACADEMIC_FRAMING = re.compile(
    r"\b(prior|previous|related|recent)\s+(work|research|studies|literature)"
    r"|\bwe\s+(evaluate|show|study|investigate|propose|consider|describe|find|measure|demonstrate|present|report)"
    r"|\bthis\s+(paper|work|study|section|approach)"
    r"|\bour\s+(results|method|approach|work|experiments|analysis)"
    r"|\b(has|have|had)\s+been\s+shown"
    r"|\b(attack|adversar|threat model|mitigat|defen[cs]e|vulnerab|countermeasure)\w*"
    r"|\bcan\s+be\s+used\s+to\b|\bwhether\b|\bsuch\s+(attacks|instructions|techniques|methods)"
    r"|\b(benchmark|dataset|baseline|ablation)\b",
    re.IGNORECASE,
)

# Match sits at the start of a sentence/clause => imperative mood => a command.
_IMPERATIVE_POSITION = re.compile(r"(?:^|[.!?:;]\s*|\n\s*)$")
# ...or is explicitly addressed at the assistant.
_ADDRESSED_AT_ASSISTANT = re.compile(r"\b(you|your|please)\s+\S{0,20}$", re.IGNORECASE)


def _sentence_around(text: str, start: int, end: int) -> str:
    left = max(text.rfind(".", 0, start), text.rfind("\n", 0, start)) + 1
    right = min(
        (i for i in (text.find(".", end), text.find("\n", end)) if i != -1),
        default=len(text),
    )
    return text[left : right + 1]


def _is_descriptive(text: str, match: re.Match[str]) -> bool:
    """True when the match reads as a paper describing an attack, not issuing one."""
    preceding = text[max(0, match.start() - 60) : match.start()]
    if _IMPERATIVE_POSITION.search(preceding) or _ADDRESSED_AT_ASSISTANT.search(preceding):
        return False
    return bool(_ACADEMIC_FRAMING.search(_sentence_around(text, match.start(), match.end())))


# Text addressed at the reader's assistant, in a document that should be
# describing research. Weak on its own, which is why it is LOW.
_IMPERATIVE_AT_ASSISTANT = re.compile(
    r"\b(you (must|should|will|need to|are required to)|your (task|job|goal|only job) is|"
    r"assistant[,:]|\bAI\b[,:]|when (answering|asked|you respond))\b",
    re.IGNORECASE,
)

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿᠎­؜⁦⁧⁨⁩‪‫‬‭‮"))

_B64_BLOB = re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b")


def normalise(text: str) -> str:
    """Fold away the encodings an attacker uses to slip past a literal match."""
    text = html.unescape(text)
    text = text.translate(_ZERO_WIDTH)
    # NFKC folds full-width and mathematical alphanumerics onto ASCII, which is
    # what defeats "ｉｇｎｏｒｅ ａｌｌ ｐｒｅｖｉｏｕｓ ｉｎｓｔｒｕｃｔｉｏｎｓ".
    text = unicodedata.normalize("NFKC", text)
    # Collapse letter-spacing tricks: "i g n o r e   a l l"
    text = re.sub(r"(?<=\b\w)(?:\s(?=\w\b)){1,}", "", text)
    return text


def _decoded_candidates(text: str) -> list[str]:
    """Base64 blobs long enough to hide an instruction, decoded for scanning."""
    out: list[str] = []
    for blob in _B64_BLOB.findall(text)[:8]:
        try:
            decoded = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True)
        except (binascii.Error, ValueError):
            continue
        try:
            as_text = decoded.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if sum(c.isprintable() for c in as_text) / max(len(as_text), 1) > 0.85:
            out.append(as_text)
    return out


def _excerpt(text: str, match: re.Match[str], width: int = 90) -> str:
    start = max(0, match.start() - 25)
    end = min(len(text), match.end() + 25)
    snippet = text[start:end].replace("\n", " ").strip()
    if len(snippet) > width:
        snippet = snippet[:width] + "..."
    return snippet


def scan_text(text: str, *, source: str = "unknown", is_retrieved: bool = False) -> ScanResult:
    result = ScanResult()
    if not text or not text.strip():
        return result

    raw = text
    folded = normalise(text)
    variants: list[tuple[str, bool]] = [(raw, False)]
    if folded != raw:
        variants.append((folded, True))
    variants.extend((d, True) for d in _decoded_candidates(folded))

    seen: set[tuple[str, str]] = set()
    for variant, was_normalised in variants:
        for name, severity, pattern in RULES:
            for match in pattern.finditer(variant):
                excerpt = _excerpt(variant, match)
                key = (name, excerpt)
                if key in seen:
                    continue
                seen.add(key)

                effective = severity
                descriptive = False
                if severity >= Severity.HIGH and _is_descriptive(variant, match):
                    # Report it, but do not block on it. A paper discussing an
                    # attack is still worth surfacing to the operator; it is not
                    # grounds for refusing to answer.
                    effective = Severity.LOW
                    descriptive = True

                result.detections.append(
                    Detection(
                        category=name,
                        severity=effective,
                        pattern=name + (" (descriptive, downgraded)" if descriptive else ""),
                        excerpt=excerpt,
                        source=source,
                        normalised=was_normalised,
                    )
                )

    # Structural heuristic, retrieved content only. A user is entitled to say
    # "you should focus on X"; a PDF is not.
    if is_retrieved:
        for match in _IMPERATIVE_AT_ASSISTANT.finditer(folded):
            excerpt = _excerpt(folded, match)
            key = ("assistant_directed_imperative", excerpt)
            if key in seen:
                continue
            seen.add(key)
            result.detections.append(
                Detection(
                    category="assistant_directed_imperative",
                    severity=Severity.LOW,
                    pattern="second-person imperative in retrieved document",
                    excerpt=excerpt,
                    source=source,
                )
            )

    return result


def scan_question(question: str) -> ScanResult:
    return scan_text(question, source="question", is_retrieved=False)


def scan_retrieved(chunks: list[tuple[str, str]]) -> ScanResult:
    """chunks: (chunk_id, text)."""
    out = ScanResult()
    for chunk_id, text in chunks:
        out.extend(scan_text(text, source=chunk_id, is_retrieved=True))
    return out


def sanitise_retrieved(text: str) -> str:
    """Neutralise what we can without destroying the evidence.

    We deliberately do not delete the offending sentence. The passage is
    reported to the operator with its detections attached, and a deleted
    sentence is a passage you can no longer audit. What we do remove is the
    machinery that makes an injection *work* mechanically: invisible
    characters, HTML comments, and role-delimiter lookalikes that could break
    out of the fence we wrap the passage in.
    """
    text = text.translate(_ZERO_WIDTH)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(
        r"<\s*/?\s*(system|im_start|im_end|s|human|assistant)\s*>", " ", text, flags=re.IGNORECASE
    )
    text = re.sub(r"<\|[^|>]{0,40}\|>", " ", text)
    text = re.sub(r"\[/?INST\]", " ", text, flags=re.IGNORECASE)
    # Our own fence and citation markers must not be forgeable from inside a
    # passage: a document containing "[C7]" could otherwise attribute a claim to
    # a passage that was never retrieved.
    text = re.sub(r"\[C(\d)", r"[ C\1", text)
    text = text.replace("<<<UNTRUSTED", "<< <UNTRUSTED").replace("UNTRUSTED>>>", "UNTRUSTED> >>")
    return text
