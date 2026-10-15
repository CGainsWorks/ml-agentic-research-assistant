"""Tool definitions and a dispatcher.

Deliberately not JSON function-calling. The default backend here is a 7B
open-weights model, and small models emit malformed JSON often enough that a
JSON protocol turns "the agent chose the wrong tool" into "the agent crashed" —
two very different failure modes, only one of which is recoverable. The line
protocol below has one bracket to get wrong:

    TOOL: search_corpus | query=hybrid retrieval reranking
    TOOL: query_table | table=corpus_metadata, op=value_counts, column=primary_category

It parses under a permissive grammar, it degrades to "no tool call" rather than
to an exception, and the same prompt works unchanged on a hosted model.

Every tool is read-only. Nothing in this inventory writes, sends, or executes.
That is a deliberate consequence of the threat model: the corpus is untrusted
input, so a successful injection must not be able to reach anything with an
effect.
"""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass, field
from typing import Any

TOOL_LINE_RE = re.compile(
    r"^\s*TOOL\s*:\s*(?P<name>[a-z_][a-z0-9_]*)\s*(?:\|\s*(?P<args>.*))?$", re.IGNORECASE
)


@dataclass(slots=True)
class ToolParam:
    name: str
    description: str
    required: bool = False
    default: Any = None


@dataclass(slots=True)
class ToolCall:
    name: str
    args: dict[str, str]
    raw: str


@dataclass(slots=True)
class ToolResult:
    name: str
    ok: bool
    content: str
    data: Any = None
    error: str | None = None
    elapsed_ms: float = 0.0

    def to_dict(self, max_chars: int = 1200) -> dict:
        return {
            "tool": self.name,
            "ok": self.ok,
            "content": self.content[:max_chars],
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


class Tool(abc.ABC):
    name: str = ""
    description: str = ""
    params: list[ToolParam] = []

    @abc.abstractmethod
    def run(self, args: dict[str, str]) -> ToolResult: ...

    def signature(self) -> str:
        if not self.params:
            return f"- {self.name}: {self.description}"
        args = ", ".join(f"{p.name}{'' if p.required else '?'}" for p in self.params)
        lines = [f"- {self.name} | {args}", f"    {self.description}"]
        lines += [
            f"    {p.name}: {p.description}"
            + ("" if p.required else f" (optional, default {p.default!r})")
            for p in self.params
        ]
        return "\n".join(lines)

    def _validate(self, args: dict[str, str]) -> dict[str, str]:
        out = dict(args)
        for p in self.params:
            if p.name not in out:
                if p.required:
                    raise ValueError(f"{self.name} requires '{p.name}'")
                if p.default is not None:
                    out[p.name] = str(p.default)
        unknown = set(out) - {p.name for p in self.params}
        for key in unknown:
            out.pop(key)
        return out


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def inventory(self) -> str:
        return "\n".join(t.signature() for t in self.tools.values())

    def names(self) -> list[str]:
        return list(self.tools)

    def dispatch(self, call: ToolCall) -> ToolResult:
        import time

        tool = self.tools.get(call.name)
        if tool is None:
            return ToolResult(
                name=call.name,
                ok=False,
                content="",
                error=f"unknown tool {call.name!r}; available: {', '.join(self.names())}",
            )
        t0 = time.perf_counter()
        try:
            result = tool.run(tool._validate(call.args))
        except Exception as exc:  # noqa: BLE001 - a tool error must not kill the graph
            return ToolResult(
                name=call.name,
                ok=False,
                content="",
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        return result


def parse_tool_calls(text: str, max_calls: int = 4) -> list[ToolCall]:
    """Extract `TOOL: name | k=v, k=v` lines. Anything else is ignored."""
    calls: list[ToolCall] = []
    for line in text.splitlines():
        match = TOOL_LINE_RE.match(line.strip())
        if not match:
            continue
        args: dict[str, str] = {}
        raw_args = (match.group("args") or "").strip()
        if raw_args:
            # Split on commas that separate `key=` pairs, so a value may itself
            # contain commas: "query=BLEU, ROUGE and METEOR" stays one value.
            parts = re.split(r",\s*(?=[a-zA-Z_][a-zA-Z0-9_]*\s*=)", raw_args)
            for part in parts:
                if "=" not in part:
                    # Bare argument: treat it as the first declared parameter.
                    args.setdefault("query", part.strip())
                    continue
                key, _, value = part.partition("=")
                args[key.strip().lower()] = value.strip().strip("\"'")
        calls.append(ToolCall(name=match.group("name").lower(), args=args, raw=line.strip()))
        if len(calls) >= max_calls:
            break
    return calls
