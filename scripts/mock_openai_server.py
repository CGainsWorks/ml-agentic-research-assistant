#!/usr/bin/env python
"""A mock OpenAI-compatible chat completions server, for CI.

The end-to-end workflow runs the real compose stack with
`ARAA_LLM_BACKEND=openai_compat` pointed at this server, so the whole HTTP path
(request shape, server-sent events, usage accounting, the /models health
check) is exercised with no model, no GPU, no key and no paid API.

It answers each pipeline step in the format that step expects, recognised from
the prompt:

    tool selection   NONE
    critique         VERDICT: PASS
    planning         two search phrases, one per line
    anything else    two sentences citing the first passage markers ([C1]...)
                     found in the prompt, so citations always point at
                     passages that were actually supplied

The answers are canned. They prove the plumbing works, not that the system
answers well. Token counts are an estimate (one token per four characters).

Standard library only, so it runs in any Python 3.11+ image:

    python scripts/mock_openai_server.py --port 8080
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "mock-llm"
_MARKER = re.compile(r"\[C(\d+)\]")


def reply_for(messages: list[dict]) -> str:
    system = " ".join(m.get("content") or "" for m in messages if m.get("role") == "system")
    user = " ".join(m.get("content") or "" for m in messages if m.get("role") != "system")

    if "TOOL: tool_name" in system:
        return "NONE"
    if "VERDICT: PASS or REVISE" in system:
        return "VERDICT: PASS\nISSUES:"
    if "planning step of a research agent" in system:
        return "evaluation methods and metrics\nbenchmark datasets and experimental setup"

    seen: list[str] = []
    for n in _MARKER.findall(user):
        if n not in seen:
            seen.append(n)
    if not seen:
        return "The supplied passages do not contain enough information to answer this."
    first = f"[C{seen[0]}]"
    second = f"[C{seen[1]}]" if len(seen) > 1 else first
    return (
        f"The retrieved papers describe the methods they evaluate and the measurements "
        f"they report {first}. They compare those results against baseline approaches "
        f"{second}."
    )


def _tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


class Handler(BaseHTTPRequestHandler):
    server_version = "mock-openai/1"

    def log_message(self, fmt: str, *args) -> None:  # quieter than the default
        print(f"{self.command} {self.path} -> {args[1] if len(args) > 1 else ''}", flush=True)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        elif self.path.rstrip("/") in ("", "/health"):
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._json(404, {"error": {"message": f"no route {self.path}"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._json(400, {"error": {"message": "body is not JSON"}})
            return
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            self._json(400, {"error": {"message": "messages must be a non-empty list"}})
            return

        text = reply_for(messages)
        prompt_chars = sum(len(m.get("content") or "") for m in messages)
        usage = {"prompt_tokens": _tokens("x" * prompt_chars), "completion_tokens": _tokens(text)}
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        model = body.get("model") or MODEL
        created = int(time.time())

        if not body.get("stream"):
            self._json(
                200,
                {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                },
            )
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def event(choices: list, **extra) -> None:
            chunk = {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": choices,
                **extra,
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        for word in re.findall(r"\S+\s*", text):
            event([{"index": 0, "delta": {"content": word}, "finish_reason": None}])
        event([{"index": 0, "delta": {}, "finish_reason": "stop"}])
        if (body.get("stream_options") or {}).get("include_usage"):
            event([], usage=usage)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock OpenAI-compatible server on http://{args.host}:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
