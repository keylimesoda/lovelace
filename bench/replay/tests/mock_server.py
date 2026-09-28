"""Tiny llama.cpp-flavoured mock of /v1/chat/completions for exercising replay.py and
capture_proxy.py without a GPU.

- streams SSE chunks: reasoning_content deltas, then content deltas, then a final chunk
  with finish_reason, `usage` (incl. prompt_tokens_details.cached_tokens) and llama.cpp
  style `timings` (cache_n, prompt_n, prompt_ms, predicted_n, predicted_ms, draft_n,
  draft_n_accepted);
- simulates a per-slot prefix cache (chars/3.2 "tokens"), prefill and decode speeds,
  and queueing when all --slots are busy;
- rejects prompts over --max-context tokens with HTTP 400 like llama-server.

  python tests/mock_server.py --port 18080 [--slots 2 --prefill-tps 50000 --decode-tps 300]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

CHARS_PER_TOKEN = 3.2


class Mock:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.slots = asyncio.Semaphore(args.slots)
        self.cache: list[str] = []  # last prompt text per slot (LRU, len <= slots)
        self.requests = 0

    def cached_prefix(self, text: str) -> int:
        best = 0
        for c in self.cache:
            n = min(len(c), len(text))
            i = 0
            # compare in blocks for speed
            while i < n and c[i : i + 4096] == text[i : i + 4096]:
                i += 4096
            while i < n and c[i] == text[i]:
                i += 1
            best = max(best, min(i, n))
        return best

    def remember(self, text: str) -> None:
        self.cache.append(text)
        del self.cache[: -self.args.slots]


def render(body: dict) -> str:
    return json.dumps(body.get("tools") or []) + "".join(
        json.dumps(m, ensure_ascii=False, sort_keys=True) for m in body.get("messages") or []
    )


def sse(obj: dict) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


async def write_chunk(w: asyncio.StreamWriter, data: bytes) -> None:
    w.write(b"%x\r\n%s\r\n" % (len(data), data))
    await w.drain()


async def respond(w: asyncio.StreamWriter, status: int, obj: dict) -> None:
    data = json.dumps(obj).encode()
    reason = {200: "OK", 400: "Bad Request", 404: "Not Found"}.get(status, "Error")
    w.write(
        f"HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {len(data)}\r\n\r\n".encode()
        + data
    )
    await w.drain()


async def completion(mock: Mock, w: asyncio.StreamWriter, body: dict) -> None:
    a = mock.args
    text = render(body)
    n_prompt = max(1, round(len(text) / CHARS_PER_TOKEN))
    if n_prompt > a.max_context:
        await respond(
            w,
            400,
            {
                "error": {
                    "code": 400,
                    "message": f"the request exceeds the available context size ({n_prompt} > {a.max_context})",
                    "type": "exceed_context_size_error",
                }
            },
        )
        return
    n_out = max(1, min(int(body.get("max_tokens") or 256), a.max_output))
    stream = bool(body.get("stream"))
    mock.requests += 1
    rid = f"chatcmpl-mock{mock.requests}"
    async with mock.slots:
        cache_n = min(n_prompt - 1, round(mock.cached_prefix(text) / CHARS_PER_TOKEN))
        prompt_n = n_prompt - cache_n
        t0 = time.monotonic()
        await asyncio.sleep(prompt_n / a.prefill_tps)
        prompt_ms = (time.monotonic() - t0) * 1000
        mock.remember(text)
        if stream:
            w.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n")
        t1 = time.monotonic()
        n_reason = n_out // 2
        pieces = []
        for i in range(n_out):
            field = "reasoning_content" if i < n_reason else "content"
            tok = f"tok{i} "
            pieces.append((field, tok))
            if stream:
                await write_chunk(
                    w,
                    sse({"id": rid, "object": "chat.completion.chunk", "model": body.get("model"),
                         "choices": [{"index": 0, "delta": {field: tok}, "finish_reason": None}]}),
                )
            await asyncio.sleep(1 / a.decode_tps)
        predicted_ms = (time.monotonic() - t1) * 1000
    usage = {"prompt_tokens": n_prompt, "completion_tokens": n_out, "total_tokens": n_prompt + n_out,
             "prompt_tokens_details": {"cached_tokens": cache_n}}
    timings = {"cache_n": cache_n, "prompt_n": prompt_n, "prompt_ms": prompt_ms,
               "prompt_per_second": prompt_n / max(prompt_ms / 1000, 1e-9), "predicted_n": n_out,
               "predicted_ms": predicted_ms, "predicted_per_second": n_out / max(predicted_ms / 1000, 1e-9),
               "draft_n": n_out // 2, "draft_n_accepted": n_out // 3}
    if stream:
        await write_chunk(w, sse({"id": rid, "object": "chat.completion.chunk", "model": body.get("model"),
                                  "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                                  "usage": usage, "timings": timings}))
        await write_chunk(w, b"data: [DONE]\n\n")
        await write_chunk(w, b"")
        return
    content = "".join(t for f, t in pieces if f == "content")
    reasoning = "".join(t for f, t in pieces if f == "reasoning_content")
    await respond(w, 200, {"id": rid, "object": "chat.completion", "model": body.get("model"),
                           "choices": [{"index": 0, "finish_reason": "length",
                                        "message": {"role": "assistant", "content": content,
                                                    "reasoning_content": reasoning}}],
                           "usage": usage, "timings": timings})


async def handle(mock: Mock, r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
    try:
        while True:
            try:
                head = await r.readuntil(b"\r\n\r\n")
            except (asyncio.IncompleteReadError, ConnectionError):
                return
            lines = head.decode("latin-1").split("\r\n")
            method, path, _ = lines[0].split(" ", 2)
            headers = {k.strip().lower(): v.strip() for k, v in (l.split(":", 1) for l in lines[1:] if ":" in l)}
            if headers.get("expect", "").lower() == "100-continue":
                w.write(b"HTTP/1.1 100 Continue\r\n\r\n")
            body = await r.readexactly(int(headers.get("content-length", "0")))
            path = path.split("?", 1)[0]
            if method == "POST" and path == "/v1/chat/completions":
                await completion(mock, w, json.loads(body))
            elif method == "GET" and path == "/v1/models":
                await respond(w, 200, {"object": "list", "data": [{"id": mock.args.model, "object": "model"}]})
            else:
                await respond(w, 404, {"error": {"message": f"no route {method} {path}"}})
            if headers.get("connection", "").lower() == "close":
                return
    except ConnectionError:
        pass
    finally:
        w.close()


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--model", default="mock")
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--prefill-tps", type=float, default=50000.0)
    ap.add_argument("--decode-tps", type=float, default=300.0)
    ap.add_argument("--max-context", type=int, default=262144)
    ap.add_argument("--max-output", type=int, default=4096)
    args = ap.parse_args()
    mock = Mock(args)
    server = await asyncio.start_server(lambda r, w: handle(mock, r, w), args.host, args.port, limit=1 << 20)
    print(f"mock server listening on http://{args.host}:{args.port}/v1", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
