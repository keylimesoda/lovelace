"""Transparent capture proxy for OpenAI/Anthropic-compatible LLM servers.

Forwards every request to --upstream unchanged and streams responses back chunk by
chunk as they arrive (SSE is never buffered). For POST /v1/chat/completions,
/v1/responses and /v1/messages it appends one JSON line per request to
DIR/capture-YYYYMMDD.jsonl: the parsed request body plus status, timing (first
byte, first token, total) and the usage/timings block reported by the server.
Capture bookkeeping happens after each chunk has been written to the client.

  uv run capture_proxy.py --listen 127.0.0.1:8081 --upstream http://127.0.0.1:8080 --out captures/
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

from corpusfmt import dumps, loads

CAPTURE_PATHS = {"/v1/chat/completions", "/v1/responses", "/v1/messages"}
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection",
    "te", "trailer", "transfer-encoding", "upgrade", "expect", "host", "content-length",
}
MAX_HEAD = 1 << 20
MAX_NONSTREAM_CAPTURE = 64 << 20


class HttpError(Exception):
    pass


async def read_request(r: asyncio.StreamReader):
    try:
        head = await r.readuntil(b"\r\n\r\n")
    except asyncio.IncompleteReadError as e:
        if e.partial.strip():
            raise HttpError("truncated request head") from e
        return None
    except asyncio.LimitOverrunError as e:
        raise HttpError("request head too large") from e
    lines = head[:-4].decode("latin-1").split("\r\n")
    try:
        method, target, version = lines[0].split(" ", 2)
    except ValueError as e:
        raise HttpError(f"bad request line {lines[0]!r}") from e
    headers: list[tuple[str, str]] = []
    for line in lines[1:]:
        k, sep, v = line.partition(":")
        if not sep:
            raise HttpError(f"bad header line {line!r}")
        headers.append((k.strip(), v.strip()))
    return method, target, version, headers


async def read_body(r: asyncio.StreamReader, w: asyncio.StreamWriter, headers: dict[str, str]) -> bytes:
    if headers.get("expect", "").lower() == "100-continue":
        w.write(b"HTTP/1.1 100 Continue\r\n\r\n")
        await w.drain()
    if "chunked" in headers.get("transfer-encoding", "").lower():
        parts = []
        while True:
            size_line = await r.readuntil(b"\r\n")
            size = int(size_line.split(b";", 1)[0].strip(), 16)
            if size == 0:
                while (await r.readuntil(b"\r\n")) != b"\r\n":  # trailers
                    pass
                return b"".join(parts)
            parts.append(await r.readexactly(size))
            await r.readexactly(2)
    n = int(headers.get("content-length") or 0)
    return await r.readexactly(n) if n else b""


class Observer:
    """Incrementally inspects a captured response for first-token time and usage."""

    def __init__(self, path: str, t0: float):
        self.path = path
        self.t0 = t0
        self.first_byte: float | None = None
        self.first_token: float | None = None
        self.last_token: float | None = None
        self.stream = False
        self.buf = b""
        self.body = bytearray()
        self.usage = None
        self.timings = None
        self.finish = None
        self.events = 0
        self.stream_error = None

    def feed(self, chunk: bytes, now: float) -> None:
        if self.first_byte is None:
            self.first_byte = now
        if not self.stream:
            if len(self.body) < MAX_NONSTREAM_CAPTURE:
                self.body += chunk
            return
        self.buf += chunk
        *lines, self.buf = self.buf.split(b"\n")
        for line in lines:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if not data or data == b"[DONE]":
                continue
            try:
                ev = loads(data)
            except ValueError:
                continue
            if isinstance(ev, dict):
                self.event(ev, now)

    def token(self, now: float) -> None:
        self.first_token = self.first_token or now
        self.last_token = now
        self.events += 1

    def event(self, ev: dict, now: float) -> None:
        if ev.get("error"):
            self.stream_error = ev["error"]
        if self.path == "/v1/chat/completions":
            if ev.get("usage"):
                self.usage = ev["usage"]
            if ev.get("timings"):
                self.timings = ev["timings"]
            for ch in ev.get("choices") or ():
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning") or d.get("tool_calls"):
                    self.token(now)
                if ch.get("finish_reason"):
                    self.finish = ch["finish_reason"]
        elif self.path == "/v1/responses":
            t = ev.get("type", "")
            if t.endswith(".delta"):
                self.token(now)
            elif t in ("response.completed", "response.incomplete", "response.failed"):
                resp = ev.get("response") or {}
                self.usage = resp.get("usage")
                self.finish = resp.get("status")
        elif self.path == "/v1/messages":
            t = ev.get("type")
            if t == "content_block_delta":
                self.token(now)
            elif t == "message_start":
                self.usage = dict((ev.get("message") or {}).get("usage") or {})
            elif t == "message_delta":
                self.usage = {**(self.usage or {}), **(ev.get("usage") or {})}
                self.finish = (ev.get("delta") or {}).get("stop_reason")

    def finish_nonstream(self) -> None:
        if self.stream or not self.body:
            return
        try:
            obj = loads(bytes(self.body))
        except ValueError:
            return
        if isinstance(obj, dict):
            self.usage = obj.get("usage")
            self.timings = obj.get("timings")
            ch = (obj.get("choices") or [{}])[0]
            self.finish = ch.get("finish_reason") or obj.get("stop_reason") or obj.get("status")


class Proxy:
    def __init__(self, upstream: str, out_dir: Path):
        self.upstream = upstream.rstrip("/")
        self.out_dir = out_dir
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None),
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=64),
            follow_redirects=False,
            headers={"accept-encoding": "identity"},  # don't add compression the client didn't ask for
        )
        self.write_lock = asyncio.Lock()

    async def handle(self, r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        try:
            while True:
                req = await read_request(r)
                if req is None:
                    return
                if not await self.exchange(r, w, *req):
                    return
        except HttpError as e:
            await self.simple(w, 400, str(e))
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            w.close()

    async def simple(self, w: asyncio.StreamWriter, status: int, msg: str) -> None:
        body = dumps({"error": {"message": msg, "type": "capture_proxy"}})
        try:
            w.write(
                f"HTTP/1.1 {status} {'Bad Gateway' if status == 502 else 'Bad Request'}\r\n"
                f"content-type: application/json\r\ncontent-length: {len(body)}\r\n\r\n".encode() + body
            )
            await w.drain()
        except ConnectionError:
            pass

    async def exchange(self, r, w, method: str, target: str, version: str, headers: list[tuple[str, str]]) -> bool:
        hmap = {k.lower(): v for k, v in headers}
        body = await read_body(r, w, hmap)
        keep_alive = version == "HTTP/1.1" and hmap.get("connection", "").lower() != "close"
        path = target.split("?", 1)[0]
        capture = method == "POST" and path in CAPTURE_PATHS
        t0 = time.monotonic()
        wall0 = time.time()
        fwd = [(k, v) for k, v in headers if k.lower() not in HOP_BY_HOP]
        obs = Observer(path, t0) if capture else None
        status = None
        error = None
        try:
            req = self.client.build_request(method, self.upstream + target, headers=fwd, content=body)
            resp = await self.client.send(req, stream=True)
        except httpx.HTTPError as e:
            error = f"upstream: {type(e).__name__}: {e}"
            await self.simple(w, 502, error)
            keep_alive = False
            resp = None
        if resp is not None:
            status = resp.status_code
            try:
                bodiless = method == "HEAD" or resp.status_code in (204, 304)
                chunked = not bodiless and "content-length" not in resp.headers
                out = [f"HTTP/1.1 {resp.status_code} {resp.reason_phrase}\r\n"]
                for k, v in resp.headers.multi_items():
                    if k.lower() not in HOP_BY_HOP:
                        out.append(f"{k}: {v}\r\n")
                if chunked:
                    out.append("transfer-encoding: chunked\r\n")
                elif "content-length" in resp.headers:
                    out.append(f"content-length: {resp.headers['content-length']}\r\n")
                if not keep_alive:
                    out.append("connection: close\r\n")
                out.append("\r\n")
                w.write("".join(out).encode("latin-1"))
                await w.drain()
                if obs:
                    obs.stream = "text/event-stream" in resp.headers.get("content-type", "")
                    if resp.headers.get("content-encoding"):
                        obs = None  # compressed; don't try to parse
                async for chunk in resp.aiter_raw():
                    if bodiless:
                        break
                    if not chunk:
                        continue
                    w.write(b"%x\r\n%s\r\n" % (len(chunk), chunk) if chunked else chunk)
                    await w.drain()
                    if obs:
                        obs.feed(chunk, time.monotonic())
                if chunked:
                    w.write(b"0\r\n\r\n")
                    await w.drain()
            except (ConnectionError, httpx.HTTPError) as e:
                error = f"{type(e).__name__}: {e}"
                keep_alive = False
            finally:
                await resp.aclose()
        if capture:
            await self.record(wall0, t0, method, path, body, status, obs, error)
        return keep_alive

    async def record(self, wall0, t0, method, path, body: bytes, status, obs: Observer | None, error) -> None:
        t_end = time.monotonic()
        try:
            req_obj = loads(body) if body else None
        except ValueError:
            req_obj = body.decode("utf-8", "replace")
        rec = {
            "ts": datetime.fromtimestamp(wall0).astimezone().isoformat(timespec="milliseconds"),
            "method": method,
            "path": path,
            "status": status,
            "error": error,
            "request_bytes": len(body),
            "total_s": round(t_end - t0, 4),
        }
        if obs:
            obs.finish_nonstream()
            rec.update(
                stream=obs.stream,
                first_byte_s=round(obs.first_byte - t0, 4) if obs.first_byte else None,
                ttft_s=round(obs.first_token - t0, 4) if obs.first_token else None,
                decode_s=round(obs.last_token - obs.first_token, 4) if obs.first_token else None,
                stream_events=obs.events,
                finish_reason=obs.finish,
                usage=obs.usage,
                timings=obs.timings,
                stream_error=obs.stream_error,
            )
            if not obs.stream and status and status >= 400 and obs.body:
                rec["response_error"] = bytes(obs.body[:4000]).decode("utf-8", "replace")
        rec["request"] = req_obj
        line = dumps(rec) + b"\n"
        target = self.out_dir / f"capture-{datetime.fromtimestamp(wall0).strftime('%Y%m%d')}.jsonl"
        async with self.write_lock:
            await asyncio.to_thread(append, target, line)


def append(path: Path, data: bytes) -> None:
    with open(path, "ab") as f:
        f.write(data)


def host_port(s: str) -> tuple[str, int]:
    host, _, port = s.rpartition(":")
    return host or "127.0.0.1", int(port)


async def amain(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    proxy = Proxy(args.upstream, args.out)
    host, port = host_port(args.listen)
    server = await asyncio.start_server(proxy.handle, host, port, limit=MAX_HEAD)
    print(f"capture proxy {host}:{port} -> {args.upstream}, capturing to {args.out}", flush=True)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    async with server:
        await stop.wait()
    await proxy.client.aclose()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", required=True, help="HOST:PORT to listen on")
    ap.add_argument("--upstream", required=True, help="upstream base URL, e.g. http://127.0.0.1:8080")
    ap.add_argument("--out", type=Path, required=True, help="capture directory")
    args = ap.parse_args()
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
