"""Capture every observability payload of a running server as fixtures.

Drives the same small, fixed workload against any server that speaks the tensorninja/ninfer-4090 API (ninfer-serve
or lovelace's llama-server) and saves what the dashboard and the omarchy-ninfer widget read:

  health.json, models.json, model.json          GET /health, /v1/models, /v1/models/{id}
  telemetry-idle.json, -busy.json, -end.json    GET /telemetry before, during and after the workload
  slots-busy.json, slots-end.json               GET /slots
  metrics-busy.txt, metrics-end.txt             GET /metrics
  events.sse                                    GET /events for the whole workload (raw SSE)
  <event>.json                                  the last record of each event type seen on /events

Workload: a routed chat turn, a streamed follow-up of the same conversation long enough to span a 5 s throughput
interval, the same conversation again (a warm reuse), an Anthropic Messages call and an OpenAI Responses call.

Usage: python3 bench/obs_capture.py --base-url http://127.0.0.1:18480 --model qwen3.8-27b --out DIR
"""

import argparse
import json
import threading
import time
import urllib.request
from pathlib import Path


def get(base, path, timeout=30):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return r.read().decode()


def post(base, path, body, timeout=600):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def save_json(out, name, text):
    (out / name).write_text(json.dumps(json.loads(text), indent=1, sort_keys=True) + "\n")


def capture(base, model, out):
    """Run the workload against `base` (no trailing slash) and write every fixture into `out`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    save_json(out, "health.json", get(base, "/health"))
    save_json(out, "models.json", get(base, "/v1/models"))
    save_json(out, "model.json", get(base, "/v1/models/" + model))
    save_json(out, "telemetry-idle.json", get(base, "/telemetry"))

    # raw SSE for the whole workload
    events = bytearray()
    stop = threading.Event()

    def reader():
        try:
            with urllib.request.urlopen(base + "/events", timeout=120) as r:
                while not stop.is_set():
                    chunk = r.read1(65536)
                    if not chunk:
                        break
                    events.extend(chunk)
        except OSError:
            pass

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    time.sleep(1.0)

    system = "You are a careful assistant. Answer concisely."
    turn1 = [{"role": "system", "content": system},
             {"role": "user", "content": "List five prime numbers and explain what a prime number is."}]
    common = {"model": model, "temperature": 0.7, "top_p": 0.95, "prompt_cache_key": "obs-fixture-session",
              "chat_template_kwargs": {"enable_thinking": False}}
    r1 = json.loads(post(base, "/v1/chat/completions", {**common, "messages": turn1, "max_tokens": 64}))
    turn2 = turn1 + [{"role": "assistant", "content": r1["choices"][0]["message"]["content"] or ""},
                     {"role": "user", "content": "Now write a long, detailed essay (at least 900 words) on the history of prime numbers."}]

    # streamed follow-up; sample the live endpoints while it decodes
    busy = {}

    def sampler():
        time.sleep(3.0)
        busy["telemetry"] = get(base, "/telemetry")
        busy["slots"] = get(base, "/slots")
        busy["metrics"] = get(base, "/metrics")

    s = threading.Thread(target=sampler)
    s.start()
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 json.dumps({**common, "messages": turn2, "max_tokens": 900, "stream": True}).encode(),
                                 {"content-type": "application/json"})
    text = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                d = json.loads(line[6:])
                for c in d.get("choices") or []:
                    text.append((c.get("delta") or {}).get("content") or "")
    s.join()
    save_json(out, "telemetry-busy.json", busy["telemetry"])
    save_json(out, "slots-busy.json", busy["slots"])
    (out / "metrics-busy.txt").write_text(busy["metrics"])

    turn3 = turn2 + [{"role": "assistant", "content": "".join(text)},
                     {"role": "user", "content": "Summarize that in one sentence."}]
    post(base, "/v1/chat/completions", {**common, "messages": turn3, "max_tokens": 48})
    post(base, "/v1/messages", {"model": model, "max_tokens": 32,
                                "messages": [{"role": "user", "content": "Say hello in French."}]})
    post(base, "/v1/responses", {"model": model, "max_output_tokens": 32, "input": "Say hello in German."})

    time.sleep(7.0)  # let the last throughput interval close
    save_json(out, "telemetry-end.json", get(base, "/telemetry"))
    save_json(out, "slots-end.json", get(base, "/slots"))
    (out / "metrics-end.txt").write_text(get(base, "/metrics"))
    stop.set()
    t.join(timeout=20)

    raw = bytes(events).decode(errors="replace")
    (out / "events.sse").write_text(raw)
    last = {}
    for frame in raw.split("\n\n"):
        data = [ln[6:] for ln in frame.split("\n") if ln.startswith("data: ")]
        if data:
            rec = json.loads("\n".join(data))
            last[rec.get("event", "message")] = rec
    for name, rec in last.items():
        (out / f"{name}.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print("captured:", ", ".join(sorted(p.name for p in out.iterdir())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    capture(args.base_url.rstrip("/"), args.model, args.out)


if __name__ == "__main__":
    main()
