# bench/replay — multi-agent trace replay

Replays real multi-agent coding traffic (recorded OMP sessions that ran against the local
Qwen3.8-27B server) against any OpenAI-compatible `/v1/chat/completions` server, so we can
compare engines under the same load: 4–6 concurrent agents, 30k–250k-token prompts,
long shared prefixes that grow turn by turn.

| file | purpose |
|---|---|
| `extract.py` | OMP session logs → `corpus/` (one replayable turn per line) |
| `replay.py` | async N-agent load driver, writes `results/<label>/` |
| `capture_proxy.py` | pass-through proxy that logs live traffic for future corpora |
| `capture-proxy.service` | systemd `--user` unit template for the proxy |
| `corpusfmt.py` | corpus reader/writer helpers shared by the scripts |
| `tests/mock_server.py` | llama.cpp-style mock server (SSE + `timings`) for GPU-free testing |
| `corpus/system_prompt.txt`, `corpus/tools.json` | system prompt and 12 (+`yield`) tool schemas sent with every request |
| `corpus/chat_template.jinja` | the model's own chat template, taken from the ninfer artifact (see below) |

## Quick start

```sh
cd bench/replay
uv run extract.py                      # builds corpus/sessions/*.jsonl + corpus/manifest.json
uv run replay.py --base-url http://127.0.0.1:8080/v1 --model qwen3.8-27b \
    --agents 4 --duration 1800 --start-turn-min-depth 30000 --label ninfer-a4
```

Try it without a GPU:

```sh
uv run tests/mock_server.py --port 18080 &
uv run replay.py --base-url http://127.0.0.1:18080/v1 --model mock --agents 4 --max-turns 40 --label mock
```

## Extraction (`extract.py`)

OMP session files (`~/.omp/agent/sessions/<cwd>/<ts>_<id>.jsonl`, subagents in
`<ts>_<id>/<agent>.jsonl`) are trees of entries linked by `id`/`parentId`. For every
assistant message produced by `--provider` (default `ninfer-local`) the extractor rebuilds
the exact `messages` array the model saw, following OMP's own context builder:

- walks the `parentId` chain root → parent, so forked branches are handled;
- applies the latest `compaction` on that path: the summary (wrapped in OMP's
  handoff/compaction template) as a user message, then entries from `firstKeptEntryId`
  up to the compaction, then everything after it;
- `custom_message` entries (async job results, IRC messages, reminders) become user
  messages (OMP sends them as `developer`, downgraded to `user` for local servers);
  `developer` messages likewise;
- assistant messages with `stopReason` error/aborted or `retryRecovery` are dropped from
  history together with their tool results, and are not replayed as turns (the retry
  that followed is);
- assistant thinking is kept as `reasoning_content`, tool calls as OpenAI `tool_calls`,
  tool results as `role: tool` messages;
- tool results OMP pruned *after* a request are restored for that request: the saved
  display content for superseded reads, otherwise deterministic filler of the recorded
  size (`[shaken ~N tokens …]` → N one-token words). The original text is not in the log.

Each turn line also carries: `gap_s` (recorded end of the previous turn → start of this
request, i.e. tool execution/think time; `after_user` marks gaps that include a human),
`rec_prompt_tokens`/`rec_output_tokens` (what the server reported at the time),
`est_prompt_tokens`, `output_chars`, `reasoning_chars`, `rec_ttft_ms`, `stop_reason`.
Lines are delta-encoded (`keep` N messages of the previous line + `append`) to keep the
corpus small; `corpusfmt.iter_turns()` materializes full arrays. Sessions with fewer than
`--min-turns` (5) turns are dropped. `manifest.json` has per-session turn counts and depth
distributions and the corpus-wide comparison with the ninfer request log (`--reqlog`).

### Corpus caveats

- **System prompt and tools are reconstructed for top-level sessions.** OMP does not log
  them. `system_prompt.txt` is the real OMP prompt taken from a subagent's
  `session_init` minus the worker/task sections; `tools.json` holds 12 hand-written
  schemas for the tools the agents actually called (`read bash edit write grep glob eval
  todo web_search task wait ask`, matching the production log's `tool_count=12`), plus
  `yield` for subagents. Subagent sessions use their real logged system prompt and tool
  list. Measured on the production server, the rebuilt preamble is ~2.2k tokens shorter
  than the real one (5515 vs 7756 prompt tokens on the same first turn).
- **Token counts marked `est_` are estimates**: chars / 3.2, calibrated against the
  recorded server token counts (est/recorded p10/p50/p90 = 0.92/1.00/1.10). Use
  `rec_*` or the server's `usage` for real numbers.
- The model's chat template renders the reasoning of **every** past assistant message
  (`preserve_thinking` defaults to on), and the recorded prompt sizes confirm the server
  saw them. Engines must use this template (`corpus/chat_template.jinja`, e.g.
  `llama-server --jinja --chat-template-file corpus/chat_template.jinja`). A stock Qwen3
  template drops reasoning from turns before the last user message, which makes prompts
  up to ~30% shorter (reasoning is ~28% of history chars) and is not a fair comparison. Check `prompt depth` in the summaries matches across engines.
- Images (one in the corpus) are replaced by `[image omitted]`.
- History always comes from the recording, never from the server's output, so all
  engines see byte-identical prefixes and the generated text does not change later turns.

## Replay (`replay.py`)

Each of `--agents N` agents takes a session nobody else is running, walks its turns in
order, and moves to an unused session when done. Before each turn after the first it
sleeps `min(gap_s * --think-scale, --max-think)` (defaults 0.25, 20 s). Recorded gaps
are short (p50 0.02 s, p90 2.2 s) because tools are fast, so agents are mostly waiting
on the GPU, as in production.

| option | meaning |
|---|---|
| `--duration S` / `--max-turns N` | stop issuing new turns after S seconds / N total requests (default `--duration 600`). In-flight requests are allowed to finish unless `--no-drain` |
| `--output-tokens recorded\|fixed:N` | `max_tokens` = recorded completion tokens clamped to [64, 4096], or fixed |
| `--start-turn-min-depth T` | skip each session's leading turns below T prompt tokens (recorded, else estimated); sessions never reaching T are not used |
| `--temperature` (0.6), `--reasoning-effort`, `--extra-body JSON` | request passthroughs; `--extra-body` is merged last |
| `--stagger S` | agent i starts after i·S seconds (default 2) |
| `--request-timeout` | per-request deadline including queueing (default 1800 s) |
| `--seed`, `--sessions a,b` | session order / restriction |

Requests carry `stream: true` and `stream_options.include_usage`. Non-200 responses
(e.g. context too long) and in-stream errors are recorded as errors and the agent moves on.

Output: `results/<label>/requests.jsonl` (one line per request, including raw `usage`
and `timings`) and `summary.json`; a table is printed at the end.

### Metrics

| metric | definition |
|---|---|
| TTFT | request sent → first streamed delta of any kind (reasoning, content or tool call). Includes server queueing and prefill |
| decode tok/s | `(completion_tokens − 1) / (last delta − first delta)` per request; `completion_tokens` from `usage`, else `timings.predicted_n`, else the number of deltas |
| mean decode tok/s per agent | mean over agents of each agent's mean per-request decode tok/s |
| aggregate output tok/s | Σ completion tokens of ok requests / (last response end − first request start) |
| prompt depth | `usage.prompt_tokens` (else the estimate) of ok and errored requests |
| cache hit ratio | Σ cached / Σ prompt tokens over requests that report caching (`usage.prompt_tokens_details.cached_tokens`, else `timings.cache_n`) |
| prefill tok/s | `timings.prompt_n / timings.prompt_ms` (tokens actually prefilled, excluding cache hits), if reported |
| draft acceptance | Σ `timings.draft_n_accepted` / Σ `timings.draft_n`, if reported |
| e2e | request sent → stream closed |
| errors | HTTP errors, in-stream errors, streams with no tokens, and timeouts; `cancelled` (from `--no-drain`) is counted separately |

## Capture proxy (`capture_proxy.py`)

```sh
uv run capture_proxy.py --listen 127.0.0.1:8081 --upstream http://127.0.0.1:8080 --out captures/
```

An asyncio HTTP/1.1 reverse proxy (keep-alive, chunked bodies, `Expect: 100-continue`).
Response bytes are written to the client as each upstream chunk arrives, so SSE is not
buffered, and parsing happens after the write. Adds ~1.5 ms to first byte once warm
(measured against the mock). For `POST /v1/chat/completions`, `/v1/responses` and
`/v1/messages` it appends to `captures/capture-YYYYMMDD.jsonl`: the parsed request
body, status, `first_byte_s`, `ttft_s`, `decode_s`, `total_s`, finish reason, and the
server's `usage`/`timings` (chat SSE, Responses events, Anthropic events, or the
non-streaming JSON). Bodies are full prompts (~0.5 MB each at 130k tokens), so watch
disk use. See `capture-proxy.service` to run it permanently; then point the agent's
provider base URL at the proxy port.
