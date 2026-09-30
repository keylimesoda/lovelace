# Observability conformance: lovelace vs tensorninja/ninfer-4090

ninfer: `/home/ric/Work/lovelace-wt/obs/bench/obs-fixtures/ninfer`  
lovelace: `/home/ric/Work/lovelace-wt/obs/bench/obs-fixtures/lovelace`  
contract: `docs/observability-parity.md`  

| payload | ninfer fields | matched | n/a | missing | type mismatch | unchecked | extension | other extra | check |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| health.json | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| models.json | 7 | 7 | 0 | 0 | 0 | 0 | 0 | 26 | 0 |
| model.json | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 10 | 0 |
| telemetry-idle.json | 122 | 95 | 27 | 0 | 0 | 0 | 154 | 0 | 0 |
| record server_start | 120 | 93 | 27 | 0 | 0 | 0 | 45 | 0 | 0 |
| record lovelace_sched (lovelace only) | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 |
| record request_done (lovelace only) | 0 | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 0 |
| record request_start (lovelace only) | 0 | 0 | 0 | 0 | 0 | 0 | 41 | 0 | 0 |
| record throughput (lovelace only) | 0 | 0 | 0 | 0 | 0 | 0 | 205 | 0 | 0 |
| **total** | **256** | **202** | **54** | **0** | **0** | **0** | **560** | **36** | **0** |

`n/a`: contract N/A and the lovelace value is the documented placeholder. `unchecked`: element schema of an array that is empty in lovelace's capture (allowed list). `extension`: lovelace-only leaves under `lovelace`, hybrid `cache` keys, lovelace-only records and `lovelace:*` series. `other extra`: other lovelace-only leaves (llama.cpp's own fields). `check`: failed lovelace-side checks of the payload (`lovelace.schema_version`, hybrid widget keys).

## Checks

- PASS: ninfer request-log.jsonl == /events data
- PASS: lovelace events.sse framing
- PASS: lovelace request-log.jsonl == /events data
- PASS: ninfer record envelope
- PASS: lovelace record envelope + lovelace.schema_version
- PASS: lovelace JSONL lovelace.seq monotonic per instance

## Failures (0)

none
