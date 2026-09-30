# Observability conformance: lovelace vs tensorninja/ninfer-4090

ninfer: `bench/obs-fixtures/ninfer`  
lovelace: `bench/obs-fixtures/lovelace`  
contract: `docs/observability-parity.md`  

| payload | ninfer fields | matched | n/a | missing | type mismatch | unchecked | extension | other extra | check |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| health.json | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| models.json | 7 | 7 | 0 | 0 | 0 | 0 | 0 | 26 | 0 |
| model.json | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 10 | 0 |
| telemetry-idle.json | 122 | 95 | 27 | 0 | 0 | 0 | 154 | 0 | 0 |
| telemetry-busy.json | 122 | 95 | 27 | 0 | 0 | 0 | 175 | 0 | 0 |
| telemetry-end.json | 122 | 95 | 27 | 0 | 0 | 0 | 175 | 0 | 0 |
| slots-busy.json | 9 | 9 | 0 | 0 | 0 | 0 | 0 | 58 | 0 |
| slots-end.json | 9 | 9 | 0 | 0 | 0 | 0 | 0 | 58 | 0 |
| record request_done | 71 | 49 | 22 | 0 | 0 | 0 | 29 | 0 | 0 |
| record request_start | 28 | 22 | 6 | 0 | 0 | 0 | 13 | 0 | 0 |
| record server_start | 120 | 93 | 27 | 0 | 0 | 0 | 45 | 0 | 0 |
| record throughput | 157 | 92 | 65 | 0 | 0 | 0 | 48 | 0 | 0 |
| record lovelace_sched (lovelace only) | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 0 | 0 |
| metrics-busy.txt | 113 | 113 | 0 | 0 | 0 | 0 | 34 | 9 | 0 |
| metrics-end.txt | 113 | 113 | 0 | 0 | 0 | 0 | 34 | 9 | 0 |
| **total** | **1000** | **799** | **201** | **0** | **0** | **0** | **722** | **170** | **0** |

`n/a`: contract N/A and the lovelace value is the documented placeholder. `unchecked`: element schema of an array that is empty in lovelace's capture (allowed list). `extension`: lovelace-only leaves under `lovelace`, hybrid `cache` keys, lovelace-only records and `lovelace:*` series. `other extra`: other lovelace-only leaves (llama.cpp's own fields). `check`: failed lovelace-side checks of the payload (`lovelace.schema_version`, hybrid widget keys).

## Checks

- PASS: ninfer events.sse framing
- PASS: ninfer request-log.jsonl == /events data
- PASS: lovelace events.sse framing
- PASS: lovelace request-log.jsonl == /events data
- PASS: ninfer record envelope
- PASS: lovelace record envelope + lovelace.schema_version
- PASS: lovelace JSONL lovelace.seq monotonic per instance

## Failures (0)

none
