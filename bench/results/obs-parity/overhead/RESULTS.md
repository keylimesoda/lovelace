# Observability overhead: telemetry OFF vs ON (same binary, flags only)

ON = `--telemetry --request-log-jsonl --metrics --slots` with /telemetry polled at 1 Hz, /metrics every 4 s and one /events subscription held for the whole run (bench/obs_overhead.py load).

| run | mode | single decode tok/s | six agg tok/s | six seq tok/s | six step ms | single warm reps | client load |
|---|---|---:|---:|---:|---:|---|---|
| 01-off | off | 101.86 | 325.60 | 66.69 | 50.86 | [100.2, 101.86, 103.42] | - |
| 02-on | on | 105.12 | 290.50 | 61.60 | 51.96 | [104.85, 105.25, 105.12] | telemetry 202 ok / 0 err, metrics 51 / 0, events 88 frames |
| 03-off | off | 99.97 | 293.90 | 63.09 | 50.74 | [99.91, 101.2, 99.97] | - |
| 04-on | on | 102.37 | 304.40 | 69.12 | 50.02 | [102.18, 102.51, 102.37] | telemetry 201 ok / 0 err, metrics 51 / 0, events 89 frames |

| figure | OFF median | ON median | ON vs OFF | OFF spread | ON spread |
|---|---:|---:|---:|---:|---:|
| single decode tok/s | 100.91 | 103.75 | +2.81% | 1.87% | 2.65% |
| six agg tok/s | 309.75 | 297.45 | -3.97% | 10.23% | 4.67% |
| six seq tok/s | 64.89 | 65.36 | +0.72% | 5.54% | 11.50% |
| six step ms (lower is better) | 50.80 | 50.99 | +0.39% | 0.24% | 3.81% |

spread = (max - min) / median over the runs of one mode; an ON vs OFF difference inside the spread is not a measurable cost.
