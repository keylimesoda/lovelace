# Prefill ubatch profile (1024 tokens, q5_0 KV, FA, per ubatch GPU time in ms)

| kernel class | depth 0 | depth 131072 |
|---|---:|---:|
| FA kernel (incl. fixup/combine) | 3.1 (1%) | 378.3 (53%) |
| MMQ iq3_s | 74.2 (23%) | 77.9 (11%) |
| DeltaNet (conv + recurrence) | 51.4 (16%) | 55.2 (8%) |
| MMQ iq3_xxs | 47.6 (15%) | 49.9 (7%) |
| MMQ iq4_xs | 42.1 (13%) | 45.3 (6%) |
| misc (elementwise, rope, copies, ...) | 28.8 (9%) | 24.5 (3%) |
| MMQ q4_K | 27.4 (9%) | 26.6 (4%) |
| norms | 14.2 (4%) | 14.1 (2%) |
| f16 conversion (FA K/V) | 0.1 (0%) | 13.9 (2%) |
| MMQ (activation quantize, fixup) | 12.8 (4%) | 12.9 (2%) |
| MMQ q3_K | 8.9 (3%) | 8.7 (1%) |
| MMQ q5_K | 3.1 (1%) | 3.6 (1%) |
| MMQ iq2_xxs | 2.2 (1%) | 2.3 (0%) |
| other matmul | 1.1 (0%) | 1.1 (0%) |
| MMQ iq2_xs | 1.0 (0%) | 1.0 (0%) |
| MMQ q6_K | 0.7 (0%) | 0.7 (0%) |
| MMQ q8_0 | 0.7 (0%) | 0.7 (0%) |
| MMQ q2_K | 0.0 (0%) | 0.0 (0%) |
| **sum of kernels** | **319.4** | **716.8** |
| ubatch wall (first to last kernel) | 331.0 | 718.6 |
| tok/s (1024 / wall) | 3094 | 1425 |

Baseline engine (main at e6171e25b, f16 MMA FlashAttention after converting q5_0 K/V to f16), llama-bench
`-p 1024 -n 0 -d D -ub 1024 -ctk q5_0 -ctv q5_0 -fa 1`, last ubatch of the run, `bench/prefill_profile.sh`.

At 128k depth FlashAttention is 53% of the ubatch (378 ms for 16 layers, 3.3e12 FLOP: ~140 TFLOPS, close to the
165 TFLOPS f16/f32-accumulate peak of the 4090); the f16 conversion of K/V is only 14 ms (2%). The weight matmuls
(MMQ, ~226 ms) and DeltaNet (~55 ms) do not grow with depth. Prefill at depth is bound by FA tensor throughput, so the
remaining lever is lower-precision tensor math for attention (int8 KQ and PV), not the conversion.
