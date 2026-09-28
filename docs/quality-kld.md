# Quality: KL divergence on the user's agent transcripts

Measured with `eval/kld` (sparse top-64 KL against a reference; the corpus is 30 windows from OMP sessions at
32k–250k tokens of depth, scored on the recorded assistant replies). All runs use `-ub 2048 -fa on`, with every
layer and all KV on the GPU (FFN weights streamed from pinned RAM where needed). Raw tables are in
`eval/kld/results/summary.txt` (untracked; regenerate with `eval/kld/run_kld.sh`).

## Weight quantization (reference: Q8_0 weights, f16 KV; buckets 32k–170k, 12,124 scored positions)

| Weights | file size | KL mean | KL median | top-1 agree | ΔNLL |
|---|---:|---:|---:|---:|---:|
| **ShapeLearn IQ4_XS-3.84bpw ("GPU-5")** | 13.1 GB | **0.0163** | 0.0011 | **95.64%** | **+0.0096** |
| ISTA-DASLab GSQ-RCO-IQ3_S (+MTP) | 12.1 GB | 0.0216 | 0.0014 | 95.41% | +0.0120 |
| ShapeLearn IQ3_S-3.23bpw ("GPU-4") | 11.0 GB | 0.0377 | 0.0029 | 93.76% | +0.0321 |

GPU-5 is kept. ISTA frees ~1 GB of VRAM (~40k KV cells) for 1.33x the KL; GPU-4 frees ~2 GB for 2.3x the KL and
3.3x the ΔNLL, a much larger gap than ByteShape's aggregate benchmark suggests (98.72% vs 99.63% of BF16).
## KV-cache quantization (reference: GPU-5 weights, f16 KV; buckets 32k–250k, 15,359 scored positions)

| K / V | bytes/token | KL mean | top-1 agree | ΔNLL | KL @ 250k |
|---|---:|---:|---:|---:|---:|
| q8_0 / q8_0 | 34 KB | 0.00095 | 98.94% | −0.00004 | 0.00120 |
| q8_0 / q5_0 | 28 KB | 0.00117 | 98.89% | +0.00050 | 0.00145 |
| **q5_0 / q5_0** | **22 KB** | **0.00155** | **98.63%** | **+0.00101** | **0.00170** |
| q8_0 / q4_0 | 26 KB | 0.00175 | 98.57% | +0.00070 | 0.00197 |
| q4_0 / q4_0 | 18 KB | 0.00328 | 98.06% | +0.00194 | 0.00367 |

None of the types shows error growing with depth through 250k.

## Decision

- **KV: q5_0 / q5_0 by default.** Its error is about 10% of the weight-quant error; q8_0/q5_0 would cut total
  error by ~2% for 27% less resident capacity. q8_0/q5_0 is the "max quality" profile.
- q4_0 KV is rejected: 2x the KL of q5_0 for 18% less memory.
- Weights dominate the error budget. Candidate larger quants must beat GPU-5 on this KLD *and* leave room for
  262k context plus session parking.
