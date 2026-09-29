"""Per-kernel-class GPU time of the last prefill ubatch in an nsys cuda_gpu_trace export (see ../prefill_profile.sh).

The ubatch is the window that starts at the last token-embedding get_rows before the last 16 FlashAttention calls
(Qwen3.8-27B has 16 attention layers) and ends with the trace.
"""
import csv
import re
import sys
from pathlib import Path

N_ATTN = 16
GGML_TYPES = {0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 8: "q8_0", 10: "q2_K", 11: "q3_K", 12: "q4_K",
              13: "q5_K", 14: "q6_K", 16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs", 19: "iq1_s", 20: "iq4_nl", 21: "iq3_s",
              22: "iq2_s", 23: "iq4_xs"}


def classify(name):
    n = name
    if "flash_attn" in n:
        return "FA kernel (incl. fixup/combine)"
    if "dequantize_block" in n or "convert_unary" in n or n.startswith("void convert"):
        return "f16 conversion (FA K/V)"
    if "gated_delta" in n or "delta_net" in n or "ssm_conv" in n or "chunk" in n.lower():
        return "DeltaNet (conv + recurrence)"
    m = re.search(r"mul_mat_q<\(ggml_type\)(\d+)", n)
    if m:
        return f"MMQ {GGML_TYPES.get(int(m.group(1)), m.group(1))}"
    if "mul_mat_q" in n or "quantize_mmq" in n or "quantize_q8_1" in n:
        return "MMQ (activation quantize, fixup)"
    if "mul_mat" in n or "gemm" in n.lower() or "gemv" in n.lower():
        return "other matmul"
    if "norm" in n:
        return "norms"
    return "misc (elementwise, rope, copies, ...)"


def last_ubatch(rows):
    fa = [i for i, r in enumerate(rows) if "flash_attn_ext" in r["Name"] and "combine" not in r["Name"] and "fixup" not in r["Name"]
          and "mask_to" not in r["Name"] and "compact" not in r["Name"]]
    # one FA kernel launch per attention layer (the main kernel); take the 16th from the end
    first_fa = fa[-N_ATTN] if len(fa) >= N_ATTN else 0
    start = 0
    for i in range(first_fa, -1, -1):
        if "get_rows" in rows[i]["Name"]:
            start = i
            break
    return rows[start:]


def main():
    out = Path(sys.argv[1])
    depths = sys.argv[2:]
    tables = {}
    totals = {}
    for d in depths:
        with open(out / f"pp-d{d}.trace.csv") as f:
            rows = [r for r in csv.DictReader(f) if r.get("Name")]
        rows.sort(key=lambda r: int(r["Start (ns)"]))
        ub = last_ubatch(rows)
        acc = {}
        for r in ub:
            c = classify(r["Name"])
            acc[c] = acc.get(c, 0) + int(r["Duration (ns)"]) / 1e6
        tables[d] = acc
        wall = (int(ub[-1]["Start (ns)"]) + int(ub[-1]["Duration (ns)"]) - int(ub[0]["Start (ns)"])) / 1e6
        totals[d] = (sum(acc.values()), wall)
    classes = sorted({c for t in tables.values() for c in t}, key=lambda c: -max(t.get(c, 0) for t in tables.values()))
    print("# Prefill ubatch profile (1024 tokens, q5_0 KV, FA, per ubatch GPU time in ms)\n")
    print("| kernel class | " + " | ".join(f"depth {d}" for d in depths) + " |")
    print("|---|" + "---:|" * len(depths))
    for c in classes:
        print(f"| {c} | " + " | ".join(f"{tables[d].get(c, 0):.1f} ({100*tables[d].get(c, 0)/totals[d][0]:.0f}%)" for d in depths) + " |")
    print("| **sum of kernels** | " + " | ".join(f"**{totals[d][0]:.1f}**" for d in depths) + " |")
    print("| ubatch wall (first to last kernel) | " + " | ".join(f"{totals[d][1]:.1f}" for d in depths) + " |")
    print("| tok/s (1024 / wall) | " + " | ".join(f"{1024e3/totals[d][1]:.0f}" for d in depths) + " |")


if __name__ == "__main__":
    main()
