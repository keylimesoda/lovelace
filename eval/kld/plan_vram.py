"""VRAM plan and runtime estimate for one lovelace-kld run with every layer and all KV on the GPU.

Everything goes on the GPU (-ngl 99). If the total doesn't fit, the FFN weights (ffn_up/gate/down)
of the layers with the largest FFN tensors move to host memory, just enough of them to fit. With
op-offload (llama.cpp's default), those weights are streamed over PCIe for every ubatch, so all
compute stays on the GPU.

stdout: the -ot value ("" when everything fits). stderr: the breakdown and the time estimate.

VRAM accounting (GiB = 2^30 bytes):
  weights     every trunk tensor except token_embd (llama.cpp keeps the input embedding on the CPU).
              MTP/nextn layers are excluded (lovelace-kld doesn't load them).
  KV          n_ctx x attention layers x n_head_kv x (key_len x size(ctk) + val_len x size(ctv))
  recurrent   conv + SSM state of the recurrent layers, f32, one sequence
  compute     ESTIMATE of llama.cpp's compute buffer for one ubatch: logits (max_outputs x n_vocab f32)
              + FFN activations (3 x ub x n_ff f32) + attention/projection activations
              (4 x ub x widest projection f32) + residual stream (6 x ub x n_embd f32).
              Quantized KV adds FA's f16 copy of K and V (n_ctx x n_head_kv x (key_len + val_len) x 2 B).
              Offloaded FFNs add staging for one layer's streamed weights.
  overhead    CUDA context, cuBLAS workspace, pools: a fixed OVERHEAD_GIB

The printed "buffer size" lines of the real run (lovelace-kld prints llama.cpp's model/KV/compute
buffer sizes) are the ground truth; if they disagree with this plan, adjust --vram-gib.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

LLAMA_ROOT = Path(os.environ.get("LLAMA_ROOT", Path.home() / "Work/src/llama.cpp"))
sys.path.insert(0, str(LLAMA_ROOT / "gguf-py"))
from gguf import GGUFReader  # noqa: E402

GIB = 2**30
OVERHEAD_GIB = 0.75
# bytes per element of the KV cache types lovelace-kld accepts on CUDA with FA
KV_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32, "q4_1": 20 / 32, "q4_0": 18 / 32}
FFN_RE = re.compile(r"^blk\.(\d+)\.ffn_(up|gate|down)\.weight$")


def field(r: GGUFReader, key: str, default=None):
    f = r.fields.get(key)
    return default if f is None else f.contents()


def needed_ctx(windows: Path, buckets: set[str] | None, max_scored: int, ctx_train: int) -> tuple[int, list[int]]:
    """n_ctx lovelace-kld will pick (longest window rounded up to 256) and each window's decode length."""
    need = []
    for line in windows.read_text().splitlines():
        if not line.strip():
            continue
        w = json.loads(line)
        if buckets and w["depth_bucket"] not in buckets:
            continue
        if "depth_tokens" not in w:
            sys.exit(f"{windows}: no exact depths; run tokcount + build_corpus.py --counts first")
        need.append(w["depth_tokens"] + min(w["reply_tokens"], max_scored) - 1)
    if not need:
        sys.exit("no windows in the selected buckets")
    fit = [n for n in need if n <= ctx_train]
    return min(ctx_train, math.ceil(max(fit) / 256) * 256), fit


def decode_calls(n_decode: int, n_prefix: int, n_batch: int, max_outputs: int) -> list[int]:
    """Chunk sizes of lovelace-kld's run_window (llama_decode calls)."""
    first_out, out, pos = n_prefix - 1, [], 0
    while pos < n_decode:
        n = min(n_batch, n_decode - pos)
        if pos + n > first_out:
            n = min(n, max(first_out - pos, 0) + max_outputs)
        out.append(n)
        pos += n
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--buckets", help="comma-separated depth buckets (default: all)")
    ap.add_argument("--ctk", default="f16")
    ap.add_argument("--ctv", default="f16")
    ap.add_argument("-b", type=int, default=2048)
    ap.add_argument("-ub", type=int, default=2048)
    ap.add_argument("--max-outputs", type=int, default=256)
    ap.add_argument("--max-scored", type=int, default=1024)
    ap.add_argument("--vram-gib", type=float, default=22.5, help="usable VRAM (RTX 4090: 23.99 GiB total)")
    ap.add_argument("--buft", default="CPU", help="buffer type for offloaded FFN weights")
    ap.add_argument("--prefill-tps", type=float, default=2000.0, help="full-GPU prefill speed (tokens/s)")
    ap.add_argument("--pcie-gbs", type=float, default=25.0, help="host->GPU streaming bandwidth (GB/s)")
    ap.add_argument("--disk-gbs", type=float, default=2.0, help="model load bandwidth (GB/s)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    r = GGUFReader(args.model)
    arch = field(r, "general.architecture")
    n_layer = field(r, f"{arch}.block_count") - field(r, f"{arch}.nextn_predict_layers", 0)
    n_embd = field(r, f"{arch}.embedding_length")
    n_ff = field(r, f"{arch}.feed_forward_length")
    n_head_kv = field(r, f"{arch}.attention.head_count_kv")
    head_dim = n_embd // field(r, f"{arch}.attention.head_count")
    k_len = field(r, f"{arch}.attention.key_length", head_dim)
    v_len = field(r, f"{arch}.attention.value_length", head_dim)
    ctx_train = field(r, f"{arch}.context_length")

    sizes = {t.name: int(t.n_bytes) for t in r.tensors}
    shapes = {t.name: [int(x) for x in t.shape] for t in r.tensors}
    n_vocab = shapes["output.weight"][1] if "output.weight" in sizes else shapes["token_embd.weight"][1]

    def layer(name: str) -> int | None:
        m = re.match(r"^blk\.(\d+)\.", name)
        return int(m.group(1)) if m else None

    trunk = {n: s for n, s in sizes.items() if (layer(n) is None or layer(n) < n_layer)}
    weights = sum(s for n, s in trunk.items() if n != "token_embd.weight")
    ffn = [0] * n_layer
    ffn_max_tensor = [0] * n_layer
    for n, s in trunk.items():
        m = FFN_RE.match(n)
        if m:
            ffn[int(m.group(1))] += s
            ffn_max_tensor[int(m.group(1))] = max(ffn_max_tensor[int(m.group(1))], s)
    attn_layers = [i for i in range(n_layer) if f"blk.{i}.attn_k.weight" in sizes]
    rec_layers = [i for i in range(n_layer) if f"blk.{i}.ssm_a" in sizes]

    buckets = set(args.buckets.split(",")) if args.buckets else None
    n_ctx, need = needed_ctx(args.windows, buckets, args.max_scored, ctx_train)

    kv = n_ctx * len(attn_layers) * n_head_kv * (k_len * KV_BYTES[args.ctk] + v_len * KV_BYTES[args.ctv])
    recurrent = 0
    if rec_layers:
        inner = field(r, f"{arch}.ssm.inner_size")
        state = field(r, f"{arch}.ssm.state_size")
        groups = field(r, f"{arch}.ssm.group_count")
        heads = field(r, f"{arch}.ssm.time_step_rank")
        kernel = field(r, f"{arch}.ssm.conv_kernel")
        per = (kernel - 1) * (inner + 2 * groups * state) + heads * (inner // heads) * state
        recurrent = len(rec_layers) * per * 4
    widest = max(shapes[n][1] for n in trunk if re.match(r"^blk\.\d+\.(attn_q|attn_qkv|attn_gate)\.weight$", n))
    ub = min(args.ub, n_ctx)
    compute = (
        min(args.max_outputs, ub) * n_vocab * 4
        + 3 * ub * n_ff * 4
        + 4 * ub * widest * 4
        + 6 * ub * n_embd * 4
    )
    fa_f16 = 0 if (args.ctk == "f16" and args.ctv == "f16") else n_ctx * n_head_kv * (k_len + v_len) * 2
    fixed = kv + recurrent + compute + fa_f16 + OVERHEAD_GIB * GIB

    budget = args.vram_gib * GIB
    offload: list[int] = []
    staging = 0
    if weights + fixed > budget:
        # largest FFNs first: fewest layers streamed, the streamed bytes overshoot by < one layer
        order = sorted(range(n_layer), key=lambda i: (-ffn[i], i))
        staging = 3 * max(ffn_max_tensor)
        deficit = weights + fixed + staging - budget
        moved = 0
        for i in order:
            if moved >= deficit:
                break
            offload.append(i)
            moved += ffn[i]
        if moved < deficit:
            sys.exit(f"{args.label}: does not fit even with every FFN on the host ({(weights + fixed - moved) / GIB:.1f} GiB)")
    offload.sort()
    streamed = sum(ffn[i] for i in offload)
    gpu_total = weights - streamed + fixed + staging

    # time: prefill at --prefill-tps + streaming of the offloaded FFNs once per ubatch
    ubatches = 0
    tokens = 0
    windows = [json.loads(l) for l in args.windows.read_text().splitlines() if l.strip()]
    for w in windows:
        if buckets and w["depth_bucket"] not in buckets:
            continue
        n_dec = w["depth_tokens"] + min(w["reply_tokens"], args.max_scored) - 1
        if n_dec > n_ctx:
            continue
        tokens += n_dec
        ubatches += sum(math.ceil(n / ub) for n in decode_calls(n_dec, w["depth_tokens"], args.b, args.max_outputs))
    t_prefill = tokens / args.prefill_tps
    t_stream = ubatches * streamed / (args.pcie_gbs * 1e9)
    t_load = sum(sizes.values()) / (args.disk_gbs * 1e9)

    ot = ""
    if offload:
        ot = "blk\\.(" + "|".join(str(i) for i in offload) + ")\\.ffn_(up|gate|down)\\.weight=" + args.buft
    g = lambda b: f"{b / GIB:6.2f}"  # noqa: E731
    print(
        f"[plan] {args.label or args.model.name}: {len(need)} windows, n_ctx {n_ctx}, KV {args.ctk}/{args.ctv}, ub {ub}\n"
        f"  GiB: weights {g(weights - streamed)} (+{g(streamed)} streamed from host, {len(offload)} layers' FFN)"
        f" | KV {g(kv)} | recurrent {g(recurrent)} | compute~ {g(compute + fa_f16 + staging)} | overhead {OVERHEAD_GIB}"
        f" | total {g(gpu_total)} of {args.vram_gib}\n"
        f"  time: {tokens} tokens, {ubatches} ubatches: prefill {t_prefill / 60:.1f} min + streaming {t_stream / 60:.1f} min"
        f" + load {t_load / 60:.1f} min = {(t_prefill + t_stream + t_load) / 60:.1f} min",
        file=sys.stderr,
    )
    print(ot)


if __name__ == "__main__":
    main()
