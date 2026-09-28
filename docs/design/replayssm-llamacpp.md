# ReplaySSM for llama.cpp: record + fold rollback for Gated DeltaNet

Status: design brief, not implemented yet. Target: Qwen3.5/3.8-family hybrid models (`LLM_ARCH_QWEN35`,
`LLM_ARCH_QWEN35MOE`; `LLM_ARCH_QWEN3NEXT` comes for free) serving 4–6 concurrent agents on one RTX 4090,
with MTP (n_draft 3) or DFlash2 (n_draft 3..7) speculative decoding.

Trees read:
- llama.cpp `~/Work/src/llama.cpp` at b11223. All `path:line` references without a prefix point into this tree.
- NInfer `~/Work/ninfer-4090-lin-hybrid`, cited as `ninfer:path:line`.

Model constants for Qwen3.8-27B (`ninfer:docs/maintainer/qwen3.8-27b-artifact.md:55-64`): 64 layers, 48 GDN + 16
full attention; GDN key heads 16×128, value heads 48×128; conv channels 10240, conv taps W=4.

---

## 0. Summary

Today llama.cpp makes rollback possible by storing a full recurrent state for each of the last `K = n_rs_seq+1`
positions. It widens every per-layer recurrent tensor to `mem_size*(1+n_rs_seq)` rows, and the verify kernel writes
up to K full 3 MiB-per-layer state planes for each sequence. For 6 sequences and n_draft=7 that is 7.0 GiB of
recurrent cache. It also costs about 6.9 GB of state writes per verify round.

Proposed change:
1. Keep exactly one committed state plane per cell.
2. During every recurrent ubatch, write the fp32 inputs of the last `K' = min(n_seq_tokens, K)` GDN transitions
   (k, v, g, beta) and conv columns into a per-layer record arena. Skip the state write for those K' tokens.
3. Fold lazily. In the next graph that touches the cell, the same GDN kernel first replays the `m` pending
   accepted records through the same device-side transition function, then stores the committed state, then runs
   the new tokens.

`m` starts at K' and is lowered by `llama_memory_seq_rm` (partial removal). This needs no new public llama API and
no server changes. Graph topology stays static, so llama graph reuse and CUDA graphs keep working.

VRAM for 6 sequences drops from 3591 MiB to 979 MiB (n_draft 3), and from 7182 MiB to 1061 MiB (n_draft 7). §3.11
has the details.

---

## 1. How llama.cpp does it today

### 1a. Who sets `n_rs_seq`, and how memory is sized

The value flows from the command line to the recurrent tensors like this:

- CLI `--spec-draft-n-max N` sets `params.speculative.draft.n_max` (`common/arg.cpp:4137-4143`). The default is 3
  (`common/common.h:327`).
- `common_params_speculative::need_n_rs_seq()` returns `draft.n_max` when any speculative type is
  `DRAFT_MTP | DRAFT_EAGLE3 | DRAFT_DFLASH | DRAFT_DSPARK`, else 0 (`common/common.h:395-401`).
- `common_context_params_to_llama` does `cparams.n_rs_seq = params.speculative.need_n_rs_seq()`
  (`common/common.cpp:1676`). The public field is `llama_context_params::n_rs_seq` (`include/llama.h:370`,
  marked EXPERIMENTAL).
- Draft and MTP contexts force it to 0: `common/speculative.cpp:2552` and the fit path `common/common.cpp:1264`.
  So only the target context pays for snapshots.
- The `llama_context` constructor copies the value into `cparams.n_rs_seq` and clamps it to 0 if
  `!llm_arch_supports_rs_rollback(arch)` (`src/llama-context.cpp:105-110`). The supported archs are listed at
  `src/llama-arch.cpp:1114-1130`: QWEN35, QWEN35MOE, KIMI_K3, QWEN4EXP, DEEPSEEK4, NEMOTRON_H(_MOE), LFM2(MOE) and
  BAILINGMOE3. QWEN3NEXT is not on the list. `llama_n_rs_seq(ctx)` exposes the clamped value
  (`src/llama-context.cpp:3893-3895`).
- `llama_model::create_memory` passes `cparams.n_rs_seq` and `recurrent_rs_size = max(1, n_seq_max)`, with f32 for
  both r and s, to `llama_memory_hybrid` (`src/llama-model.cpp:2700-2718`; the iswa variant is at 2661-2678). The
  hybrid builds `llama_memory_recurrent(model, type_r, type_s, offload, rs_size, n_seq_max, n_rs_seq, filter)`
  (`src/llama-memory-hybrid.cpp:54-65`).
- The recurrent constructor sets `n_rows = mem_size * (1 + n_rs_seq)` and allocates, per recurrent layer,
  `cache_r_l = F32[n_embd_r, n_rows]` and `cache_s_l = F32[n_embd_s, n_rows]` (`src/llama-memory-recurrent.cpp:101-107`).
  It also sets `rs_idx.assign(n_seq_max, 0)` (`:36`). Row `p*mem_size + cell` is "plane p" of that cell.
- Sizes for qwen35:
  - `n_embd_r = (d_conv-1) * (d_inner + 2*n_group*d_state) = 3*10240 = 30720` floats (`src/llama-hparams.cpp:229`).
  - `n_embd_s = d_state * d_inner = 128*6144 = 786432` floats (`src/llama-hparams.cpp:257`).
  - One plane per cell is 48 × (3 MiB + 120 KiB) = 149.625 MiB.
- MTP draft context: `mtp_on_hybrid_qwen` makes a `LLAMA_CONTEXT_TYPE_MTP` context on QWEN3NEXT/QWEN35/QWEN35MOE/
  BAILINGMOE3 skip the hybrid memory (`src/llama-model.cpp:2606-2609, 2624`) and use a plain attention KV cache,
  because the MTP head is one full-attention block (`src/models/qwen35.cpp:93-116, 486-489`).
- DFlash draft: `LLM_ARCH_DFLASH` falls through to the default non-recurrent path (`src/llama-model.cpp:2575-2600`).

In short, only the target context's GDN state is affected, for both MTP and DFlash.

### 1b. Writing the K snapshot slots

**GDN state.** In `llm_build_delta_net_base::build_recurrent_attn` (`src/models/delta-net-base.cpp:527-606`):
- With `n_rs_seq == 0`: `build_delta_net` (fused → `ggml_gated_delta_net(..., K=1)`, `:402`, or composite ops,
  `:425-447`) is followed by one `ggml_cpy` of the final state into `ssm_states_all` rows at `kv_head` (`:548-560`).
- With `n_rs_seq > 0`: `K = n_rs_seq+1` (`:564`), then `gdn_out = ggml_gated_delta_net(ctx0, q,k,v,g,b,s,K)` (`:567`)
  is marked as a fused GDN node (`:568-572`).
  - The first `S_v*H_v*n_tokens*n_seqs` floats are the attention output (`:577-582`).
  - `n_written = min(n_seq_tokens, K)` state slots are copied by one strided `ggml_cpy` into
    `ssm_states_all[plane i, kv_head+s]`, with plane stride `mem_size*row_size` (`:585-603`).
- Op contract (`ggml/include/ggml.h:2639-2660`; `ggml/src/ggml.c:6366-6419`):
  - q,k are `[S_k,H_k,T,B]`; v is `[S_v,H_v,T,B]`; g is `[1|S_v,H_v,T,B]`; beta is `[1,H_v,T,B]`; state (s0) is
    `[S_v,S_v,H_v,B]`. Everything is F32 (`ggml.c:6382-6387`).
  - The output is `[S_v*H, T*B + K*S_v*B]`: scores, then K snapshots, most recent first (slot s = state s tokens
    back). K is op param 0 (`ggml.c:6404-6408`).
  - When the heads differ, the model repeats q/k to `H_v` unless both fused paths are enabled
    (`src/models/qwen35.cpp:438-442`). The kernel itself broadcasts with `iq1 = h % H_k`
    (`ggml-cuda/gated_delta_net.cu:37`).

**Conv state.** In `build_conv_state` (`src/models/delta-net-base.cpp:449-525`):
- `conv_states = build_rs(...)` gathers `[W-1, C, B]` (`:463-466`), and `conv_input = concat(conv_states, qkv_mixed^T)`
  (`:469-472`).
- For `n_rs_seq>0`, K views of `conv_input` at column offsets `ne0-(W-1)-K+t` are each copied to plane `K-t`
  (`:497-521`), so every slot receives a full (W-1)-column window. qwen35 then runs `ggml_ssm_conv → silu`,
  splits q/k/v and L2-normalises q,k in separate ops (`src/models/qwen35.cpp:391-430`). The GDN kernel therefore
  consumes **post-conv, post-L2-norm fp32 k** and fp32 v, g, beta.

### 1c. Rollback path

1. The server verifies `[anchor, d1..dD]` as `n_draft+1` tokens per slot (`tools/server/server-context.cpp:3924`).
2. After sampling it computes `n_rollback = n_draft + 1 - accepted.size()` (`:3935`) and calls
   `slot.mem.seq_rm(slot.id, pos_next, -1)` (`:4009`). If `seq_rm` fails, `common_context_seq_rm` aborts
   (`common/common.cpp:1577-1581`). When `draft.size() > llama_n_rs_seq(ctx)` it uses full state checkpoints
   instead (`server-context.cpp:3082-3101, 3937-3967`).
3. `llama_memory_recurrent::seq_rm` handles a partial tail removal `0<p0<=cell.pos<p1`
   (`src/llama-memory-recurrent.cpp:193-204`):
   - It computes `rollback = cell.pos-(p0-1)`, requires no pending rollback and `1 <= rollback <= n_rs_seq`, then
     sets `rs_idx[seq]=rollback` and `cell.pos=p0-1`.
   - Rollbacks are single-use: a second partial removal while one is pending returns false (`:196-203`). This is
     asserted in `tests/test-recurrent-state-rollback.cpp:174-175`.
   - The validity check against how many slots the last ubatch actually wrote is missing (see the TODO at
     `tests/test-recurrent-state-rollback.cpp:402-403`).
4. At the next graph, `llama_memory_recurrent_context::s_copy(i)` returns `rs_idx[seq]*size + src0` and resets
   `rs_idx[seq]=0` as a side effect of `set_input` (`src/llama-memory-recurrent.cpp:1346-1364`).
   `llm_graph_context::build_rs` then does `ggml_get_rows(states_all, s_copy_main)` (`src/llama-graph.cpp:3487-3521`),
   so the graph reads the snapshot plane as s0.
5. Inputs are filled at `src/llama-graph.cpp:334-342` (recurrent) and `:1105-1113` (hybrid). `can_reuse` checks the
   `s_copy` shape, head and `rs_z` (`:352-358`, `:1128-1134`).

### 1d. The `split_equal` `n_keep_tail` constraint

- `llama_memory_recurrent::init_batch` calls `balloc.split_equal(n_ubatch, true, n_rs_seq>0 ? n_rs_seq+1 : 0)`
  (`src/llama-memory-recurrent.cpp:442-445`). The hybrid does the same with `sequential = !unified`
  (`src/llama-memory-hybrid.cpp:81-89`; same in `-iswa.cpp:83-88` and `-idx.cpp:88-93`).
- `split_equal` keeps in a ubatch only the sequences that either finish there or leave at least `n_keep_tail`
  tokens for a later ubatch. It truncates or extends the first sequence to satisfy this, and asserts
  `n_ubatch > n_keep_tail` (`src/llama-batch.cpp:637-701`).
- Why: snapshots are written only for the last `min(n_seq_tokens,K)` tokens of each ubatch
  (`delta-net-base.cpp:498-500`, the `[TAG_RECURRENT_ROLLBACK_SPLITS]` comment). If a sequence's final K tokens were
  split across two ubatches, the second ubatch would overwrite the earlier slots and the rollback window would be
  wrong.

### 1e. CUDA kernels and fusions

**`gated_delta_net_cuda<S_v,KDA,keep_rs_t>`** (`ggml/src/ggml-cuda/gated_delta_net.cu:4-167`):
- Grid is `(H, n_seqs, S_v/4)`, block is `(32, 4)`. Each warp owns one state column; each lane holds `S_v/32` rows
  in registers (`:31-54`, `:180-184`). The kernel is sequential over tokens (`:63`).
- Per token (non-KDA):
  - `g=expf(g_t)`, `kv = warp_reduce_sum(Σ s·k)`, `delta = (v[col] - g·kv)·beta`
  - `s = g·s + k·delta`, `attn = warp_reduce_sum(Σ s·q)·scale` (`:84-111`)
- If `keep_rs_t` (K>1), after every token t with `target_slot = n_tokens-1-t < K` it stores the full column to
  slot `target_slot` (`:145-157`). Otherwise it stores once at the end (`:160-166`). The input state is s0 only
  (`:42-47`).
- Host side: `K = op_params[0]`, `keep_rs = K>1` (`:285-287`). State goes to the dst tail, or to the cache when fused
  (`:289-295`).

**Cache fusion.** `ggml_cuda_try_gdn_cache_fusion` (`ggml/src/ggml-cuda/ggml-cuda.cu:2751-2811`) matches
`GATED_DELTA_NET` followed by the strided snapshot `CPY` and lets the kernel write the K planes straight into
`cache_s_l`. It is invoked at `ggml-cuda.cu:3463-3470`.

**Conv.** `GGML_OP_SSM_CONV` is `ggml-cuda/ssm-conv.cu`, with an SSM_CONV+SILU (+ADD) fusion at
`ggml-cuda.cu:4171-4179`.

**MTP/DFlash-specific hybrid handling.** The only special case is `mtp_on_hybrid_qwen` / `mtp_on_hybrid_nemotron`
memory selection for the draft context (§1a). Nothing on the target side is MTP- or DFlash-specific. The server
treats every draft type the same through `llama_n_rs_seq` and `seq_rm`.

**CUDA graphs** (`ggml-cuda.cu:2592-2632`):
- The graph key is `cgraph->nodes[0]`. The graph is re-captured when the node count or any node's
  `ggml_tensor` bytes (op_params, data pointers) or its srcs' data/ne/nb change. Changes to input tensor data do not
  trigger a re-capture.
- Warmup and update logic is at `:4441-4490`. Concurrent-stream reordering only happens with
  `GGML_CUDA_GRAPH_OPT=1` (`:4621-4624`).

**llama graph reuse** (`src/llama-context.cpp:1402-1446`): if `res->can_reuse(gparams)` the graph is kept;
otherwise `ggml_backend_sched_reset` runs and the graph is rebuilt and reallocated. Note that `memory_update()`
resets all previous graph results whenever a memory module reports an update (`src/llama-context.cpp:870-877`), and
the K-shift update runs its own graph on the shared scheduler (`src/llama-kv-cache.cpp:857-886`).

### 1f. State save/restore today

- `llama_memory_recurrent::state_write` serialises one plane per cell: the logical source row
  `rs_idx*size + src` (`src/llama-memory-recurrent.cpp:766-845, 897-990`; see the comment at `:918-919`).
  `state_read` writes plane 0 and clears `rs_idx` (`:873-875`, `:1093-1229`).
- Serialized size is therefore already one plane. The VRAM widening is the problem, not the file size.

---

## 2. NInfer ReplaySSM (what we are porting)

### 2.1 Math (`ninfer:docs/maintainer/replayssm-gdn.md`)

For each GDN layer and value head, the state is an fp32 matrix S ∈ ℝ^{V×K}, with K = V = 128. With
$\bar k = k/\sqrt{\sum k^2+\epsilon}$ and $\alpha_t = e^{g_t}$, one transition is:

$$u_t=\beta_t\,(v_t-\alpha_t S_{t-1}\bar k_t),\qquad S_t=\alpha_t S_{t-1}+u_t\bar k_t^{\mathsf T},\qquad y_t=c\,S_t\bar q_t,\ c=1/\sqrt{128}$$

(doc §2.1, L91-150).

- The transition needs only $R_t=(k_t^{raw},v_t,g_t,\beta_t)$; q only affects the output (L168-183).
- Verify runs T=D+1 inputs from the committed $S_0$, emits all outputs and records every $R_t$, but never publishes
  $S_1..S_T$ (L229-248).
- Once the commit length m (0 ≤ m ≤ A+1) is known, fold computes $S_m=F_{fp}(\dots F_{fp}(S_0,R_1)\dots,R_m)$. With
  m=0 the state is untouched, and the rejected suffix is never read (L250-267).

Exactness is by induction. If fold executes *the same finite-precision* $F_{fp}$ on the same record bits, then
$S_m^{fold}=S_m^{verify}$ bit for bit (L273-307). The doc stresses that "algebraically equivalent" forms drift,
because normalisation, reduction order, FMA contraction and exp paths all differ, and drift propagates across rounds
(L311-436). It lists the verification checklist: m=0,1,T and middle values, long chains, extreme gates, grouped
heads, and rejected suffix mutation (L438-462).

Conv history for width W keeps only the last W−1 projection columns. Commit is the exact gather
$H_m=\mathrm{tail}_{W-1}(H_0\Vert[p_1..p_m])$. Recording one column per token is 1/3 of a snapshot for W=4
(L466-498).

### 2.2 Record layout and bytes

Spec fields: layers, `record_capacity` (rows ≤ 8), width, conv_channels, qk_heads, value_heads, key_dim, value_dim
(`ninfer:src/core/gdn_replay_records.h:11-20`). There are four 256-B aligned, disjoint planes
(`ninfer:src/core/gdn_replay_records.cpp:11-12, 78-91, 96-113`):

| plane | dtype | shape (innermost first) |
|---|---|---|
| conv | BF16 | `[conv_channels, width, layers*rows]` |
| key | BF16 | `[key_dim, qk_heads, width, layers*rows]` (raw, **pre**-normalisation) |
| value | BF16 | `[value_dim, value_heads, width, layers*rows]` |
| gate | FP32 | `[2, value_heads, width, layers*rows]`, ordered {g, beta} |

- The outer index is `layer*record_capacity + row` (`gdn_replay_records.h:45-46`; kernel side
  `ninfer:.../recurrent.cuh:502-504`).
- The arena is Program-owned. It is planned only when a speculative backend is configured, with
  `record_capacity = max_concurrency` and `width = draft_window+1`
  (`ninfer:src/targets/qwen3_6/impl/runtime/layouts_impl.h:155-167`), and bound in `program_impl.h:879-885`.
- Bytes per token: $P_{gdn}=2L_g(H_qK+H_vV)+8L_gH_v$ and $P_{conv}=2L_gC_p$ (doc L528-539). For 27B that is
  0.7676 MiB (GDN) + 0.9375 MiB (conv) = 1.705 MiB/token, versus 146.8 MiB per snapshot position (doc L568-578).

### 2.3 Verify-with-record and fold kernels

Everything goes through one shared sequence function,
`run_recurrent_sequence<Normalize, Effects>` (`ninfer:src/ops/linear_attention/gated_delta_net/recurrent.cuh:616-640`),
which calls `apply_gdn_transition` (`:104-122`: `alpha=expf(g)`; `partial=warp_sum(Σ S·k)`;
`delta=beta·(v−alpha·partial)`; `S=alpha·S+delta·k`). The effect policy decides what happens as a side effect:

- `OutputEffects` publishes y.
- `RecordEffects` publishes y and stores the raw record bits (`:192-214`). Stores are single-writer: the key is
  written once per qk head (tile 0, warp 0, `value_head % group == 0`), the value by lane 0, and the gate by one
  thread (`:421-446`).
- `FoldEffects` does nothing per token (`:216-230`).

The verify-record kernel (`recurrent_record_kernel`, `:680-689`) reads the state through a `const float*`
(`ninfer:.../recurrent.cu:88`), so the committed checkpoint is never written during verify.

Fold (`recurrent_fold_kernel`, `recurrent.cuh:691-702`) is a single all-layer launch:
- Grid is `(value_heads, active_rows, layers*8 state tiles)`, block is `(32,4)` (`recurrent.cu:122-126`). The
  coordinates decode `layer = z>>3, tile = z&7, qk_head = value_head/group` (`recurrent.cuh:475-495`).
- Each warp owns 4 value rows and each lane 4 key columns (`recurrent.cuh:13-16`). The kernel is sequential over the
  m accepted tokens and parallel over layers × heads × state tiles × rows.
- m=0 means immediate return (`:696`). After the loop it calls `store_final_state` (`:542-552`) and
  `publish_final_conv_history`, which handles commits 1, 2 and ≥3 as pure gathers from old history plus records
  (`:554-594`).

Conv records are the BF16 projection column `p` that the baseline would have shifted into history
(`RecordColumnPublish`, `ninfer:src/ops/gdn_input_proj/gdn_conv.cuh:27-37`). By contrast `SnapshotHistoryPublish`
writes all 3 columns per token (`:11-25`).

### 2.4 Integration, batch rows, CUDA graphs

- Row descriptor: `{source_state_slot, destination_state_slot, commit_columns}`
  (`ninfer:include/ninfer/ops/gdn_replay.h:13-17`). Row b always maps to physical record row b; in-place is allowed;
  destinations must not overwrite another active row's source (`:22-35`).
- Verify uses `GdnStateAction::RecordForReplay` (`ninfer:.../speculative_target_impl.h:15`; TextContext
  `text_context_impl.h:915-964`). Prefill uses `UpdateInPlace` with no records (`text_prefill_impl.h:40`). Batched
  in-place update is width-1 only (`text_context_impl.h:905-906`).
- Variable per-row widths inside one fixed-width graph are handled by `valid_columns` masking
  (`recurrent.cuh:379-383`) and zeroing padded outputs (`:642-651`).
- The fold runs **eagerly, outside the verify CUDA graph**, once the committed prefix is known
  (`program_impl.h:10146-10195`). Row descriptors are passed by value as `__grid_constant__`
  (`recurrent.cuh:693`), so a captured fold graph would need re-capture whenever the descriptors change
  (`gdn_replay.h:40-42`). Verify runs inside per-batch-size CUDA graph families (`program_impl.h:11850-11853,
  11989-11992, 12182-12185`).

### 2.5 Numerics and tests

- State is fp32. Records hold the exact represented bits: BF16 raw k/v (NInfer's verify consumes BF16 and normalises
  inside the kernel with `NormalizeInputs=true`, `recurrent.cuh:622, 687, 699`) and FP32 g/beta. The conv history
  is BF16 (`layouts_impl.h:140`).
- `ninfer:tests/ops/test_gdn_replay_fold.cpp` asserts `memcmp` equality of folded state against the Nth sequential
  snapshot (`:383`) and of conv history (`:400`). It also checks that sources and inactive slots are untouched
  (`:409, 450-451`), that CUDA-graph replay of the fold matches (`:345-353`), that record and snapshot projection
  outputs are bitwise equal (`:678-688`), and that multi-round record→fold chains match snapshots (`:698-714`).

---

## 3. Proposed llama.cpp design

### 3.1 Invariants

The design is an adaptation of §2.5 and §7 of the NInfer doc.

1. Every cell (physical recurrent row) holds exactly one committed plane: fp32 `S_c` and `H_c` (conv).
2. A recurrent ubatch with `n = n_seq_tokens` tokens per sequence and `K' = min(n, K)` (with `K = n_rs_seq+1`)
   leaves each touched cell in this state:
   - `S_c` = state after the pending fold plus the first `n−K'` new tokens;
   - `H_c` = conv history at the same point;
   - `rec[row][0..K')` = the exact fp32 inputs of the last K' transitions and conv columns;
   - `rec_m[row] = K'`.
3. `seq_rm` partial removal of r tokens is valid iff `r ≤ rec_m[row]`. It sets `rec_m[row] -= r`.
4. The next ubatch touching the cell replays `rec_m[src]` records in-kernel, with the same device function as the
   verify loop, before processing new tokens. Replaying 0 records is a strict no-op.
5. Records are read (phase 1) before any record is written (phase 2) inside the same ggml op. This makes swapped
   source and destination rows safe.

Rollback depth is unchanged: at most `n_rs_seq` after any ubatch, with `n_keep_tail = K` still guaranteeing the tail.
It becomes more exact than today (see 1c.3), because the depth is checked against the records actually written.

```mermaid
sequenceDiagram
  participant S as server
  participant M as llama_memory_recurrent (host)
  participant G as decode graph (per GDN layer)
  S->>G: verify ubatch [anchor,d1..dD] (T=D+1<=K)
  Note over G: rr-op: S0=get_rows(src) → replay rec_m(prev) → store S_c → T steps (outputs only)<br/>phase 2: rec[dst]=k,v,g,β of T tokens
  M->>M: find_slot: fold_in[i]=rec_m[src0(i)]; rec_m[dst]=T
  S->>M: seq_rm(seq, pos_next, -1)  → rec_m[dst] = T - r
  S->>G: next ubatch (draft round or decode)
  Note over G: rr-op replays rec_m records from rec[src] first (m = A+1), then continues
```

### 3.2 Record arena: where it lives and how big it is

Add two tensors per recurrent layer to `llama_memory_recurrent`, allocated in the same per-buffer-type contexts as
`r_l`/`s_l` (`src/llama-memory-recurrent.cpp:77-115`):

- `rec_l[il]`: F32 `[n_embd_rec, K, mem_size]`. Row layout per (cell, slot) is `[k (S_k·H_k) | v (S_v·H_v) |
  g (G·H_v) | beta (H_v)]`, where G = 1, or S_v for KDA.
  New helper `llama_hparams::n_embd_rec() = ssm_d_state*ssm_n_group + ssm_d_inner + 2*ssm_dt_rank`
  = 2048 + 6144 + 96 = **8288 floats** for qwen35.
- `recx_l[il]`: F32 `[C, K, mem_size]`, with `C = n_embd_r()/(ssm_d_conv-1) = 10240`. This holds the raw `qkv_mixed`
  column the baseline would shift into `H`.

`r_l`/`s_l` go back to `n_rows = mem_size` in replay mode (the `(1+n_rs_seq)` factor at `:101` applies only in
snapshot mode).

Records must be fp32. They are exact copies of what `ggml_gated_delta_net` consumes (post-conv, post-L2-norm k and
fp32 v, g, beta; `ggml.c:6382-6387`) and of the f32 conv state type (`llama-model.cpp:2710-2711`). BF16, as NInfer
uses, would be lossy here because llama.cpp produces those tensors in fp32 arithmetic.

Bytes per token, per layer: k 8192 + v 24576 + g/beta 384 + conv 40960 = **74,112 B**. Across 48 GDN layers that is
**3,557,376 B = 3.393 MiB/token** (1.518 MiB GDN + 1.875 MiB conv). This is 2× NInfer's figure because of fp32.

Arena = `mem_size × K × 3.393 MiB`, with `mem_size = n_seq_max`:

| n_seq_max | n_draft (K) | arena |
|---|---|---|
| 6 | 3 (K=4) | 24 × 3.393 = **81.4 MiB** |
| 6 | 7 (K=8) | 48 × 3.393 = **162.8 MiB** |

Host metadata, indexed by **physical row** because the data lives in rows while `find_slot` swaps cell metadata
(`src/llama-memory-recurrent.cpp:628-650`):

- `std::vector<uint8_t> rec_m` (pending fold count per row). `prepare()` saves and restores it alongside `cells`
  (`:481-502`).
- Per-ubatch `std::vector<int32_t> rs_fold_in` of size n_rs, captured in `find_slot`:
  - `(m, src_row) = (cells[i].src < 0 ? 0 : rec_m[src0], src0)`;
  - then `rec_m[head+s] = K'` for rows in the ubatch;
  - extra rows copy `rec_m` from their source.

### 3.3 New ggml ops

A new op is used instead of widening `GGML_OP_GATED_DELTA_NET`. The existing op is implemented by about ten backends
(cpu, cuda, metal, vulkan, sycl, opencl, webgpu, hexagon, et and more; grep `case GGML_OP_GATED_DELTA_NET`), and
adding srcs to it would silently break all of them.

```c
// ReplaySSM gated delta net. q,k,v,g,beta: as ggml_gated_delta_net (k/q may be H_k-broadcast).
// state : F32 [S_v,S_v,H_v,n_seqs]  gathered committed state of each seq's source row
// rec   : F32 [n_embd_rec, K, n_rows] this layer's record arena (READ rows rs[1][s], WRITE rows dst_row0+s)
// rs    : I32 [2, n_seqs]  rs[0][s] = m (records to replay, 0..K), rs[1][s] = physical source row
// op params: [0]=K, [1]=dst_row0 (=kv_head)
// result: F32 [S_v*H_v, n_tokens*n_seqs + S_v*n_seqs] = scores, then ONE committed state per seq:
//   state after replaying m records and the first n_tokens-K' new tokens, K' = min(n_tokens,K).
// side effect (strictly after all reads): rec[dst_row0+s][u] = (k,v,g,beta)[token n_tokens-K'+u], u<K'.
GGML_API struct ggml_tensor * ggml_gated_delta_net_rr(
        struct ggml_context * ctx, struct ggml_tensor * q, struct ggml_tensor * k,
        struct ggml_tensor * v, struct ggml_tensor * g, struct ggml_tensor * beta,
        struct ggml_tensor * state, struct ggml_tensor * rec, struct ggml_tensor * rs,
        int32_t K, int32_t dst_row0);

// ReplaySSM conv history fold. conv_state: F32 [W-1, C, n_seqs] gathered committed history.
// recx: F32 [C, K, n_rows]; x_new: F32 [C, n_tokens, n_seqs] (qkv_mixed, not transposed); rs as above.
// result: F32 [W-1, C, n_seqs] = tail_{W-1}( conv_state || recx[rs[1][s]][0:rs[0][s]] )
// side effect (after all reads): recx[dst_row0+s][u] = x_new[:, n_tokens-K'+u, s].
GGML_API struct ggml_tensor * ggml_ssm_conv_fold(
        struct ggml_context * ctx, struct ggml_tensor * conv_state, struct ggml_tensor * recx,
        struct ggml_tensor * x_new, struct ggml_tensor * rs, int32_t K, int32_t dst_row0);
```

The output layout is identical to `ggml_gated_delta_net(..., K=1)`. That means `ggml_cuda_try_gdn_cache_fusion`
(`ggml-cuda.cu:2751-2811`) only needs to accept the new op code to keep writing the committed plane straight into
`cache_s_l`. The rest of the matcher is unchanged.

Writing into a non-output src has precedent: `CPY` writes into src[1] and `SET_ROWS` into src[0]. Both
implementations must `GGML_ASSERT` that `rec` lives in a buffer of the executing backend, because otherwise the
scheduler's input copy would drop the writes.

**CUDA kernel.** Refactor `gated_delta_net_cuda` (`gated_delta_net.cu:4-167`) so that one
`__device__ __forceinline__ gdn_step(s_shard, k_reg, v_col, g, beta)` is used by both the replay and the verify loop.
This is the llama.cpp equivalent of NInfer's `run_recurrent_sequence` / `apply_gdn_transition`. Pseudo-code:

```
load s_shard from state (gathered S of src row)
m = rs[0][seq]; src = rs[1][seq]
for t in [0,m):  k_reg,v,g,β ← rec[src][t];  gdn_step(...)          // no readout
j = n_tokens - min(n_tokens,K)
for t in [0,n_tokens):
    if t == j: store s_shard → committed (dst tail or fused cache ptr)
    k_reg,q_reg,v,g,β ← inputs[t]; gdn_step(...); attn[t] = warp_reduce(Σ s·q)·scale
```

- Phase 2 is a second, trivially parallel copy kernel launched from the same op: tokens `[j,n)` of k (H_k heads),
  v, g and beta go into `rec[dst_row0+s]`. Stream order separates phase 1 reads from phase 2 writes without any
  grid sync. Two kernels are needed because every column-group block reads the full k vector of its qk head
  (`gated_delta_net.cu:75-82`), so an in-kernel overwrite would race across blocks.
- `ggml_ssm_conv_fold` is one small kernel for phase 1 (thread per channel × seq, gather W−1 values) plus a phase-2
  copy.
- Pin the arithmetic in `gdn_step` with explicit `__fmaf_rn` / `__fmul_rn` / `__fsub_rn`, keeping the existing
  `expf` and `warp_reduce_sum`. The CUDA build uses `-use_fast_math` (`ggml/src/ggml-cuda/CMakeLists.txt:186`),
  which lets nvcc contract `g*s + k*delta` either way. Pinning removes any chance that the replay loop (no readout)
  and the verify loop compile to different FMA trees. Apply the same pinning to the K=1 path so non-speculative
  decode goes through the identical transition.

**CPU kernel.** Extend `ggml_compute_forward_gated_delta_net_one_chunk` (`ggml/src/ggml-cpu/ops.cpp:10895`) with the
replay prologue and committed-store point. Run phase 2 after `ggml_barrier(params->threadpool)`, following the
pattern at `ggml/src/ggml-cpu/ggml-cpu.c:1378`. Work-size accounting goes next to `ggml-cpu.c:3019-3025`.

Other backends: `supports_op` returns false, and §3.12 covers the fallback.

### 3.4 Graph wiring (`delta-net-base.cpp`, `llama-graph.cpp`)

**Graph inputs.** Add `ggml_tensor * rs_fold` (I32 `[2, n_rs]`) and its view `rs_fold_main` (`[2, n_seqs]`) to
`llm_graph_input_rs` (`src/llama-graph.h:271-284`):
- create them in `build_rs_inp_impl` (`src/llama-graph.cpp:3523-3544`);
- fill them from `mctx->rs_fold_in` in both `set_input` sites (`:334-342`, `:1105-1113`);
- add `rs_fold->ne[1] == n_rs` to both `can_reuse` checks (`:352-358`, `:1128-1134`).

In replay mode `s_copy(i)` returns plain `src0`; the `rs_idx` plane arithmetic and its side-effect reset
(`llama-memory-recurrent.cpp:1350-1363`) are dropped.

**`build_rs` extra rows** (`src/llama-graph.cpp:3513-3518`): in replay mode also copy `rec_l`/`recx_l` rows with
the same `s_copy_extra` (`get_rows` + `cpy`), so that moved cells take their pending records with them. This follows
the same graph-order assumption as the existing extra-state copy.

**`build_conv_state`** (replay branch replacing `:497-521`):
```
conv_states = build_rs(...)                       // [W-1,C,B] committed
H1 = ggml_ssm_conv_fold(conv_states, recx_l, qkv_mixed, rs_fold_main, K, kv_head)
conv_input = concat(H1, transpose(qkv_mixed))     // as today
committed = view(conv_input, cols [n-K', n-K'+W-1)) → cpy to conv_states_all rows kv_head+s (plane 0)
```

**`build_recurrent_attn`** (replay branch replacing `:562-603`): `ggml_gated_delta_net_rr(q,k,v,g,b,s, rec_l, rs_fold_main, K, kv_head)`,
then views and the plane-0 `cpy` exactly as in the `K=1` path (`:548-560`, `:409-420`). `add_fused_node` stays
as it is.

**qwen35 / qwen35moe / qwen3next.** In replay mode, skip the `ggml_repeat_4d` of q/k (`src/models/qwen35.cpp:438-442`
and the equivalents in the other two files), so records hold H_k heads. The new op broadcasts the same way as the
existing kernel.

**Other archs.** kimi-k3, bailingmoe3 and qwen4exp also call `build_recurrent_attn` but have their own conv
(`kimi-k3.cpp:411-414`, `bailingmoe3.cpp:200-203`, `qwen4exp.cpp:1164-1167`). They stay in snapshot mode until their
conv paths are ported. The mode is a per-arch predicate (§3.12).

### 3.5 Commit / fold timing

The fold is **lazy and fused into the next graph**. There is no eager fold graph and no new llama API. The reason:
a separate fold graph would go through `memory_update()` → `init_update()`, which resets every previous graph result
(`src/llama-context.cpp:870-877`), and would reset the shared scheduler the way K-shift does
(`src/llama-kv-cache.cpp:866-880`). That would force a full graph rebuild and reallocation every speculative round
(`src/llama-context.cpp:1420-1446`).

Fusing also saves one full state read and write per sequence per round compared with NInfer's separate fold launch.
The state is loaded into registers once, m steps are replayed, and the committed plane is stored where the kernel
already stored the final state.

`seq_rm` stays a pure host operation. Replacement for `src/llama-memory-recurrent.cpp:193-204` in replay mode:
```
r = cell.pos - (p0-1);  row = seq tail cell's physical row
if (cell.seq_id.size() > 1 || r > rec_m[row]) return false;   // shared cell / beyond records
rec_m[row] -= r;  cell.pos = p0 - 1;  return true;             // composes; no single-use rule
```
Other operations:
- All-accepted (no `seq_rm`, or `p0 > cell.pos`) leaves `rec_m = K'`, so the next ubatch folds everything.
- `seq_rm` rm_all, `clear()`, `seq_keep` and cell frees set `rec_m[row] = 0`.
- `seq_cp` shares the row. The records stay with the row because the first writer after divergence is moved to a
  fresh cell (`find_slot`, `:598-611`).

Server and common code stay unchanged. `llama_n_rs_seq` keeps its meaning, so `use_ckpt_tgt`
(`server-context.cpp:3082-3084, 3937-3939`) behaves as before.

### 3.6 Non-speculative decode and prefill

- **`n_rs_seq == 0`** (no speculative type): the current code path is untouched. There is no arena, no rr-op, and
  fused or composite GDN is used as today (`delta-net-base.cpp:548-560`).
- **Replay mode, plain decode (n=1):** K'=1. The op replays ≤1 pending record, stores committed, runs 1 token and
  records it. State traffic is one plane read and one plane written, the same as non-speculative decode. The
  per-token cost is one extra register-resident transition plus a 3.4 MiB record write per sequence.
- **Replay mode, prefill (n > K):** the same op. It stores the committed state at `j = n−K` and records the last K
  tokens. This matches today's replay-mode behaviour, since with `n_rs_seq>0` every ubatch already goes through
  `ggml_gated_delta_net(K)` (`delta-net-base.cpp:563-567`), and it keeps the "rollback ≤ n_rs_seq after any ubatch"
  contract that server prompt-cache code relies on (`server-context.cpp:3455`). The `n_keep_tail` constraint is
  kept (§1d).

### 3.7 `--kv-unified` and multi-sequence batches

- With a unified KV cache the hybrid passes `sequential=false` to `split_equal` (`llama-memory-hybrid.cpp:81-89`).
  Seq ids in a ubatch can then be non-contiguous, but `find_slot` still gathers the touched cells into
  `[head, head+n)` and records `src0` per cell (`llama-memory-recurrent.cpp:589-713`).
- Records are addressed by physical row: they are read from `rs[1][s] = src0` and written to `kv_head+s`. That makes
  them independent of seq-id order. Swaps (A←B, B←A) are safe because of the phase 1 / phase 2 split.
- Six sequences each verifying T tokens form one ubatch (`n_seqs=6, n_seq_tokens=T`). If draft lengths differ,
  `split_equal` produces one ubatch per length class, as it does today. Each graph has its own static shape.
  NInfer-style `valid_columns` masking (§2.4) is a later optimisation and is not required.

### 3.8 Prompt cache save/restore

- `state_write` writes the committed plane as today (`llama-memory-recurrent.cpp:897-990`, now always plane 0), then
  for each cell appends `rec_m` (u32) and `rec_m` rows of `rec_l`/`recx_l` for every layer.
- `state_read` restores them into the destination row and sets `rec_m`.
- The format changes, so add a version word after `s_trans` (`:898-902`) and reject old files.
- Extra size is ≤ K × 3.39 MiB per cell (27 MiB at K=8) on top of 149.6 MiB. `LLAMA_STATE_SEQ_FLAGS_ON_DEVICE` works
  unchanged because `io.write_tensor` handles device buffers.
- Folding on the host at save time would be wrong: CPU arithmetic is not bit-identical to the CUDA verify
  trajectory. Folding with a graph at save time would reset graph reuse. Serialising the records avoids both.

### 3.9 CUDA graph and llama graph reuse

- The graph topology depends only on (mode, K, ubatch shape, head). Per-round values (`m`, source rows) live in the
  `rs_fold` input tensor data.
- `can_reuse` therefore holds across rounds with the same shape. The CUDA graph node properties compare pointers,
  shapes and op_params but not input contents (`ggml-cuda.cu:2613-2629`), so steady-state speculative rounds replay
  the captured CUDA graph.
- `dst_row0 = kv_head` is an op_param. A change of head already invalidates both levels today (`llama-graph.cpp:357`,
  `:1133`).

### 3.10 Mode selection

Add `enum llama_rs_mode { NONE, SNAPSHOT, REPLAY }` to `llama_cparams`. Resolve it in the `llama_context` constructor
**before** memory creation, since the number of planes depends on it:
- `n_rs_seq==0` → NONE.
- Arch in {QWEN35, QWEN35MOE, QWEN3NEXT} and `ggml_backend_dev_supports_op(model.dev_layer(il), probe_rr_op)` for
  all recurrent layers → REPLAY. The probe is built in a no-alloc context, following the existing
  `buft_supported` probe pattern (`src/llama-model.cpp:2252-2277`).
- Else, if `llm_arch_supports_rs_rollback` → SNAPSHOT (the current path).
- Else NONE (today's clamp, `llama-context.cpp:105-110`).

Add QWEN3NEXT to `llm_arch_supports_rs_rollback`. Add a debug override `LLAMA_RS_ROLLBACK=snapshot`, modelled on
`LLAMA_GRAPH_REUSE_DISABLE` (`llama-context.cpp:280-281`), so one binary can A/B the two modes. The mode is passed to
`llama_memory_recurrent` in place of the bare `n_rs_seq`.

### 3.11 VRAM and traffic, before and after (Qwen3.8-27B, 6 sequences)

| | n_draft 3 (K=4) | n_draft 7 (K=8) |
|---|---|---|
| today: recurrent planes = 6 × K × 149.625 MiB | **3591.0 MiB** | **7182.0 MiB** |
| replay: 6 × 149.625 MiB committed | 897.75 MiB | 897.75 MiB |
| replay: record arena | 81.4 MiB | 162.8 MiB |
| **replay total** | **979.2 MiB** | **1060.6 MiB** |
| freed for KV | 2611.8 MiB | 6121.4 MiB |

Freed VRAM converts to extra attention-KV context at the target's per-token KV cost (16 attention layers × 4 KV heads ×
256 × 2 (K,V) × type size). That is 64 KiB/token at f16, so the n_draft 7 saving is about 98k more tokens of shared
KV at f16. This figure is [INFERENCE] from the head counts in `qwen3.8-27b-artifact.md:58-61`, not measured.

Compute buffer: the `gdn_out` state tail shrinks from K×3 MiB×n_seqs per layer (72 / 144 MiB at 6 seqs) to 18 MiB,
because the result shape in `ggml.c:6404-6405` goes from K planes to 1. [INFERENCE: the actual peak depends on
gallocr reuse and reservation shapes.]

Per-round GDN state writes: today `n_written = T` planes per sequence (`gated_delta_net.cu:145-157`), i.e.
6×4×144 = 3456 MiB (n=3) or 6×8×144 = 6912 MiB (n=7). Replay writes 6×144 = 864 MiB, plus 81/163 MiB of records,
plus reads of the replayed prefix at ≤1.52 MiB per token. At about 1 TB/s that saves roughly 2.5 ms and 6 ms per
round. [INFERENCE: DRAM-bound estimate, not measured.]

---

## 4. Risks

1. **Bitwise equality of replay vs verify.** It holds only if both loops execute the identical instruction sequence
   (`-use_fast_math`, FMA contraction). Mitigation: a single `gdn_step` with pinned intrinsics and the bitwise unit
   test in §6.2. Any mismatch fails the test instead of drifting silently. This is NInfer doc §4, L311-436.
2. **Graph-order WAR hazards.**
   - Within an op there is no hazard: phase 1 and phase 2 are separate kernels.
   - The extra-row record copy in `build_rs` happens before the layer's rr-op in graph order, the same assumption
     the existing extra-state copy and the KV `set_rows`→view reads already make.
   - `GGML_CUDA_GRAPH_OPT=1` concurrent streams (`ggml-cuda.cu:4621-4624`) must be checked not to move the rr-op or
     the record copies off the main stream. The alternative is to disable that optimisation in replay mode.
3. **The rr-op writes into a src** (`rec`). If the scheduler ever placed the op on a different backend than the
   arena buffer, the writes would be lost. Mitigation: assert on the buffer type, and make `supports_op` gate the
   mode (§3.10).
4. **Stricter `seq_rm`.** It returns false when `r > rec_m` or the cell is shared, and
   `common_context_seq_rm` aborts on false (`common/common.cpp:1577-1581`). The server acceptance path always rolls
   back ≤ T−1 < K' and is safe. Today the same situation silently reads a stale plane (1c.3). Other callers still
   need an audit: `server-context.cpp:3455` (prompt-cache reuse) and `:2962` (context shift, a middle removal that
   fails for recurrent memory today too).
5. **Spec vs non-spec token identity is not guaranteed by llama.cpp.** Verify runs T-token GEMMs and attention while
   plain decode runs 1-token kernels, so logits may differ in the last ulps and flip greedy near-ties. This happens
   independently of ReplaySSM. [INFERENCE: not measured on this model.] §6.4 defines how to tell this apart from a
   ReplaySSM bug.
6. **Record traffic in plain decode with `n_rs_seq>0`** is 3.4 MiB/token/seq of extra writes. That is negligible
   next to the 144 MiB state read, but visible in microbenchmarks.
7. **State file format change** (§3.8) invalidates existing prompt-cache files.
8. **Multi-backend.** Metal, Vulkan and other backends keep snapshot mode until they implement the two ops. There
   is no silent fallback to CPU placement because of risk 3.

---

## 5. Step plan (ordered commits)

1. **ggml: ops + CPU.**
   - Add `GGML_OP_GATED_DELTA_NET_RR` and `GGML_OP_SSM_CONV_FOLD`, their constructors, op names, and CPU forward
     functions with a barrier between the two phases.
   - Factor the CPU transition into one inline function.
   - Add `test-backend-ops` cases: `S_v=128`, `H_k=16/H_v=48` broadcast and `H_k=H_v`; n_tokens ∈ {1,4,8,20};
     n_seqs ∈ {1,6}; K ∈ {4,8}; per-sequence m ∈ {0,1,K/2,K}; permuted and swapped source rows. The harness compares
     outputs, committed state and written records against CPU.
2. **ggml-cuda: kernels.**
   - Implement `gdn_step` with pinned intrinsics, shared by the existing K≥1 kernel and the new replay kernel.
   - Add the record copy kernel and the conv-fold kernel, `supports_op`, and extend `try_gdn_cache_fusion` to the rr
     op.
   - Add `tests/test-gdn-replay.cpp` (bitwise, §6.2).
3. **llama: mode + memory.**
   - `llama_rs_mode` resolution in `llama_context` (§3.10) and `n_embd_rec()`.
   - `llama_memory_recurrent`: `rec_l`/`recx_l` allocation, `n_rows = mem_size` in replay, `rec_m`,
     `rs_fold_in` capture in `find_slot`, save/restore in `prepare`, the new `seq_rm` logic, and
     clear/keep/cp handling.
   - `llama_memory_recurrent_context` accessors `get_rec_l`, `get_recx_l`, `rs_fold(i)`.
4. **llama: graph.**
   - `rs_fold` input, `set_input`/`can_reuse` changes, extra-row record copy.
   - `delta-net-base` replay branches in `build_conv_state` and `build_recurrent_attn`; no q/k repeat in replay for
     qwen35, qwen35moe and qwen3next; add QWEN3NEXT to the rollback arch list.
   - In replay mode, remove `rs_idx` use from `s_copy`.
5. **State I/O.** Versioned recurrent state format carrying pending records (§3.8).
6. **Tests + docs.**
   - Extend `tests/test-recurrent-state-rollback.cpp` (§6.3).
   - Update the `n_rs_seq` comments at `include/llama.h:370` and `src/llama-memory-recurrent.h:73-77`, plus the
     `[TAG_RECURRENT_ROLLBACK_SPLITS]` notes.
   - Measure VRAM (`llama_memory_recurrent` log at `llama-memory-recurrent.cpp:133-137`) and tok/s with MTP n=3 and
     DFlash n=7 at 6 slots.
7. **Optional.** Batch the 48 per-layer rr-op launches through a CUDA fusion; add per-row `valid_columns` masking
   for mixed draft lengths.

---

## 6. Test strategy

### 6.1 `test-backend-ops` (cross-backend correctness)
These are the cases from commit 1. The usual NMSE comparison against CPU covers every output, the committed state,
and the record arena after phase 2 (read back as an extra "output" by viewing `rec`).

### 6.2 Fold vs sequential recurrence, bitwise, same backend (`tests/test-gdn-replay.cpp`)
Run the following for each backend that supports the op, with random fp32 inputs including tiny key norms, extreme
g (±20) and beta ∈ {0,1}:

1. **Reference.** `ggml_gated_delta_net(K=T)` on T tokens from S0 gives snapshots. The state after i tokens is
   slot `T−i`.
2. **Record.** The rr-op on the same T tokens with `rs={0, row}` and `K=T`. Assert that committed == S0 bitwise
   (since `j=0`) and that the scores equal the reference scores bitwise.
3. **Fold.** For every m in 0..T, run the rr-op with 1 new token and `rs={m,row}`.
   - Assert `memcmp(committed, snapshot_slot(T−m)) == 0`; for m=0 compare against S0.
   - Assert the new token's score equals `ggml_gated_delta_net(K=1)` run from that snapshot, bitwise.
   - Repeat for the conv fold against `conv_input` column windows.
4. **Chain.** 50 rounds with random T ≤ K and m ≤ T, compared bitwise against a single long `ggml_gated_delta_net`
   run over only the accepted inputs.
5. **Isolation.** Sources and inactive rows are unchanged. Mutating the rejected-suffix records does not change
   the result (the NInfer test list, §2.5).

### 6.3 llama-level rollback (`tests/test-recurrent-state-rollback.cpp`)
The existing multi-sequence cases (`:107-303`) must pass in replay mode. New cases:
- **A/B.** Compare replay against `LLAMA_RS_ROLLBACK=snapshot` with identical ubatch shapes. Require bitwise-equal
  logits, a stricter bar than the current NMSE 1e-5 at `:199-242`.
- **Mixed rollback.** Six sequences with different r in one ubatch, with `kv_unified` both true and false.
- **Composed rollback.** Two stacked partial `seq_rm` calls succeed and match a single rollback of the combined
  depth.
- **Invalid rollback.** `seq_rm` beyond `rec_m` returns false.
- **Save/restore.** Save and restore with pending records, then check the continuation is bitwise equal to the
  original context.

### 6.4 End-to-end exactness (acceptance gate)

Use greedy sampling (temp 0, top-k 1) on Qwen3.8-27B, 6 parallel slots, `llama-server` or `llama-cli`, over a fixed
set of 50 prompts × 512 generated tokens. Run:

- (a) non-speculative, `n_rs_seq=0`, fused GDN on;
- (b) MTP n_draft=3;
- (c) DFlash2 n_draft=3 and n_draft=7;

each in replay and snapshot mode.

Pass criteria:
1. Replay and snapshot mode produce identical token streams. This gate is hard.
2. Each speculative run matches (a) token for token.
3. If criterion 2 fails at some position, the divergence must also occur in snapshot mode, and the top-2 logit
   margin at that position in run (a) must be below 1e-3 relative. That identifies a pre-existing batch-shape
   near-tie (risk 5) rather than a ReplaySSM bug. Any divergence present only in replay mode fails the gate.

---

## Appendix: record/fold math as implemented here

For layer ℓ, value head h, qk head $h_q = h \bmod H_k$ (llama broadcast rule, `gated_delta_net.cu:37`), and
column c:

- Verify input t is $x_t = (\hat k_t\in\mathbb R^{128}, v_t, g_t, \beta_t)$, where $\hat k_t$ is the L2-normalised
  fp32 key produced by the graph.
- Record $R_t = x_t$ (fp32 bit copy), plus the conv column $p_t\in\mathbb R^{10240}$ (fp32 `qkv_mixed`).
- Step (column c, warp-parallel over rows i):
  $\kappa=\mathrm{wsum}_i(S_{ic}\hat k_i)$, $\delta_c=\big(v_c-\mathrm{fl}(e^{g})\kappa\big)\beta$,
  $S_{ic}\leftarrow\mathrm{fma}(\hat k_i,\delta_c,\mathrm{fl}(e^{g})S_{ic})$.
  The same `gdn_step` is used in both the replay and the verify loops.
- Commit after the next graph's prologue is $S_c = \mathrm{step}^{m}(S_c^{prev}; R_{0..m-1})$ and
  $H_c=\mathrm{tail}_3(H_c^{prev}\Vert p_{0..m-1})$, where $m=\text{rec\_m}$ after all `seq_rm` calls. The tail is a
  pure gather, so it is exact.
