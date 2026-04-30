# Phase 8 — Hybrid prefix cache for Qwen3.5/3.6 GDN models on Orin AGX (serving TTFT)

**Date opened:** 2026-04-29
**Predecessor context:** [phase4-perf.md](phase4-perf.md), [phase7-mxfp4-kv-cache.md](phase7-mxfp4-kv-cache.md). Phase 4B shipped MTP self-spec including `set_with_history` per-position rnn_state capture (only used by spec-verify today). Phases 5/6/7 closed out KV-format experiments with the conclusion that **single-request decode on 35B is BW-bound at 81% of Orin peak — no further wall-clock lever there**. Phase 8 shifts focus to **multi-request serving**, where the real lever is **TTFT savings on shared prompts**.

**Goal:** Make MLC's radix prefix cache work on hybrid GDN+attention models (Qwen3.5-0.8B, Qwen3.6-35B-A3B). Today it's globally disabled at [bench_mlc.py:101](../../bench_mlc.py#L101) and similar entry points because the engine's PopN-on-prefix-match path can't roll the GDN recurrent state back to a multi-token prefill point. Wire up rnn_state checkpointing at radix-tree node boundaries so cross-request prefix matching skips re-prefill of the shared portion.

**Headline projection.** At PP=512 prefill rate ~210 tps on 35B, a 512-token shared system prompt costs **~2.4 s of redundant TTFT per request**. Multi-user chat workloads typically share 80–95 % of the prefill across requests. Realistic deployment win: **5–20× reduction in TTFT** on the cached portion. Decode tps unchanged.

---

## Why this is the right next phase

After Phase 7:

- Single-request **decode** is at 1.79× llama.cpp Q4_K_S on the 35B (52.6 tps). Decode is BW-bound at 81% peak; further wall-clock work needs smaller weights or kernel-level BW improvements (no easy lever).
- Single-request **prefill** trails llama.cpp by ~3–4× (210 vs 583 tps PP=512). Real but separate kernel-tuning problem; not what blocks deployments.
- Multi-request **serving** has prefix-cache disabled, so every request re-prefills the entire prompt. **This is the lever where MLC actually leaves measurable wall-clock on the floor.**

The lever is well-scoped (engine + rnn_state plumbing, not kernel work), the machinery already exists (`set_with_history`, `RollbackVerifyAppend`), and the win lands in a deployment-relevant metric (TTFT) rather than a synthetic decode-tps that's already saturated.

---

## What's structurally broken today

`MLCEngine` with `prefix_cache_mode="radix"` (the default) does the following per request:

1. Tokenize the new prompt.
2. Walk the radix tree to find the longest cached prefix.
3. For the matching portion: reuse cached **PagedKVCache** pages (full-attn layers) — **works**.
4. For the matching portion: roll **RNNState** back to the match boundary — **fails on hybrid models.**
5. Re-prefill only the unmatched suffix.

Step 4 fails because `RNNState.PopN(n)` requires `n <= available_history_num`, and after a multi-token prefill, [rnn_state.cc:273-275](../../3rdparty/tvm/src/runtime/vm/rnn_state.cc#L273-L275) sets `available_history_num = 0`:

```cpp
} else if (seq_length > 1) {
  // We cannot rollback the prefill input
  it->second.available_history_num = 0;
  it->second.history_slot_id = (it->second.history_slot_id + 1) % max_history_;
}
```

**The intermediate per-token rnn_states inside a prefill are not stored.** Only the post-prefill state lives in the circular history buffer.

So when the engine looks for "rnn_state at token position 384 of a 512-token cached prefix", there's no slot to restore from. The engine either crashes or silently returns the wrong state — both unsafe. Hence the global disable.

`set_with_history` (added in Phase 4B for spec-verify) is the missing link: it scatters per-position rnn_state into history slots `[H+1..H+seq_len]`. Spec-verify uses it for the verify step (γ ≤ 4 tokens). The same machinery, with bigger `max_history`, would let prefill snapshots survive across requests.

---

## Code reuse — what's already in tree

- **`set_with_history` PackedFunc** ([rnn_state.cc:323-336](../../3rdparty/tvm/src/runtime/vm/rnn_state.cc#L323-L336)) — already wired through TVM runtime + the `f_sets_with_history_` array.
- **`set_with_history` TIR kernel codegen** ([rnn_state.py:323-404](../../python/mlc_llm/nn/rnn_state.py#L323-L404)) — handles per-position scatter into circular history slots.
- **`gdn_func_history_kernel`** ([qwen35_model.py:394+](../../python/mlc_llm/model/qwen35/qwen35_model.py#L394)) — emits the full `(b, s, n_vh, K, V)` per-position state needed for `set_with_history`. Currently called only from `forward_with_history` in spec verify.
- **`RollbackVerifyAppend`** ([rnn_state.cc:441-456](../../3rdparty/tvm/src/runtime/vm/rnn_state.cc#L441-L456)) — single multi-token rollback API.
- **Radix prefix cache** ([cpp/serve/](../../cpp/serve/)) — works fine for full-attn KV. The infrastructure (tree walk, cache hit, partial reuse) is solid.
- **`max_history` parameter** on `RNNState.create()` — currently set per-model, typically 2 (1 current + 1 prior for single-step rollback). Trivially configurable.

**Net new work is engine-side plumbing to (a) drive the prefill through `forward_with_history`, (b) request `max_history` large enough to survive cache eviction, and (c) restore rnn_state on cache hit by indexing the history slot at the matched token position.**

---

## Memory budget — the gating constraint

GDN state per layer per token, fp32:
- 35B-A3B: 32 value heads × 128 K × 128 V × 4 B = **2 MiB / layer / token**
- 0.8B: 32 value heads × 128 × 128 × 4 B = 2 MiB / layer / token (same per-layer footprint)

Per token across all GDN layers:
- 35B: 30 GDN layers × 2 MiB = **60 MiB / token**
- 0.8B: 18 GDN layers × 2 MiB = 36 MiB / token

For a 512-token cached prefix, full per-position snapshots:
- 35B: 30 GiB. **Infeasible** in 64 GB Orin shared memory.
- 0.8B: 18 GiB. Tight but feasible.

For 4096-token cached prefix:
- 35B: 240 GiB. Definitely no.
- 0.8B: 144 GiB. No.

**Per-token snapshots don't scale.** We need a sparser strategy.

### Three viable approaches

**(A) Node-boundary snapshots (recommended).** The radix tree only ever cuts at node boundaries — when two requests' prompts diverge. The set of *cacheable rnn_state checkpoints* equals the set of distinct radix nodes, **not** the set of token positions. For typical chat traffic (one shared system prompt + diverging user messages), the node count is O(unique_user_msgs), not O(total_tokens). Memory cost = `num_nodes × 60 MiB` on 35B. With 100 distinct cached prefixes, that's 6 GiB — workable.

**(B) Coarse checkpoints.** Snapshot every K tokens (e.g., K=64). Prefix matches only succeed at K-aligned boundaries — anything between checkpoints requires re-prefill from the prior checkpoint. K=64 → 32× memory savings vs per-token. 35B at 512-token prefix = 1 GiB. Trades partial-cache-hit fidelity for memory.

**(C) Quantized snapshots.** Store rnn_state in fp16 instead of fp32. Halves memory (35B per-token = 30 MiB). Still needs (A) or (B) to fit. Adds quant noise; probably acceptable since rnn_state already operates on fp16-cast outputs at the kernel boundary.

**Combining (A) + (C) is the practical landing point.** Snapshot at radix-tree node boundaries in fp16. 35B with 100 cached prefixes = **3 GiB total** for rnn_state checkpoints. Realistic.

PagedKVCache prefix cache is already shape-aware and copy-efficient via `_copy_single_page` / `_compact_kv_copy` (from Phase 6). RNNState's checkpoint copies can mirror that pattern (small TIR kernel for fp32 → fp16 + page).

---

## Stages and land criteria

### Stage 8.1 — Engine wiring: drive prefill through `forward_with_history` (1 session)

Today, prefill calls `forward()` which uses `set()` (single-slot post-prefill write). Spec-verify uses `forward_with_history()` which uses `set_with_history()`. We need a *third path*: prefix-cacheable prefill, which uses `set_with_history()` with a sufficiently sized `max_history`.

Tasks:
- Add a new prefill entry point `batch_prefill_with_history` (analogous to `batch_verify_to_last_hidden_states`) that uses `forward_with_history`. Or — simpler — add a `cache_prefill: bool` flag on the existing prefill that toggles which `forward_*` is called.
- Configure `max_history >= max_prefix_cache_seqlen + 1` at engine init. Probably 4096 or 8192.
- Smoke: prefill 512 tokens, then `RollbackVerifyAppend(seq_id, 256)`, then continue prefilling — should produce same output as a fresh 256-token prefill.

**Land criterion:** smoke test passes; no parity regression on the existing decode + spec-verify paths.

### Stage 8.2 — Radix tree node boundary checkpoints (1–2 sessions)

The existing radix prefix cache (in [cpp/serve/](../../cpp/serve/)) tracks PagedKVCache page references at each tree node. Extend it to also track an `rnn_state_history_slot` for each GDN layer at each node.

Tasks:
- At node creation: snapshot `RNNState.history_slot_id` (mod `max_history`) for the just-prefilled span.
- At node hit: call `RollbackVerifyAppend(seq_id, total_tokens_after_match)` to restore rnn_state to the match boundary. Or use the saved `history_slot_id` to set up a new `Sequence` whose `history_slot_id` points at the cached snapshot.
- Eviction: when a tree node is evicted from the radix cache (LRU), the saved history slot doesn't need explicit cleanup — slots cycle naturally as new sequences arrive. But the bookkeeping needs to handle the case where the slot has been overwritten.

**Land criterion:** two requests sharing a 256-token prefix → second request's TTFT measured to be < 50 ms (vs ~1.2 s without prefix cache). Output coherent and identical to fresh-prefill on a parity-controlled prompt.

### Stage 8.3 — fp16 snapshot quantization (0.5 session)

State storage today is fp32 to match the kernel's accumulator dtype. For *checkpoint* slots (those used for cross-request reuse), store fp16. The active `state_in_buf` fed to `gdn_func` continues to be fp32 (cast on read).

Tasks:
- Add an fp16 view to `RNNStateImpObj::storages_` for checkpoint slots. Or make `init_layer_value` dtype configurable per state_id.
- Modify `set_with_history` to cast on write; `get` to cast on read.
- Re-bench parity vs fp32-snapshot baseline.

**Land criterion:** memory halved, parity ≥ 4/5 prompts EXACT after cache-hit prefill.

### Stage 8.4 — Memory budget + eviction (0.5 session)

Per-deployment configuration:
- 35B: 100 cached prefixes × 30 layers × 32 heads × 128 × 128 fp16 = 3 GiB. Default `max_cached_prefixes=100`.
- 0.8B: 1000 cached prefixes × 18 layers = 9 GiB. Default `max_cached_prefixes=500`.

Tasks:
- Engine flag `--rnn-state-prefix-cache-budget-mb` (or auto-derived from total memory).
- LRU eviction on cache budget exceeded.
- Telemetry: cache-hit rate, evictions per minute.

**Land criterion:** synthetic load test (100 distinct system prompts, 500 user messages each) shows expected hit rate (>80 %) and bounded memory.

### Stage 8.5 — End-to-end serving bench (0.5 session)

A synthetic "shared prompt + diverging continuation" workload to measure the realistic deployment win.

Tasks:
- Bench harness that submits N=100 requests with shared 512-token system prompt and 32-token user-msg suffix; measures TTFT distribution and total throughput.
- Compare prefix-cache-on vs off, both with hybrid models.

**Land criterion:** TTFT p50 reduction ≥ 5× on shared-prompt traffic. Decode tps unchanged (within ±2%).

---

## Risk register

1. **Radix-cache eviction races.** Tree node deleted while a request's history_slot_id still points at it. Fix: ref-count per-slot, defer eviction until refcount=0. Standard pattern; existing Paged KV cache solves this.
2. **State drift across fp16 round-trip.** rnn_state is the recurrent accumulator; small noise compounds across positions. Mitigation: keep the *active* state buffer fp32, only cast at checkpoint boundaries. Quant noise applies only to the cached prefix — same noise budget as a single position's state, not 512 positions.
3. **`max_history` blow-up at compile time.** The TIR `set_with_history` kernel uses `max_history` in its modulo arithmetic. If we bump `max_history` from 2 → 4096, kernel codegen + cudagraph capture might balk. Worth pre-checking with a smoke compile before committing.
4. **Spec-verify interaction.** Phase 4B's `set_with_history` and our new "prefix-cache history" reuse the same `f_sets_with_history_` PackedFunc and same `max_history`. If a request alternates between spec-verify and cache-hit prefill, the history slots must not collide. Likely fine because both use `(history_slot_id + 1 + t) mod max_history` — but worth proving with a regression test.
5. **PagedKVCache + rnn_state checkpoint atomicity.** The radix cache assumes prefix matches restore *both* the KV cache pages AND the rnn_state to a consistent point. If only one of the two is correctly checkpointed, decode silently produces wrong output. Hard to test for absent careful parity comparison.

---

## Out of scope

- Single-request decode tps (Phases 4/5/6/7 closed that line on Orin).
- Single-request prefill tps (separate kernel-tuning problem; not where serving deployment leaves money).
- Distributed prefix cache across nodes (single-Orin scope).

## Land-criteria summary

| Gate | Metric | Bar |
|---|---|---|
| 1 (smoke) | RollbackVerifyAppend round-trip | Output identical to fresh prefill |
| 2 (TTFT) | Two-request shared-prompt TTFT | ≥ 5× reduction on cached portion |
| 3 (parity) | Cache-hit greedy decode vs fresh | ≥ 4/5 prompts EXACT |
| 4 (memory) | Per-cached-prefix RAM | < 60 MB on 35B (with fp16 + node-boundary) |
| 5 (deployment) | 100-request shared-prompt sweep | TTFT p50 ≥ 5× faster, decode tps within ±2% |

If gates 1–3 land, ship as opt-in (`prefix_cache_mode="radix"` works for hybrid models). Gates 4–5 are the polish for production deployment.

If gate 1 fails — meaning `set_with_history` + larger `max_history` interacts badly with the existing forward path — fall back to **node-boundary-only snapshots** (Stage 8.2 alone, no per-token capture), accepting that prefix matches can only succeed at node boundaries. Still useful for shared-system-prompt traffic.
