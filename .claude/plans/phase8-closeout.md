# Phase 8 close-out — open items before declaring done

**Date opened:** 2026-04-30
**Parent plan:** [phase8-hybrid-prefix-cache.md](phase8-hybrid-prefix-cache.md)
**Status of parent:** Stages 8.1 + 8.2 shipped. 0.8B (TTFT 109.7 → 15.5 ms, 7.1×) and 35B (TTFT 1410 → 79.8 ms, 17.7×) both pass round-trip parity smoke with prefix cache active. Code is functionally correct on the tested path; this doc captures the *residual* work between "smoke green" and "shippable without surprises."

The work splits into **real gaps** (genuinely missing — risk of silent regression or untested adjacent path) and **nice-to-haves** (refinements; defer until a concrete deployment asks).

---

## Real gaps — close these before declaring done

### 1. Memory estimator under-reports hybrid + prefix-cache by ~7.5 GB

**Where:** [cpp/serve/config.cc:899](../../cpp/serve/config.cc#L899) — `InferForKVCache` reports `Estimated total single GPU memory usage` but doesn't add the rnn_state buffer that's allocated when `kv_state_kind == kHybrid && prefix_cache_mode != kDisable`. The `InferForRNNState` path *does* compute `rnn_state_base_bytes` (lines 1005-1006); the same calc needs to fire in the hybrid branch.

**Why:** We hit it during the 35B smoke as a near-OOM. The estimator said "32904 MB used"; actual peak with `max_history=64` was ~40 GB. Orin's 64 GB headroom carried us; a tighter system (or a future user who reads the printed estimate and provisions accordingly) would silent-OOM at engine load.

**Effort:** ~30 min. Add the rnn_state byte calc inside `InferForKVCache` when `any_hybrid && prefix_cache_mode != kDisable`, pulling the per-layer/per-slot state size from model_config (vh × K × V × 4 + conv state). Multiply by `(max_num_sequence + prefix_cache_max_num_recycling_seqs) * max_history_size`. Add to the `Estimated total` line and to the `MemUsageEstimationResult.total_memory_bytes`. Verify the printed number matches `nvidia-smi`-equivalent on Orin within ~5 %.

**Land criterion:** Estimator's printed total covers the actual rnn_state allocation. Smoke output on 35B with `--mode on` shows ~40 GB total instead of ~32 GB.

---

### 2. MTP self-spec + prefix cache untested on 35B

**Where:** Production 35B runs use MTP self-spec ([dist/qwen3_6-35B-A3B-q4f16_1-mtp-draft/lib.so](../../dist/qwen3_6-35B-A3B-q4f16_1-mtp-draft/lib.so)). With prefix cache on, the engine arms `set_use_history_mode(true)` for the prefill (Phase 8) **and** for the verify forward (Phase 4B). Both write to the same `f_sets_with_history_` PackedFunc on the same sequence.

**Why:** This is the actual production workload. If there's a history-slot interaction bug between cache-prefill and spec-verify on the same sequence, we won't find it from the current smoke (which uses target-only generation).

**Risk model:**
- Cache-prefill writes per-position state to slots `[H+1..H+seq_len]` and advances `history_slot_id` by `seq_len`.
- Subsequent decode (single-token) advances `history_slot_id` by 1, growing `available_history_num` by 1 each.
- Spec-verify on a 4-token speculative chain uses `set_with_history` over slots `[H+1..H+4]`, advancing by 4.
- After verify, partial accept of (say) 2 tokens calls `PopNFromRNNStateOnly(seq, 2)` to roll back the 2 unaccepted tokens.
- If `available_history_num` is correctly tracked through this entire dance, it works. The plan's risk register #4 flagged this as worth proving with a regression test.

**Effort:** ~1 session.
- Smoke: target_only-with-prefix-cache req-A then spec(γ=1)-with-prefix-cache req-B sharing 256-token prefix. Expect req-B's accept_rate to be in the same ~72 % range as the 04-29 worklog's baseline (no prefix cache), and decode output to be parity vs target_only.
- Use [scratch_phase8_stage2_smoke.py](../../scratch_phase8_stage2_smoke.py) as a starting point; add MTP via `additional_models=[mtp_draft_lib_path]` on the EngineConfig and `speculative_mode="eagle"` (MTP uses the EAGLE engine path).
- Bonus: run with `--shared-len 1024` so the verify gets a long enough decode window to actually exercise multiple verify steps after the cache hit.

**Land criterion:** MTP+prefix-cache produces parity output vs target_only+prefix-cache, and accept rate stays within ±3 % of pre-Phase-8 baseline.

---

### 3. EAGLE engine path wired but never exercised

**Where:** [cpp/serve/engine_actions/eagle_new_request_prefill.cc](../../cpp/serve/engine_actions/eagle_new_request_prefill.cc) — Stage 8.2 added `cache_prefill` flag plumbing and post-`ForkSequence` `PopNFromRNNStateOnly` in the EAGLE-shift-by-1 fork branch. None of it has been run.

**Why:** Lower risk than #2 because EAGLE isn't the active spec-decode mode in production (the 35B uses MTP). But it's claimed-working code that's never been validated. The shift-by-1 trick documented in eagle_new_request_prefill.cc is subtle; the PopN amount calc `forked_parent_seq_length - (prefilled_offset - 1)` could be off-by-one in edge cases.

**Effort:** ~30 min if a small EAGLE-mode model lib exists in `dist/`; ~1-2 sessions if we have to compile one. Likely sufficient to share most of #2's harness with `speculative_mode="eagle"` and the MTP draft swapped for an EAGLE draft.

**Alternative:** Note explicitly in the worklog that EAGLE+prefix-cache is unverified, and gate the runtime path on a config flag if anyone enables it.

**Land criterion:** Either a smoke test runs through the EAGLE+cache_prefill path with parity output, or the runtime asserts/warns when entered.

---

### 4. Backward-compat check on pre-Phase-8 libs

**Where:** [dist/qwen3_5-0.8B-q4f16_2/lib.so](../../dist/qwen3_5-0.8B-q4f16_2/lib_pre_phase7.so.bak), [dist/qwen3_6-35B-A3B-q4f16_1/lib_pre_phase8.so.bak](../../dist/qwen3_6-35B-A3B-q4f16_1/lib_pre_phase8.so.bak), and any other `.bak` libs we want to remain loadable.

**Why:** Phase 8 changed `Model::BatchPrefill` and `Model::BatchPrefillToLastHidden` ABI (added `cache_prefill = false` default arg) and added `IsCachePrefillSupported()` virtual. These are vtable-positional methods; if a pre-Phase-8 lib's metadata doesn't expose `batch_prefill_with_history`, the new function-table init returns null, the new dispatch path safely falls through to the standard prefill, and the old lib should load+run unchanged. **Should**, but unverified — the Phase 6 worklog records a binary-compat regression that exact same pattern caused.

**Effort:** ~10 min. Use [scratch_phase7_followup_smoke.py](../../scratch_phase7_followup_smoke.py) as the template — it boots an engine on a backed-up lib and runs 1 prompt × 10 tokens. Run it against:
- `lib_pre_phase8.so.bak` (35B fp16 saved on 2026-04-30 06:50)
- Any other backup lib in `dist/*/lib_pre_*.so.bak` we care about.

**Land criterion:** Each backup lib loads, generates coherent text in 10 tokens, and produces output identical to (or at least string-equal-to) what it produced pre-rebuild.

---

## Nice-to-have — defer unless deployment asks

### 5. Longer-prefix TTFT sweep on 35B

The 256-token smoke showed 17.7× win. Theory says it scales linearly with prefix length. Confirm with `--shared-len {512, 1024, 2048}` and record a small table. Quick win for a concrete deployment-relevant headline; ~15 min of bench work assuming the smoke already exists.

### 6. `disagg_*.cc` consistency

Currently both [disagg_prepare_recv.cc](../../cpp/serve/engine_actions/disagg_prepare_recv.cc) and [disagg_remote_send.cc](../../cpp/serve/engine_actions/disagg_remote_send.cc) keep `cache_prefill=false` (engine default). Disaggregated serving on a hybrid model won't benefit from the prefix cache. Out of current scope (Orin single-GPU), but document so the next person in this code knows it's intentional.

### 7. `bench_mlc.py --prefix-cache=radix` steady-state regression bench

Confirm decode tps doesn't regress when prefix cache is on. The flag default is `disable` so existing benches in worklog.md are unchanged; a single sweep with `radix` would catch any silent decode-path slowdown introduced by the rnn_state buffer's larger memory footprint or the extra bookkeeping.

### 8. Build cascade investigation

Touching `cpp/serve/*.h` re-fires the heavy CUTLASS NVCC compiles (~12 min wall on Orin per iteration). Worth one focused session to figure out the spurious dep — likely a transitive `#include` chain through `metadata/model.h` → some TVM header that CUTLASS pulls in. Cosmetic but it'll keep biting on every Phase-8-adjacent iteration.

---

## Suggested attack order

1. **#4 (backward compat smoke, 10 min)** — cheapest, highest signal-to-noise, catches binary-ABI regressions before they hide.
2. **#1 (memory estimator fix, 30 min)** — only item with a code bug. Closing it makes the printed estimate trustworthy and avoids future silent OOM.
3. **#2 (MTP+prefix-cache, 1 session)** — the actual production path. The "real" Phase 8 win is in this combo.
4. **#3 (EAGLE smoke or runtime warning, 30 min)** — finish the "tested or guarded" promise.
5. Items 5-8 as time/deployment-need allows.

---

## Land-criteria summary for "Phase 8 closed"

| Gate | Status | Bar |
|---|---|---|
| 1 — 0.8B parity | ✓ shipped | TTFT 109.7 → 15.5 ms, decode bit-exact |
| 2 — 35B parity | ✓ shipped | TTFT 1410 → 79.8 ms, decode bit-exact |
| 3 — Memory estimator accurate | ⏳ #1 | printed total within 5 % of actual peak |
| 4 — MTP+prefix-cache parity | ⏳ #2 | output matches target_only, accept ±3 % of baseline |
| 5 — EAGLE+prefix-cache | ⏳ #3 | smoke green or runtime guard in place |
| 6 — pre-Phase-8 lib regression | ⏳ #4 | backup libs still load + generate |

When 3-6 land, Phase 8 closes for real.
