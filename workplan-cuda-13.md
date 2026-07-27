# Workplan: Qwen3.6-35B-A3B on JetPack 7.2 / CUDA 13.2

**Sessions:** 2026-07-24 (Stage 0/1), 2026-07-25a (kernel attribution, re-scope, options 1 and 2
landed, concurrent serving fixed), 2026-07-25b (conv-state fusion, §9 item 3 refuted, everything
committed), 2026-07-25c (history-path conv fusion — §14), 2026-07-25d (history-path *recurrent*
fusion — §15), 2026-07-26a (the 35B state gate, items 1 and 5 refuted by measurement — §16.1–§16.4),
2026-07-26b (the lane-split recurrence built, measured and gated — §16.5), 2026-07-26c (the MoE
GEMM measured and its padding CTAs skipped — §16.7–§16.11), 2026-07-26d (`BLK_K` 32→64 lands
+14.1% pp512; the real CTAs rooflined at 85–87% of the wall; item 0h built and parked; **the bench
prompt found to be picking winners** — §17)
**Status:** **35B-A3B tg512 54.13 → 60.00 (+10.8%)** from four landed changes: the GDN
input-projection merge (§10), the in-place recurrent state (§11), concurrent serving on hybrid
models (§12), and the in-place conv state (§13). Everything since — §14, §15 and §16.5 — is on the
**prefill** path and leaves decode where it is. **Prefill on the *default* `prefix_cache_mode=radix`
was never measured until 2026-07-25c, and it was getting barely half the headline number** — §14,
§15 and §16.5 close that gap and then some:

| pp512, `radix` (the default) | start of 2026-07-25c | after §15 | after §16.5 | after §16.11 | **now (§17)** |
|---|---:|---:|---:|---:|---:|
| 35B-A3B | 355 | 628 | 642 | 769 | **875 (+147% overall, +14.1% from §17)** |
| 0.8B | 1469 | 3912 | **4888 (+233% overall)** | — | — (no MoE; §17 cannot reach it) |

> ⚠️ **Every number in that row is a filler-prompt number and is ~4% optimistic** (§17.9). §18 measured
> the shipped lib on real prose at **827 tps** pp512 / **945** pp2048, and mapped a frontier of
> alternative `BLK_M` configurations reaching **995–1063 tps at pp2048** — §18.9. Defaults are
> unchanged; the choice is a workload judgement and the evidence for making it is now in one table.

§16.5 is the lane-split GDN recurrence; decode is neutral on both models (≤0.1%). The two models
differ by 10× on it because the kernel's grid is `(num_value_heads, batch)` — 16 blocks on the 0.8B,
32 on the 35B — so only the 0.8B was ever grid-starved. **This is the standing trap for anything
that tunes this kernel: a ratio measured on one model does not transfer to the other.**

Everything from §16.7 on is the MoE expert GEMM, which is the 35B's dominant prefill cost and which
the 0.8B does not have at all — so the two models' prefill numbers have diverged by construction
since §16.11, and a 0.8B A/B is blind to all of it.

**Everything through §16.11 is committed.** **Next session: start at §9's open list** — it is down to
item **0c.2** (chunked recurrence, de-prioritised at a +12.5% ceiling) plus the VL re-gate, which
is **not** blocked — the checkpoint is already local (§9's item entry corrects the old claim). §17 refuted both of the candidates §16.11 queued.
**Primary target:** Qwen3.6-35B-A3B · **Fast-iteration vehicle:** Qwen3.5-0.8B

> **Two traps that cost most of session 2026-07-25b. Read before benching or gating anything.**
> 1. **Compiling the 35B requires `MLC_MOE_GEMM_V2=1`** and nothing warns you — §8. Without it,
>    pp512 reads 225 instead of 645 and it looks like the model change did it.
> 2. **`prefix_cache_mode` decides which forward path prefill takes.** Under the default `radix`
>    prefill goes through `forward_with_history` (copy path); only `disable` uses the fused path.
>    A prefill gate run at the default tests nothing — §13.
>
> 3. **The bench harness could not measure the default configuration at all** — it hardcoded
>    `prefix_cache_mode="disable"`, *and* reused one prompt across runs so radix runs 2+ are pure
>    cache hits. Both fixed 2026-07-25c; §14.1 is the number that was hiding behind them.
>
> Generalised: **an A/B is only an A/B if the two libs differ by the change under test**, and **a
> benchmark only measures what its harness lets it configure.** Every "the change broke it" moment
> across these sessions was the harness.

> **Read §4.6 before §4.5.** The 2026-07-25 session re-derived the decode breakdown and three of the
> first pass's conclusions did not survive: GPU idle is **5.3%, not 13.3%**; `rnn_state_get/set`
> move **245 MiB/token**, not "~nothing"; and the two unidentified mid-size kernels turned out to be
> the 4096→2048 down-projection (75% of the wall) and `in_proj_z` (60%) — i.e. there *is* kernel
> headroom left. §5 is rewritten accordingly.

---

## 1. Why this exists

Everything in [qwen3_5.md](qwen3_5.md) §14 was measured on Ubuntu 22.04 / JetPack 6.2.2 /
**CUDA 12.6** / LLVM 15 between 2026-04-27 and 2026-04-30. The box was re-bootstrapped onto
JetPack 7.2 / CUDA 13.2 / LLVM 18. The question was: *what performance are we leaving on the
table in the new environment?*

**Answer: none in CUDA 13 itself.** The toolchain move is a wash. But re-measuring surfaced
where the real headroom is, and it is not where §14 says.

---

## 2. Environment (verified this session)

Hardware unchanged — AGX Orin Developer Kit, tegra234, module **p3701-0005 (64 GB)**, **sm_87**,
16 SMs, 12× Cortex-A78.

| | was | now |
|---|---|---|
| OS | Ubuntu 22.04 | Ubuntu 24.04.4 |
| JetPack | 6.2.2 | **7.2-b187** (L4T R39.2.0) |
| CUDA | 12.6 | **13.2** (nvcc V13.2.78) |
| Driver | — | 595.78 |
| LLVM | 15 | 18.1.3 |
| cuDNN / TRT | — | 9.20 / 10.16 |

**Power/clocks are already maxed — no headroom there.** `nvpmodel` is MAXN
(`/etc/nvpmodel/nvpmodel_p3701_0000.conf`, all 12 cores online, `TPC_PG_MASK 0`, every
`MAX_FREQ -1`). GPU ceiling 1300.5 MHz; EMC pinned at 3199 MHz = the full 204.8 GB/s.

**`jetson_clocks` DOES matter — worth ~1.4%.** The `nvhost_podgov` governor reaches 1300.5 MHz
under load, but coarse frequency sampling hides real throughput loss. Always run
`sudo nvpmodel -m 0 && sudo jetson_clocks` before benching. Unpinned tg512 measured 53.13 vs
54.13 pinned.

**`ncu` is blocked**: `/proc/driver/nvidia/params` has `RmProfilingAdminOnly: 1`, so hardware
counters need root. Not resolved this session. `nsys` tracing works unprivileged and was
sufficient. To unblock:
```bash
echo 'alfie ALL=(root) NOPASSWD: /usr/local/cuda/bin/ncu' | sudo tee /etc/sudoers.d/ncu-profiling
sudo chmod 440 /etc/sudoers.d/ncu-profiling
```

---

## 3. Stage 0 — the chain is green

The Thrust/CUB/CUTLASS drift risk flagged at [qwen3_5.md:121](qwen3_5.md#L121) **did not
materialize**. TVM built clean (646 ninja edits), bundled CUTLASS is 4.1.0 (CUDA-13-era).

Verified working: TVM → mlc-llm runtime (`compute_87` in `build.ninja`, links `libcudart.so.13`,
all deps resolved) → convert_weight → gen_config → compile with **FlashInfer live on sm_87**
(`create_flashinfer_paged_kv_cache` is emitted) → coherent generation.

TVM device detect is correct: `sm_87`, 16 SMs, 48 KB shared/block, 1024 max threads — the attrs
the dlight passes key on.

### Corrections to the documented recipe

- **`accelerate` is missing from the §2.1.1 pip list.** transformers 5.x needs it for
  `device_map`; `validate.py --reference-only` hard-fails without it. Installed 1.14.0.
- **`cutlass=1` is inert on sm_87.** [op/extern.py:43-47](python/mlc_llm/op/extern.py#L43-L47)
  gates it to `sm_90a`/`sm_100a`, and `CUTLASS.cmake:59,65` gates the CUDA sources to `90a`/`100a`,
  leaving `tvm_cutlass_objs` empty. [qwen3_5.md:226](qwen3_5.md#L226) still calls it a
  "long-standing Orin-tuned flag" — it does nothing. (`cublas_gemm=1` being a no-op is already
  documented at §2.3; `faster_transformer` is hard-disabled at `extern.py:48`.)
- **torch 2.11.0+cu130 lacks sm_87 in `get_arch_list()`** (`sm_80,90,100,110,120`) but every
  native kernel runs correctly via PTX JIT. It is a valid correctness oracle. Do not "fix" this.
- **`profile_decode.py` is hardcoded to the 0.8B** and uses `next(model_dir.glob("*.so"))` — the
  exact footgun §14.1 warns about. The 35B dir holds both `lib.so` and `lib_nofi.so`.
- **CUDA 13 landmine:** configuring mlc-llm without an explicit arch falls through
  `3rdparty/tvm/cmake/modules/CUDA.cmake:153-156` → `75;80;86;89;90`, and **sm_75 was removed in
  CUDA 13**. Always pass `-DCMAKE_CUDA_ARCHITECTURES=87`. Also `USE_NVTX OFF` is load-bearing
  (CUDA 13 dropped `libnvToolsExt`).

---

## 4. Stage 1 — measurements

### 4.1 CUDA 13.2 vs CUDA 12.6 (clocks pinned, §14.4 protocol, 3 runs + warmup)

**Qwen3.6-35B-A3B** (`dist/qwen3_6-35B-A3B-q4f16_1/lib.so`, MoE GEMM v2 + FlashInfer):

| tg | 12.6 tg_tps | **13.2 tg_tps** | delta | 13.2 pp_tps |
|---:|---:|---:|---:|---:|
| 512 | 54.46 | **54.13** | −0.6% | 566.33 |
| 1024 | 54.30 | **54.00** | −0.6% | 566.32 |
| 2048 | 54.07 | **53.83** | −0.4% | 566.32 |
| 4096 | 53.69 | **53.34** | −0.7% | 565.59 |
| 8192 | 53.00 | **52.68** | −0.6% | 565.65 |

pp512: 561.5 → **566.33 (+0.9%)**.

**Qwen3.5-0.8B** (`q4f16_g16e` + FlashInfer): tg512 134.82 → **133.47** (−1.0%), flat to
**128.16** at tg8192 (was 129.59). pp512 2870 → **2889 (+0.7%)**.

Run-to-run spread across engine loads is ~0.5%, so the decode delta is at the edge of noise.
**Verdict: parity.** Prefill marginally up, decode marginally down, depth-flat behaviour preserved.

Raw: `tuning/mlc_tg_sweep_35b_cuda13_20260724_210926.json`,
`tuning/mlc_tg_sweep_0.8b_cuda13_20260724_213627.json`.

### 4.2 nvcc 13.2 reproduces 12.6 codegen

`bench_moe_kernel.py` under CUDA 13.2: gate_up **1.002 ms**, down **0.948 ms**.
`baseline_moe_v0_cta1024.json` (CUDA 12.6, CTA_COUNT=1024): **1.0016 / 0.9495 ms**. Match within
0.1%.

⚠️ **Do not compare against `baseline_moe.json`** — that is the **CTA_COUNT=64** config
(0.4915 / 0.2613 ms). Current source has CTA_COUNT=1024, restored in Phase 9 Stage 9.2. Comparing
against the wrong baseline looks like a 2–3.6× regression and is not one.

Side observation worth a look someday: CTA=64 is 2–3.6× faster at B=8 while the two are close at
B=1024 (14.26 vs 14.68 ms). The Phase 9 restoration optimized prefill at a large small-batch cost.
Only reachable via the v1 `dequantize_group_gemm`, which the shipped 35B path does not use
(prefill goes v2, decode goes gemv) — so this only matters if spec-decode verify ever routes there
(see §9.9 pitfall).

### 4.3 Achievable bandwidth — the number the docs never had

A native sm_87 read-dominated kernel (128-bit vectorized, 2 GiB buffer, far beyond the 4 MB L2):

**156.0 GB/s achievable = 76.2% of the 204.8 GB/s spec peak.** Re-measured 2026-07-25 after
promoting the probe to [scripts/bw_probe.cu](scripts/bw_probe.cu): **156.2 GB/s (76.3%)** — the
figure is stable to ~0.1%.

Every roofline in [qwen3_5.md](qwen3_5.md) used the marketing 204.8 figure, which overstates
headroom by ~31%. Corrected there 2026-07-25 (§14 preamble).

### 4.4 The 35B roofline (§14.1 asserts "weight-BW bound" but computes none)

Exact active set per decode token, from safetensors headers — excludes the vision tower (446.6 M),
the MTP draft head (844.6 M) and the embed table; routed experts counted at top-8/256:

| bucket | full params | active/token |
|---|---:|---:|
| moe routed | 32,212,254,720 | 1,006,632,960 |
| gdn linear-attn (30 layers) | 1,011,553,920 | 1,011,553,920 |
| lm_head | 508,559,360 | 508,559,360 |
| full-attention (10 layers) | 272,634,880 | 272,634,880 |
| moe shared expert | 125,911,040 | 125,911,040 |
| moe router | 20,971,520 | 20,971,520 |
| **total / active** | **35,951,822,704** | **2,946,429,568** |

Total matches the converter's reported param count exactly. At 4.345 bits/param:

- **1.600 GB/token**
- Roofline @ 204.8 spec peak: **128.0 tps**
- Roofline @ **156.0 GB/s achievable: 97.5 tps**
- Measured 54.13 tps = 86.6 GB/s = **55.5% of the achievable wall**

### 4.5 Where the token budget actually goes

**Superseded by §4.6.** The first pass at this table (a) guessed the byte counts for two of the
six biggest kernels, (b) called `rnn_state_get/set` "moves ~nothing", and (c) derived GPU idle by
subtracting a *traced* kernel sum from an *untraced* bench wall-clock. All three were wrong. The
numbers below are kept only so the corrections in §4.6 are legible.

| kernel | inst | avg µs | ms/tok | achieved BW | % of 156 GB/s |
|---|---:|---:|---:|---:|---:|
| `moe_dequantize_gemv_kernel` | 5040 | 66.3 | 2.609 | 137.5 GB/s | **88%** |
| `fused_dequantize1_NT_matmul_kernel` (GDN qkv) | 3780 | 66.1 | 1.953 | 137.8 GB/s | **88%** |
| `fused_dequantize4_NT_matmul3_kernel` | 5040 | 38.8 | 1.529 | ~~unknown~~ | — |
| `moe_dequantize_gemv1_kernel` | 5040 | 38.5 | 1.516 | 118.3 GB/s | 76% |
| `fused_dequantize2_…_silu1_multiply1` | 3780 | 48.6 | 1.436 | ~~unknown~~ | — |
| `fused_dequantize7_NT_matmul8_kernel` | 1260 | 72.6 | 0.714 | — | — |
| **`rnn_state_get_0` + `rnn_state_set_0`** | 3840×2 | ~20 | **1.198** | ~~moves ~nothing~~ | — |
| `gdn_func_kernel` | 3840 | 61.2 | 0.505 | — | — |
| 39 smaller kernels | — | — | 3.644 | — | — |
| **sum of kernel time** | | | ~~16.016~~ | | |
| **GPU idle / launch gap** | | | ~~2.458~~ | | ~~13.3%~~ |

### 4.6 Corrected breakdown — every kernel mapped, idle measured in-trace

Same `q35_decode.nsys-rep`, re-analyzed by [scripts/analyze_decode_trace.py](scripts/analyze_decode_trace.py).
Three methodology fixes:

1. **Steps are cut on `parallel_sampling_from_prob_kernel`** (fires exactly once per engine step),
   and steps longer than 60 ms are classified as prefill. The trace holds 2 prefill + 125 decode
   steps. Without this split, prefill instances of a shared kernel name blend into the decode
   average — `depthwise_conv1d` is 4016 µs in prefill and 11 µs in decode under one name.
2. **Wall-clock and kernel time come from the same trace.** Traced decode wall is **19.122
   ms/token (52.29 tps)** — nsys `--cuda-graph-trace=node` costs ~3.5% vs the 18.474 ms / 54.13 tps
   bench. Percentages below are of the traced budget, which is internally consistent; the old
   13.3% idle figure was an artifact of mixing the two runs.
3. **Every kernel ≥0.09 ms/token is identified from launch geometry**, not guessed. The dlight GEMV
   schedule emits `block=(16,32,1)` with one CTA per 64 output elements, so `gridX × 64` is the
   output width; instances-per-token gives the layer multiplicity (40 = all layers, 30 = GDN, 10 =
   full-attention, 1 = once per token). Together those pin each kernel to exactly one projection.

| kernel | what it is | /tok | ms/tok | %budget | GB/s | % of 156 |
|---|---|---:|---:|---:|---:|---:|
| `moe_dequantize_gemv` | routed experts gate_up, top-8 (2048→2×512) | 40 | 2.650 | 13.9% | 137.5 | **88%** |
| `fused_dequantize1_NT_matmul` | GDN `in_proj_qkv` (2048→8192) | 30 | 1.984 | 10.4% | 137.8 | **88%** |
| `fused_dequantize_fused_NT_matmul9_cast4` | `lm_head` (2048→248320) | 1 | 1.769 | 9.3% | 156.1 | **100%** |
| `fused_dequantize4_NT_matmul3` | **GDN `out_proj` AND attn `o_proj`** (4096→2048, one shared kernel) | 40 | 1.553 | 8.1% | 117.4 | 75% |
| `moe_dequantize_gemv1` | routed experts down, top-8 (512→2048) | 40 | 1.540 | 8.1% | 118.4 | 76% |
| `fused_dequantize2_…_silu1_multiply1` | **GDN `in_proj_z` + `silu(z)*core_out`** (2048→4096) | 30 | 1.459 | 7.6% | 93.7 | 60% |
| `fused_dequantize7_NT_matmul8` | attn `c_attn` (2048→9216) | 10 | 0.726 | 3.8% | 141.3 | **91%** |
| `rnn_state_get_0` | GDN recurrent state get (2 MiB fp32 slot) | 30 | 0.612 | 3.2% | 205.6 | **132%** |
| `rnn_state_set_0` | GDN recurrent state set | 30 | 0.583 | 3.1% | 215.7 | **138%** |
| `gdn_func` | the recurrence itself | 30 | 0.569 | 3.0% | — | — |
| `fused_dequantize5_NT_matmul5` | shared expert gate_up (2048→2×512) | 40 | 0.478 | 2.5% | 95.3 | 61% |
| `NT_matmul4` | MoE router gate (2048→256, **fp16**) | 40 | 0.447 | 2.3% | 93.9 | 60% |
| `depthwise_conv1d1` | GDN causal conv1d | 30 | 0.332 | 1.7% | — | — |
| `fused_dequantize6_…_multiply5_add1` | shared expert down + gate·add (512→2048) | 40 | 0.330 | 1.7% | 69.0 | 44% |
| `fuse_add_norm_prefill` | residual add + RMSNorm | 80 | 0.310 | 1.6% | — | — |
| `top8_softmax` | router top-k | 40 | 0.268 | 1.4% | — | — |
| `fused_dequantize3_NT_matmul2` | GDN `in_proj_a` (2048→**32**) | 30 | 0.242 | 1.3% | 4.4 | 3% |
| `fused_dequantize3_…_sigmoid_cast2` | GDN `in_proj_b` + sigmoid (2048→**32**) | 30 | 0.232 | 1.2% | 4.6 | 3% |
| `rnn_state_get_1` / `set_1` | GDN conv state get/set (3×8192) | 30+30 | 0.294 | 1.5% | 15 / 29 | 10 / 19% |
| `BatchPrefillWithPagedKVCache` | FlashInfer full-attn decode | 10 | 0.189 | 1.0% | — | — |
| 32 smaller kernels | — | — | 1.351 | 7.1% | — | — |
| **sum of kernel time** | | | **18.106** | **94.7%** | | |
| **GPU idle / launch gap** | | | **1.016** | **5.3%** | | |

Roll-ups: identified weight GEMVs **13.409 ms (70.1%)**; `rnn_state` get/set (all four)
**1.489 ms (7.8%)**; `in_proj_a` + `in_proj_b` **0.474 ms (2.5%)**.

Three corrections that matter:

- **`rnn_state_get/set` are memory-bound, not launch-bound.** A `get` reads one 2 MiB state slot
  (`32 v-heads × 128 × 128` fp32) and writes a 2 MiB destination — 4 MiB per call, ×60 calls/token
  = **245 MiB/token**, 15% on top of the 1.600 GB of weights. At 205–216 GB/s they run *above* the
  156 GB/s DRAM wall, which is only possible because the 4 MiB working set fits the 4 MB L2 and the
  consumer reads it right back. This matches [worklog.md:2312](worklog.md#L2312) ("at peak DRAM BW …
  truly at ceiling") and contradicts §4.5. **Cudagraph capture cannot recover this 1.489 ms** — it
  is real traffic. Only eliminating the copies can.
- **GPU idle is 5.3%, not 13.3%.** Measured wall minus measured kernel time, same trace. This
  roughly halves the ceiling on any launch-overhead work.
- **The two unknown kernels are now known, and both are below the wall.** `fused_dequantize4_NT_matmul3`
  is the 4096→2048 down-projection *shared* between GDN `out_proj` and attention `o_proj` (identical
  shapes → dlight emits one kernel, hence 40 calls/token) at **75%**;
  `fused_dequantize2_…_silu1_multiply1` is `in_proj_z` fused with the GDN output gate at **60%**.
  The old guess had both at shared-expert size — off by 4× and 2×. Confirmed against
  [qwen35_model.py:560-566](python/mlc_llm/model/qwen35/qwen35_model.py#L560-L566), which declares
  `in_proj_qkv` / `in_proj_z` / `in_proj_a` / `in_proj_b` / `out_proj` as five separate `nn.Linear`s.

### 4.7 Cudagraph coverage (`--cuda-graph-trace=node`)

In steady-state decode, **131 launches/token stay eager**:

| eager in decode | launches/tok | ms/tok | why it matters |
|---|---:|---:|---|
| `rnn_state_get_0/1` + `set_0/1` | 120 | 1.489 | the whole eager population, effectively |
| `lm_head` | 1 | 1.769 | one 1.77 ms kernel — launch cost is noise |
| `BatchPrefillWithPagedKVCache` (FlashInfer) | 10 | 0.189 | external kernel, outside the captured region |

Everything else — `gdn_func`, the conv1d, all MoE and projection GEMVs — is inside a graph.
`op.topk`/thrust (the `cudaErrorStreamCaptureImplicit` breaker at [worklog.md:2529](worklog.md#L2529))
did **not** regress: `top8_softmax` is captured. So the 5.3% idle is ~131 launches, of which 120 are
the `rnn_state` wrappers. Capturing those is worth at most ~4% and realistically less.

---

## 5. The conclusion that changes the plan

The first version of this section said: the big kernels are done, the budget is overhead, go chase
launch gaps. **With every kernel now identified (§4.6), that is backwards.** Overhead is 5.3%, not
20%. The headroom is still in kernels — just not the ones that were already measured.

**Tier 1 — genuinely finished.** Four kernels sit at 88–100% of the 156 GB/s wall: routed-expert
gate_up (88%), GDN `in_proj_qkv` (88%), attention `c_attn` (91%), and `lm_head` (100.1% — exactly
at the wall). That is **7.13 ms/token, 37% of the budget**, with nothing left in it. Phase 2C's
"Pareto-optimal, no further tile gains" verdict holds for this tier under nvcc 13.2.

**Tier 2 — six kernels between 44% and 76%, and this is the real headroom.**

| kernel | ms/tok | now | at 88% | saving |
|---|---:|---:|---:|---:|
| GDN `out_proj` + attn `o_proj` (shared) | 1.553 | 75% | 1.328 | 0.225 |
| routed-expert down | 1.540 | 76% | 1.328 | 0.212 |
| GDN `in_proj_z` + output gate | 1.459 | 60% | 0.996 | 0.463 |
| shared-expert gate_up | 0.478 | 61% | 0.332 | 0.146 |
| MoE router (fp16) | 0.447 | 60% | 0.305 | 0.142 |
| shared-expert down | 0.330 | 44% | 0.165 | 0.165 |
| **total** | **5.807** | | **4.454** | **1.353** |

**Tier 3 — two structural inefficiencies that are not bandwidth at all.**

- **`in_proj_a` + `in_proj_b`: 0.474 ms/token (2.5%) to move 142 KB.** Both are `2048→32`, so the
  64-outputs-per-CTA GEMV schedule emits `grid=(1,1,1)` — **one CTA, 1/16 of the GPU**, twice per
  GDN layer. This is parallelism starvation, not bandwidth; at 4.4 GB/s they are not even close to
  what a single SM could pull.
- **`rnn_state_get/set`: 1.489 ms/token (7.8%) of real, unnecessary traffic.** 245 MiB/token copied
  slot→temp→slot around a recurrence that could read and write the slot directly. Running at
  205–216 GB/s (L2-assisted), so there is no tiling win here — the fix is to not do the copy.

### Recommended re-scope (revised)

Drop the knob re-sweep and the ptxas dials, as before — §4.2 shows the constants transferred to
nvcc 13.2 cleanly, and tier 1 has no room. But **do not redirect to launch-overhead work**: at 5.3%
total idle across 131 eager launches, that whole lane is worth ≤4% and option B below makes most of
it moot.

In expected-value order:

1. **Merge the four GDN input projections into one GEMV** — `in_proj_qkv` ∥ `in_proj_z` ∥
   `in_proj_a` ∥ `in_proj_b` → a single `2048→12352`. **IMPLEMENTED AND MEASURED — see §10.**
   Predicted ~0.87 ms/token (4.5% decode); **measured +2.8% decode and −1.1% prefill.** The
   prediction assumed the fused 12352-wide GEMV would reach the 137.8 GB/s its 8192-wide
   predecessor does, and it did not; it also ignored the prefill cost of losing the
   `silu(z)·core_out` fusion. Subsumes tier-3's `in_proj_a/b` problem and tier-2's `in_proj_z` row.
2. **Eliminate the recurrent-state copies** — up to **1.489 ms/token (7.8%)**, the single largest
   line item. **IMPLEMENTED AND MEASURED — see §11.** Landed the recurrent half (state 0,
   1.195 ms/token of the 1.489); **measured +6.0% decode on the 35B with prefill neutral**, i.e.
   93% of its estimate and no offsetting cost — the best-behaved prediction in this document.
   The conv half (state 1, 0.294 ms) is left for a follow-on.
   ⚠️ **The rest of this item is superseded twice — read the two bold correction blocks
   below before acting on any of it.** The mechanism it proposes is cudagraph-unsafe, and the
   obvious implementation corrupts the *default* configuration. The corrected design is at
   the end, and is what §11 built. Upstream TVM already flags exactly this:
   `// TODO(siyuan): support zero-copy when seq_len is one` at
   [rnn_state.cc:304](3rdparty/tvm/src/runtime/vm/rnn_state.cc#L304), and
   `GetStatePtrBySeqHistory` ([rnn_state.cc:482](3rdparty/tvm/src/runtime/vm/rnn_state.cc#L482))
   already constructs the exact zero-copy `DLTensor` view — the slice
   `(seq_slot, history_slot, …)` is contiguous. **Gated on pitfall §9.2**: aliasing the read and
   write of recurrent state is what broke SGLang #20791, so `gdn_func` must first be shown to touch
   each state element exactly once per step. For a single decode token the recurrence is
   elementwise in `S`, so this looks safe — but prove it before writing code, and keep the
   copy path for the multi-token / history-mode forwards.

   **Feasibility analysed 2026-07-25 — aliasing is SAFE, but the recommended mechanism is WRONG.**

   *The aliasing question is settled, favourably.* In `gdn_func`
   ([qwen35_model.py](python/mlc_llm/model/qwen35/qwen35_model.py)) thread `(b_idx, h_idx, col)` owns
   exactly one column of the state matrix: it reads `state_in_buf[b,h,row,col]` once per `row` on
   entry, holds the column in registers across every pass and every `t`, and writes
   `state_out_buf[b,h,row,col]` once per `row` on exit. Read set and write set are identical,
   per-element, and disjoint across threads — no thread ever reads state another thread wrote. That is
   a stronger guarantee than "elementwise". §9.2/SGLang #20791 was a *scheduler* introducing aliasing
   into a kernel with cross-thread state reads; this kernel structurally has none. (Caveat: the
   kernel declares `tirx.noalias: True`, so that attribute has to change with any aliasing.)

   *But "just return `GetStatePtrBySeqHistory`'s view" does not work.* The slot byte offset is
   `(seq_slot_id * max_history_ + history_slot_id) * state_size`, and `EndForward` advances
   `history_slot_id = (history_slot_id + 1) % max_history_` every step. With prefix caching off
   `max_history_` clamps to 1 and the address is stable — but Phase 8 sets `max_history_ = 64`
   ([config.cc](cpp/serve/config.cc)), and then the address **rotates every step**. Baking that
   pointer into a captured cudagraph reads the wrong slot silently. Since the point of this work is
   partly to get these 120 launches *into* the graph (§4.7), a raw-pointer view is the wrong mechanism.
   Two further obstacles: `get()` uses `call_dps_packed`, which by construction allocates a
   destination the builtin must fill, so zero-copy needs a different Relax op; and
   `tirx.noalias: True` on `gdn_func` would be violated.

   **Revised design — fuse the state access into `gdn_func` instead of aliasing buffers.** Pass the
   *whole* storage buffer plus the `seq_slot_ids` / `history_slot_ids` device tensors into the kernel
   and index inside it, exactly as the existing `rnn_state_get_0` kernel already does
   (`f_gets_[state_id](state, seq_slot_ids_view_, history_slot_ids_view_, o_data)`). Then:
   addresses are computed from device tensor data at runtime, so it is cudagraph-safe under any
   `max_history_`; both copy kernels disappear rather than being merely captured; and the per-element
   ownership proof above still applies unchanged. Cost: a new builtin to expose the storage tensor and
   the id views to the model, a TIR signature change, and the copy path retained for prefill and
   history mode. Requires rebuilding `libtvm.so` in `3rdparty/tvm/build` separately (§2.1.1).

   **CRITICAL correction — `get`/`set` are a ring-buffer advance, not a redundant copy.**
   [rnn_state.py](python/mlc_llm/nn/rnn_state.py) `create_get_func` reads
   `storage[seq_id, history_id, ...]` but `create_set_func` writes
   `storage[seq_id, (history_id + 1) % max_history, ...]` — **different slots**. `EndForward` then
   advances `history_slot_id`, so the previous state survives at slot `h`, which is exactly what
   Phase 8's `PopN` prefix-cache rollback reads.

   This matters because `prefix_cache_mode` defaults to **`"radix"`**
   ([config.py:157](python/mlc_llm/serve/config.py#L157)) and
   [config.cc:931](cpp/serve/config.cc#L931) then sets `max_history = 64` for hybrid models. The
   1.489 ms/token in §4.6 was traced under `prefix_cache_mode="disable"` (which every bench harness
   sets), i.e. `max_history = 1`, where `(h+1) % 1 == h` and the pair degenerates to a pure copy.
   **A naive in-place fusion would have sped up the benchmark configuration while destroying the
   history the default configuration depends on.**

   *The fix is not to abandon the fusion — it is to use the right write index.* The fused kernel
   loads from `storage[seq, hist, h, row, col]` and flushes to
   `storage[seq, (hist + 1) % max_history, h, row, col]`, mirroring `create_set_func`. Then both
   copy kernels vanish in **both** configurations, ring semantics are preserved exactly, and the
   aliasing question disappears — the slots are distinct whenever `max_history > 1`, and identical
   only when `max_history == 1`, where the per-element ownership proof above already guarantees
   safety. The full 1.489 ms/token is recoverable without trading away prefix caching.

   **Progress 2026-07-25:** `vm.builtin.rnn_state_storage` / `_seq_slot_ids` / `_history_slot_ids`
   added to [rnn_state.cc](3rdparty/tvm/src/runtime/vm/rnn_state.cc), `libtvm.so` rebuilt, all three
   verified resolvable via `get_global_func`. `RNNState.storage()` / `.slot_ids()` added to
   [rnn_state.py](python/mlc_llm/nn/rnn_state.py). Remaining: the fused TIR kernel, model wiring
   (fused for decode, copy path retained for prefill and history mode), and validation.
3. **Retune the two big tier-2 shared kernels** — `out_proj`/`o_proj` (75%, 40 calls/token) and
   routed-expert down (76%, 40 calls/token), 3.09 ms/token combined. Worth ~0.44 ms (2.3%) at 88%.
   Treat 88% as optimistic: both have K = 512 or 4096 rather than the 2048 the tier-1 kernels use,
   so some of the gap is shape, not schedule.
4. **Capture the 120 `rnn_state` launches into the cudagraph** — ≤4%, and only if option 2 is
   ruled out. Superseded by 2, not additive with it.

1 + 2 + 3 ≈ 2.8 ms/token = **14.6%** → roughly **54.13 → 62 tps** *if every option lands at its
estimate*. Option 1 has since landed at **62% of its estimate** (§10), so treat the other two as
optimistic by a similar factor until measured. The 97.5 tps weight roofline stays out of reach
regardless, because §4.6 shows 30% of the budget is kernels that do not stream weights at all.

> **Estimation lesson, worth carrying into options 2 and 3.** The option-1 model was
> "bytes ÷ the best GB/s any kernel of similar shape achieves". It missed on both sides: the wider
> fused GEMV did not inherit the narrower one's efficiency, and the estimate covered only decode
> while the change also cost prefill. For options 2 and 3, predict from a *measured* kernel at the
> target shape, and always state the prefill effect.

---

## 6. Correctness status

| gate | result |
|---|---|
| 0.8B `q0f16` greedy parity vs HF fp16 | ✅ **5/5 prompts, 50/50 tokens each** |
| 0.8B `q0f16` greedy parity **after the `in_proj_qkvzab` merge** | ✅ **5/5 prompts, 50/50 tokens each** (§10) |
| 0.8B `q0f16` greedy parity **after the in-place state update**, radix **and** disable | ✅ **5/5 prompts, 50/50 tokens each, both modes** (§11) |
| 0.8B `q0f16` prefix-cache round-trip (PopN rollback), radix | ✅ **4/4 checks**, identical to the copy path (§11) |
| 0.8B `q0f16` serial vs **6-way concurrent** decode, `disable` **and** radix | ✅ **6/6 identical**, on the in-place **and** copy-path libs (§12) |
| 0.8B `q0f16` 4-way concurrent **JSON-schema** generation (jump-forward) | ✅ no abort, schema-valid output (§12) |
| 0.8B `q0f16` greedy parity **after the in-place conv state**, radix **and** disable | ✅ **5/5 prompts, 50/50 tokens, both modes** (§13) |
| 0.8B `q0f16` greedy parity **after the history-path conv fusion**, radix **and** disable | ✅ **5/5 prompts, 50/50 tokens, both modes** (§14) |
| 0.8B `q0f16` greedy parity **after the history-path recurrent fusion**, radix **and** disable | ✅ **5/5 prompts, 50/50 tokens, both modes** (§15) |
| 0.8B `q0f16` **bit-exactness** after the history-path recurrent fusion, radix | ✅ **5/5 byte-identical** — the first history-path change that manages it (§15) |
| 0.8B `q0f16` long prompt (5814 tok, **3** prefill chunks) on the **history** path, radix | ✅ **byte-identical** (§15) |
| fused **history recurrent** kernel vs the copy-path kernel, 9 seq_lens × 2 head configs | ✅ **output bit-exact, ring bit-exact, non-target slots untouched**; fp64 rel ≤4.1e-6 (§15) |
| 35B-A3B fp8 tier-2 gate after the history recurrent fusion, **radix** | ✅ **1/15/2/5/50, identical to `lib_histconv`** (§15) |
| 0.8B `q0f16` long prompt (3133 tok, crosses the 2048 prefill chunk) on the **history** path, radix | ✅ **byte-identical** (§14) |
| fused **history** conv1d kernel vs fp64, 12 shapes × 2 widths × 4 ring configs incl. ring wrap | ✅ **within fp16 rounding; state bit-exact; non-target slots untouched** (§14) |
| 35B-A3B fp8 tier-2 gate after the history conv fusion, **radix** | ✅ **1/15/2/5/50, identical to `lib_convfused`** (§14) |
| 0.8B `q0f16` bit-exactness + long-prompt prefill after the conv fusion | ✅ **byte-identical**, incl. ~3.5 k-token prompt on the fused path (§13) |
| fused conv1d kernel vs fp64, 12 shapes × both conv widths | ✅ **within fp16 rounding; state bit-exact; ring slots clean** (§13) |
| 35B-A3B fp8 tier-2 gate after the conv fusion | ✅ **identical to `lib_inplace`, 1/15/2/5/50** (§13) |
| 35B-A3B high-margin gate after **`BLK_M=64` + the item-0h hoist** (§17.10), radix **and** disable | ✅ **139/139 at τ=2.0 in both modes, every column identical to `lib_blkk64`** — the hoist is bit-exact end-to-end (not shipped; see 0h) |
| v2 MoE GEMM bit-exactness across the **item-0h hoist**, `BLK_M` 16/32/64 × B 4096–16384 × 2 shapes × 2 routings (§17.10) | ✅ **48/48 exact**, incl. `HOIST=1` at `BLK_M=16` measuring 1.00× — the check that the reorder itself is sound |
| 35B-A3B high-margin gate after **`BLK_K=64`** (§17), radix **and** disable | ✅ **139/139 at τ=2.0 in both modes, every column identical to `lib_skippad`** — same τ=1.0 counts, same near-tie counts |
| v2 MoE GEMM bit-exactness across **`BLK_K` 32/64/128** and **`BLK_M` 16/32/64**, 2 shapes × 2 routings × B=4096/16384 (§17) | ✅ **exact at every value** — neither parameter touches the split over `K` |
| **35B-A3B `q4f16_1` high-margin gate** vs HF fp8, **both libs × both modes** (§16.1) | ✅ **139/139 wide-margin positions (τ=2.0), all four runs identical** — the first 35B gate with a pass/fail bar |
| 0.8B `q0f16` high-margin gate vs HF fp16, radix **and** disable (§16.1) | ✅ **400/400 positions, every margin** — including all 39 near-ties |
| 0.8B `q4f16_g16e` high-margin gate vs HF fp16 (calibration, §16.1) | ✅ 361/361 at τ=2.0; 11 flips total, **all at margin ≤ 1.031** |
| high-margin gate **negative control** (`stale1`, state one step stale) (§16.1) | ✅ **fails 342/361** on a lib that otherwise scores 361/361 — the gate is not vacuous |
| 0.8B `q4f16_g16e` vs HF fp16 | 1/5 — **quantization divergence, not a bug** |
| 35B-A3B greedy parity | ❌ **not reproducible on this box — see §6.1, neither leg fits** |
| 35B-A3B concurrent decode | ⛔ **blocked by design, not by a bug** — `batch_decode` is compiled with batch pinned to 1 (§12) |

⚠️ **Gate hybrid recurrent-state changes under `--prefix-cache-mode radix` as well as `disable`.**
The two configurations give `max_history` 64 and 1, and a whole class of state-indexing bug is
green under `disable` and broken under radix — which is the *default*. §11 demonstrates this with a
built negative control. Every bench harness sets `disable`.

The `q0f16` pass proves the CUDA 13.2 build is numerically exact. The `q4f16_g16e` result is a
4-bit-vs-fp16 comparison: every divergence is at a genuine near-tie ("Paris." vs "Paris,",
`n <= 1` vs `n == 0`, both reaching **42**), all outputs coherent and correct, and the high-margin
Fibonacci prompt is 50/50. Not a valid gate — use `q0f16` for correctness.

### 6.1 Why a bf16 35B gate cannot run on Orin — and the fp8 route that can

Two independent size walls, not one.

**Wall 1 — the HF reference leg**: bf16 `Qwen/Qwen3.6-35B-A3B` needs ~72 GB and this box
has 61 GB. The worklog recipe ([worklog.md:3226](worklog.md#L3226)) was run on a Blackwell machine
and `reference_outputs*.pt` is gitignored, so the cache was lost in the re-bootstrap.

**Wall 2 — the MLC leg, missed the first time round.** `--greedy-parity` is only a *bit-exact* gate
when MLC also runs fp16, and a `q0f16` 35B is 34.66 B text-only params at 2 B = **~69 GB**. Also does
not fit. So regenerating the 3.5 kB reference cache off-box is necessary but **not sufficient** —
there is no configuration in which a bit-exact 35B-vs-bf16 comparison runs on a 64 GB Orin.

#### The fp8 release fits, and a software path exists

`Qwen/Qwen3.6-35B-A3B-FP8` is **37.5 GB** (block-wise e4m3, `weight_block_size [128, 128]`, with
`modules_to_not_convert` keeping the routers, norms and every `linear_attn.in_proj_a` in bf16).
Getting transformers to actually execute it on sm_87 took three findings:

1. `FineGrainedFP8HfQuantizer.validate_environment` warns below compute capability 8.9 and sets
   `quantization_config.dequantize = True` — which materializes the model in bf16 and puts us back at
   72 GB. It has to be overridden, not worked around.
2. With the quantized path forced, dispatch lands on the Triton kernel (DeepGEMM is SM90+), and
   Triton **cannot compile it on Ampere**: `ValueError("type fp8e4nv not supported in this
   architecture. The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')")`. There is no fp8 *arithmetic*
   on sm_87 — this is a hardware limit, not a missing package. (`kernels==0.15.2` installs cleanly
   and changes nothing.)
3. The fp8 to bf16/fp16 **cast** does work on sm_87. So a weight-only dequant is viable.

[fp8_software_dequant.py](fp8_software_dequant.py) replaces `fp8_linear` — the single dispatcher both
`FP8Linear.forward` and `FP8Experts.linear` route through — with a pure-PyTorch dequant: weights stay
fp8 in memory, the per-128x128-block scale is expanded, and the matmul runs in the activation dtype.
`FP8Experts.forward` loops only over *hit* experts, so a decode step dequantizes top-8, not all 256.
`validate.py::load_hf` installs it automatically when the checkpoint declares fp8, and skips the
`.half()`/`.float()` forcing that would otherwise cast the fp8 weights straight back up.

**Read the result correctly: this is W8A16, not the W8A8 a real fp8 deployment runs.** Activations are
never quantized, which makes it a *more* accurate oracle than genuine fp8 inference and closer to the
bf16 master — but weights still carry e4m3 rounding (3 mantissa bits per block). Against a 4-bit MLC
build, expect near-tie flips exactly like the 0.8B `q4f16_g16e` row above.

So the 35B now has three tiers of check, and it matters which one a claim rests on:

| check | where | strength |
|---|---|---|
| `--greedy-parity`, MLC `q0f16` vs HF bf16 | **off-box only** (>=72 GB) | bit-exact; the only thing that proves the port itself |
| `--greedy-parity`, MLC `q4f16_1` vs **HF fp8 W8A16** | **Orin, works** | weaker than first claimed — see §6.2. Coherence + *comparative* signal only; the match count is not a pass/fail bar |
| [scripts/greedy_snapshot.py](scripts/greedy_snapshot.py) before/after | **Orin, works** | **bit-exact for refactors** — no reference model needed at all |

Tier 3 is what a layout merge or kernel fusion actually needs: "same lib, same prompts, identical
tokens" is a stronger statement than any tolerance comparison, and it runs in ~2 min. It cannot tell
you the port was right to begin with — only that a change did not alter it. Tier 2 covers that gap
and, unlike tier 1, runs on this hardware.

```bash
# tier 2 — fp8 reference cache, on Orin. The shim installs itself; ~37.5 GB resident.
python validate.py --reference-only --model Qwen/Qwen3.6-35B-A3B-FP8 --device cuda:0 \
    --cache reference_outputs_35b_fp8.pt --no-layer-hooks
python validate.py --greedy-parity  --model Qwen/Qwen3.6-35B-A3B-FP8 --device cuda:0 \
    --mlc-model-dir dist/qwen3_6-35B-A3B-q4f16_1 \
    --mlc-lib dist/qwen3_6-35B-A3B-q4f16_1/lib.so --cache reference_outputs_35b_fp8.pt

# tier 3 — bit-exactness across a refactor, no reference model
python scripts/greedy_snapshot.py --model-dir <dir> --model-lib <dir>/lib.so \
    --out tuning/greedy_35b_before.json          # before the change
python scripts/greedy_snapshot.py --model-dir <dir> --model-lib <dir>/lib.so \
    --compare tuning/greedy_35b_before.json      # after; nonzero exit on divergence

# tier 1 — still needs another machine, if a bit-exact port gate is ever wanted
python validate.py --reference-only --model Qwen/Qwen3.6-35B-A3B --device cuda:0 \
    --cache reference_outputs_35b.pt --no-layer-hooks
git add -f reference_outputs_35b.pt   # 3.5 kB; .gitignore has an exception for it
```

### 6.2 What the fp8 tier-2 gate actually measures (measured 2026-07-25)

`reference_outputs_35b_fp8.pt` is built and committed-able (3.5 kB, 5 prompts x 50 tokens). Ran it
against both 35B libs:

| prompt | baseline `q4f16_1` | fused `q4f16_1` |
|---|---:|---:|
| 1 `The capital of France is` | 1/50 | 1/50 |
| 2 `def fibonacci(n):` | 15/50 | 15/50 |
| 3 `7 multiplied by 6` (chat) | 2/50 | 2/50 |
| 4 `Once upon a time...` | 5/50 | 5/50 |
| 5 `1, 1, 2, 3, 5, 8, 13, 21,` | **50/50** | **50/50** |

**The absolute match counts are low, and §6.1's first framing of this gate was too generous.** It
claimed the gate "catches every gross porting bug"; at 1/50 agreement on an open-ended prompt, a real
bug and 4-bit-vs-8-bit quantization noise are not distinguishable by match count. The `>=48/50` bar in
`validate.py` is calibrated for fp16-vs-fp16 and applying it here is a category error — the same
mistake §6 already flags for the 0.8B `q4f16_g16e` row.

What the run does establish, and it is not nothing:

1. **Comparative identity.** Both libs agree with the reference *identically, prompt for prompt*,
   despite generating different text from each other (prompt 3's trailing `<|im_start|>` run differs,
   4 vs 5). Whatever drives the divergence is common to both, so **the `in_proj_qkvzab` merge does not
   degrade agreement**. That is the question the gate was run to answer, and it answered it cleanly.
2. **Coherence.** All five outputs on both libs are fluent and factually right (Paris/Seine,
   a correct recursive fibonacci, `7 x 6 = 42`, an intact fable, exact Fibonacci continuation). A
   wrong concat order, a broken RoPE or mis-routed experts produce garbage, not this.
3. **Margin dependence is the tell.** Prompt 5 — the only high-logit-margin prompt in the set — is
   **50/50 on both**. Everything else is an open-ended continuation where the next token is a
   near-tie, so the two quantizations part ways within a token or two and cascade.

**Concrete improvement for next session:** point 3 says how to turn this into a real gate. Build a
high-margin prompt set — arithmetic, exact-continuation sequences, closed-form factual lookups, the
kind of prompt where the top-1 logit gap is wide — and require e.g. 48/50 on those. Prompt 5 shows the
signal is there when the margin is; the current set is simply the wrong instrument, inherited from an
fp16-vs-fp16 era.

> ✅ **Done — §16.1**, and the diagnosis above was only half of it. A high-margin prompt set alone
> would not have fixed this gate, because the dominant defect is **cascade**: both sides free-run,
> so the counts above measure *when the two first diverged*, not how often they disagree. Teacher
> forcing removes it. The 35B now scores **139/139 wide-margin positions**, identically across both
> libs and both prefix-cache modes.


---

## 7. Artifacts (all uncommitted)

Full uncommitted inventory — including the **TVM submodule**, which is easy to miss — is in §9.
Data artifacts:
```
tuning/mlc_tg_35b_cuda13_20260724_210647.json        # 35B tg512 single point
tuning/mlc_tg_sweep_35b_cuda13_20260724_210926.json  # 35B full depth sweep
tuning/mlc_tg_sweep_0.8b_cuda13_20260724_213627.json # 0.8B full depth sweep
tuning/moe_kernel_cuda13_20260724.json               # MoE microbench, CUDA 13.2
tuning/mlc_tg_35b_fused_20260725.json                # 35B after the in_proj merge (§10)
tuning/mlc_tg_35b_copypath_20260725.json             # 35B option-2 A/B baseline (§11)
tuning/mlc_tg_35b_inplace_20260725.json              # 35B after the in-place state update (§11)
tuning/greedy_35b_before.json  tuning/greedy_35b_after.json   # tier-3 snapshots (§6.1)
reference_outputs.pt              # 0.8B HF fp16 reference (gitignored)
reference_outputs_35b_fp8.pt      # 35B fp8 W8A16 reference — COMMIT THIS (3.5 kB, §6.1/§6.2);
                                  # .gitignore already has the exception for the 35B name
dist/qwen3_5-0.8B-q4f16_g16e/  dist/qwen3_5-0.8B-q0f16/  dist/qwen3_5-0.8B-q0f16_fused/
dist/qwen3_6-35B-A3B-q4f16_1/  dist/qwen3_6-35B-A3B-q4f16_1_fused/     # dist/ is gitignored
```

**Both `_fused` dirs now hold three libs, so `glob("*.so")` is more dangerous than ever** (§14.1's
footgun, and §3's note that `profile_decode.py` still trips it). Always pass `--model-lib`:

| lib | what it is |
|---|---|
| `lib_rowspec64.so` | **§19.4 — item 0l.** `BLK_M=64` + hoist + `ROWSPEC`. **Not shipped**: −9.1% pp128 / +2.3% pp512 / +12.2% pp2048, i.e. `lib_hoist64` plus 2.4 points at pp512 and nothing at either end. It is the evidence that padding-row compute was never the wide tile's problem |
| `lib_hoist64.so` | §17.10 — `lib_blkk64` + `BLK_M=64` + the hoist. **Not shipped**: +12.6% pp2048 but −10.4% pp128. Keep as the A/B leg for the runtime-specialisation work |
| `lib_hoist32.so` | §17.10 — same with `BLK_M=32`. +7.8% / −5.5%; proves 32 is not a safe middle, only a smaller version of the same trade |
| `lib_blkk64.so` | **§17 — the current 35B build.** `lib_skippad` plus `BLK_K=64` (item 0g) and the whole-body padding guard. pp512 **875.37**, decode 59.97, state gate identical to `lib_skippad` in both modes. Bench and gate against this |
| `lib_skippad.so` | §16.11 — `lib_ksplit4` plus item 0f's k_o_o padding-CTA skip. **The §17 A/B baseline**; re-measured 2026-07-26d at pp512 **767.20**, decode 59.92 |
| `lib_ksplit4.so` | §16.5's lane-split build — the §16.11 A/B baseline. pp512 644 |
| `lib_gdnhist.so` | §15 — everything in `lib_histconv` plus the history-path recurrent fusion. Was "the current build" until §16.5 superseded it |
| `lib_histconv.so` | §14 history-path conv fusion — the §15 A/B baseline |
| `lib_convfused.so` | §10 in_proj merge + §11 in-place recurrent state + §13 in-place conv state — the §14 A/B baseline |
| `lib_inplace.so` | the §11 build — the §13 A/B baseline |
| `lib_copypath.so` | same tree, same flags, `MLC_QWEN35_INPLACE_STATE=0` — the §11 A/B baseline. Note that toggle now disables the conv fusion too, since both hang off `state_io` |
| `lib.so` | the older §10 build (35B: byte-size-identical to `lib_copypath.so` at 166 MB) |

⚠️ **The 35B `lib_convfused.so` is the only one of these built with `MLC_MOE_GEMM_V2=1`
deliberately verified** (`nm -D | grep -c 'group_gemm_v2\|moe_dispatch_tables'` → 4). Any 35B lib
rebuilt without that env var is not comparable to these — see the §8 warning.

The `_fused` dirs are the §10 builds and are the ones to bench against; the non-fused 35B dir is the
pre-merge baseline, worth keeping until the merge is committed.

`qwen3_5.md` had uncommitted edits from a parallel session; §9.1 folds the CUDA-13 findings in on
top of them.

**Session scripts — promoted into `scripts/` (2026-07-25), all re-verified after the move:**

| file | purpose |
|---|---|
| [scripts/bw_probe.cu](scripts/bw_probe.cu) | achievable read-BW probe, native sm_87. Reproduces **156.2 GB/s** |
| [scripts/active_params.py](scripts/active_params.py) | exact active-param/roofline calc from safetensors headers. Now takes `--snapshot/--bits/--tps` |
| [scripts/profile_decode_35b.py](scripts/profile_decode_35b.py) | parameterized nsys/ncu decode harness (explicit `--model-lib`) |
| [scripts/analyze_decode_trace.py](scripts/analyze_decode_trace.py) | **new** — the §4.6 breakdown: step-segmented per-token ms, in-trace idle, kernel→weight map |
| [scripts/greedy_snapshot.py](scripts/greedy_snapshot.py) | **new** — tier-3 bit-exactness gate (§6.1): capture greedy tokens, diff after a refactor |
| [scripts/prefix_cache_roundtrip.py](scripts/prefix_cache_roundtrip.py) | **new (§11)** — PopN rollback gate under radix. Needs no reference model, so it runs on the 35B |
| [scripts/batch_decode_parity.py](scripts/batch_decode_parity.py) | **new (§11)** — serial vs concurrent decode; covers per-batch state-slot indexing. **Runs and passes as of §12**, and now also reports the concurrency speedup |
| [scripts/long_prompt_gate.py](scripts/long_prompt_gate.py) | **new (§15)** — bit-exactness across a prompt long enough to span several prefill chunks, which is the only way `history_slot_id` advances *mid-prompt*. A single-chunk gate cannot reach that. Promoted from the ad-hoc check §14 ran |
| [scripts/gdn_kernel_check.py](scripts/gdn_kernel_check.py) | **new (§15)** — the recurrent analogue of `conv1d_kernel_check`. Gates both GDN in-place kernels against the *copy-path kernel* (so the bar is bit-exactness, not a tolerance) plus an fp64 reference of the recurrence, across 9 seq_lens × 2 head configs including 5 that wrap the ring. Separates "ring misindexed" from "output wrong" — the off-by-one control leaves `out_bit` at 0 while `state_err` hits 130. No model, no weights, no engine. **§16.6:** `--v-block N` adds `vb_exact` — output *and* ring compared against `v_block=V` at the same `k_split`, required to be **exactly 0**, since re-gridding changes no reduction order. Stricter than the §16.5 bars and the item-0d claim rests on it. **§16.5:** `--k-split N` gates the lane-split kernel, where output and ring necessarily drop to relative tolerances (the reduction is re-associated by design) while **"every other ring slot byte-clean" stays exact** and the fp64 check becomes the primary bar. Also memoizes the compile on `(kind, n_kh, n_vh, k_split)` — it was recompiling per `seq_len`, which is a *runtime* dimension, and ptxas costs 30–42 s on a split kernel |
| [scripts/conv1d_kernel_check.py](scripts/conv1d_kernel_check.py) | **new (§13), extended (§14)** — now gates the history variant too, including ring-wrap shapes. Numerical unit gate for the fused conv1d: kernel vs fp64 across 12 shapes and both conv widths. Separates "wrong" from "rounded differently", which no token-diff can do on the 35B. Needs no model, no weights, no engine; runs in seconds |
| [scripts/moe_blkm_check.py](scripts/moe_blkm_check.py) | **new (§17.1)** — sweeps the v2 GEMM's row-blocking factor. `BLK_M` regroups output rows into CTAs and does **not** touch the split over `K`, so the bar is exact equality against `BLK_M=16`, not a tolerance. Takes `--batches` because the conclusion only holds once B=16384 (the real 2048-token prefill chunk) is measured as well as B=4096, and `--hoist` to sweep item 0h's loop order (§17.10). ⚠️ Its synthetic routings are for the **bit-exactness bar, not for ranking** — they got item 0h's sign wrong (§17.9) |
| [scripts/moe_gemm_roofline.py](scripts/moe_gemm_roofline.py) | **new (§17.7)** — rooflines the v2 GEMM's *real* CTAs separately from its padding ones, by **fitting** `n_real*c_real + n_pad*c_skip` rather than assuming a padding ratio (§16.10's 0.933 does not carry to guarded CTAs). Reports issued *and* unique bytes; only the unique column carries information, since issued/time is constant by construction |
| [scripts/make_prose_corpus.py](scripts/make_prose_corpus.py) | **new (§17.9)** — builds a natural-language corpus for `--prompt-file`. Exists because the harness' built-in filler is one sentence repeated, which concentrates MoE routing and can flip which kernel config wins |
| [scripts/moe_skippad_ab.py](scripts/moe_skippad_ab.py) | **new (§17.2)** — three-way A/B of item 0f: no guard / §16.10's k_o_o guard / §17's whole-body guard, all required byte-identical. Inverting its three timings is what showed §16.10's "20% residue" to be CTA launch overhead rather than the store tail |
| [scripts/moe_rowspec_ab.py](scripts/moe_rowspec_ab.py) | **new (§19.2)** — three-way A/B of items 0k and 0l against no guard at all, in one process off identical inputs, so the two mechanisms are ranked under one clock state. Its `BLK_M=16` leg is the point: there both guards are logically inert, so any delta is the *mechanism's* overhead and nothing else. That cell is what turned §18.11's inference into a measurement (0k 0.91×, 0l 1.00×) |
| [fp8_software_dequant.py](fp8_software_dequant.py) | **new** — software W8A16 fp8 path so the 37.5 GB fp8 checkpoint can be an HF reference on sm_87 (§6.1) |

Build/run:
```bash
nvcc -O3 -arch=sm_87 -o /tmp/bw scripts/bw_probe.cu && /tmp/bw
python scripts/active_params.py
nsys export --type sqlite -o dec.sqlite q35_decode.nsys-rep
python scripts/analyze_decode_trace.py dec.sqlite --top 26
```

`prof_decode35.py`'s `decode-steady` NVTX range **never made it into the trace** — the only NVTX
rows are CCCL/thrust. torch's `nvtx.range_push` is a no-op here (TVM is built `USE_NVTX OFF` and the
sbsa torch cannot initialize CUDA on sm_87). `analyze_decode_trace.py` segments on the sampling
kernel instead and does not need NVTX; leave the range in place but do not rely on it.

---

## 8. Reproduction

> ⚠️ **Compiling the 35B needs `MLC_MOE_GEMM_V2=1` in the environment, and nothing warns
> you.** It is an env-var opt-in read at *compile* time
> ([moe_matmul.py:875](python/mlc_llm/op/moe_matmul.py#L875)); without it the int4 MoE
> GEMM silently falls back to the v1 persistent-loop kernel and `moe_dispatch_tables` /
> `dequantize_group_gemm_v2` never get emitted. Cost when this was hit on 2026-07-25:
> **pp512 560 → 225 tps (−60%)**, which read as a catastrophic regression in an A/B whose
> two libs differed by a *model* change. The flag is in the [qwen3_5.md:183](qwen3_5.md#L183)
> compile recipe but was missing from this section. It is not in `.envrc.local` either, so
> it has to be typed on every 35B compile. Check a build with:
> `nm -D --defined-only <lib>.so | grep -c 'group_gemm_v2\|moe_dispatch_tables'` — expect 4,
> not 0. The 0.8B has no MoE and is unaffected, so a 0.8B A/B will not catch it.

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks     # REQUIRED — worth ~1.4%
source .envrc.local

# compile the 35B — MLC_MOE_GEMM_V2=1 is REQUIRED, see the warning above
MLC_MOE_GEMM_V2=1 python -m mlc_llm compile dist/qwen3_6-35B-A3B-q4f16_1_fused/mlc-chat-config.json \
  --device cuda --opt "flashinfer=1;cudagraph=1" \
  -o dist/qwen3_6-35B-A3B-q4f16_1_fused/<name>.so

# bench (always pass --model-lib; glob picks lib_nofi.so otherwise → ~82% of headline)
# --prefix-cache-mode radix is the DEFAULT configuration; the harness default 'disable'
# is not what a user gets on a hybrid model (§13/§14.1).
python scratch_mlc_tg_sweep.py \
  --model-dir dist/qwen3_6-35B-A3B-q4f16_1 \
  --model-lib dist/qwen3_6-35B-A3B-q4f16_1/lib.so \
  --pp 512 --tg 512,1024,2048,4096,8192 --runs 3 --warmup 1 \
  --prefix-cache-mode radix --json-out tuning/<name>.json

# ⚠️ For ANY MoE A/B whose mechanism touches tile counts, expert counts or routing,
# add --prompt-file. The built-in filler is one sentence repeated (11 distinct tokens
# per 512) and concentrates the router; it got item 0h's *sign* wrong (§17.9).
python scripts/make_prose_corpus.py --out /tmp/prose_corpus.txt
python scratch_mlc_tg_sweep.py ... --prompt-file /tmp/prose_corpus.txt

# correctness (q0f16 only — q4 vs fp16 is not a valid gate)
python validate.py --reference-only --model Qwen/Qwen3.5-0.8B --device cuda:0 \
    --cache reference_outputs.pt
python validate.py --greedy-parity --model Qwen/Qwen3.5-0.8B --device cuda:0 \
    --mlc-model-dir dist/qwen3_5-0.8B-q0f16 \
    --mlc-lib dist/qwen3_5-0.8B-q0f16/lib.so --cache reference_outputs.pt

# MoE GEMM row-fragment guards (items 0k and 0l). Bit-exactness first, always, then rank.
# The BLK_M=16 leg is the control that separates a mechanism's cost from its benefit (§19.2).
python scripts/moe_rowspec_ab.py --indptr-file tuning/expert_hist_35b.npz --indptr-key prose_len512
python scripts/moe_rowspec_ab.py --batches 1024      # pp128 scale; only synthetic routings exist here

# item 0l end to end — compile-time env vars, all three needed together
MLC_MOE_GEMM_V2=1 MLC_MOE_GEMM_V2_BLKM=64 MLC_MOE_GEMM_V2_HOIST=1 MLC_MOE_GEMM_V2_ROWSPEC=1 \
  python -m mlc_llm compile dist/qwen3_6-35B-A3B-q4f16_1_fused/mlc-chat-config.json \
  --device cuda --opt "flashinfer=1;cudagraph=1" \
  -o dist/qwen3_6-35B-A3B-q4f16_1_fused/lib_rowspec64.so

# VL — current lib is lib_vl2.so. Both §20 knobs are at their defaults here, so this
# is plain --opt with nothing set; that is deliberate, it is the shipping config.
python -m mlc_llm compile dist/qwen3_5-0.8B-vl-q0f16/mlc-chat-config.json \
  --device cuda --opt "flashinfer=1;cublas_gemm=1;cudagraph=1" \
  -o dist/qwen3_5-0.8B-vl-q0f16/lib_vl2.so
python validate.py --greedy-parity-vl5 --mlc-model-dir dist/qwen3_5-0.8B-vl-q0f16 \
  --mlc-lib dist/qwen3_5-0.8B-vl-q0f16/lib_vl2.so --device cuda:0   # 184/184, ~13 s

# The two §20 knobs are coupled (§20.6). To reproduce the earlier legs:
#   MLC_QWEN35_VL_PRESCALE_Q=0 MLC_BLAS_SKIP_FP32=1  -> lib_nofp32blas (§20.3), ttft 443
#   MLC_QWEN35_VL_PRESCALE_Q=0 MLC_BLAS_SKIP_FP32=0  -> lib_cublas     (§20.1), ttft 487

# VL performance (§20.1) — three separate numbers. NOT MLCEngine: it cannot drive this
# vision tower (llava image_embed signature + hardcoded ImageData embed_size).
python validate.py --perf-vl5 --mlc-model-dir dist/qwen3_5-0.8B-vl-q0f16 \
  --mlc-lib dist/qwen3_5-0.8B-vl-q0f16/lib_vl2.so

# The tower's attention block alone, no compile (§20.5). Fidelity bar: 1.3% vs the trace.
python scripts/vit_attn_bench.py --static --prescale

# kernel unit gates — no model, no engine. Run these FIRST on any state change.
python scripts/gdn_kernel_check.py                     # §15, recurrent, ~51 s; --seq-lens narrows
python scripts/gdn_kernel_check.py --k-split 4         # §16.5, the lane-split kernel, ~120 s
python scripts/gdn_kernel_check.py --k-split 4 --max-history 1   # ...and the `disable` ring
python scripts/conv1d_kernel_check.py                  # §13/§14, conv

# GDN recurrence A/B on the kernels MLC compiles (§16.5). Includes the ring flush, unlike the
# probe below, and uses the real grid — n_vh blocks, so 32 on the 35B and 16 on the 0.8B.
python scripts/gdn_kernel_bench.py                     # add --trace-share for an Amdahl bound

# lane-split recurrence: COMPILE-time flag, read while tracing (like MLC_QWEN35_INPLACE_STATE).
# 4 is the default; 1 restores §15's bit-exact kernel for an A/B.
MLC_QWEN35_GDN_KSPLIT=1 python -m mlc_llm compile ... -o .../lib_ksplit1.so

# high-margin state gate (§16.1) — the only 35B gate that can adjudicate a state change.
# Capture once per reference model, then check any lib against it. Run BOTH modes.
python scripts/high_margin_gate.py --capture --model Qwen/Qwen3.6-35B-A3B-FP8 \
    --out tuning/high_margin_ref_35b_fp8.json --num-prompts 6 --num-tokens 24
for m in radix disable; do
  python scripts/high_margin_gate.py --check tuning/high_margin_ref_35b_fp8.json \
      --model-dir dist/qwen3_6-35B-A3B-q4f16_1 \
      --model-lib dist/qwen3_6-35B-A3B-q4f16_1/lib.so --prefix-cache-mode $m --num-tokens 0
done
# prove the gate is not vacuous before trusting a pass (expect a loud FAIL):
python scripts/high_margin_gate.py --check tuning/high_margin_ref_0.8b_fp16.json \
    --model-dir dist/qwen3_5-0.8B-q0f16_fused \
    --model-lib dist/qwen3_5-0.8B-q0f16_fused/lib_gdnhist.so --negative-control stale1

# GDN recurrence design probe (§16.2) — no model, no TVM; ~30 s
nvcc -arch=sm_87 -O3 -o /tmp/gdn_probe scripts/gdn_recurrence_probe.cu && /tmp/gdn_probe 1 20

# MoE dispatch + GEMV shape/schedule (§16.3, §16.4).
# MLC_MOE_GEMM_V2=1 matters here exactly as it does at compile time — without it the
# bench measures the v1 fallback and the b=1 ratio changes by 2.6x.
MLC_MOE_GEMM_V2=1 python bench_moe_kernel.py --shapes gate_up_gemv,down_gemv,gate_up,down
MLC_GEMV_TSTR="16,32,1" python bench_moe_kernel.py --shapes ksweep_n2048_k4096

# recurrent-state gates — run BOTH modes; `disable` alone is blind to a whole bug class (§6)
python scripts/prefix_cache_roundtrip.py \
    --model-dir dist/qwen3_5-0.8B-q0f16_fused \
    --model-lib dist/qwen3_5-0.8B-q0f16_fused/lib_inplace.so
for m in disable radix; do
  python scripts/batch_decode_parity.py --prefix-cache-mode $m \
      --model-dir dist/qwen3_5-0.8B-q0f16_fused \
      --model-lib dist/qwen3_5-0.8B-q0f16_fused/lib_inplace.so
done   # also prints the concurrency speedup (§12); 0.8B only — the 35B is pinned to batch 1

# decode trace + the §4.6 breakdown
nsys profile -t cuda,nvtx --cuda-graph-trace=node -o q35_decode -f true \
  python scripts/profile_decode_35b.py \
  --model-dir dist/qwen3_6-35B-A3B-q4f16_1 --model-lib dist/qwen3_6-35B-A3B-q4f16_1/lib.so
nsys export --type sqlite -o q35_decode.sqlite q35_decode.nsys-rep
python scripts/analyze_decode_trace.py q35_decode.sqlite --top 26
```

**Engine-side C++ changes need `ninja -C build`, not just `source .envrc.local`.** Everything in §12
lives in `libmlc_llm.so`; a Python-only workflow will silently keep running the old engine. The TVM
half (`3rdparty/tvm/build`) is a separate build again — see §2.1.1.

Do not read `nsys stats --report cuda_gpu_kern_sum` directly for per-token cost: it averages prefill
and decode instances of the same kernel name together (`depthwise_conv1d` is 4016 µs in prefill,
11 µs in decode) and gives no idle figure. Use `analyze_decode_trace.py`.

**`ncu` is still blocked** — `sudo` on this box requires a password, so the §2 sudoers entry could
not be added non-interactively. Everything in §4.6/§4.7 came from `nsys`, which needs no privileges.
Add the entry by hand if per-kernel hardware counters are wanted:
```bash
echo 'alfie ALL=(root) NOPASSWD: /usr/local/cuda/bin/ncu' | sudo tee /etc/sudoers.d/ncu-profiling
sudo chmod 440 /etc/sudoers.d/ncu-profiling
```

Monitoring: **`nvidia-smi` does not report iGPU utilization or processes on Tegra** — it shows
`Not Supported` / `N/A` / "No running processes found" even under full load. Use `tegrastats`, or
`cat /sys/devices/platform/bus@0/17000000.gpu/load` (per-mille).

---

## 9. Status and next steps

### Done 2026-07-25

- **Every decode kernel identified; the budget re-derived from one trace (§4.6).** Closed both
  "which weights are those two kernels" questions; refuted the "moves ~nothing" claim about
  `rnn_state_get/set` and the 13.3% idle figure.
- **Cudagraph coverage audited (§4.7).** 131 eager launches/token, 120 of them `rnn_state`.
  `op.topk` has not regressed.
- **§5 re-scoped** off launch-overhead onto tier-2/tier-3 kernels.
- **§5 option 1 landed and measured (§10)** — `in_proj_qkvzab`. +2.8% tg / -1.1% pp. **Kept**, by
  decision, after the fp8 gate showed it correctness-neutral.
- **The 35B got its first on-box reference (§6.1, §6.2)** via the fp8 checkpoint plus a software
  W8A16 shim, and the gate was then honestly de-rated in §6.2.
- **§5 option 2 LANDED and measured (§11)** — in-place GDN recurrent-state update. **35B +6.04%
  decode, prefill neutral**; 0.8B +2.25%. Validated under both `prefix_cache_mode` settings, and
  the §5 trap was confirmed by building the wrong version on purpose.
- **Eight scripts promoted/added under `scripts/` (§7).**
- **Concurrent serving fixed and gated (§12)** — the break was multi-sequence *prefill*, not decode.
  0.8B now serves 6 requests concurrently at **2.67×** serial wall clock, 6/6 identical output under
  both prefix-cache modes and on both libs. Two pre-existing grammar-path aborts fixed on the way.
  Closes §11's coverage gap on the fused kernel's per-batch slot indexing.

### Done 2026-07-25d

- **§9 item 0a landed (§15)** — in-place GDN *recurrent* state on the history path. **35B pp512
  +59.1%, 0.8B +119.4%**, decode neutral, `disable` path unmoved (645.4 → 646.1). The recurrent
  pair fell 365.2 ms → 95.6 ms (**3.82×**) in an A/B trace whose whole-run delta (268 ms) is
  accounted for by those two kernels alone (269.6 ms) — no side effects.
- **The default configuration is no longer the slow one.** radix vs `disable` prefill: 35B
  1.64× → **1.03×**, 0.8B 2.77× → **1.04×**.
- **First bit-exact history-path change** — `greedy_snapshot` 5/5 byte-identical under radix, and
  the unit gate predicted it (output bit-exact against the copy-path kernel).
- **`scripts/gdn_kernel_check.py` added** with **three** negative controls, one of which
  (guard removed entirely → still passes) refuted the assumption that the skip guard is what makes
  the kernel correct. It is a pure optimization; a second control shows it is nonetheless placed
  exactly on the boundary.
- **A latent race removed rather than preserved** — `create_set_with_history_func` writes every
  `t` from a flat parallel grid, so at prefill (512–2048 positions into 64 slots) `t` and
  `t + max_hist` race for the same slot. §15.3.
- **Estimation post-mortem worth reading (§15.2)**: the headline prediction landed in band, but two
  mechanism errors cancelled. New rule — renormalize against a trace of the *actual* A/B baseline,
  and never mix a whole-run denominator with a prefill-only metric.

### Done 2026-07-25b

- **§9 item 2 landed (§13)** — in-place GDN *conv* state. **35B +2.40% tg, +15.3% pp**; 0.8B
  +1.28% / +42.7%. Decode hit 109% of a traced estimate. The prefill win was unpredicted and is
  the larger number: the TE conv it replaces is **~42× off roofline**.
- **§9 item 3 REFUTED (§13)** — the cudagraph allowlist should not be built. Eager launches went
  *up* 131 → 183/token after §11, idle did not move, and an eager launch costs ~0.9 µs at the
  margin.
- **`analyze_decode_trace.py` was reporting wrong bandwidth** and is fixed (commit `6749a630`).
  Kernel names are reused across fusion passes, so it now verifies output width against
  `gridX*64` every run. It had the fused `in_proj` at 60% of wall; it is at **91%**.
- **`scripts/conv1d_kernel_check.py` added** — numerical unit gate, no model needed (§7).
- **Everything committed**, including the TVM submodule.

### Committed state (was: nothing committed)

All work from the 2026-07-25 and 2026-07-26 sessions is in git on branch `qwen3_5`:

| commit | what |
|---|---|
| `e3b099e8` | §10 in_proj merge + §11 in-place recurrent state |
| `838c2d1b` | §12 concurrent serving on hybrid models |
| `fce533aa` | §6.1 fp8 software dequant + six gates/probes + fp8 reference cache |
| `54777593` | §1–§12 workplan, qwen3_5.md corrections, tuning data |
| `6749a630` | `analyze_decode_trace` geometry verification |
| `3ff691f5` | §13 conv-state fusion + `conv1d_kernel_check.py` |
| `c2fd8691` | §13 workplan, §8 build warning, §9 item 3 refutation |
| `51d8bfd8` | §15 history-path recurrent fusion + `gdn_kernel_check.py` |
| `dc695262` | **§16.1** `high_margin_gate.py`, `prefix_cache_roundtrip` prompt set, `gdn_recurrence_probe.cu`, bench sweeps |
| `8ce123dc` | **§16.2/§16.3/§16.4** items 0c.1, 5 and 1 measured; MoE comment corrected |
| `2341bb92` | **§16.1** 35B fp8 high-margin reference + the four gate runs |
| `0c4af0e5` | **§9** restructured — 0b/1/5 closed, 0c step 1 done, VL blocker pinned down |
| `cd314bb7` | **§9** handoff block, by-file table, the unpushed-submodule analysis |
| `ad56b584` | **§16.5** the lane-split GDN recurrence in TIR + `gdn_kernel_bench.py`, `--k-split` on the gate, two §16.2 corrections, item **0d** filed |
| `dfd7fec9` | commit-table backfill for `ad56b584` |
| `3c6edc2b` | **§16.5** flush isolated with `--max-history 1`; item 0d's mechanism corrected before building it |
| `7865c5de` | **§16.6** item 0d — `v_block` + `MLC_QWEN35_GDN_VBLOCK` + the `vb_exact` gate bar; opt-in at default `0` |
| `6c0ef467` | **§16.7** the 35B prefill trace; item 0c.2 de-prioritised, item **0e** filed |
| `8562fade` | **§9** handoff rewritten as a cold-start block; traps list gains the concurrency failure |
| `5294dae4` | **§16.8** item 0e measured — v2 is CTA-bound, 27–50% padding CTAs; item **0f** filed; `bench_moe_kernel.py` large-B mode + two instrument fixes |
| `0dd23758` | commit-table backfill for `5294dae4` and `8562fade` |
| `6daef3e2` | **§16.9** item 0f's two obvious routes refuted; `MLC_MOE_GEMM_V2_BLKM` A/B knob (inert at 16); §16.8's dequant attribution corrected |
| `88fce949` | commit-table backfill for `6daef3e2` |
| `1aa29196` | commit-table order + two missing rows; §16.8 heading no longer quotes the top of its own range |
| `31283474` | **§16.9** node classes located (`tvm.tirx`), 0f's blocker narrowed |
| `ec051589` | **§16.10** item 0f built — padding-CTA skip via a zero loop extent, bit-exact, 1.28–1.73×; `scripts/moe_gemm_check.py` |
| `4b7d1be6` | item 0f marked built; handoff repointed at the end-to-end measurement |
| `3f53eb24` | commit-table backfill |
| `dbe3a51f` | **§16.11** item 0f measured end-to-end — pp512 644 → 769 (+19.4%), gate identical to baseline, default flipped to `1` |
| `5cdcb68c`, `463f95b4` | commit-table backfill; worklog gains the two missing sessions |
| `76bceaec` | **§17** item **0g** — `BLK_K` 32 → 64, pp512 767 → 875 (+14.1%); whole-body padding guard; both guards found by annotation, not extent; `moe_blkm_check.py` + `moe_skippad_ab.py` |
| `723af51e` | **§17** session record — item **0g** landed, **0h** refuted (⚠️ retracted by `08997dd6`), §16.10's 20% residue re-attributed to launch overhead |
| `25465da0` | commit-table backfill |
| `08997dd6` | **§17.7–§17.8** real CTAs rooflined (85–87% of the wall); **§17.1's claim about the hoist retracted**; `moe_gemm_roofline.py` |
| `0a4fd1e4` | VL entry corrected — never blocked, no download; `Qwen/Qwen3.5-0.8B` *is* the VL checkpoint |
| `1100cb18` | **§17.9–§17.10** item **0h** built and measured end-to-end, parked (no Pareto `BLK_M`); **the bench prompt was picking winners**; `--prompt-file` + `make_prose_corpus.py` |
| `61b5e055` | workplan brought current for a fresh session |
| `5a9873d9` | the real expert histogram filed as item **0i** and queued |
| `9422dff2` | **§18.1–§18.11** item **0i** lands; **§17.7's roofline retracted** — 45% of the wall, not 85–87%; items **0j** refuted and **0k** measured; the 4×3 frontier mapped |
| `83061601` | **§18.12–§18.13** the VL re-gate runs; "not cleared" rather than passed |
| `7c6789ff` | **§18.15** the 35B against the original, measured end to end rather than chained |
| `7f71e2ab` | **§18.14** the VL gate gets a margin — the one divergence is a 0.05-nat near-tie, PASS |
| `8da5f2fc` | both remaining leads filed as items **0l** and **0m** |
| `741bca7f`, `b7922474`, `baade217` | **§19** session record — 0l refuted, 0m fixed; the padding-row-compute premise retracted where asserted; item **0o** filed and queued |
| `0a563a09` | **§20** item **0o** — the first VL performance numbers; cuBLAS net-negative on the tower; `MLC_BLAS_SKIP_FP32` (default `1`), ttft −8.9%, gate 184/184 exact with a to-the-digit `=0` control; `--perf-vl5`; the gate's `image_embed` hoist; item **0p** filed |
| `12e7a95d`, `06760850`, `d0df9b9f` | commit-table backfill; the submodule loose end closed (`dff702c` pushed, pointer advanced by `3281f97f`); the VL default-lib trap flagged |
| `0e285ea7` | **§20.5–§20.7** item **0p** half done — `vit_attn_bench.py` (1.3% fidelity); **the wall is 184.8 GB/s, not 156**, so softmax is at 98.5% and closed; symbolic shapes refuted (6.5%); prescaling worth **0.00 ms alone**, which retires §20.3's guard and unlocks 70 ms via cuBLAS. `image_embed` 293 → **223**, ttft → **372.88**, gate 184/184 exact. Item **0q** filed |

✅ **The TVM submodule commit that §11–§15 depend on IS pushed.** The parent's `3rdparty/tvm`
pointer is `4624d97` (branch `qwen35-inplace-rnn-state` on the `alansrobotlab2/relax` fork),
carrying the three `vm.builtin.rnn_state_*` accessors. Verified 2026-07-25d and again 2026-07-26:
`git ls-remote origin qwen35-inplace-rnn-state` returns `4624d972…`, identical to the pointer.
This section said otherwise for three sessions and was wrong.

✅ **The second submodule commit, `dff702c`, is now pushed too, and the parent pointer is advanced
(2026-07-27b).** It is the §16.3 `MLC_GEMV_TSTR` probe hook in
`python/tvm/s_tir/dlight/gpu/gemv.py`. It was unpushable non-interactively for three sessions — the
submodule's remote is HTTPS with no credential helper, so a push attempt failed with
`could not read Username for 'https://github.com'` — and was pushed from an interactive shell.
Verified by the authoritative test: `git -C 3rdparty/tvm ls-remote origin` returns
`dff702c… refs/heads/qwen35-inplace-rnn-state`, matching the parent pointer that `3281f97f` now
records.

**Both loose ends in this section are therefore closed**, and `git status` is down to a single
entry, `?? COLCON_IGNORE`. A fresh clone now gets the hook, so §8's `MLC_GEMV_TSTR` reproduction
line works there.

⚠️ **The check that produced the false alarm is the thing to remember.** That clone's
`remote.origin.fetch` was narrowed to `+refs/heads/mlc:refs/remotes/origin/mlc` only, so no
remote-tracking ref existed for this branch and **`git branch -r --contains HEAD` returned empty
even though the commit was on the remote**. Fixed by adding the branch to the refspec and setting
an upstream, so `git status -sb` now reads `...origin/qwen35-inplace-rnn-state` and the ordinary
checks work. **On a submodule with a narrowed refspec, `git ls-remote` is the only trustworthy
"is it pushed?" test** — everything local can be a false negative.

`.gitignore`'s reference-cache exception named only `reference_outputs_35b.pt`, so the fp8 cache
that §7 said to commit was still ignored; the exception now covers both names and the file is in
`fce533aa`.

### Start here next session

> **Handoff, end of 2026-07-27b.** Branch `qwen3_5`. **Item 0o closed, item 0p half closed, and the
> VL model is 23% faster to first token.** The first VL performance number ever taken on this box was
> also the thing that overturned §19.6 — and then §20.5 overturned §20.3 one section later. 35B lib
> unchanged (**`lib_blkk64.so`**); current VL lib is **`lib_vl2.so`**, built at plain `--opt` with no
> environment variables set.
>
> | VL 0.8B, cat fixture, `radix` | `lib_cublas` (was default) | `lib_nofp32blas` (§20.3) | **`lib_vl2`** |
> |---|---:|---:|---:|
> | `image_embed` | 337.28 ms | 293.33 | **223.33** (−33.8%) |
> | `prefill` (652 tok) | 149.37 | 149.87 | 149.54 |
> | `decode` | 11.36 (88.1 tok/s) | 11.38 | 11.38 |
> | **ttft** | 486.65 | 443.20 | **372.88** (−23.4%) |
>
> Gate **184/184 exact, margin PASS** at every step. **`image_embed` was 69% of ttft and had never
> been measured**; the gate called it "cheap" and ran it five times per pass (now hoisted — gate is
> 184/184 in 13.2 s).
>
> **Two knobs, coupled — read this before touching either.** `MLC_QWEN35_VL_PRESCALE_Q` (default
> `1`) applies the attention scale to `q` instead of to the scores, using
> `matmul(q,k^T)·c ≡ matmul(q·c,k^T)`. `MLC_BLAS_SKIP_FP32` (default `0`) declines fp32 cuBLAS
> offloads. §20.3 shipped the second at `1` and §20.6 retired it: the fusion it was protecting was
> worth **0.00 ms**, so protecting it bought 3.0 ms/iter and gave up **70**. Set `PRESCALE_Q=0` and
> `SKIP_FP32` is worth `1` again — that configuration pays §20.2's 47 ms.
>
> ⚠️ **The 156 GB/s wall does not apply to this model.** §20.5 measured 184.8 GB/s by D2D copy at the
> score tensor's exact size. 156 is the MoE's *strided* figure and is correct there; the tower streams
> contiguously. Quoting either out of its shape is how §20.2's roofline came out wrong.
>
> **Next: §20.7 — price the P@V fusion, which is now the tower's largest kernel** (84.34 ms/iter,
> 3.8× off bound). It is the identical question §20.5 just answered for QK^T and it has not been
> asked here; `scripts/vit_attn_bench.py` answers it with no compile. Item **0n** (the MoE per-CTA
> cost) is still open and still a measurement, not a build.
>
> ✅ **The submodule loose end is closed** (not by me — pushed interactively this session).
> `dff702c` is on the remote and `3281f97f` advances the parent pointer, so the three-session
> "`M 3rdparty/tvm` is deliberate" caveat is retired and **`git status` is down to `?? COLCON_IGNORE`
> alone**.
>
> <details><summary>Handoff, end of 2026-07-27 (superseded)</summary>
>
> **Handoff, end of 2026-07-27.** Branch `qwen3_5`. **Both items §18 left open are closed, and
> neither changed a default.** Item **0l** is refuted — the mechanism works and the premise behind it
> does not (§19.1–§19.4). Item **0m** is fixed — a VL-only compile break, down to two ops and a
> pattern check (§19.5–§19.7). Current 35B lib is still **`lib_blkk64.so`**.
>
> | 35B-A3B, prose, `radix`, one clock state | `lib_blkk64` (shipped) | `lib_rowspec64` (item 0l) |
> |---|---:|---:|
> | pp128 | **550.51** | 500.17 (−9.1%) |
> | pp512 | 836.56 | **855.89** (+2.3%) |
> | pp2048 | 946.32 | **1061.53** (+12.2%) |
>
> Decode neutral (59.0–60.1 both legs). ⚠️ `jetson_clocks` needs an interactive sudo, so absolutes sit
> ~1–3% under §18.9's; the A/B is unaffected and `lib_blkk64` at pp2048 reproduces §18.9's 945.00 to
> **0.14%**, which is the control that makes the sessions comparable.
>
> ### Read this before touching the MoE again
>
> **The reason a wide `BLK_M` loses at short prompts is not padding-row compute.** §17.10, §18.7 and
> §18.11 all say it is. §19.3 removed that compute entirely and recovered **2 points of a 31-point
> gap**. What remains is per-CTA, and §19.8 names three surviving `BLK_M`-scaled terms straight off
> the emitted CUDA — shared footprint 20.25 → 27.0 kB, and 4× (not 1×) `X_shared` stores and
> `load_matrix_sync` per k-step, both of which sit *above* the loop 0k and 0l guarded. That is item
> **0n**, and it is a measurement, not a build. Do not cost another wide-tile variant until it is
> answered.
>
> §18's two instrument traps still stand: the bench prompt picks winners (§17.9 — use `--prompt-file`)
> and the microbench's synthetic routings are wrong at B=4096 (§18.2 — use
> `--indptr-file tuning/expert_hist_35b.npz`). Note that B=1024 is the one shape where synthetic
> routings are all that exist; §19.3's short-prompt numbers use them and say so.
>
> ### Start here: item 0o, the first VL performance number
>
> `dist/qwen3_5-0.8B-vl-q0f16/lib_cublas.so` is the first VL lib at **default `--opt`** (cuBLAS on)
> and it gates **184/184**, so §18.12's `cublas_gemm=0` workaround is retired — and `lib.so`, the
> old workaround build, is now the other half of a free A/B. **No VL performance number has ever
> been taken on this box in any configuration**, which makes it the largest unmeasured surface here
> and the cheapest thing to fix: one harness, no compiles.
>
> Three separable numbers — `image_embed`, `prefill`, `decode` — from an extension of the
> `--greedy-parity-vl5` driver ([validate.py:1180](validate.py#L1180)), which already makes all three
> calls on the raw VM. **Do not reach for `scratch_mlc_tg_sweep.py`: `MLCEngine` cannot drive this
> model's vision tower** (its `image_embed` call site is the llava signature, and `ImageData`
> hardcodes the embed size). Item 0o has the details and the two other traps.
>
> Item **0n** is the MoE follow-up and is a measurement too, but it is the deeper hole; 0o is the one
> that costs nothing and closes a gap nobody has looked at.
>
> <details><summary>Handoff, end of 2026-07-26d (superseded)</summary>
>
> **Handoff, end of 2026-07-26d.** Branch `qwen3_5`, five commits, **no uncommitted work**.
> `git status` shows exactly `M 3rdparty/tvm` and `?? COLCON_IGNORE`, both deliberate (see the end).
> Current 35B lib is **`lib_blkk64.so`**.
>
> | 35B-A3B | filler prompt | real prose (§17.9) |
> |---|---:|---:|
> | pp512 `radix` | 875.37 (from 767.20, **+14.1%**) | **838.06** |
> | pp2048 `radix` | 953.26 | **945.81** |
> | tg512 | 59.97 (unchanged) | — |
>
> ### Read these two before touching the MoE
>
> 1. ⚠️ **The bench prompt picks winners (§17.9).** `PROMPT_FILLER` is one sentence repeated — **11
>    distinct tokens per 512**, against 219 for prose. Harmless on a dense model; on this MoE the
>    router keys on hidden states, so it concentrates routing, and expert concentration sets the
>    GEMM's tile count. It made every pp512 figure here ~4% optimistic, inflated one A/B 3×, and made
>    the microbench predict item 0h's **wrong sign**. Use `--prompt-file` (build with
>    `scripts/make_prose_corpus.py`) for anything routing-dependent. `moe_blkm_check.py` /
>    `moe_skippad_ab.py` synthetic routings are for the **bit-exactness bar, not for ranking.**
> 2. **§17 corrects itself twice.** §17.8 retracts §17.1's claim about the hoist; §17.10 then
>    measures it. Do not act on §17.1 alone.
>
> ### State of the MoE GEMM — largely done
>
> **§17.7 roofline'd the real CTAs: 85–87% of the 156 GB/s wall on balanced routing, 16% of the
> tensor ceiling.** That is §5's tier-1 band. `BLK_K` 32→64 (§17.3) is the whole reason (57%/68% →
> 85%/87%) and was the session's win. The padding lane is closed (§17.2): §16.10's "20% residue" is
> **CTA launch overhead**, not the store tail, and only ~a sixth of it was recoverable.
>
> ### Where to start: item 0i, and do it before ranking anything else
>
> **Dump the real expert histogram (item 0i).** Every ranking decision in this kernel rests on an
> assumed routing distribution, and §17.9 showed *both* instruments in use are unrepresentative — the
> bench prompt concentrates the router, and the microbench's synthetic routings predicted item 0h's
> **wrong sign**. §16.8 and §16.11 both filed this as "worth having" and skipped it; §17.9 is what
> that cost.
>
> It needs **no compile and no MLC instrumentation**: the routers are
> `model.language_model.layers.{i}.mlp.gate`, plain `Linear(2048, 256)` kept in bf16 by the fp8
> checkpoint, so forward hooks under the existing fp8 HF path (`fp8_software_dequant.py`, §6.1) give
> top-8 assignments per token per layer. Produce `sum_e ceildiv(count_e, BLK_M)` for
> `BLK_M` ∈ {16, 32, 64}, the padding share and the hit-expert count, at pp512 **and** pp2048; then
> feed the real `indptr` into `moe_blkm_check.py` in place of its synthetic routing. ~Half a day.
>
> **Then item 0h**, whose numbers 0i will move. The hoist is
> built and bit-exact (`MLC_MOE_GEMM_V2_HOIST=1`, inert at `BLK_M=16`). Measured on prose:
> `BLK_M=64`+hoist is **+12.6% pp2048 / −10.4% pp128**, `BLK_M=32` is +7.8% / −5.5%; crossover
> ≈ pp450. **No compile-time `BLK_M` is Pareto**, so defaults are unchanged. Shipping it means
> emitting `(BLK_M=16)` and `(BLK_M=64)` dispatch+GEMM pairs and branching on `x.shape[0]` near
> **B ≈ 3600** — `LowBatchGemvSpecialize` is the in-tree precedent. Worth **+12.6% on ≥2048-token
> prefill at no short-prompt cost**, on a 262144-context model chunked at 2048. `lib_hoist64.so` and
> `lib_hoist32.so` are already built as A/B legs.
>
> Also open, both smaller: **item 0c.2** (chunked recurrence, capped at +12.5% by Amdahl, changes the
> arithmetic so bit-exactness is off the table) and **the VL re-gate** — *not* blocked, no download;
> see its entry. The VL path has silently inherited five state-path changes (§11, §13, §14, §15,
> §16.5) and has not been gated since `f667b07e`, which makes it the largest ungated surface here.
>
> §17.5 records one dead end: widening the W fetch to a whole `uint32` per thread hits TVM's 4-lane
> `Ramp` ceiling.
>
> ### Two non-code loose ends, carried forward unchanged
>
> `3rdparty/tvm` commit `dff702c` is still unpushed (needs an interactive shell or an SSH remote), so
> `M 3rdparty/tvm` is deliberate; `COLCON_IGNORE` is an untracked ROS artifact predating this work.
>
> <details><summary>Handoff, end of 2026-07-26b (superseded)</summary>
>
> Branch `qwen3_5`, six commits this session (`ad56b584`,
> `dfd7fec9`, `3c6edc2b`, `7865c5de`, `6c0ef467`, and this one). **No uncommitted work** — nothing is
> half-finished, every change is gated, measured and committed. `git status` shows exactly two
> entries, `M 3rdparty/tvm` and `?? COLCON_IGNORE`, both deliberate and both explained below.
>
> **Performance as of this session's end:**
>
> | | decode tg512 | prefill pp512 (`radix`, the default) |
> |---|---:|---:|
> | **35B-A3B** | **60.09** tps (from 54.13, +11.0%) — 61.6% of the 97.5 tps achievable roofline | **769** (from 355 at the start of 2026-07-25c, **+117%**) |
> | **0.8B** | ~90 tps | **4888** (from 1469, +233%) |
>
> **What landed today.** §16.5: the lane-split GDN recurrence, `MLC_QWEN35_GDN_KSPLIT` default **4**
> — 0.8B pp512 **+25.0%**, 35B **+2.3%**, decode neutral on both, and the full gate battery passes
> (361/361 and 139/139 on the state gates under *both* prefix-cache modes, negative control still
> failing 342/361). §16.6: item 0d's `v_block`, **opt-in at default `0`** because it is +15.6% on the
> 0.8B and −8% on the 35B — so the default configuration and every gate result above are unchanged
> by it.
>
> ### Where to start: no queued item — pick from three, none costed
>
> §16.7 traced 35B prefill *after* §16.5/§16.6 and **§9's priority order was wrong**:
> `dequantize_group_gemm_v2`+`_v21` are **52.5% of prefill** (MoE machinery overall ~64%), while the
> GDN recurrence that four sections called "the biggest prefill item by 5×" is **11.1% and third**.
> That framing was measured on the 0.8B, and §16.5 cut it further. Consequently **item 0c.2, the
> chunked reformulation, is capped at +12.5% on the 35B by Amdahl** — do not start it first.
>
> **§16.8 then measured the GEMM (item 0e, closed).** It uses tensor cores and is bound by neither
> wall — 5.5% of the fp16 tensor ceiling, 28.6% of the 156 GB/s one. It is **CTA-bound**: cost is
> `n_real·c_real + n_pad·c_pad` with `c_pad ≈ 0.93 c_real`, fitting all 12 sweep points to ≤1.6%.
> And **27–50% of its CTAs are dispatch-table padding that runs the full dequant + wmma and discards
> the result**, because the sentinel path guards only the `X` read and the store.
>
> **§16.8–§16.11 closed items 0e and 0f. 35B prefill went 642 → 769 tps (+19.4%).** The MoE expert
> GEMM was never bandwidth- or compute-bound: it is CTA-bound, and 27–50% of its CTAs were dispatch-
> table padding running a full dequant + wmma and discarding it. `MLC_MOE_GEMM_V2_SKIPPAD` (now
> default `1`) gives the `k_o_o` loop a zero trip count on those — a `Select` on the loop *extent*,
> **not** an `IfThenElse`, because `ThreadSync` refuses a `__syncthreads()` inside a condition.
> Bit-exact, and the state gate is *identical* to the pre-change lib in both modes, not merely
> passing. Current 35B lib is **`lib_skippad.so`**.
>
> **There is no queued item.** The open list is item 0c.2 (de-prioritised, +12.5% ceiling) and the
> blocked precondition. Three candidates, none costed:
> 1. **Register-blocking the v2 inner loop.** §16.9 found `i_o` sits outside the k-loop, so widening
>    `BLK_M` re-runs the whole dequant per row-fragment (0.64×/0.39×). Hoisting the shared loads above
>    `i_o` is both the fix and the standard blocking that would lift v2 off **5.5% of the fp16 tensor
>    ceiling**. Largest remaining prize in the MoE, still unquantified.
> 2. **The last 20% of a skipped CTA** (§16.10) — zero the trailing store loops too. ~3% end-to-end,
>    small and well understood, but the extents are not unique so it needs a targeted match.
> 3. **A direct indptr histogram.** §16.11 *inferred* ~40% padding in production from the end-to-end
>    number; observing it would check that inference.
>
> Do not re-litigate what §16.9/§16.10 closed: a source-level `if` dies in `sch.compute_at`; an
> `IfThenElse` around the scheduled body dies in `ThreadSync`; widening `BLK_M` alone regresses.
>
> ### Two non-code loose ends
>
> 1. **`3rdparty/tvm` commit `dff702c` is still unpushed** (the §16.3 `MLC_GEMV_TSTR` hook), and the
>    parent pointer is deliberately not advanced — see "Committed state" for the commands and why.
>    Re-verified 2026-07-26b: `git ls-remote origin` has no ref containing it, and the remote is
>    HTTPS with no credential helper, so it needs an interactive shell or an SSH remote. **Nothing
>    from today touches TVM**, so this is unchanged rather than newly blocking; `git status` showing
>    `M 3rdparty/tvm` is the expected, deliberate state, not dirt.
> 2. `COLCON_IGNORE` is untracked and predates this work — a ROS-workspace artifact, left alone
>    rather than folded into an unrelated commit.
>
> ### Environment
>
> **`source .envrc.local` before anything.** Nothing is pip-installed: TVM and `mlc_llm` come from
> the source tree and `.venv/` holds only `tvm_ffi`, so a bare `python` fails on `import tvm`, then
> on `tvm_ffi`, then on `nvcc`. GPU clocks are already pinned at max
> (`/sys/class/devfreq/17000000.gpu`, min == max == 1300.5 MHz), so benches need no `jetson_clocks`.
>
> **Five traps this session paid for, in order of how much time they cost:**
> 1. **Never run two benchmarks at once, and do not conclude one has died because it is quiet.** A
>    split kernel spends 30–42 s *per config* in ptxas, so a sweep looks idle for many minutes.
>    A live run was declared dead — **`ps -C python` does not find it**, because the process shows as
>    `timeout NNNN python ...` — and two more were started on top of it. Three benches shared the GPU
>    and that table was thrown away and re-measured. Use `pgrep -af gdn_kernel_bench`.
> 2. **In a probe that A/Bs a hand-written baseline against a hand-written variant, the *baseline* is
>    the dangerous half.** §16.2's variant was within 1.6% of TIR; its `base` was 35% slow, which is
>    the whole of its over-prediction. It was checkable without building anything — §15.2 had traced
>    the real kernel at 2.65 ms/call and the probe's `base` read 3.49.
> 3. **Do not quote `gdn_recurrence_probe.cu` ratios against the 35B.** Its grid is `n_kh = 16`,
>    which is the 0.8B's `n_vh`; the 35B launches 32 blocks and behaves qualitatively differently
>    (`k_split=2` is a *regression* there). Use `gdn_kernel_bench.py`, which uses the real grid.
> 4. **ptxas takes 30–42 s on a split kernel**, so anything that recompiles in a loop gets slow
>    fast. The gate now memoizes; check before adding a sweep axis.
> 5. `Executable` has no `time_evaluator` — it lives on `.mod`.
>
> **And the pattern behind three of this session's corrections:** every one of them was an
> *extrapolation from a measurement made under different conditions* — the probe's grid, §15.6's
> model, item 0d's occupancy arithmetic, §9's priority order. Each was checkable cheaply and none
> was checked until it had already been written down as fact. When a number is quoted from another
> section, re-read what configuration it was measured on before building on it.
>
> </details>
>
> </details>
>
> </details>
>
> ### Environment traps that still bite, carried forward
>
> **`source .envrc.local` before anything** — nothing is pip-installed, so a bare `python` fails on
> `import tvm`. GPU clocks are pinned at max already; `jetson_clocks` itself needs an interactive
> sudo this box does not have non-interactively.
>
> ⚠️ **`--mlc-lib` is not optional on the VL model any more.** `validate.py` defaults to
> `<model-dir>/lib.so`, and for VL that is still §18.12's `cublas_gemm=0` build — now the **worst of
> the three** on ttft (459.25 vs `lib_nofp32blas`'s 443.20). Same shape as the 35B, whose current lib
> is `lib_blkk64.so` rather than `lib.so`; name the lib explicitly on every VL run.
>
> **Never run two benchmarks at once, and do not conclude one has died because it is quiet.**
> `pgrep -f "mlc_llm compile"` **matches the polling shell itself**, because that string is in the
> poll loop's own command line — a finished 35B compile looked like it was still running for ~20
> minutes. Check for the output artifact (`ls -la <lib>.so`), not for the absence of a process.
> **The 35B compile takes ~14 min**; a full `mlc_llm compile` of the 0.8B VL model takes ~3 min; one
> `scratch_mlc_tg_sweep.py` leg on the 35B is ~2 min including load.
>
> **One non-code loose end left.** ✅ `3rdparty/tvm` commit `dff702c` was pushed interactively and the
> parent pointer advanced (`3281f97f`, 2026-07-27b), so `M 3rdparty/tvm` is **gone** — see "Committed
> state". `COLCON_IGNORE` remains an untracked ROS artifact predating this work, and is the only
> entry `git status` now shows.

> **Item IDs are stable, not sequential.** They are referenced from §12–§16 and from the Done
> sections above, so closed items keep their number rather than being renumbered away. Ordering
> below is by measured expected value as of **2026-07-26b**, re-ranked by §16.7's prefill trace.

#### Open

> ### ⬆️ Superseded by §18 — read §18.11 first.
>
> **Item 0i is done** (§18.1) and it moved more than expected: it retracted §17.7's roofline (the
> kernel is at **45%** of the bandwidth wall, not 85–87% — §18.4), corrected §16.11's padding share
> (29.3%, not ~40%), and turned item 0h from "wrong sign at pp512" into "wins at every measured
> shape". Two new items were built and measured on the back of it: **0j (tile order) is refuted**
> (§18.6) and **0k (row-fragment skip) works but only interpolates** along the same frontier (§18.7).
> §18.9 maps that frontier across 4 libs × 3 prompt lengths; **no configuration is Pareto** and the
> defaults are unchanged. The list below is kept for the items §18 did not touch.
>
> | item | state | worth |
> |---|---|---|
> | **0n** | **(§19.8)**; a measurement, not a build. Why is `BLK_M=64` 30–40% slower at B=1024 when it launches the *same* CTAs over the *same* rows? | it is the only unexplained term left in the MoE GEMM, and every wide-tile idea has died on it |
> | **0o** | ✅ **DONE (§20)** — first VL numbers taken; cuBLAS was net-negative and `MLC_BLAS_SKIP_FP32` fixes it, ttft −8.9%, gate 184/184 | — |
> | **0p** | 🔶 **half done (§20.5–§20.7)** — softmax is closed (98.5% of the *measured* 184.8 GB/s wall), QK^T is closed (cuBLAS, after prescaling made it eligible), `image_embed` 293 → **223 ms**. `scripts/vit_attn_bench.py` built, 1.3% fidelity | — |
> | **0q — START HERE** | **new (§20.7)**; price the `astype` fusion on `matmul(attn_probs, v32)`, now the tower's largest kernel at 84.34 ms/iter and 3.8× off bound | it is the *same question* §20.5 answered for QK^T — the epilogue writes 3.87 MB against a 305 MB read, so it is probably protecting nothing — and the microbench answers it with **no compile** |
> | ~~**0l**~~ | ✅ **built, §19.1–§19.4 — and it refutes its own premise.** The mechanism works (`BLK_M=16` control: 1.00× where 0k cost 5–9%); the hypothesis it was built on does not | closed. pp128 is still −9.1% end-to-end, so no wider tile is Pareto and the defaults are unchanged |
> | ~~**0m**~~ | ✅ **fixed, §19.5–§19.7** — two ops, both in the VL patch merger; the guard declines exactly those two matches | a default-`--opt` VL lib now compiles and gates **184/184** |
> | ~~**0i**~~ | ✅ **done, §18.1** | did what it was for — see §18.11's before/after table |
> | ~~**VL re-gate**~~ | ✅ **cleared, §18.14** — margin scoring added; 0 wide-margin divergences | the five inherited state-path changes are gated on VL for the first time since `f667b07e` |
> | **0h** | re-costed by §18.9; the runtime branch is the **wrong shape of fix** (§18.11). 0l did *not* make it unnecessary (§19.4) | the frontier's upper envelope, ≤ +12.5% at pp2048 only — and `lib_rowspec64` now reaches it without a branch, at the same pp128 cost |
> | **0c.2** | de-prioritised; changes the arithmetic, so bit-exactness is off the table | ≤ +12.5% by Amdahl |
>
> **What §18 changed about how to read the rest of this document.** §17 believed it had closed the
> MoE lane — §17.7 put the real CTAs at 85–87% of the bandwidth wall and concluded "there is no
> bandwidth story left in this kernel". **§18.4 retracted that**: measured on real routing the figure
> is **45%**, and the gap is tile fragmentation. Items 0b, 1, 5 closed 2026-07-26a; **0c.1** landed in
> §16.5, **0d** opt-in in §16.6, **0e/0f** closed 2026-07-26c, **0g** (`BLK_K`) landed at +14.1%.
>
> ⚠️ **Two traps before ranking anything new in this kernel.** Read **§17.9** (the bench prompt
> concentrates the router — use `--prompt-file`) and **§18.2** (the microbench's synthetic routings
> are wrong at B=4096 specifically — use `--indptr-file tuning/expert_hist_35b.npz`). Both are now
> fixed in the tooling; neither is fixed by default.

**0n. ⬅️ NEXT — find the wide tile's per-CTA cost, which is not what four sections said it was.**

*The question, stated so it cannot be answered by argument.* At B=1024 (the pp128 scale) no expert
holds more than 64 rows, so `BLK_M=16` and `BLK_M=64` launch **the same number of CTAs** over the
**same real rows** and fetch the **same weight tiles**. `BLK_M=64` is nonetheless **30–40% slower**
(§19.3). Everything that differs is per-CTA. Which term is it?

*Three candidates, already read off the emitted CUDA* (§19.8 has the table; `scripts/moe_dump_cuda.py`
at `BLK_M` 16 vs 64+hoist+`ROWSPEC`). All three live *above* the row-fragment loop that items 0k and
0l guarded, which is why predicating that loop bought 0–3%:

(a) **Shared-memory footprint** — 20.25 kB → 27.0 kB, so 8 resident CTAs become 6 on a 164 kB/SM
budget. 1.33×, so probably not the whole 30–40% by itself. Same effect §17 blamed for the `BLK_K=128`
regression. (b) **The `X_shared` cooperative store** — 4 fragments per k-step instead of 1. The load
is predicated on `row_end` so padding rows cost no DRAM, but the shared store and the `condval` are
paid anyway. (c) **`load_matrix_sync` of the X fragments** — 4 per k-step instead of 1, because
`A_mat` is attached at `k_o_i`, outside the guarded loop.

*How to settle it.* (b) and (c) are both testable by pushing the same predicate one level up — the
`A_mat` cache-read loop is annotatable exactly as the fragment loop was, and the `X_shared` store's
`ax0_ax1_fused_0` extent is a candidate for item 0k's *extent* trick (it carries no barrier of its
own). If instead (a) dominates, the follow-up is not a wider tile at all but a **narrower `BLK_N`** at
`BLK_M=64`, trading the same shared budget the other way. `ncu` would rank the three outright but is
still blocked (§8).

*Why it matters.* §18.9's frontier has killed every wide-tile idea at the short-prompt end, and
§19.3 shows the mechanism everyone assumed is worth 0–3% of a 31–40% gap. Until this term is named,
any further wide-tile work is guessing — which is precisely how items 0h, 0k and 0l were each costed.

**0q. ⬅️ NEXT — price the P@V cast fusion, the way 0p priced the QK one.**

*The question.* `matmul(attn_probs, v32)` then `astype(..., fp16)` fuses into one kernel,
`fused_matmul9_cast17`, and at **84.34 ms/iter (7.03 ms/layer, 3.8× off bound)** it is now the
tower's largest. cuBLAS never took it — in *both* §20.2 legs it stayed a generated kernel — and the
`astype` epilogue is the likely reason. §20.5 asked exactly this about the QK matmul's `multiply`
epilogue and the answer was **0.00 ms**, which is where the 70 ms came from. Nobody has asked it here.

*Why it is probably the same answer.* The epilogue writes `(h, s, d)` fp16 = **3.87 MB** against a
`(h, s, s)` fp32 = **305 MB** read of `attn_probs`. There is no arithmetic to rearrange — just a cast
that could move after an unfused matmul. If the fusion is worth ~nothing, cuBLAS gets a bare fp32
GEMM at roughly the QK one's 3.63 ms/layer against today's 7.03, i.e. **~41 ms/iter**, and
`image_embed` goes 223 → ~182 ms.

*How, and it costs nothing.* `scripts/vit_attn_bench.py` already builds this exact block. Add the
split variant, time `matmul` and `cast` separately against the fused 7.04 ms, and the answer is one
run — no compile. **Do that before building anything.** If it comes back non-zero, the fusion is real
and the lane is closed; if it comes back ~zero, it is the same one-line graph change as §20.6.

⚠️ Do not assume the QK result transfers. §20.5's null was measured, not reasoned, and the two
epilogues differ: a scalar `multiply` reads and writes the same 305 MB, while this `astype` reads
305 MB and writes 3.87 MB. The traffic argument that made prescaling free does not obviously apply.

**0p. 🔶 HALF DONE, §20.5–§20.7 — two of three kernels closed, and the wall was wrong. Entry below.**
The instrument (`vit_attn_bench.py`, 1.3% fidelity) and two hypotheses: symbolic shapes are worth
6.5% not 4× (refuted), and prescaling is worth **0.00 ms on its own** — which is what proved the QK
fusion protects nothing and unlocked cuBLAS for 70 ms. **The wall is 184.8 GB/s, not the 156 this
entry assumed**; re-based, softmax is at 98.5% and closed. What remains of the original framing is
P@V (now item 0q) and, behind it, flash attention.

**0p (historic). Stop materializing the vision tower's score matrix.**

*The shape of it.* At `lib_nofp32blas`, `image_embed` is 293 ms and **84% of it is three fp32 kernels
that all move the same `(12, 2520, 2520)` fp32 = 305 MB score tensor**: `fused_NT_matmul8_multiply22`
(QK^T + scale, 113.5 ms), `fused_matmul14_cast17` (P@V + cast, 84.3 ms) and `softmax` (48.4 ms).
§20.2 measured what one round trip of that tensor costs — **3.91 ms/layer at the 156 GB/s wall** —
and the whole of §20.3's win was removing exactly one of them.

*Why a better GEMM is the wrong target.* cuBLAS demonstrated 2.69 TFLOP/s against the generated
kernel's 1.03 and **still lost**, because the tensor dominates the GEMM. Tiling the attention so the
scores never reach DRAM — flash attention — is the only change that addresses the term that is
actually large. HF runs this tower with eager attention, so there is no reference kernel to copy;
the parity constraint is that the math stays fp32
([qwen3_vl_vit.py:151](python/mlc_llm/model/vision/qwen3_vl_vit.py#L151) — fp16 collapses tower
parity at max diff 2.03 / rel 39%).

*Cheaper things to rule out first, in order.* (a) `softmax` at 48.4 ms is 610 MB of traffic for one
pass — check it against the wall before assuming it is optimal. (b) The tower is 12 layers of
identical shape, so a single-layer microbench is cheap to build and would let this be costed before
committing to a fused kernel. (c) `seq_len` is 2520 for one image at this fixture size; confirm how
it scales before optimising for this shape alone.

**0o. ✅ DONE, §20 — the numbers exist, and they changed the default. Historic entry below.**
Three numbers taken (`image_embed` 337.28 / `prefill` 149.37 / `decode` 11.36 ms on the then-default
lib). The A/B **inverted**: the `cublas_gemm=0` build §19.6 retired as a "workaround" was 5.6% better
on ttft. Split by call — as this entry insisted — it resolved into cuBLAS winning the fp16 FFN GEMMs
and losing the fp32 QK^T by 47.3 ms. `MLC_BLAS_SKIP_FP32=1` takes both wins: ttft 486.65 → **443.20**
(−8.9%), gate 184/184 exact. The entry's own closing question is answered **backwards** — see §20.4.
All three traps below were real: the host merge, the "cheap" `image_embed` (337 ms, five times a
gate) and the engine.

**0o (historic). Take the first VL performance number — and do not reach for the engine to do it.**

*Why now.* No VL performance number has ever been taken on this box, in any configuration. §19.6 made
the missing half of the A/B exist: `dist/qwen3_5-0.8B-vl-q0f16/lib.so` is §18.12's `cublas_gemm=0`
build and `lib_cublas.so` is the default-`--opt` one with 20 cuBLAS offloads, so **the first
measurement costs one harness and no compiles.** Both libs are gated (§18.14, §19.7).

*What separates into three numbers.* `image_embed` (vision tower + patch merger, once per image),
`prefill` of the merged sequence, and `decode`. They have different shapes and different consumers —
a chat turn pays `image_embed` once and `decode` hundreds of times — so a single ttft figure would
hide the thing worth knowing.

*How, and what to reuse.* Extend the `--greedy-parity-vl5` driver
([validate.py:1180](validate.py#L1180)): it already loads lib + params, preprocesses the fixture, and
calls `image_embed` / `embed` / `prefill` / `decode` directly on the VM. A perf harness is that driver
minus the reference comparison, plus `dev.sync()` and timing around each call.

*⚠️ Three traps, all of them already visible in that driver.*

1. **Do not time the loop.** It merges image embeddings into the text embedding **on the host** —
   `image_embeds.numpy()`, a numpy scatter, then a re-upload — per prompt. Timing the loop measures
   numpy. Time the three VM calls individually, with a sync on each.
2. **`image_embed` is called inside the per-prompt loop**, under a comment reading "same image across
   prompts; could cache but cheap". **"Cheap" is untested.** It is the first thing to measure, and if
   it is wrong the gate has been paying it five times over.
3. **`MLCEngine` cannot drive this model's vision tower, so `scratch_mlc_tg_sweep.py` is not the
   instrument.** [cpp/serve/model.cc:144](cpp/serve/model.cc#L144) calls
   `image_embed(image, resize_h, resize_w, crop_h, crop_w, params)` — the llava signature — while
   Qwen3.5-VL's takes `(pixel_values, pos_embeds, rotary_cos, rotary_sin, params)`; and `ImageData`
   hardcodes `embed_size` to 576/1921 ([serve/data.py:107](python/mlc_llm/serve/data.py#L107)).
   **Serving the VL model is a separate and much larger piece of work than measuring it** — do not
   let the two merge. Every 35B/0.8B throughput number in this document came from the engine; none of
   that tooling transfers here.

*What the A/B can and cannot answer.* Both libs differ in the text stack as well as the tower — a
`q0f16` text build offloads 13 matmuls on its own (§19.6) — so a whole-model delta does not attribute
itself. Split by call before comparing. If the tower turns out to be where cuBLAS matters, the
follow-up question is whether the **two declined merger matmuls** (3072×3072 and 1024×3072, over
`num_patches // 4` rows) are worth recovering by giving them a bare-`tir.Var` shape, which would make
them eligible again.

**0l. ✅ DONE, §19.1–§19.4 — the mechanism works and the premise does not. Historic entry below.**
Built as `MLC_MOE_GEMM_V2_ROWSPEC=1`, though as a *predicate on a constant-extent loop* rather than
the four-way static specialisation described here — same effect, a quarter of the code. The
`BLK_M=16` control came in at **1.00×** against 0k's 0.91×, which confirms §18.11's diagnosis of 0k.
But the prediction below — that this makes a wide tile beat `BLK_M=16` at every prompt length — is
**refuted**: pp128 is still −9.1% end-to-end, and at the kernel level removing *all* padding-fragment
reduction closes only 2 points of a 31-point gap. See item 0n for what the cost actually has to be.

**0l (historic). Statically specialise the row-fragment count, so a wide `BLK_M` stops paying for
its own padding.**

*The one-line version:* `BLK_M=64` already cuts tiles **1.90× at pp512 and 2.84× at pp2048** (§18.1),
and tile count is what sets weight traffic — but it pays that back in padding-row compute, and item
0k's fix for the padding costs more than it saves *because it made a constant loop extent dynamic*.
Make the trip count static again by specialising on it, and the win should survive.

*Why this is the lever.* §18.4 established the kernel is at **45% of the bandwidth wall** with issued
bytes **2.9× unique** — it is fragmentation-bound, not bandwidth-bound. With the hoist, a CTA loads
one `BLK_N × K` weight tile, so total weight traffic is proportional to
`n_real = sum_e ceildiv(count_e, BLK_M) · tiles_per_n`. Cutting tiles cuts weight traffic
proportionally, and that is exactly why `BLK_M=64` wins 1.56× at pp2048 (§18.3). §18.9 then showed
every wider tile also *loses* at short prompts, monotonically, because a real prefill puts only
**~24 rows on each hit expert** (§18.1) so a 64-row tile is mostly padding.

*Why item 0k did not already solve it.* 0k gives the row-fragment loop a runtime extent
`min(ceildiv(rows, 16), BLK_M/16)`, which is correct and bit-exact — and measured **0.91× at
`BLK_M=64`** (§18.7). The tell is `BLK_M=16`, where the guard is logically inert (extent 1) and still
costs **5–9%**: the loss is not the skipping, it is that a runtime extent blocks the unroll. 0k pays
that on every CTA and only recovers it where padding dominates, which is why it nets out at `BLK_M=32`
and nowhere else.

*The change.* At `BLK_M=64` there are only **four** possible active-fragment counts (1, 2, 3, 4). Emit
four statically-unrolled bodies and select with a `Select` on the loop extent the same way item 0f
selects between 1 and 0 — or, if that is unwieldy, hoist the choice into four `T.serial` nests under
a CTA-uniform branch. Every fragment count keeps its unroll; empty fragments cost nothing. The
uniformity and bit-exactness arguments are unchanged from 0f and 0k: the trip count depends only on
`tm[bx]` and `indptr`, so it is CTA-uniform, and a skipped fragment's global store is predicated off
by `m_offset + i < row_end` regardless of what its accumulator holds.

*What success looks like, and what refutes it.* The prediction is that `BLK_M=64` + hoist + static
fragment specialisation beats `BLK_M=16` **at every prompt length**, i.e. it crosses the frontier
§18.9 mapped rather than sliding along it. Concretely: recover most of 0k's `BLK_M=64` loss (0.91× →
≥1.0× on the kernel) while keeping the 1.59×/1.51× that plain `M=64`+hoist already has at pp2048.
**It is refuted if the pp128 end-to-end leg still loses** — that is the cell that has killed every
wide tile so far, and it is the first thing to measure, not the last. Gate with
`scripts/moe_skiprows_ab.py` (bit-exactness) then `scripts/moe_blkm_check.py --indptr-file` before
compiling anything.

*Also considered and probably dead: mixed tiles.* The obvious way to eliminate the remainder entirely
is to let one CTA cover rows from two experts, since the dispatch table already stores a per-CTA row
offset. **The weight tile is per-expert**, so a mixed tile needs two `BLK_N × K` weight loads — which
doubles the dominant cost to save one tile. Recorded so the next session does not re-derive it.

*If 0l works, item 0h's runtime branch is moot* — the whole point of the branch was to get the wide
tile's long-prompt win without its short-prompt cost.

**0m. ✅ DONE, §19.5–§19.7 — it was two ops, both in the patch merger, and the fix is a pattern
check.** `FuseOpsByPattern` appends a `tir_vars: R.Shape([...])` parameter to any lifted region with a
free symbolic var, and the BYOC JSON serializer requires every parameter to be a tensor. The VL merger
feeds `linear_fc1`/`linear_fc2` a `R.Tensor((num_patches // 4, 3072))`, which *uses* `num_patches`
without *defining* it. `BLASDispatch` now declines exactly those matches (VL 22 → 20 offloads, text
13 → 13), a default-`--opt` VL lib builds, and it gates 184/184. Historic entry below.

**0m (historic). The VL-only `BLASDispatch` compile break — diagnosed to one pass, not yet to one
op.**

*Symptom.* Compiling `--model-type qwen3_5_vl` at default `--opt` dies in
[compiler_pass/blas_dispatch.py:40](python/mlc_llm/compiler_pass/blas_dispatch.py#L40), inside
`FuseOpsByPattern` / `RunCodegen`:

```
tvm.error.InternalError: Check failed: (tensor_sinfo) is false:
    Expect TensorStructInfo, but received: relax.ShapeStructInfo
```

*What is already established (§18.12), so it need not be redone.* It is **VL-specific**: the
text-only `dist/qwen3_5-0.8B-q0f16` config compiles cleanly through the *same* default pipeline with
cuBLAS enabled. And it has been invisible until now because `_cublas_gemm`
([compiler_flags.py:103](python/mlc_llm/interface/compiler_flags.py#L103)) enables the pass **only for
unquantized weights** — `q0f16`/`q0bf16`/`q0f32`/fp8 — so every `q4f16_1` build in this document
skipped it entirely. The VL build is the first `q0f16` compile since the box moved to CUDA 13.2.

*Where to look.* Something in the vision tower hands a cuBLAS matmul pattern an argument whose
struct-info is a `ShapeStructInfo` rather than a tensor — most likely an op whose operand is a
`ShapeExpr` (a reshape target, or a `strided_slice` bound) sitting inside the matched region.
`python/mlc_llm/model/qwen3_5_vl/qwen3_5_vl_model.py` (`image_embed`) and
`python/mlc_llm/model/vision/qwen3_vl_vit.py` are the two files that produce it. Bisecting by
`entry_functions` — `BLASDispatch` already filters which functions it enters — should localise it to
one Relax function in minutes.

*Workaround in use.* `--opt "flashinfer=1;cublas_gemm=0;cudagraph=1"` compiles and is what
`dist/qwen3_5-0.8B-vl-q0f16/lib.so` was built with. **Correctness is not at stake** — §18.14 gates
that lib and it passes — so this is about not shipping a VL build with a compiler pass switched off,
and about the fact that **no VL performance number has ever been taken on this box**, with or
without cuBLAS.

*Worth noting for scope:* if a VL build is only ever wanted quantized, `cublas_gemm` never turns on
and this never fires. The bug is real either way, but its priority depends on whether a `q0f16` VL
lib is a deliverable or only a gating vehicle.

**0i. ✅ DONE, §18.1 — dump the real expert histogram, because both current instruments are wrong.**
Historic entry below; the results and what they overturned are in §18.1–§18.4.
Every ranking decision in this kernel rests on an assumed routing distribution, and §17.9 showed
both available proxies are unrepresentative: the bench prompt has **11 distinct tokens per 512** and
concentrates the router, while `moe_blkm_check.py`'s synthetic `even` / uniform-`random` routings
bracket nothing real — they predicted item 0h's **wrong sign**, not merely the wrong magnitude.
§16.8 and §16.11 both filed this as "worth having" and it was skipped twice.

*What to produce:* per-layer expert counts from a real prefill, and from them the only numbers this
kernel's cost actually depends on — `sum_e ceildiv(count_e, BLK_M)` for `BLK_M` ∈ {16, 32, 64}, the
padding share, and the number of experts actually hit. §16.11 *inferred* ~40% padding from an
end-to-end ratio; this measures it.

*How, without touching MLC:* the routers are `model.language_model.layers.{i}.mlp.gate` — 41 of them
(40 layers + the MTP draft), plain `Linear(2048, 256)` kept in bf16 by the fp8 checkpoint's
`modules_to_not_convert`. So a forward hook on each `mlp.gate` under the **existing** fp8 HF path
(`fp8_software_dequant.py`, §6.1 — already runs on this box at 37.5 GB) yields top-8 assignments per
token per layer with no MLC instrumentation and no compile. Prompts should be real prose
(`scripts/make_prose_corpus.py`) at pp512 **and** pp2048, since B differs 4× between them.

*Then:* feed the real `indptr` into `moe_blkm_check.py` / `moe_skippad_ab.py` in place of
`make_inputs`'s synthetic routing, and re-derive item 0h's crossover and §17.2's padding share from
it. Expect the 0h numbers to move — the question is by how much, and whether pp ≈ 450 survives.

*Watch for:* routing may differ by layer depth and between prefill and decode; report the spread, not
just a mean. And the MTP draft layer's router is in that list — exclude it, it is not on the decode
path measured here.

**0c. The GDN recurrence is parallelism-starved — still the biggest prefill item after §16.5.**
§15.6 measured it on the **0.8B**: `gdn_func_history_inplace` is **95.6 ms against 19.3 ms for the
next kernel**, at **202 GFLOP/s, ~3.8% of sm_87 fp32 peak**, because it launches `batch × n_vh`
blocks of `V` threads (**16 blocks of 128 threads** there, on a 16-SM GPU) and each walks the
sequence sequentially.
  1. ✅ **Landed, §16.5 — the lane split, worth +25.0% pp512 on the 0.8B, and it is *not* worth
     what §16.2 predicted.** `k_split=4` in TIR: 0 spill, 0 barriers, 361/361 on the state gate in
     both prefix-cache modes. The kernel A/B is **1.94× on the 0.8B and 1.21× on the 35B** against
     the probe's 2.79×, and **`k_split=2` is a 0.96× regression on the 35B** — the probe's grid is
     the 0.8B's `n_vh = 16`, and the 35B's `n_vh = 32` already fits 2 blocks/SM. Default flipped to
     4; `MLC_QWEN35_GDN_KSPLIT=1` restores §15's bit-exact kernel.
  2. ⬇️ **De-prioritised on the 35B by §16.7's trace — worth at most +12.5% there.** The chunked
     linear-attention formulation (matmuls over a chunk of C positions instead of a scalar loop, as
     in `../flash-linear-attention/fla/layers/gated_deltanet.py` and vLLM's `qwen3_next.py`) is
     still the only lever with a ~15× ceiling *on the kernel*, but the kernel is now **11.1% of 35B
     prefill**, not the dominant item §15.6 measured on the 0.8B. Amdahl caps a perfect version at
     `1/(1 − 0.111)` = **+12.5% prefill**, for the largest and riskiest change in this document: it
     **changes the arithmetic**, so bit-exactness is off the table, item 0b is a prerequisite for
     judging it, and it needs its own chunk-state intermediate. It does **nothing for decode** —
     at `seq_len=1` the chunked form degenerates. Still worth ~+27% on the 0.8B, which is the
     iteration vehicle rather than the target. **Do item 0e first.**

**0e. ✅ Measured, §16.8 — the MoE expert GEMM is CTA-bound, not bandwidth- or compute-bound.**
§16.7's trace put `dequantize_group_gemm_v2` + `_v21` at **395.9 ms of a 754 ms pp512 prefill**. The
three questions it posed are answered: it **does** use tensor cores (`nvcuda::wmma::mma_sync`), and at
B=4096 it sits at **5.5% of the fp16 tensor ceiling and 28.6% of the 156 GB/s wall** — so neither.
Its time is `n_real·c_real + n_pad·c_pad` with `c_pad ≈ 0.93 c_real`, a two-parameter fit that
predicts all 12 sweep points to ≤1.6%. §16.4's "flat in batch" was the right observation with the
wrong mechanism: the grid is 97% padding at B=8, not reading all 256 experts. Closes into **0f**.

**0f. ✅ CLOSED, §16.11 — shipped on by default, +19.4% pp512 on the 35B (644.26 → 769.18 tps).**
Bit-exact: 8/8 on `scripts/moe_gemm_check.py` under `np.array_equal`, and the 35B state gate is
*identical* to the pre-change lib in both prefix-cache modes — same τ=1.0 mismatch count, same
near-tie count — which is the end-to-end version of the same claim. Decode neutral (60.16 → 60.09).
`MLC_MOE_GEMM_V2_SKIPPAD=0` restores the un-skipped kernel. Historic detail below.

<details><summary>§16.10's entry — the build, before the end-to-end numbers</summary>

**Was: built, opt-in, pending the end-to-end measurement.** Bit-exact on all 8 gate cases and **1.28×–1.73×** on the kernel; the
guard is a `Select` on the `k_o_o` loop *extent*, not an `IfThenElse`, because `ThreadSync` refuses
to place a `__syncthreads()` inside a condition. A skipped CTA still costs **20%** of a full one
(accumulator fill + `O_tile` store + the predicated-off store loop), so 80% of the padding cost is
recovered, not all of it. **To close it:** rebuild the 35B lib with the flag on, A/B pp512, run the
state gate, then flip the default on those numbers — the projection is +14.6% to +28.0% and it is a
projection. Optional follow-on worth ~7% more: zero the trailing store loops too (extents 1 and 2,
which are not unique, so it needs a targeted match rather than extent equality).

</details>

<details><summary>Original entry — the analysis that motivated it</summary>

v2 launches `UPPER = (ceildiv(B, BLK_M) + Ne) · tiles_per_n` CTAs, where the `+ Ne` slack gives each
expert a private index range without a prefix scan. Slack CTAs carry sentinel `te[bx] = -1`, and
[moe_matmul.py:698-727](python/mlc_llm/op/moe_matmul.py#L698-L727) guards only the `X` read and the
store — the `W_shared` dequant and the entire wmma reduction run unconditionally on `e_safe = 0`. So
those CTAs cost 92–94% of a real one and produce nothing. At B=4096 they are **50% of the grid under
a perfectly balanced router and 27% under uniform-random routing** (§16.8 measured both).

The change is to predicate the `W_shared` load and the compute block on `e_v >= 0` as well. **It is
bit-exact by construction** — those results are already discarded by the store predicate — which
makes `high_margin_gate.py` a regression check rather than a judgement call, unlike 0c.2.

⚠️ **§16.9 tried the obvious form and it does not work.** A source-level `if e_v >= 0:` around the
loop nests dies at `sch.compute_at(w_shared, k_o_o)` with `InternalError: unordered_map::at` — an
`IfThenElse` between an sblock and its target loop breaks the scope bookkeeping. **The guard has to
be applied after `_schedule_v2()` returns**, as a stmt mutator over the scheduled body. The nodes and
the mutator live in **`tvm.tirx`** / `tvm.tirx.stmt_functor.ir_transform` (§16.9 has the details);
the remaining unknown was thought to be `thread_extent` hoisting. **It was not** — §16.10 found the
real blocker one pass earlier, `ThreadSync` refusing a barrier inside a condition, and the fix was to
drop the conditional entirely in favour of a zero loop extent. The no-conditional
fallback is to stop *launching* those CTAs — compact the dispatch table with an exclusive scan over
`ceildiv(count_e, BLK_M)` — but the grid extent is a compile-time shape expression, so that needs a
host round-trip per call and should be costed first. Not needed now.

Also: the **+34% end is the balanced-router case**; quote the range, not the top of it, until a real
prefill's indptr histogram is dumped (§16.8's second open item). And do **not** reach for widening
`BLK_M` as a shortcut — §16.9 measured it at 0.64× and 0.39×, because `i_o` sits outside the k-loop
and every extra row-fragment re-runs the whole dequant.

</details>

**0g. ✅ LANDED, §17.3 — `BLK_K` 32 → 64, +14.1% pp512 on the 35B (767.20 → 875.37 tps).**
The v2 GEMM's k-step also sets how many bytes of each `W` row are fetched per step (`BLK_K/2`); at 32
that was 16 bytes, **half a 32-byte sector**, plus a `__syncthreads()` pair per 16 bytes/row. 64
makes a row-chunk one sector and halves the barriers — 1.27×–1.39× on the kernel, bit-exact, decode
neutral, and the 35B state gate is *identical* to `lib_skippad` in both prefix-cache modes. 128
regresses (0.80×–0.93×) on shared memory. `MLC_MOE_GEMM_V2_BLKK=32` restores the old kernel. This
parameter had never been swept.

**0h. ⚖️ BUILT AND MEASURED END-TO-END (§17.10) — the hoist works; it needs a runtime branch to
ship.** Register-blocking `BLK_M` by hoisting the cooperative loads above `i_o`, which §16.11 called
"the largest remaining prize in the MoE". `MLC_MOE_GEMM_V2_HOIST=1`; inert and bit-exact at the
default `BLK_M=16`.

*Measured, on real prose (§17.9 — the filler prompt gets this wrong by a factor of 3, and the
synthetic microbench routings get the **sign** wrong):* `BLK_M=64`+hoist is **+12.6% pp2048** and
**−10.4% pp128**; `BLK_M=32`+hoist is +7.8% / −5.5%. Crossover ≈ pp450. Decode neutral, state gate
identical to `lib_blkk64` in both prefix-cache modes.

*Verdict:* **no compile-time `BLK_M` is Pareto**, so defaults stay at 16/off and the shipped kernel
is unchanged. Widening inflates the X and O traffic a tile moves whether or not its rows are real,
and a 128-token prompt has 4 rows per expert to amortize that over.

*To ship it:* specialise on `B` — emit `(BLK_M=16)` and `(BLK_M=64)` dispatch+GEMM pairs and branch
on `x.shape[0]` near **B ≈ 3600**. `LowBatchGemvSpecialize` is the in-tree precedent. **+12.6% on
≥2048-token prefill at no short-prompt cost**, on a 262144-context model chunked at 2048. Cost: two
variants in the binary (the dispatch table is `BLK_M`-dependent and cannot be shared) plus a branch
on a symbolic shape.

**0d. ✅ Built and measured, §16.6 — `v_block`, worth +15.6% on the 0.8B and −8% on the 35B.**
Ships as an opt-in knob (`MLC_QWEN35_GDN_VBLOCK`, default `0` = inert), so the default configuration
and every gate result above are unchanged. It works by **block granularity at fixed occupancy**, not
by the occupancy this item first claimed nor by the `k_split=8` unlocking it then claimed — both
predictions were wrong and §16.6 records why. The text below is the pre-measurement argument, kept
because two-thirds of it was refuted:

Nothing in the recurrence crosses value columns. Every term is indexed by `col`: the state slice is
`storage[seq, hist, h, row, col]`, both dots reduce over `row` *within* a column, `coef` depends on
`v[.., col]` and `dot_sk[col]`, and the output and ring scatter are both per-`col`. So **splitting
`V` across blocks needs no communication at all** — unlike `K`, which is why §15's block-split
proposal was unbuildable and §16.5 had to split lanes instead.

🚫 **What it does NOT buy is occupancy, which is what this item originally claimed.** Warps/SM is set
by registers/thread, and registers/thread is set by the state slice `K/k_split` — which the `V` split
does not touch. `Vb=32, k_split=4` is 128 threads at 117 registers → 4 blocks/SM × 4 warps =
**16 warps/SM, exactly what `k_split=4` already achieves** with one 512-thread block. It repackages
the same warps into more, smaller blocks.

✅ **What it does buy is a `(Vb, k_split)` space the current kernel cannot express.** Today `k_split`
alone sets both the register footprint *and* the block size (`V × k_split` threads), so reaching
`k_split=8`'s 62 registers/thread forces a **1024-thread block** — one block per SM, a coarse
scheduling unit with a tail, which is the likely reason `k_split=8` measured *worse* than 4 at
seq_len 128 and 512 despite halving the state again. Decoupling them allows `Vb=16, k_split=8`:
**128 threads, ~62 registers → 8 blocks/SM = 32 warps/SM**, double anything measured in §16.5, at
128 blocks on the 0.8B and 256 on the 35B. That configuration is currently unreachable.

Whether 32 warps/SM helps is the open question, and §16.5 gives reason for doubt: `k_split=8` already
reached 33 warps/SM in theory and lost at 512, so occupancy may no longer be the binding constraint
at that length. Worth one measurement precisely because it is cheap — a grid and index change to an
existing, gated kernel.

Cost: `k`/`q`/`gate`/`beta` get re-read by each column block (broadcast reads, already L2-resident);
the per-block preload/flush loops shorten rather than duplicate. Gate it with `gdn_kernel_check.py`
at the **full exact bar** — a pure grid refactor at fixed `k_split` changes no reduction order, so it
should be bit-exact, unlike §16.5.

**The VL path has not been re-gated. It is NOT blocked — an earlier version of this entry said it
needed "a multi-GB download before any of it can start", and that is wrong.** Re-checked 2026-07-26d:
`Qwen/Qwen3.5-0.8B` **is** the VL checkpoint. Its architecture is `Qwen3_5ForConditionalGeneration`,
its `config.json` carries a `vision_config`, and **153 of its 488 tensors are the vision tower**
(`model.visual.*`) — inside the single 1.7 GB shard that has been in `~/.cache/huggingface/hub` since
the first session. `qwen3_5_vl_loader.HF_VISUAL_PREFIX` is `"model.visual"`, so the names line up.

What is actually missing is only that **all three `dist/qwen3_5-0.8B-*` builds were compiled as
`model_type: qwen3_5`**, the text-only path, which drops the vision tower. A VL build is a local
`convert_weight` + `gen_config` + `compile --model-type qwen3_5_vl` against a checkpoint already on
disk — **no download at all**. The mistake was inferring "no VL checkpoint" from "no directory named
VL"; the text and VL models share one repo.

The package is registered
(`python/mlc_llm/model/qwen3_5_vl`, `model.py:487`) and all three loaders already carry the 4-way
`in_proj` concat, so the rebuild itself should be uneventful. `Qwen35VLLMHeadModel` reuses `Qwen35Model.forward` and
`forward_with_history`, so it inherits §11's in-place state, §13's conv fusion, §14's history conv
and §15's history recurrence — but there is **no compiled VL model on this box**, so the 176/180
multimodal gate from `f667b07e` has not been re-run since. Rebuild and re-gate before trusting a
VL build. All three loaders were already updated for the 4-way `in_proj` concat, so the rebuild
should just work.

#### Closed — IDs kept because §12–§15 reference them

| id | outcome |
|---|---|
| **0** | ✅ **Landed, §14** — history-path conv fusion. 35B pp512 +10.9%, 0.8B +22.4% under radix; 107% of its traced estimate. The session also found that every prefill number in this document predated any measurement of the default configuration (§14.1) and fixed the two harness bugs that made it unmeasurable |
| **0a** | ✅ **Landed, §15** — history-path *recurrent* fusion, both compounding wins as scoped. 35B pp512 395 → 629 (+59.1%), 0.8B 1793 → 3934 (+119.4%), decode neutral; the recurrent pair 365.2 → 95.6 ms (3.82×). Closes §14.1's 2.77× default-configuration penalty and is the first bit-exact history-path change |
| **0-old** | 🗑 **Deleted** — was a verbatim duplicate of item 0's pre-completion text |
| **2** | ✅ **Landed, §13** — in-place GDN *conv* state, 35B +2.40% tg / +15.3% pp. The decode estimate was right and the stated rationale was not: the real prize was the TE conv itself at ~42× off roofline, not the state copies |
| **3** | ❌ **Refuted by measurement, §13 — do not build.** Allowlisting the `rnn_state_*` handle builtins as static in `rewrite_cuda_graph.cc` to make the fused kernel capturable. The capture prediction was correct and everything built on it was wrong: eager launches went *up* 131 → 183/token, idle did not move (1.016 → 1.065 ms/token), and an eager launch costs ~0.9 µs at the margin, so the whole lane is worth ≤0.38 ms/token. §13 has the trace |
| **0b** | ✅ **Landed, §16.1** — a deterministic 35B state gate. Teacher forcing removes the cascade that made §6.2's counts uninterpretable, and margin-gated scoring replaces the inherited 48/50 bar. 35B: **139/139 wide-margin positions**, identical across both libs × both prefix-cache modes. `prefix_cache_roundtrip` moved to the same prompt set: **4/4** vs **0/2** for the legacy set on the identical lib. Calibrated on the 0.8B (4-bit flips 11 of 400, all at margin ≤1.031) with a `stale1` negative control that fails 342/361 |
| **4** | ➡ **Promoted to 0b** (2026-07-25b) — a high-margin prompt set now unblocks two gates rather than one |
| **1** | ✅ **Settled by measurement, §16.4 — keep the b=1 specialization.** The "~6×" source comment understates it by 8×: at b=1 the top-8 gemv pair is **0.100 ms vs 5.112 ms** for `dequantize_group_gemm` v2 (**51×**; v1 is 1.957 ms, 20×). v2 is a dispatch-table kernel sized for all 256 experts, so its cost is flat in batch and crossover is ~50 sequences. Option (a) (a Relax `If`) is therefore pointless — the dynamic path never wins. If batched decode is ever wanted, widen the per-token gemv split that already ships for spec-decode verify (option **(d)**, which the original option list missed). Comment corrected in source |
| **5** | ❌ **Refuted by measurement, §16.3 — do not build.** The premise was wrong twice over. At fixed N=2048 efficiency *rises* with K (48% → 68% → 84% → **90%** at K=4096), so K=4096 is the best case rather than the shortfall; and across six tile configurations the shipped sm_87 tile is within 0.5% of the best at every shape, with nothing improving K=4096 at all. `o_proj`'s traced 75% is a memory-system effect — it streams 40 distinct weight tensors per token with no reuse — not a schedule defect, so a GEMV retune cannot recover it. The sweep also exposed an instrument bug worth remembering: `bench_moe_kernel.py` reuses one weight tensor, which inflates small-footprint kernels by up to 20% via L2 (`lm_head` at 70× L2 agrees with the trace to 3.3%; `o_proj` at 1.2× L2 is 20.6% high) |

### What changed, by file (all committed — see "Committed state" above)

**The TVM submodule change is committed and pushed** (verified 2026-07-25d — see "Committed
state" above for why the earlier "NOT pushed" claim was a false alarm). It still has to be built
separately: the mlc-llm build will not rebuild it (§2.1.1):

```
3rdparty/tvm (fork alansrobotlab2/relax) -> 4624d97 on branch qwen35-inplace-rnn-state
  src/runtime/vm/rnn_state.cc      # the three new builtins — on the remote, verified
```

Main repo:

| file | what |
|---|---|
| `python/mlc_llm/model/qwen35/qwen35_model.py` | **§16.6** `v_block` on the ksplit kernel + `_gdn_v_block()` (`MLC_QWEN35_GDN_VBLOCK`, **default 0 = inert**): grid becomes `(n_vh × V/v_block, batch)`, block `v_block × k_split` threads. Bit-exact with `v_block=V` by construction. **§16.5** `create_gated_delta_net_func_with_history_inplace_ksplit` + `_gdn_k_split()` (`MLC_QWEN35_GDN_KSPLIT`, **default 4**) and the two-way selection at the `forward_with_history` call site. Uses `T.tvm_warp_shuffle` — there is no `_xor` variant in this TVM, so the partner lane is computed as `(tid % 32) ^ d`; it lowers to a real `__shfl_sync` at sm_87 (the legacy `__shfl` compat macro is gated on `__CUDA_ARCH__ < 700`). **§15** `create_gated_delta_net_func_with_history_inplace` + the recurrent half of `forward_with_history` behind `state_io`. **§14** `create_causal_conv1d_func_with_history_inplace` + `state_io` threaded through `forward_with_history`; the per-model hoist block factored into `_maybe_hoist_state_io`. **§10–13:** `in_proj_qkvzab` + `_in_proj()` helper; **§11** `create_gated_delta_net_func_inplace`, `_GDNStateIO`, `_hoist_gdn_state_io`, `MLC_QWEN35_INPLACE_STATE` toggle; **§13** `create_causal_conv1d_func_inplace` + `conv_storages` on `_GDNStateIO` |
| `python/mlc_llm/op/moe_matmul.py` | **§16.9/§16.10** — `MLC_MOE_GEMM_V2_BLKM` (A/B knob, inert at 16) and `MLC_MOE_GEMM_V2_SKIPPAD` (item 0f's padding-CTA skip, default `0` pending pp512). The skip is a post-schedule rewrite of the `k_o_o` loop *extent* to `Select(e_v >= 0, K/BLK_K, 0)` — **not** an `IfThenElse`, because `ThreadSync` refuses to put a barrier inside a condition |
| `python/mlc_llm/compiler_pass/blas_dispatch.py` | **§19.6** `_region_needs_tir_vars` — decline a cuBLAS match that would need a `tir_vars` parameter (the VL merger matmuls; fixes a compile abort). **§20.3** `_region_is_fp32` + `MLC_BLAS_SKIP_FP32` — decline fp32 offloads. **§20.6 dropped its default to `0`**: superseded by prescaling, which removes the fusion it protected instead of working around it. Unreachable on the 35B/0.8B text models — `_cublas_gemm` only enables the pass for `q0f16`/`q0bf16`/`q0f32`/fp8, and those are `q4f16_1` |
| `python/mlc_llm/model/vision/qwen3_vl_vit.py` | **§20.5** `MLC_QWEN35_VL_PRESCALE_Q` (default `1`) — apply the attention scale to `q` rather than to the scores, via `matmul(q,k^T)·c ≡ matmul(q·c,k^T)`. Worth 0 ms by itself; worth **70 ms** because it makes the QK matmul a bare GEMM cuBLAS can take. Coupled to `MLC_BLAS_SKIP_FP32` — see §20.6 |
| `scripts/vit_attn_bench.py` | **§20.5 new** — the tower's attention block alone, same passes and dlight schedules, each PrimFunc timed separately. Reproduces the traced per-layer numbers to **1.3%**; `--static` and `--prescale` are the two hypothesis legs. This is the loop for item 0q, and it needs no compile |
| `validate.py` | **§20.1** `--perf-vl5` — the first VL perf harness; times `image_embed`/`prefill`/`decode` as three separate VM calls with a sync each, both `max_history` rings. **§20.4** `--greedy-parity-vl5` hoists `image_embed` out of the per-prompt loop (was 5× a 337 ms call for one fixture) |
| `python/mlc_llm/model/qwen3_5_moe/qwen3_5_moe_model.py` | **§11** same state_io wiring (the 35B's model file; VL reuses `Qwen35Model` and needed none). **§13**: it has its *own* `_hoist_gdn_state_io` call site — changing that helper's signature breaks the 35B compile while the 0.8B still builds |
| `scripts/{prefix_cache_roundtrip,batch_decode_parity}.py` | **§11 new** — rollback and batch-slot gates; `batch_decode_parity` gained phase timing in **§12** |
| `cpp/serve/engine_actions/batch_prefill_base.cc` | **§12** one-sequence prefill cap for RNN-state models; no decode-folding |
| `cpp/serve/engine_actions/batch_decode.cc` | **§12** multi-token decode cap + retokenization history guard |
| `cpp/serve/engine_actions/batch_jumpforward.cc` | **§12** skip jump-forward when its rollback exceeds the RNNState ring |
| `cpp/serve/engine.cc` | **§12** startup warning: speculative decoding on hybrid is single-sequence only |
| `python/mlc_llm/model/{qwen35,qwen3_5_moe,qwen3_5_vl}/*_loader.py` | 4-way concat |
| `python/mlc_llm/nn/rnn_state.py` | `storage()` / `slot_ids()` accessors (`storage()` now `match_cast`s the shape; both had a `Tensor(_name=...)` kwarg that does not exist and would have failed on first use) |
| `python/mlc_llm/quantization/quantization.py` | comment: ft-quant fallback no longer hit on the 35B |
| `validate.py` | fp8 detection, shim install, bf16 forcing; **§11** `--prefix-cache-mode` |
| `fp8_software_dequant.py` | **new** — software W8A16 fp8 path |
| `scripts/{analyze_decode_trace,greedy_snapshot,active_params,profile_decode_35b}.py`, `scripts/bw_probe.cu` | **new/promoted**. `analyze_decode_trace` reworked in **§13** to verify kernel identity against launch geometry |
| `scripts/conv1d_kernel_check.py` | **§13 new** — numerical unit gate for the fused conv1d |
| `scripts/high_margin_gate.py` | **§16.1 new** — the 35B state gate. `--capture` builds a margin-annotated reference from an HF model; `--check` teacher-forces an MLC lib against it and scores only where the reference had margin. `--negative-control stale1` proves it is not vacuous |
| `scripts/gdn_recurrence_probe.cu` | **§16.2 new** — standalone CUDA probe, four variants of the GDN recurrence (base / acc4 / ksplit2 / ksplit4). No model, no TVM. **§16.5** added a header warning: its `grid(n_kh, batch)` is the 0.8B's geometry, and quoting its ratios against the 35B over-predicts by 2.3× |
| `scripts/gdn_kernel_bench.py` | **§16.5 new** — the GDN recurrence A/B on the kernels MLC actually compiles: real grid (`n_vh` blocks, so 32 on the 35B), ring flush included, `k_split` 1/2/4/8 × seq_len. `--trace-share` prints an Amdahl bound. Timing goes through `mod.mod.time_evaluator`, **not** `Executable.time_evaluator`, which does not exist. **§16.6** `--v-blocks` crosses the `v_block` axis with `--k-splits`; header warns to run only one instance at a time and to check with `pgrep -af`, not `ps -C python` |
| `scripts/moe_gemm_check.py` | **§16.10 new** — the numerical gate for `dequantize_group_gemm_v2`. Builds the kernel twice in one process (`MLC_MOE_GEMM_V2_SKIPPAD` off/on) and requires **`np.array_equal`**, not a tolerance, on the §16.6 `vb_exact` precedent. Sweeps even / uniform-random routing and `B=777`, which is a multiple of neither `BLK_M` nor `Ne` and is the case a wrong guard breaks |
| `scripts/prefix_cache_roundtrip.py` | **§16.1** — prompt families replaced with the high-margin set; `--legacy-prompts` reproduces §13's numbers. **Propagate any new flag to the subprocess `common` list** — the two phases run as separate processes and mismatched sets fail everything |
| `bench_moe_kernel.py` | **§16.3/§16.4** — K-sweep at fixed N and N-sweep at fixed K, plus achieved-bandwidth reporting against the 156 GB/s wall. ⚠️ it reuses one weight tensor, so absolute numbers are L2-inflated for small kernels (§16.3); A/Bs at a fixed shape are fine. **§16.8** added the prefill-scale sweep (`{gate_up,down}_b512…b8192`), FLOP reporting against both the tensor and CUDA-core ceilings, `v2_grid()` (replays v2's dispatch table to count real vs padding CTAs), and `spread="random"` routing. Two fixes went with it: the active-expert count is now taken from the indptr instead of assumed to be `top_k`, which understated prefill traffic by 32×, and the indptr is drawn **once** and shared by the timed call and the accounting — two draws under random routing would report a different routing than the one measured |
| `python/mlc_llm/model/qwen3_5_moe/qwen3_5_moe_model.py` | **§16.4** — the batch-1 MoE comment now carries the measurement (51×, not ~6×) and points at option (d) |
| `python/mlc_llm/support/auto_target.py` | **§14.6 new** — `MLC_NVCC_OPTIONS` / `MLC_DUMP_CUDA` hooks + nvcc phase timing |
| `scratch_mlc_tg_sweep.py` | **§14.1 new** — `--prefix-cache-mode` and per-run prompt salting; without both, the default config was unmeasurable |
| `scripts/greedy_snapshot.py` | **§14** — `--prefix-cache-mode`; it hardcoded `disable`, where a history-path change is inert and the gate passes vacuously |
| `scripts/profile_decode_35b.py` | **§14** — `--prefix-cache-mode` + prompt salting so a radix trace contains prefill |
| `qwen3_5.md` | §14.5 + 5 stale-item corrections |
| `workplan-cuda-13.md` | **new** — this file |
| `.gitignore` | exception for `reference_outputs_35b.pt` **and `reference_outputs_35b_fp8.pt`** — the fp8 name was still ignored, so §7's "commit this" silently had not happened |

**§16.5 artifacts:** `tuning/mlc_tg_{0.8b,35b}_{gdnhist,ksplit4}_radix_20260726.json` — all four legs
of the A/B, committed. `dist/qwen3_5-0.8B-q0f16_fused/lib_ksplit4.so` and
`dist/qwen3_6-35B-A3B-q4f16_1_fused/lib_ksplit4.so` are the built libs (`dist/` is gitignored); both
were compiled with `MLC_QWEN35_GDN_KSPLIT=4` explicitly and the 35B with `MLC_MOE_GEMM_V2=1`.

Artifacts: `reference_outputs_35b_fp8.pt` (3.5 kB, **committed** in `fce533aa` — it is the only 35B
reference that runs on this box), `tuning/greedy_35b_{before,after}.json`,
`tuning/mlc_tg_35b_fused_20260725.json`, plus the four 2026-07-24 tuning files.
`dist/qwen3_6-35B-A3B-q4f16_1_fused/` and `dist/qwen3_5-0.8B-q0f16_fused/` are the merged builds
(`dist/` is gitignored).

### Open questions

- Why are `out_proj`/`o_proj` (75%) and routed-expert down (76%) short of the 88% the same schedule
  reaches at K=2048? Both differ in K (4096 and 512). Shape-driven or schedule-driven decides
  whether §5 option 3 is worth doing.
- Can the -1.1% prefill regression from §10 be recovered? The lost `silu(z)*core_out` fusion and the
  wider v2 group-GEMM output are the two candidates; neither has been profiled.
- Does the CTA_COUNT=64-vs-1024 small-batch gap (§4.2) reach any shipped path?
- The baseline/fused A/B was not compile-flag-identical (§10 caveat). A rebuild with matching flags
  would close it.
- **Is the 35B's batch-1 MoE specialization still worth ~6%?** The "~6× faster than
  `dequantize_group_gemm` at b=1 top-8" claim at
  [qwen3_5_moe_model.py:632](python/mlc_llm/model/qwen3_5_moe/qwen3_5_moe_model.py#L632) is a source
  comment, not a measurement in this document, and it predates both the Phase 9 CTA_COUNT=1024
  restoration (§4.2) and nvcc 13.2. Measuring it is what decides §9 item 1 — and `moe_dequantize_gemv`
  already sits at 88% of the 156 GB/s wall (§4.6), so the honest question is how far the
  dynamic-batch path falls short of *that*, at b=1, today. `bench_moe_kernel.py` is the instrument.

---

## 10. Landed: the GDN input-projection merge (2026-07-25)

§5 option 1, implemented. `in_proj_qkv` / `in_proj_z` / `in_proj_a` / `in_proj_b` are now one
`in_proj_qkvzab` Linear (`2048 -> 12352` on the 35B, `1024 -> 8224` on the 0.8B), split back into
four tensors right after the projection.

**Code**: [qwen35_model.py](python/mlc_llm/model/qwen35/qwen35_model.py) — one `nn.Linear` plus a
`_in_proj()` helper that both `forward` and `forward_with_history` call (the helper carries the
small-static-seq per-token GEMV dispatch the spec-verify path needs). The `QWEN35_NO_QUANT` bisect
hook still accepts the four legacy names, but they now all mark the fused Linear — per-sub-projection
granularity is gone, so a bisect needing it has to un-fuse first. Loader concat added to all three
loaders that map this layer ([qwen35](python/mlc_llm/model/qwen35/qwen35_loader.py),
[qwen3_5_moe](python/mlc_llm/model/qwen3_5_moe/qwen3_5_moe_loader.py),
[qwen3_5_vl](python/mlc_llm/model/qwen3_5_vl/qwen3_5_vl_loader.py)) — same treatment `c_attn`
already gives q/k/v. The MTP drafts do not instantiate a GDN layer and needed no change.

### What the weights do — exactly as predicted

Group quantization runs along the reduction axis, so row-concatenation changes nothing:

| check | result |
|---|---|
| 0.8B `q0f16`: fused fp16 tensor vs `concatenate([qkv, z, a, b])` | **bit-identical** |
| 35B `q4f16_1`: fused `q_weight` (int4) vs concat | **bit-identical** |
| 35B `q4f16_1`: fused `q_scale` vs concat | **bit-identical** |
| 35B convert totals | **18.187 GB / 35,951,822,704 params / 4.345 bits** — unchanged |
| 0.8B tensor count | 296 -> 242 (18 GDN layers x 3 fewer tensors) |

### What the arithmetic does — NOT as predicted

§5 called this "bit-exact, not an approximation". **That claim was too strong and is corrected
here.** The *weights* are bit-exact. The *arithmetic* is not: a 12352-wide GEMV gets a different
dlight reduction split than the 8192/4096/32/32 kernels it replaces, so fp16 accumulation order
changes, and at 4 bits that flips near-ties.

- 0.8B `q0f16` greedy parity vs HF: **5/5 prompts, 50/50 tokens** — the gate holds. fp16 has enough
  margin that the reordering changes nothing observable.
- 35B `q4f16_1` [greedy_snapshot](scripts/greedy_snapshot.py) vs pre-merge: **3/5 diverged**. Every
  divergence is a single early token flip that then cascades (prompt 4 splits at char 10 on
  `<think>\n` -> `\n` vs `Here`), and all five outputs stay coherent and correct.

**Method lesson: `greedy_snapshot.py` is the wrong gate for a change that alters kernel shape.** It
is the right gate for changes that preserve arithmetic — launch reordering, cudagraph capture,
memory-layout moves that keep the same reduction. Anything that makes dlight schedule differently
needs the tier-2 semantic gate (§6.1) instead.

### Measured throughput

Baseline `dist/qwen3_6-35B-A3B-q4f16_1/lib.so` vs fused, MAXN + `jetson_clocks`, `--pp 512
--runs 3 --warmup 1`, both libs carrying 58 FlashInfer symbols:

| | baseline | fused | delta |
|---|---:|---:|---:|
| tg 512 | 54.13 | **55.66** | **+2.8%** |
| tg 1024 | 54.00 | **55.47** | +2.7% |
| tg 4096 | 53.34 | **54.74** | +2.6% |
| pp 512 | 566.33 | **559.83** | **-1.1%** |
| `lib.so` | 200 MB | 166 MB | -34 MB |

Raw: `tuning/mlc_tg_35b_fused_20260725.json`.

Predicted +4.5% decode, got +2.8% — the estimate assumed the fused 12352-wide GEMV would inherit the
137.8 GB/s its 8192-wide predecessor reaches, and it did not. The prefill cost was not predicted at
all; the `silu(z)*core_out` fusion is gone (z now comes from a split, so Relax cannot fold the
multiply into the matmul) and the v2 group-GEMM sees a different output width.

**The trade is workload-dependent.** End-to-end at pp512/tg512 the fused lib is ~2.5% faster; at
pp8192/tg64 it is ~0.9% *slower*. Decode-bound interactive use wins, prefill-heavy short-generation
use loses slightly.

### Correctness verdict

With bit-exactness unavailable (arithmetic changed) the deciding evidence is the §6.2 tier-2 run:
**both libs agree with the fp8 reference identically, 1/15/2/5/50 prompt for prompt**, while
generating slightly different text from each other. The divergence is common to both, so it is not
caused by the merge. Combined with the 0.8B `q0f16` 5/5 gate and bit-identical quantized weights,
the merge is **correctness-neutral**. Note this is weaker than a bit-exact proof, and §6.2 explains
why the match counts themselves are not a pass/fail bar.

> **Compile-flag caveat.** The baseline was built with
> `flashinfer=1;cublas_gemm=1;cudagraph=1;cutlass=1`; the fused lib with `flashinfer=1;cudagraph=1`.
> Per §3 both dropped flags are inert here (`cublas_gemm` is silently disabled at `q4f16_1`,
> `cutlass` is gated to sm_90a/sm_100a), and both libs export 58 FlashInfer symbols — but the A/B
> is not flag-identical, and a rebuild with matching flags would remove the last doubt.

---

## 11. Landed: in-place GDN recurrent-state update (2026-07-25)

§5 option 2, implemented as the corrected design in that section prescribes — fuse the state
*access* into `gdn_func` rather than aliasing buffers or handing out a slot pointer.

**Result: the largest single win in this workplan, and the only one that cost nothing elsewhere.**

| | 35B `q4f16_1` | 0.8B `q0f16` |
|---|---:|---:|
| tg512 before | 55.61 | 87.64 |
| tg512 after | **58.97** | **89.61** |
| delta | **+6.04%** | **+2.25%** |
| pp512 before → after | 560.18 → 560.59 (+0.07%) | 2838.45 → 2846.25 (+0.27%) |
| `lib.so` | 166 → 153 MB | — |

Five runs each, spread ≤0.1% on both libs (35B: 55.54–55.64 vs 58.93–58.99). **Clocks were not
pinned** — `sudo` needs a password on this box (§8) — so absolute numbers sit ~1.4% below the
pinned figures in §4.1; the A/B ran back-to-back under identical conditions.

Predicted 1.195 ms/token of an 18.474 ms budget = 6.5%; measured 6.04%, i.e. **93% of estimate**.
That is the estimation lesson in §5 being applied rather than re-learned: the prediction came from
a *measured* pair of kernels at the real shape (`rnn_state_get_0` + `set_0` in §4.6), not from a
bytes-÷-best-observed-bandwidth model, and it also predicted the prefill effect (none — the copies
are per-call, not per-token, so prefill amortizes them away already).

### The A/B is source- and flag-identical

§10 shipped with a caveat that its baseline and fused libs were built with different `--opt`
strings. That is closed here: both libs are built from **the same tree** with the **same flags**
(`flashinfer=1;cudagraph=1`), differing only by `MLC_QWEN35_INPLACE_STATE=0`, a new env toggle that
compiles the old copy path. The 35B `lib_copypath.so` and the pre-existing §10 `lib.so` are both
166 MB, which is the consistency check on that claim.

### What the kernel does

[`create_gated_delta_net_func_inplace`](python/mlc_llm/model/qwen35/qwen35_model.py) takes the whole
`(max_batch, max_hist, H, K, V)` storage buffer plus the device-side `seq_slot_ids` /
`history_slot_ids` arrays, and does the slot addressing itself:

```
load  from storage[seq_slot,       hist_slot,            head, row, col]
flush into storage[seq_slot, (hist_slot + 1) % max_hist, head, row, col]
```

Both copy kernels disappear from the decode path. `max_batch_size` and `max_history` are bound from
the storage tensor's own shape by `T.match_buffer` — they are *runtime* arguments to
`create_rnn_state`, so nothing in Relax scope names them, and `RNNState.storage()` introduces them
with a `match_cast`. One kernel is therefore correct at any `max_history`.

Wiring: handles are hoisted **above** the layer loop (`_GDNStateIO` / `_hoist_gdn_state_io`). That
placement is load-bearing — TVM's cudagraph pass calls `EndRegion()` on every `vm.builtin.*` call
([rewrite_cuda_graph.cc:383](3rdparty/tvm/src/relax/transform/rewrite_cuda_graph.cc#L383)), so
emitting 30 storage handles inside the loop would cut the capture region 30 extra times. The
`forward_with_history` path kept the copy path unchanged until §15 fused it too.

**Hoisting the handles out of the loop is only safe because the pointers are stable across steps,
and that was audited rather than assumed:** in [rnn_state.cc](3rdparty/tvm/src/runtime/vm/rnn_state.cc)
`storages_` and both slot-id views are allocated once in the constructor, and `CreateView` uses
byte offset 0 — so a handle fetched once before the layer loop stays valid for every subsequent
step, at any `max_history`. (What *does* move every step is `history_slot_id`, which is why the
kernels take the device-side index arrays and do the addressing themselves; see §5.) This audit
was originally recorded under the now-refuted §9 item 3 and is kept here because it is the reason
the §11/§13/§14/§15 wiring is correct, independently of that item.

### Correctness

All gates on the 0.8B `q0f16`, which is the only configuration where a bit-exact comparison exists
(§6):

| gate | radix (`max_history=64`, **default**) | disable (`max_history=1`, what benches use) |
|---|---|---|
| `--greedy-parity` vs HF fp16 | ✅ 5/5 prompts, 50/50 tokens | ✅ 5/5 prompts, 50/50 tokens |
| `scripts/prefix_cache_roundtrip.py` | ✅ 4/4 checks | ✅ 4/4 checks (control) |

The copy-path lib produces identical text on the round-trip gate, so this is behavioural identity
on the rollback path, not merely self-consistency.

**On the 35B `q4f16_1` the round-trip gate is comparative, not pass/fail** — the same category
error §6.2 flags for the fp8 gate, now confirmed for this one. Both libs score 19/20:

| check | in-place | copy path (pre-existing) |
|---|---|---|
| pass1 vs cold | **4/5** | 5/5 |
| pass2 vs pass1 (exact-match reuse) | 5/5 | **4/5** |
| pass3 ext vs cold (fork) | 5/5 | 5/5 |
| pass3 base vs cold (**fork + PopN rollback**) | **5/5** | **5/5** |

The single divergence is the *same prompt* on both libs ("The three primary colors are") between
the *same two orderings* ("red, yellow, and blue" ↔ "red, blue, and yellow") — a genuine near-tie
that flips run to run — and it lands on a **different check each time**. That is the signature of
nondeterminism, not of a mechanism bug: a real state bug takes a whole check from 5/5 to 0/5, as
the negative control below does. **The rollback check itself is 5/5 on both libs**, which is the
question the gate was run to answer.

### The trap in §5 is real — built and measured

§5 warned that writing back into the *current* slot "would have sped up the benchmark configuration
while destroying the history the default configuration depends on". That was an analytical claim;
it is now a measurement. A deliberate negative-control lib with `hist_out = hist_in`:

| | radix (default) | disable (benches) |
|---|---|---|
| `--greedy-parity` | ❌ all 5 prompts, **3–16/50 tokens** | — |
| `prefix_cache_roundtrip.py` | ❌ 13 divergences (0/5, 2/5, 0/5) | ✅ **4/4 — fully green** |

**One correction to §5's framing.** It says a naive fusion corrupts the default *silently*. It does
not: under radix, `EndForward` advances into a slot the kernel never wrote, so ordinary generation
breaks immediately and loudly. The real hazard is narrower and more specific — the failure is
invisible **only if you gate exclusively under `disable`**, which is what every bench harness sets.
Since `validate.py` inherits the engine default (radix), it would have caught this. The genuine
risk was benching under `disable`, seeing a clean speedup, and shipping.

### New gates

`validate.py` gained `--prefix-cache-mode`, because the mode is the variable that decides whether
this class of bug is observable at all.

[scripts/prefix_cache_roundtrip.py](scripts/prefix_cache_roundtrip.py) closes the gap flagged at
[worklog.md:946](worklog.md#L946) ("the actual PopN-round-trip parity test") and guards the risk at
[worklog.md:1112](worklog.md#L1112) ("a fused kernel would need to preserve that PopN-able history
path"). It needs **no reference model**, so unlike `--greedy-parity` it runs on the 35B, where §6.1
shows no bit-exact reference fits in 64 GB. Cold-vs-warm engine, exact-match reuse, and
fork-plus-rollback are checked separately so a failure says which mechanism broke.

### Found while gating: hybrid models cannot serve more than one sequence at a time

> **Fixed in §12 — and this diagnosis is wrong about where.** The failing forward is
> multi-sequence *prefill*, not decode: `BatchDecode` already passes `(num_seq, 1, h)` and always
> agreed with `cur_batch_size_`. Read §12 before acting on anything below. The reassuring second
> bullet is also half-wrong: batch > 1 *was* reachable on the 0.8B and was being blocked by an
> engine bug; on the 35B it is blocked instead by a deliberate compile-time MoE specialization.

[scripts/batch_decode_parity.py](scripts/batch_decode_parity.py) was written to cover the one thing
the other gates miss — the fused kernel indexes `storage[seq_slot_ids[b], ...]` per batch element,
and every other gate runs at batch 1. It cannot run, because **multi-sequence decode is broken on
hybrid models independently of this work**:

```
ValueError: Mismatched output.shape[0] on argument #3 when calling:
  rnn_state_get_1(storage: Tensor([max_batch_size, max_history, 3, 6144], float16),
                  seq_slot_ids: Tensor([batch_size], int32), ...,
                  output: Tensor([batch_size, 3, 6144], float16))
  expected to match seq_slot_ids.shape[0]
```

**Reproduced identically on `lib_copypath.so`**, and the failing kernel is `rnn_state_get_1` — the
*conv* state, which §11 does not touch. The cause is the invariant the model has always assumed:
`RNNState.get` sizes its destination from `hidden_states.shape[0]`, while the runtime fills
`cur_batch_size_` rows, and those diverge as soon as more than one sequence is in flight. This is
presumably why every harness in this repo uses `mode="interactive"` (max_batch_size 1).

Two consequences, and the second is the reassuring one:

- The per-batch slot indexing in the fused kernel is **still unverified**, and cannot be verified
  until the above is fixed. Stated plainly rather than buried: it is the one gap in §11's coverage.
- **The risk is correspondingly small** — batch > 1 is not a reachable configuration on hybrid
  models in this tree at all, for the copy path or the fused path. Nothing regressed; a door that
  was already shut stayed shut.

### Three harness traps, all of which cost time this session

Any future multi-engine gate on this box will hit these. All three are already handled in
`prefix_cache_roundtrip.py`; copy that file's `new_engine` / `_gen` / driver shape rather than
rediscovering them.

1. **`engine._generate` hangs, GPU at 0%.** Already documented at
   [worklog.md:815](worklog.md#L815) but easy to miss. It survived on the 0.8B and on the 35B's
   *first* engine, then hung on the second — so it fails late and looks like a model bug rather
   than a harness bug. Use `engine.completions.create(..., model=<model_type>, stream=False)`. It
   also deadlocks outright when driven from several threads.
2. **`mode="interactive"` without explicit sizes routes through the auto-config path** — the
   `max_total=262144, prefill_chunk=2048` combination in the worklog.md:815 hang. Pass
   `max_num_sequence` / `max_total_sequence_length` / `prefill_chunk_size` explicitly.
3. **Two 35B engines cannot coexist, and neither `terminate()` nor `gc.collect()` frees in time.**
   At ~41 GB each (18.6 GB params + 5.2 KV + 7.9 rnn_state at `max_history=64` + 9.2 temp), the
   second `MLCEngine(...)` dies in `cudaMalloc`. Only **process exit** reliably releases the C++
   engine's device memory, so the script re-invokes itself once per phase and passes results
   through JSON.

---

## 12. Landed: concurrent serving on hybrid models (2026-07-25)

§9 item 1. [scripts/batch_decode_parity.py](scripts/batch_decode_parity.py) now runs, and passes
6/6 under both `prefix_cache_mode` settings and on both the in-place and copy-path libs. **This
closes §11's one stated coverage gap**: the fused kernel's per-batch `storage[seq_slot_ids[b], …]`
indexing has now been exercised with six sequences in distinct slots.

Everything below is **engine-side C++. No model change, no recompile** — `lib_inplace.so` and
`lib_copypath.so` were both gated exactly as they were already built.

### It was prefill, not decode

§11 called this "hybrid models cannot serve more than one sequence at a time" and named
`rnn_state_get_1`, both correct. But it placed the failure in decode, and the stack places it
somewhere else entirely:

```
NewRequestPrefillActionObj::Step   new_request_prefill.cc:148
  ModelImpl::BatchPrefill          model.cc:355
    RNNStateImpObj::Get            rnn_state.cc:307
      ValueError: Mismatched output.shape[0] ... expected to match seq_slot_ids.shape[0]
```

`BatchDecode` was never broken. [model.cc:538](cpp/serve/model.cc#L538) views the embeddings as
`(num_sequence, 1, h)`, which is exactly what `cur_batch_size_` expects, and the 0.8B's
`batch_decode` spec is `["batch_size", 1, hidden]` to match. `BatchPrefill` at
[model.cc:316](cpp/serve/model.cc#L316) instead views them as `(1, total_length, h)` — the batch is
**concatenated**, and per-sequence boundaries exist only inside the PagedKVCache's own
`BeginForward` — while [model.cc:302](cpp/serve/model.cc#L302) hands `RNNState::BeginForward` all N
sequence ids. The model sizes the `get` destination from `hidden_states.shape[0]`, which is the
literal `1` in the `batch_prefill` spec, and the assert fires.

**The shape mismatch is the shallow half.** Reconcile the shapes and it is still wrong: `gdn_func`'s
`for t in range(seq_len)` and the causal conv1d would both run straight across the boundary between
two concatenated sequences, silently mixing one request's recurrent state into the next. *One
sequence per prefill forward is what a recurrent layer means*, not a workaround for an assert.
That is why the fix caps the batch rather than reconciling the shapes.

### The changes

| file | change |
|---|---|
| [batch_prefill_base.cc](cpp/serve/engine_actions/batch_prefill_base.cc) | cap the prefill batch to one sequence when the model carries an RNN state, and skip `PrefillMode::kHybrid`'s decode-folding, which would otherwise re-add the running sequences to the same forward |
| [batch_decode.cc](cpp/serve/engine_actions/batch_decode.cc) | a decode step whose entries do not *all* want exactly one token falls through to `BatchPrefill`; when that happens, advance only the one multi-token entry. Picking the multi-token entry, not the first, is what avoids starving it |
| [batch_decode.cc](cpp/serve/engine_actions/batch_decode.cc) | `CommitTokenMayRetokenize`: skip retokenization when the rollback exceeds the RNNState history ring |
| [batch_jumpforward.cc](cpp/serve/engine_actions/batch_jumpforward.cc) | skip the jump forward when *its* rollback would exceed the ring |
| [engine.cc](cpp/serve/engine.cc) | warn at startup that speculative decoding on a hybrid model is still single-sequence only |

Nothing changes for pure-attention models: every guard is behind
`kv_state_kind ∈ {kRNNState, kHybrid}`. Pure-RNN models (RWKV6) get the prefill cap too, and want
it — [rwkv6_model.py:483](python/mlc_llm/model/rwkv6/rwkv6_model.py#L483) declares the same
`[1, "seq_len", h]` prefill spec against a per-`batch` `state.get`, so the bug is upstream MLC's,
not something this port introduced.

### One more bug fell out, at two call sites — the grammar path

Not reachable without structured output, and not known before this session. It surfaced because
unblocking concurrency is what first let four grammar-constrained requests run together.

**Jump-forward decoding aborts the engine loop on any hybrid model.** Jump-forward commits several
tokens at once and then rolls back to re-tokenize; the rollback goes through `PopNFromKVCache`,
which on a hybrid model also rewinds the GDN recurrent state — and that reaches only as far as the
RNNState history ring:

```
tvm.error.InternalError: Check failed: n <= it->second.available_history_num (1 vs. 0)
```

With `prefix_cache_mode="disable"` the ring is **one slot deep**, so `available_history_num` is
pinned at 0 and *no* rollback is possible at all. Retokenization only refines token boundaries so
the committed ids match a fresh tokenization of the same text — it is not a correctness
requirement — so both call sites now check the ring first and fall back to committing the sampled
token as-is: `CommitTokenMayRetokenize` in `batch_decode.cc` and `HandleRollback`'s caller in
`batch_jumpforward.cc`, one root cause reachable from two places.

⚠️ **It should hit at batch 1 too, but that is reasoned, not measured.** `available_history_num` is
per-sequence and has nothing to do with batch size, so nothing about the crash *depends* on
concurrency. The single-request schema run done this session did not trigger it — that run simply
never produced a jump forward long enough to need a rollback. If a batch-1 repro matters, force one
with a longer schema before claiming it.

**Measured before/after**, 4 concurrent requests against a JSON schema with long mandatory key
names (the shape that gives xgrammar something to jump over): before, the engine thread died in
`PopN` and every request hung. After, all four return, three of them schema-valid JSON and the
fourth only truncated by `max_tokens` — the same output the single-request run gives.

### The 35B cannot use this, and that is a design decision rather than a bug

Running the gate on `dist/qwen3_6-35B-A3B-q4f16_1_fused` fails differently and much earlier:

```
RuntimeError: Check failed: input_shape[i] == reg (6 vs. 1)
  ErrorContext(fn=batch_decode, param=input_embeds, annotation=R.Tensor((1, 1, 2048), "float16"))
```

[qwen3_5_moe_model.py:637](python/mlc_llm/model/qwen3_5_moe/qwen3_5_moe_model.py#L637) pins
`batch_decode` to a **literal** batch of 1, and the comment above it says why: it makes the MoE
block's `if num_tokens == 1:` resolve statically so decode routes through `dequantize_gemv` instead
of `dequantize_group_gemm`, worth ~6× at top-8 on Orin. That specialization is load-bearing for the
58.97 tps in §11. The 0.8B ([qwen35_model.py:1771](python/mlc_llm/model/qwen35/qwen35_model.py#L1771))
keeps `["batch_size", 1, hidden]` and is therefore the model this work unblocks.

So **§11's "batch > 1 is not a reachable configuration on hybrid models in this tree at all" is now
half true.** It is reachable on the 0.8B and was blocked by the engine bug above. On the 35B it is
blocked by a deliberate compile-time specialization, and making it reachable is a separate decision
with a real cost — see §9.

The same run is the regression evidence for the 35B: its **serial phase ran to completion** (six
prompts, one at a time, through `lib_inplace.so`) before the concurrent phase hit the pinned spec.
The prefill cap and the decode guard are therefore neutral for the single-sequence path that every
benchmark in this document measures.

### What is still not covered

- **Speculative decoding at batch > 1 on a hybrid model still aborts.** `BatchVerify`,
  `BatchVerifyToLastHidden` and the `batch_draft` multi-token path all pack `(1, total_len, h)`
  across sequences the same way `BatchPrefill` does, and none of them is capped. The engine now
  warns at startup instead of failing obscurely mid-run. Fixing it wants the same treatment prefill
  got — one sequence per forward — but `batch_draft` and `batch_verify` have to agree on which
  sequences are in flight, so capping one without the other strands the other's draft tokens. Not a
  two-line change.
- **Disaggregated serving** ([disagg_remote_send.cc:148](cpp/serve/engine_actions/disagg_remote_send.cc#L148))
  calls `BatchPrefill` directly and is uncapped. Not a combination this box runs.
- **The VL path.** Same as §9: `Qwen35VLLMHeadModel` reuses `Qwen35Model`, so it inherits the
  dynamic-batch decode spec and should work, but there is no compiled VL model here to gate.

### What it buys, and what it costs

`batch_decode_parity.py` now times both of its phases, so the gate reports the win as well as the
correctness answer. 0.8B `q0f16`, `lib_inplace.so`, 6 requests × 40 tokens, one warmup request
outside each timed region (the first request pays kernel JIT and cudagraph capture, which would
otherwise land entirely on the serial phase and flatter the concurrent one):

| | wall clock |
|---|---:|
| serial (one at a time, `max_num_sequence=1`) | 2.72 s |
| concurrent (all six in flight) | **1.02 s** |
| | **2.67×** |

The cost: prefill for N new requests now takes N engine steps rather than one. Single-request
benchmarks are untouched — every `scratch_mlc_tg_sweep.py` number in this document runs one
sequence. The loss is real only when many *short* prompts arrive together, and it is what buys the
2.67× on the decode that follows.

---

## 13. Landed: in-place GDN *conv* state, and what the trace said about §9 (2026-07-25)

§9 items 2 and 3. Item 2 landed; **item 3 was refuted by measurement and should not be built.**

| | 35B `q4f16_1` | 0.8B `q0f16` |
|---|---:|---:|
| tg512 before → after | 58.79 → **60.20** | 89.57 → **90.72** |
| | **+2.40%** | +1.28% |
| pp512 before → after | 559.95 → **645.37** | 2841.7 → **4054.1** |
| | **+15.3%** | **+42.7%** |
| ttft @ pp512 | 914 → 793 ms | 180 → 126 ms |

Three runs each, spread ≤0.1%; A/B back-to-back, clocks unpinned (§8). Predicted 2.2% decode
from §4.6's traced `get_1`/`set_1`/`update_conv_state1` line; measured **2.40%**, i.e. 109% of
estimate. That is two for two on the §5 estimation lesson — predict from a measured kernel at
the target shape.

**The prefill win was not predicted at all, and it is the larger number.** §9 called this item
"1.5%, harder than option 2". The decode part was right. What the estimate missed is that the
fused kernel also replaces the TE conv itself, which was catastrophically mis-scheduled.

### The TE depthwise conv was ~42× off roofline

`_te_depthwise_conv` is a 4-tap reduction that dlight scheduled as
`grid=(196608,1,1) block=(16,16,1)` — **50.3 M threads for 3.1 M output elements**, 16 threads
per output — at **3404 µs per call**. The data it moves is 12.6 MB, which is **81 µs** at the
156 GB/s wall. Replacing it with one hand-written kernel took the same prefill work from
**709 ms to 55 ms (13×)** on the 35B.

So the ordering of prizes in this item was inverted: the state copies were the stated target
(0.383 ms/token, 2.2% of decode) and the conv schedule was not mentioned, but the conv schedule
was worth more.

### ⚠️ The prefill win only exists under `prefix_cache_mode="disable"`

Under the **default** `"radix"`, prefill routes through `batch_prefill_with_history` →
`forward_with_history`, which keeps the copy path and the TE conv. Verified by trace: a radix
run emits `gdn_func_history` / `set_with_history` / `depthwise_conv1d` and **zero**
`conv1d_inplace`. Every bench harness sets `disable`, so the pp numbers above are real but are
*not* what a default-configured user sees.

This is the same blind spot the §6 warning describes, pointing the other way: there, gating only
under `disable` hides a bug; here, benching only under `disable` shows a win that the default
configuration does not get.

**That makes the history path the biggest remaining prefill item in this document.** Its conv is
still the 3404 µs/call kernel, it is what the default config uses, and the same treatment
applies — with the extra work that `set_with_history` writes per-position rather than one slot.

### §9 item 3 is refuted — do not build it

Item 3 proposed allowlisting the `rnn_state_*` handle builtins as static in
`rewrite_cuda_graph.cc` so the fused kernel could be captured, reasoning from pass source that
it currently is not. Re-traced against `lib_inplace.so` with `--cuda-graph-trace=node`:

- **The prediction was right about capture**: `gdn_func_inplace` is eager, 30/token.
- **But eager launches went *up*, 131 → 183/token.** §11 pushed `gdn_func_inplace` and
  `fused_cast3` out of the graph (their storage argument comes from a `vm.builtin`), so it
  traded 60 captured launches for 60 eager ones and *still* won +6.04%.
- **Idle did not move**: 1.016 ms/token before, **1.065 ms/token** after — up slightly in
  absolute terms while the budget shrank. Halving the eager `rnn_state` population changed
  nothing measurable.

Measured cost of being eager, from inter-kernel gaps in the same trace:

| | gap per launch |
|---|---:|
| graph-captured kernels | **0.36 µs** |
| eager kernels | **~2.2 µs** |

The eager total is dominated by two once-per-token kernels at the sampling boundary
(`fused_dequantize_take2` 331 µs, `rms_norm` 153 µs) which are host-side engine work, not launch
overhead. What is actually recoverable by capturing the GDN region is ~0.1–0.38 ms/token, and
§9's "the whole decode body could become one captured region" does not follow from any of it.
**Item 3's ≤4% ceiling was already generous; the measured marginal cost is ~0.9 µs per launch.**

### Correctness — and one gate that lied

The kernel is gated three ways because on this model no single gate is sufficient.

| gate | 0.8B `q0f16` | 35B `q4f16_1` |
|---|---|---|
| [scripts/conv1d_kernel_check.py](scripts/conv1d_kernel_check.py) — kernel vs fp64, 12 shapes | ✅ | ✅ same kernel, both widths |
| `greedy_snapshot` bit-exactness vs previous lib | ✅ **5/5 identical** | ❌ 3/5 diverge |
| long prompt (~3.5 k tok, crosses prefill chunks), fused path | ✅ identical | — |
| `--greedy-parity` vs HF fp16, radix **and** disable | ✅ 5/5 × 50/50 both | n/a (§6.1) |
| `prefix_cache_roundtrip`, radix **and** disable | ✅ 4/4 both | comparative — see below |
| `batch_decode_parity` 6-way, radix **and** disable | ✅ 6/6 both | n/a (batch pinned to 1) |
| fp8 tier-2 semantic gate | — | ✅ **1/15/2/5/50, identical to `lib_inplace`** |

**The bit-exactness asymmetry is rounding order, not a bug, and the unit gate is what proves
it.** At `conv_dim=6144` dlight happens to accumulate the 4-tap reduction in the same ascending
order the fused kernel uses, so the 0.8B is byte-identical; at 8192 it does not, and the 35B's
dense near-ties cascade from the first flipped token.
[scripts/conv1d_kernel_check.py](scripts/conv1d_kernel_check.py) settles this in seconds with no
model loaded: against an fp64 reference the output lands within fp16 rounding
(rel ≤7.3e-4, fp16 eps 9.8e-4) at **both** widths and at seq_len 1/2/3/4/17/512, while the new
state is **bit-exact** and every history slot outside `{hist, hist+1}` is **untouched**. A
mis-indexed ring would fail the last two; rounding order cannot.

New standing rule, since §10 already learned half of it: `greedy_snapshot` is bit-exact only for
changes that preserve the *schedule*. Replacing a dlight-scheduled kernel with a hand-written one
does not, even when the arithmetic is written to match.

#### The 35B rollback gate is flaky on *both* libs — quantified this time

§11 asserted this from two runs. It was re-run 4–5 times per lib to check whether the conv change
made it worse:

| lib | runs | runs with ≥1 divergence | which checks | worst run |
|---|---:|---:|---|---:|
| `lib_inplace` (unmodified baseline) | 4 | **3** | pass1 ×1, pass2 ×2 | **18/20** |
| `lib_convfused` | 5 | 4 | pass1 ×1, pass2 ×1, pass3base ×3 | 18/20 |

**The baseline fails this gate at essentially the same rate**, and its worst run (18/20) is worse
than the fused lib's best (a clean **20/20**). Every failure is 4/5 on the same near-tie prompt
("The three primary colors are"), and no check ever collapsed to 0/5 — §11's stated signature of a
real state bug, and what the deliberate negative control there produced.

⚠️ **Not fully excluded**: the PopN-rollback check itself diverged in 3/5 fused runs and 0/4
baseline runs. On this sample, of a prompt that demonstrably flips run to run, that is not
significant — but it is the one asymmetry in the data and it is recorded rather than rounded away.
The deterministic evidence for the ring mechanism is much stronger than this gate can be:
`conv1d_kernel_check.py` shows the new state bit-exact and non-target slots untouched at the 35B's
own `conv_dim=8192` with `max_history=64` and a non-zero start slot, and the 0.8B — where a
bit-exact reference exists — is 4/4 on this same gate under both modes.

**This gate needs a deterministic replacement before it can carry weight on the 35B.** Its prompt
set is inherited from an fp16-vs-fp16 era, which is the same complaint §6.2 makes about the fp8
gate; the fix is the same high-margin prompt set (§9 item 4).

### Two mistakes worth not repeating

**1. A gate that passed without testing anything.** The first long-prompt bit-exactness run was
done at the engine default (radix), where prefill uses the history copy path — so it compared the
fused build against itself on a path neither build changes, and passed vacuously. Only the trace
revealed it. Any prefill-path gate on a hybrid model must pass `--prefix-cache-mode disable`
explicitly, or it is not testing the fused path.

**2. `MLC_MOE_GEMM_V2=1` was omitted from the 35B build** — see the warning now at the top of §8.
This produced pp512 **560 → 225 (−60%)** and a 5/5 bit-exactness divergence, both of which read
as a disastrous model-change regression. They were a missing compile-time env var. The 0.8B A/B
was clean throughout because it has no MoE, which is exactly what made the 35B result look like a
model bug. The `nm -D | grep -c group_gemm_v2` check in §8 takes two seconds and would have
caught it before the bench ran.

The lesson generalizes past this flag: an A/B is only an A/B if the two libs differ by the change
under test. Both of this session's "the change broke it" moments were the harness, not the change.

---

## 14. Landed: the history-path conv fusion, and the radix prefill number nobody had taken (2026-07-25c)

§9 item 0. The kernel landed; the more important result is the **baseline measurement that
motivated it**, which had never been taken because no harness in this repo could take it.

### 14.1 The finding: every prefill number in this document is from a non-default configuration

`scratch_mlc_tg_sweep.py` hardcoded `prefix_cache_mode="disable"`
([line 90](scratch_mlc_tg_sweep.py#L90)), so §13's "0.8B pp512 4054" and the 35B's "645" describe a
mode users do not run. Adding `--prefix-cache-mode` and re-measuring the **same libs**:

| 0.8B `q0f16`, `lib_convfused` | pp512 tps | ttft | tg512 tps |
|---|---:|---:|---:|
| `disable` (fused conv path) | 4053 | 126 ms | 90.83 |
| **`radix` — the engine default** | **1462** | **350 ms** | 90.40 |

**A 2.77× prefill gap, +224 ms of ttft, on the configuration that ships.** §13 predicted this
qualitatively ("the pp numbers are real but are *not* what a default-configured user sees"); this
is the number. Decode is unaffected — the split is prefill-only, because
`prefix_cache_mode` only selects which *prefill* forward path runs.

⚠️ **A second harness bug had to be fixed to measure this at all.** `build_prompt` returned the
same prompt every run, so under radix runs 2+ are full cache hits and `pp_tps` measures a cache
lookup rather than prefill. The sweep now salts each run's prompt (`--unique-prompts`, on by
default whenever the mode is not `disable`) and reports the *re-encoded* length. Without that fix
the radix column reads as a spectacular win instead of a 2.77× loss.

### 14.2 Where the radix prefill budget goes (0.8B, nsys, 36 GDN-layer instances)

| kernel | avg µs/call | % GPU time | what it is |
|---|---:|---:|---|
| `rnn_state_set_with_history_0` | 5188 | **24.8%** | scatter recurrent state into the ring |
| `gdn_func_history` | 5003 | **23.9%** | recurrence emitting per-position state |
| `depthwise_conv1d` | 3026 | 14.5% | the TE conv §13 measured at ~42× off roofline |
| `NT_matmul5` | 2831 | 11.3% | |
| `update_conv_state_history` | 477 | 2.3% | materializes the per-position conv history |
| `rnn_state_set_with_history_1` | 184 | 0.9% | scatter conv state into the ring |

**66% of all GPU time on this path is history-path GDN work.** §9 item 0 scoped the conv; the
trace says the conv is the *third* prize (17.7% combined) and the **recurrent pair is 48.7%**.

### 14.3 The kernel

`create_causal_conv1d_func_with_history_inplace` collapses four kernels — `rnn_state_get_1`,
`update_conv_state_history`, the TE conv, and `rnn_state_set_with_history_1` — into one, mirroring
`RNNState.create_set_with_history_func`'s addressing: load from `storage[seq, hist]`, flush position
`t` to `storage[seq, (hist + 1 + t) % max_hist]`.

Two things make it harder than §13's kernel:

1. **The flush wraps the ring.** `seq_len` is a prefill chunk (512, up to 2048) against
   `max_history=64`, so slot `hist` — the one the conv reads — is itself overwritten mid-kernel.
2. **Most of the scatter is dead.** `EndForward` caps `available_history_num` at `max_history - 1`,
   so only the last 63 positions are ever reachable. Writing all 512 and letting 449 be overwritten
   is pure waste; the kernel skips any `t` a later position provably overwrites, turning a
   512-position scatter into a 64-position one (32× at a full 2048 chunk) with no semantic change.

That skip guard turns out to do double duty — see 14.5.

### 14.4 Measured

Both libs verified `MLC_MOE_GEMM_V2=1`-clean (`nm -D | grep -c` → 4) and differing only by this
change. Three runs each, spread ≤0.3%, `--prefix-cache-mode radix`:

| | 35B `q4f16_1` | 0.8B `q0f16` |
|---|---:|---:|
| pp512 before → after | 355.3 → **393.8** | 1468.6 → **1797.6** |
| | **+10.9%** | **+22.4%** |
| ttft @ pp512 | 1441 → **1300 ms** | 350 → **284 ms** |
| tg before → after | 60.02 → 59.77 (−0.4%) | 90.69 → 90.77 (+0.1%) |
| pp512 under `disable` (regression check) | — | 4053 → 4070 (unchanged) |

Predicted from 14.2's traced share (17.7% of GPU time conv-related); 0.8B ttft fell 18.9% =
**107% of estimate**. That is three for three on the §5 estimation lesson — predict from a measured
kernel at the target shape. Decode is untouched by construction (this path is prefill-only under
radix and verify-only otherwise), and the measurement confirms it.

**The 35B gains less than the 0.8B (+10.9% vs +22.4%)** because the 35B's radix prefill is more
dominated by the MoE GEMMs, so the same absolute GDN saving is a smaller share. The remaining
`radix`-vs-`disable` gap on the 35B is still large — 394 vs 645 — and 14.2 says the rest is the
recurrent pair.

### 14.5 Correctness — and a mechanism claim that did not survive its own test

| gate | result |
|---|---|
| [conv1d_kernel_check.py](scripts/conv1d_kernel_check.py), history variant: 12 shapes × 2 widths, 4 ring configs | ✅ output within fp16 rounding, **state bit-exact, non-target slots untouched** |
| ” incl. `seq_len=512 > max_history` (WRAP), `max_history=1`, `hist_slot=63`, `max_history=4, batch=3` | ✅ all |
| `--greedy-parity` vs HF fp16, radix **and** disable | ✅ **5/5 prompts, 50/50 tokens, both** |
| `prefix_cache_roundtrip` (PopN rollback), radix | ✅ **4/4** |
| `batch_decode_parity` 6-way, radix **and** disable | ✅ **6/6 both** |
| long prompt (3133 tok, crosses the 2048 prefill chunk), radix | ✅ **byte-identical** |
| `greedy_snapshot` bit-exactness, radix | 1/5 diverge — rounding order, see below |
| 35B fp8 tier-2 semantic gate, radix | ✅ **1/15/2/5/50, identical to `lib_convfused` prompt for prompt** |

**The 1/5 divergence is rounding order, and this time it was confirmed from launch geometry rather
than assumed.** The baseline history conv runs `grid=(174720,1) block=(16,16)` — **44.7 M threads
for 3.1 M outputs, ~14 threads per 4-tap reduction**, i.e. a cross-thread tree reduction. A
sequential ascending fp16 sum cannot match that bitwise, so bit-exactness was never available on
this path (and that geometry is also *why* the TE conv was 42× off roofline). §13 got bit-exactness
on the 0.8B only because the *decode* conv happened to be scheduled with a matching order.

⚠️ **A negative control refuted the safety mechanism this kernel was built around.** The kernel
stages the one write that targets `hist_slot` and flushes it after all reads, on the reasoning that
the ring wrap would otherwise clobber state the conv still needs. Building the un-staged version
and running the gate: **it passes all 12 shapes.** The skip guard in 14.3 already removes exactly
the early positions (`t < ks-2`) whose flush would re-read the old state, so at `kernel_size=4` no
write to `hist_slot` can precede a read of it at any `(max_history, seq_len)`. The staging is kept
as defence-in-depth — it costs 3 registers and the argument above depends on `ks-1 == 3` — but
**it is not what makes this kernel correct, and the docstring now says so.**

**The gate does have teeth**: a deliberate ring off-by-one (`hist + t` instead of `hist + 1 + t`)
fails all 12 shapes with `state_err ~3.0` while `out_rel` stays clean — exactly separating "state
misindexed" from "output wrong", which is what this instrument exists for.

### 14.6 Compile-time investigation (asked mid-session: "why is the 35B compile single-threaded?")

Measured phase split:

| phase | 0.8B | 35B |
|---|---:|---:|
| TVM/Relax/dlight passes (Python, 1 core of 12) | ~140 s (65%) | **~11 min (83%)** |
| `nvcc` fatbin + link | 64 s (29%) | 133 s (17%) |

So `nvcc` is the minority cost. On the 3.5 MB translation unit TVM emits (sm_87, nvcc 13.2,
`--fatbin -O3`):

| flags | time |
|---|---:|
| baseline | 41.5 s |
| **`-split-compile=12`** | **31.6 s (−24%)** |
| `-split-compile=0` (auto) | 31.8 s |
| `-t 12` | 41.7 s — **no effect** |
| `-split-compile-extended=12` | 41.5 s — no effect |

`-t/--threads` is inert because it only parallelizes across `-gencode` targets and we build one
arch; `-split-compile` parallelizes the device-code optimizer *within* the single unit TVM emits.

⚠️ **`-split-compile` is a codegen change, not just a build-speed knob.** The fatbins disassemble
to different SASS (23382 differing lines, +48 instructions), so a lib built with it is **not
A/B-comparable** against one built without — the §8 trap. It is therefore opt-in via
`MLC_NVCC_OPTIONS`, not a default. `MLC_DUMP_CUDA=<path>` was added alongside it so nvcc flags can
be benchmarked against a real unit in ~40 s instead of a 13-minute rebuild.

**The real lever is the pass pipeline, which has no parallelism knob — so reduce its input
instead.** The 35B spec compiles 18 entry points, **14 of them full 48-layer traversals**, and 4 of
those (`batch_verify_g1..g4`) exist only for speculative decoding. Skipping them on
correctness-iteration builds should cut ~20–25% of the dominant phase. Not implemented — it changes
what the lib can do, so it needs a deliberate opt-in flag rather than a silent default.
---

## 15. Landed: the history-path *recurrent* fusion — the radix prefill gap closes (2026-07-25d)

§9 item 0a, the item §14.2's trace had just promoted to "the biggest in this document". It was,
and it is the largest single measured win in the whole workplan.

### 15.1 Measured

Both libs `MLC_MOE_GEMM_V2=1`-clean (`nm -D | grep -c` → 4) and differing only by this change.
Three runs each, spread ≤0.3%, clocks unpinned (§8), `--prefix-cache-mode radix`:

| | 35B `q4f16_1` | 0.8B `q0f16` |
|---|---:|---:|
| pp512 before → after | 395.4 → **629.1** | 1793.0 → **3934.5** |
| | **+59.1%** | **+119.4%** |
| ttft @ pp512 | 1294.8 → **813.7 ms** (−37.2%) | 285.5 → **130.1 ms** (−54.4%) |
| tg512 before → after | 59.86 → 59.68 (−0.3%) | 90.47 → 90.54 (+0.08%) |
| pp512 under `disable` (regression check) | 645.4 → **646.1** (unchanged) | 4053 → 4070 (unchanged) |
| `lib.so` | 160 → **113 MB** | 22 → **18 MB** |

**The 2.77× default-configuration prefill penalty §14.1 found is gone on the 0.8B.** That section
measured `radix` at 1462 against `disable`'s 4053 and called it "a 2.77× prefill gap on the
configuration that ships". After §14 and this change the same comparison is **3934 vs 4070 —
1.035×**. The default path now does essentially the same work as the fast path, which is what it
should always have done: the only remaining difference is scattering 64 ring slots instead of
advancing one. The 35B closes the same way: **1.64× (394 vs 645) → 1.027× (629 vs 646)**.

The `disable` row is the regression check that matters here — this change touches only
`forward_with_history`, so the fused-prefill path must not move, and it does not (645.4 → 646.1,
within run-to-run spread).

### 15.2 The trace, and an honest scoring of the prediction

Both libs traced identically (`nsys`, 0.8B, radix, 36 GDN-layer history calls), so this is
apples-to-apples rather than a comparison against §14.2's pre-§14 table:

| kernel | `lib_histconv` | `lib_gdnhist` |
|---|---:|---:|
| `rnn_state_set_with_history_0` | 5176.5 µs × 36 = **186.4 ms** (10.1%) | **gone** |
| `gdn_func_history` | 4966.6 µs × 36 = **178.8 ms** (9.7%) | — |
| `gdn_func_history_inplace` | — | 2654.7 µs × 36 = **95.6 ms** (6.1%) |
| **the recurrent pair** | **365.2 ms** | **95.6 ms — 3.82×** |
| `conv1d_history_inplace` (untouched control) | 363.7 µs × 36 = 13.1 ms | 365.8 µs × 36 = 13.2 ms |
| **whole-trace GPU time** | **1847 ms** | **1579 ms** |

The whole-trace delta is **268 ms** and the two kernels account for **269.6 ms** of it. Nothing
else moved — the untouched conv kernel is within 0.6%, and every `NT_matmul` is within 0.3%. That
is the cleanest no-side-effects evidence in this document, and it is worth noting that it only
exists because the change was traced against its own A/B baseline rather than against an older
table.

**The prediction was recorded before any lib was built** and is scored here in full:

| claim | predicted | measured |
|---|---|---|
| `set_with_history_0` removed outright | 100% | ✅ 100% |
| `gdn_func_history` is store-bound, so most of it goes | **60–90%** | ❌ **46.6%** (4966.6 → 2654.7 µs) |
| total saving | **47–56%** | ✅ **54.4%** (0.8B ttft 285.5 → 130.1 ms) |
| 0.8B pp512 | **3200–3900** | ✅ **3934** |
| radix approaches `disable` from below without reaching it | — | ✅ 3934 vs 4070 |
| decode unchanged | 0% | ✅ +0.08% / −0.3% |

**Four for four on the headline, but the mechanism accounting had two errors that cancelled**, and
that is worth more than the score. The store-bound inference was too strong: `gdn_func_history`'s
537 MB at 5003 µs reads as 107 GB/s, but removing 87.5% of those stores bought only 46.6% of the
kernel, so the recurrence arithmetic is a bigger share than the achieved-bandwidth figure implied.
Offsetting that, the renormalization used §14.2's **whole-run** percentages as if they were
**prefill** percentages — two different denominators — which understated the pair's share of the
thing ttft actually measures. Right answer, partly for the wrong reason.

**Rule for the next estimate: renormalize against a trace of the actual A/B baseline, not against
the table from two sessions ago, and never mix a whole-run denominator with a prefill-only
metric.** Both halves of that were avoidable here — the baseline trace this section is built on
took three minutes and would have caught both.

### 15.3 The kernel

`create_gated_delta_net_func_with_history_inplace` collapses three kernels — `rnn_state_get_0`,
`gdn_func_history`, `rnn_state_set_with_history_0` — into one, with the same ring addressing
§14's conv kernel uses: load from `storage[seq, hist]`, flush position `t` to
`storage[seq, (hist + 1 + t) % max_hist]`.

**The copy elimination is the smaller half.** The bigger half is not materializing dead state.
The kernel it replaces emits a `(batch, seq_len, n_vh, K, V)` fp32 tensor — **537 MB per layer
per call** at pp512 on the 0.8B — which the scatter then reads straight back. But `EndForward`
caps `available_history_num` at `max_history - 1`, so only the last 63 of those 512 positions can
ever be read: **~87% of the tensor is overwritten in the ring before anything can reach it**, 32×
at a full 2048 chunk. Guarding the flush on `t + max_history >= seq_len` turns 537 MB written +
537 MB read + a scatter into one ~67 MB scatter.

**It is also better-defined than the path it replaces.** `create_set_with_history_func` documents
the precondition "caller must guarantee `max_history >= seq_len + 1` so writes do not collide" —
and prefill violates it on every chunk (512–2048 positions into 64 slots). Its writes are a flat
parallel `T.grid(batch_size, seq_len)`, so `t` and `t + max_hist` race for the same slot and the
winner is whichever warp retires last. That was only harmless because the racing writes all land
in slots nothing can read — *except* at the boundary. Skipping the doomed writes removes the race
by construction: each ring slot now has exactly one writer.

**No register staging, unlike §14's conv kernel.** Every read of the storage buffer happens in the
preload loop before the `t` loop starts, so even when the flush wraps far enough to overwrite
`hist_slot` itself, no read can observe it. §14.5 discovered its staging was defence-in-depth
rather than load-bearing; here the read/write ordering makes the question disappear.

### 15.4 Correctness — bit-exact, which no previous history-path change managed

| gate | result |
|---|---|
| [scripts/gdn_kernel_check.py](scripts/gdn_kernel_check.py) — 9 seq_lens × 2 head configs, incl. 5 wrapping | ✅ **output bit-exact vs the copy-path kernel, ring bit-exact, non-target slots untouched** |
| ” fp64 reference of the recurrence | ✅ rel ≤4.1e-6 |
| `--greedy-parity` vs HF fp16, radix **and** disable | ✅ **5/5 prompts, 50/50 tokens, both** |
| `greedy_snapshot` bit-exactness, radix | ✅ **5/5 byte-identical** |
| long prompt (5814 tok, **3** prefill chunks), radix | ✅ **byte-identical** |
| `prefix_cache_roundtrip` (PopN rollback), radix | ✅ **20/20** |
| `batch_decode_parity` 6-way, radix **and** disable | ✅ **6/6 both**, 2.68× concurrency |
| 35B fp8 tier-2 semantic gate, radix | ✅ **1/15/2/5/50, identical to `lib_histconv` prompt for prompt** |

**This is the first change on the history path that is bit-exact end-to-end**, and the unit gate
predicted it before the engine ran. §10, §13 and §14 all had to argue their divergences were
rounding order; here there is nothing to argue, because the fused kernel runs the same two passes
in the same order over the same fp32 registers as `gdn_func_history` — only the destination of the
flush changed. §14's conv fusion could not be bit-exact because the TE conv it replaced used a
cross-thread tree reduction; this one replaces a kernel that was already sequential per thread.

New rule to go with §13's: **when a fusion changes only where a result is written, demand
bit-exactness.** It is available, and settling for a tolerance would have hidden the ring bugs the
negative controls below produce.

### 15.5 Three negative controls, one of which refutes a design assumption

`gdn_kernel_check.py` is only worth running if it fails when it should. All three were built:

| control | result |
|---|---|
| ring off-by-one (`hist + t` instead of `hist + 1 + t`) | ❌ **fails**, `state_err` 3.2–130, `other_slots` fires — while `out_bit` stays **0** |
| skip guard one position tighter (`>` for `>=`) | ❌ **fails** the wrapping shapes, `state_err` 2.4–520 |
| skip guard removed entirely (write every `t`) | ✅ **passes every shape** |

The first is the instrument working exactly as designed: a misindexed ring leaves the *output*
untouched, so a token-diff can never see it, and `state_err`/`other_slots` separate it from
rounding cleanly.

The third is the interesting one. **The skip guard is a pure optimization, not what makes the
kernel correct** — writing all 512 positions is equally correct, because within a thread the `t`
loop is sequential so the last write wins deterministically. The guard is worth ~8× of this
kernel's store traffic and nothing else. The second control shows it is nevertheless placed
exactly: one position tighter and reachable state is lost. So the guard is neither loose nor
tight by one, and both sides of that were measured rather than argued — the §14.5 lesson applied
prospectively for once.

### 15.6 What is now the biggest prefill item, and why it is a different kind of problem

With both state fusions landed, the remaining radix prefill budget on the 0.8B looks like this
(same trace, prefill-only kernels — 36 instances = 18 GDN layers × 2 chunks, 48 = attention):

| kernel | ms | what it is |
|---|---:|---|
| `gdn_func_history_inplace` | **95.6** | the GDN recurrence itself |
| `NT_matmul8_kernel_2` | 19.3 | attention `c_attn` |
| `NT_matmul6_kernel_2` | 18.5 | GDN `in_proj_qkvzab` |
| `conv1d_history_inplace` | 13.2 | the §14 fused conv |
| `NT_matmul9_kernel_2` | 9.8 | attention `o_proj` |
| `fused_split7_silu5_multiply5` | 8.5 | |
| `split5` | 7.6 | |

**The recurrence is now 5× the next item, and it is not a bandwidth problem — it is parallelism
starvation.** The kernel launches `batch × n_vh` blocks of `V` threads: on the 0.8B that is
**16 blocks of 128 threads = 2048 threads total**, on a GPU with 16 SMs. Each thread walks the
sequence strictly sequentially because the recurrence is sequential in `t`. Measured:
2048 threads × 512 positions × 2 passes × 128 rows × 2 flops = 537 MFLOP in 2655 µs =
**202 GFLOP/s, about 3.8% of the sm_87 fp32 peak**. Removing the stores was worth 1.88× and there
is nothing left to remove — the kernel is now latency-bound on a dependency chain.

This is the same diagnosis §5 tier-3 gave `in_proj_a`/`in_proj_b` (`grid=(1,1,1)`, 1/16 of the
GPU), and it has the same shape of fix: **more parallelism, which for a linear-attention
recurrence means the chunked formulation** — process the sequence in chunks of C positions with
matmuls over the chunk instead of a scalar loop, which is exactly what
`../flash-linear-attention/fla/layers/gated_deltanet.py` and vLLM's `qwen3_next.py` do and why
their prefill is fast. That is a substantially bigger piece of work than anything in §10–§15: it
changes the arithmetic (so bit-exactness is off the table and the tier-2 gate is what decides it),
and it needs its own chunk-state intermediate. **Scope it deliberately; do not start it as a
follow-on to this session.** Note also that it would not help decode at all — at `seq_len=1` the
chunked form degenerates to the current one, and `gdn_func_inplace` is already only 2.1% of the
whole-run budget.

Cheaper things to check first, in case they move it without the restructure: the kernel uses
`K=128` fp32 registers per thread, which at 128 threads/block is 64 KB of registers per block and
almost certainly caps occupancy at 1 block/SM — worth confirming, since a smaller register
footprint (splitting `K` across two blocks with a cross-thread reduction per `t`) might raise
occupancy without changing the algorithm.

---

## 16. Session 2026-07-26 — the gates get teeth, two items refuted, and the last one built

> §16.1–§16.4 are session **2026-07-26a**; §16.5 is **2026-07-26b**, which built the kernel §16.2
> designed and **corrected two of §16.2's conclusions in the process** — see the callout there.

Four §9 items were open at the start of this session: **0b** (a deterministic 35B state gate),
**0c** (the parallelism-starved GDN recurrence), **1** (should the 35B decode more than one
sequence), and **5** (the tier-2 GEMV retune). All four are resolved here — 0b landed, 1 settled,
5 refuted, and 0c's step 1 answered and measured (step 2, the chunked reformulation, is the only
thing left open in §9).

**Two of the four ended by refuting their own premise**, which is the pattern worth carrying
forward: item 5's question assumed K=2048 was the good case when K=4096 is the best case, and
item 1's "~6×" was a source comment nobody had measured. Item 3 went the same way in §13. Every
time, the write-up had a plausible mechanism and the measurement said the premise was wrong.

### 16.1 Item 0b — LANDED. The 35B finally has a gate that can adjudicate a state change

§6.2 left this with a diagnosis and a one-line prescription ("build a high-margin prompt set").
Carrying it out surfaced a second defect that the prescription alone would not have fixed.

**Defect 1 — the prompt set.** As diagnosed. The old five are open-ended continuations where the
next token is a near-tie, so 4-bit-vs-8-bit noise decides them.

**Defect 2 — cascade, which is the larger one and was not in the diagnosis.** Both sides
free-run. One near-tie flip and the two sequences are in *different contexts* for every later
position, so the score is dominated by where the first flip happened rather than by how often the
model disagrees. That is why §6.2 read 1/50, 15/50, 2/50, 5/50, 50/50: those are five samples of
"when did it first diverge", not five measurements of agreement.

[scripts/high_margin_gate.py](scripts/high_margin_gate.py) fixes both:

1. **Teacher forcing.** At position `i`, MLC is asked for exactly one token given the reference's
   own prefix `prompt + ref_tokens[:i]`. Every position is scored in the context the reference
   saw, so a flip at position 3 cannot contaminate position 4. Implemented over the public engine
   API — `_generate` accepts a token-id list, so no new plumbing.
2. **Margin-gated scoring.** The capture step records `logprob(top1) - logprob(top2)` at every
   position; the check step asserts agreement only where that margin clears `--tau`. A wide-margin
   flip is a bug; a near-tie flip is quantization. The pass bar stops being an inherited constant.

Under `radix`, teacher forcing is also a harder exercise of the history path than anything else in
the tree: each of the 400 requests extends the previous by exactly one token, which is the
fork-and-extend case the recurrent ring exists to serve.

**τ = 2.0 is calibrated, not asserted.** The 0.8B is the control, because it has both an fp16
reference *and* a 4-bit build — the same relationship the 35B has to its fp8 reference:

| lib vs `Qwen3.5-0.8B` fp16 reference | τ=0 | τ=0.5 | τ=1.0 | τ=2.0 | τ=4.0 | τ=8.0 |
|---|---:|---:|---:|---:|---:|---:|
| `q0f16` (fp16, radix) | 400/400 | 385/385 | 376/376 | 361/361 | 267/267 | 64/64 |
| `q0f16` (fp16, disable) | 400/400 | 385/385 | 376/376 | 361/361 | 267/267 | 64/64 |
| `q4f16_g16e` (4-bit, radix) | 389/400 | 381/385 | 375/376 | **361/361** | 267/267 | 64/64 |

Two things fall out. The fp16 build agrees with HuggingFace at **every one of 400 positions**,
including all 39 near-ties — a stronger statement than the old 5×50/50. And 4-bit quantization
flips 11 positions, **every one of them at margin ≤ 1.031**, so τ=2.0 clears the highest observed
quantization flip by ~2×. The same 4-bit lib scores 1/5 prompts under the old free-running gate.

**A negative control, because a gate that cannot fail is not a gate.** `--negative-control stale1`
feeds a prefix one token short while still scoring against the reference — the outward signature
of an off-by-one in history-state indexing. On the same fp16 lib that scores 361/361:

| run | agreement at τ=2.0 |
|---|---|
| normal | **361/361 (100%)** |
| `stale1` | **19/361 (5.3%)** |

Genuine 4-bit noise and a one-step-stale state are separated by ~19× at the pass bar. The
distribution is flat across τ (4.7% at τ=0, 6.3% at τ=8), which is the signature of a *mechanism*
break rather than a rounding difference — margin does not protect you from computing on the wrong
state, and that asymmetry is what makes the gate diagnostic rather than merely sensitive.

**The second gate it unblocks.** §9 item 0b claimed a high-margin set would fix two gates, and
[scripts/prefix_cache_roundtrip.py](scripts/prefix_cache_roundtrip.py) was the other: §13 measured
it failing on the *unmodified 35B baseline* in 3 runs out of 4, always on the primary-colours
prompt, always the same near-tie ("red, yellow, and blue" vs "red, blue, and yellow"). Its five
prompt families were the same fp16-era set. They are now the high-margin families, each extension
being the model's own measured continuation (checked against
`tuning/high_margin_ref_0.8b_fp16.json`), with `--legacy-prompts` retained so §13's numbers stay
reproducible.

**Measured, and it is the cleanest A/B in this document** — same unmodified 35B `q4f16_1` baseline
lib, same engine config, same `radix` mode, *only the prompt set differs*:

| prompt set | runs | result |
|---|---|---|
| **high-margin (new default)** | 4 | ✅ **4/4 ALL CHECKS PASS** |
| legacy (`--legacy-prompts`) | 2 | ❌ **0/2** — 3 and 2 divergences |

The legacy failures also reproduce §13's *character*, not just its rate: they land on different
prompts and different checks each run (run 1 on prompts 4, 4, 3; run 2 on prompts 3, 3), always on
`Once upon a time…` or `The three primary colors are`, and `pass3 base vs cold` — the PopN rollback
check, the one the gate exists for — passes in both. **A wandering failure set is the signature of
a near-tie instrument, and §13 already said as much; the fix is to stop asking the model questions
it does not have an opinion about.**

**The 35B now has a hard pass/fail bar.** `tuning/high_margin_ref_35b_fp8.json` — 6 prompts × 24
positions against `Qwen/Qwen3.6-35B-A3B-FP8` through the §6.1 software W8A16 shim, 578 s once
loaded. The prompt set carries over: **139/144 positions (96.5%) clear margin 2.0** on the 35B, and
all six continuations are exactly right (Fibonacci, powers of two, squares, arithmetic progression,
the 3× table, weekdays).

Both 35B libs, both prefix-cache modes, `q4f16_1` vs the fp8 reference:

| τ | scored | mismatch | agreement |
|---:|---:|---:|---:|
| 0.0 | 144 | 3 | 97.92% |
| 1.0 | 142 | 1 | 99.30% |
| **2.0** | **139** | **0** | **100.00%** |
| 4.0 | 99 | 0 | 100.00% |
| 8.0 | 17 | 0 | 100.00% |

**All four runs — `lib.so` and `lib_gdnhist.so`, under `radix` and `disable` — produced this table
identically**, down to which three positions mismatched. Set against §6.2's 1/15/2/5/50 on the same
model, that is the difference between a number nobody could act on and a bar a change can be held
to. And the 4-bit-vs-fp8 flips land exactly where the 0.8B calibration predicted: all three below
τ=2.0, the highest between 1.0 and 2.0 (0.8B: 11 flips, highest 1.031).

Two cautions on reading it. Identical scores across four libs would *also* be what a gate that
cannot see the difference produces — the `stale1` control is what rules that out, and §15 already
established these libs are bit-exact to each other, so agreement is the correct expected result.
And 24 positions × 6 prompts is a smaller sample than the 0.8B's 400; the τ=8.0 row rests on 17
positions and should not be quoted on its own.

### 16.2 Item 0c.1 — the "cheap first" step, answered. Two corrections to the hypothesis

§15's closing paragraph guessed that `K=128` fp32 registers per thread "almost certainly caps
occupancy at 1 block/SM" and proposed splitting `K` across two *blocks* with a cross-thread
reduction. Confirmed by `ptxas -v` and `nvdisasm` on the kernel TVM actually emits — with two
corrections, one of which invalidates the proposed mechanism.

**Correction 1 — registers do not cap it at 1 block/SM; the *grid* does.** ptxas lands on **255
registers/thread**, the hardware ceiling, which permits **2** blocks/SM at 128 threads. But the
kernel launches `(n_kh, batch)` = **16 blocks at batch=1 on a 16-SM GPU**, so only **1.00 blocks/SM
is available to schedule**. Occupancy is 4 warps of the 48 an SM holds — **8.3%** — and it is
grid-bound, not register-bound. A change that only reduced register pressure would buy nothing.

**Correction 2 — the state does not fit in registers, and §15's "no bandwidth left" is DRAM-only.**
`float state_local[128]` is 512 bytes/thread; ptxas keeps ~81 rows in registers and **spills 47** —
`192 bytes stack frame, 188 bytes spill stores, 188 bytes spill loads`, and exactly **47 LDL + 47
STL** in the SASS. Both inner loops touch all 128 rows every timestep, so those 47 rows are
reloaded and restored *per position*. It is L1/L2 traffic rather than DRAM, which is why §15's
bandwidth analysis did not see it, but it is memory traffic in the inner loop.

**And the finding neither the hypothesis nor §15 had — the dot products are one serial FMA chain.**
Disassembling the emitted kernel: 256 FFMAs per timestep (2 dots × 128 rows, fully unrolled), of
which **115 consecutive FFMAs accumulate into a single register, R124**:

```
FFMA R124, R125, R158, R124 ;
FFMA R124, R125, R157, R124 ;
FFMA R124, R125, R156, R124 ;      <- 115 of these in a row, one accumulator
```

At ~4-cycle FFMA latency that is a ~1000-cycle dependency chain per position, and with 4 warps/SM
(one per scheduler) there is nothing to interleave against it: each scheduler issues one FFMA every
4 cycles, ~25% of issue capacity. That is the mechanism behind §15.6's "202 GFLOP/s, 3.8% of peak"
— **not** bandwidth, and only partly occupancy.

**Why the proposed fix cannot be built as proposed.** `K` is reduced *inside* a thread: `threadIdx.x`
indexes `V`, each thread owns one value column and loops over all 128 key rows. Splitting `K` across
two **blocks** would need a cross-**block** reduction *every timestep* — a global barrier per
position, fatal for a sequential recurrence. Splitting `K` across **lanes** of the same block is the
viable form, and if the split is laid out `tid = v*2 + half` the two halves land on adjacent lanes,
so the reduction is a single `__shfl_xor_sync` with no `__syncthreads` at all.

[scripts/gdn_recurrence_probe.cu](scripts/gdn_recurrence_probe.cu) builds four variants to separate
the three causes. Static resource usage, before timing:

| variant | threads | floats/thread | registers | spill | warps/SM |
|---|---:|---:|---:|---:|---:|
| `base` (as shipped) | 128 | 128 | 255 | 240 B | 4 |
| `acc4` (4 accumulators) | 128 | 128 | 255 | **424 B** | 4 |
| `ksplit2` (K/2 across lanes) | 256 | 64 | 168 | **0** | 8 |
| `ksplit4` (K/4 across lanes) | 512 | 32 | **97** | **0** | 16 |

`acc4` isolates the chain: it shortens the dependency 4× but spills *more*, so if it still wins,
latency dominates spill. `ksplit4` vs `ksplit2` discriminates occupancy from spill — both are
spill-free, only occupancy differs.

**Measured** (Orin, clocks pinned, 20 iters, batch=1, n_kh=16 — so 16 blocks, the shipped
geometry). ms per kernel call:

| seq_len | base | acc4 | ksplit2 | ksplit4 | acc4 × | ks2 × | **ks4 ×** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0777 | 0.0778 | 0.0376 | 0.0206 | 1.00 | 2.07 | **3.78** |
| 128 | 0.7295 | 0.7268 | 0.2986 | 0.2627 | 1.00 | 2.44 | **2.78** |
| 512 | 3.4900 | 2.7491 | 1.5546 | 1.2494 | 1.27 | 2.24 | **2.79** |
| 2048 | 13.5119 | 10.5220 | 5.9638 | 4.8470 | 1.28 | 2.27 | **2.79** |

Max relative deviation from `base` is 5.7e-7 across every variant and length — fp32 reduction
order, not a different answer.

Reading it:

- **The single accumulator is real but not the main cost.** `acc4` breaks a 115-deep chain into
  four and buys only **1.27×**, and nothing at all below 512 positions. It spills *more* (424 B vs
  240 B), and that gives back most of the ILP it gains — which is the answer to "is it latency or
  spill": at fixed register pressure the two are entangled, and you cannot fix one alone.
- **Getting the state into registers is what matters.** `ksplit2` halves `state_local` to 64
  floats, drops from 255 registers to 168 with **zero spill**, and is worth **2.24–2.44×** — nearly
  double `acc4` for the same 2× reduction in chain length. The difference between them is exactly
  the 47 LDL + 47 STL per timestep.
- **Occupancy adds, with diminishing returns.** `ksplit4` doubles warps/SM again (8 → 16) and adds
  only **1.23×** on top of `ksplit2` (2.79 vs 2.27). Both are spill-free, so that increment is
  occupancy and shorter chains alone — and it is already flattening, so a 8-way split is unlikely
  to be worth its reduction cost.

**So the "cheap first" step is worth ~2.8× on the recurrence, and the answer to §15's proposal is
"right instinct, wrong axis".** Splitting `K` was correct; splitting it across *blocks* was not,
and the reason it helps is primarily that it takes the state off the 255-register ceiling rather
than that it raises occupancy.

**Bounding the claim honestly.** The probe measures the recurrence only — no ring flush (see the
header comment), one head config, and a synthetic input distribution matched to the model's ranges.
`gdn_func_history_inplace` is 95.6 ms of the **0.8B** pp512 budget (§15.6); at 2.79× that lane would
fall to ~34 ms, but the flush is excluded and Amdahl applies to the rest of prefill, so treat ~2.8×
as an upper bound on the kernel and *not* as a prefill prediction. Per §15.2's rule, the estimate
to publish should be renormalized against a trace of the actual A/B baseline before any lib is
built.

> ⚠️ **Corrected 2026-07-26b, §16.5 — two claims above did not survive being built.** This section
> originally said "the 35B pp512 budget"; §15.6's trace is the **0.8B**, which is also the geometry
> the probe's `grid(n_kh, batch)` = 16 blocks reproduces. So the probe was faithful to its target,
> but **its ratios do not transfer to the 35B**, where the real grid is `(n_vh, batch)` = 32 blocks
> and `k_split=2` is a *regression*. And the ranking of causes above — "primarily that it takes the
> state off the 255-register ceiling rather than that it raises occupancy" — is **geometry-dependent
> rather than general**: at 32 blocks the spill is equally gone at `k_split=2` and the kernel is
> still slower. Measured in TIR: **1.94× (0.8B) and 1.21× (35B)** at seq_len=512, against 2.79× here.

**Status: built in TIR and landed — see §16.5.** `create_gated_delta_net_func_with_history_inplace_ksplit`,
`MLC_QWEN35_GDN_KSPLIT` default 4. There is no `tvm_warp_shuffle_xor` in this TVM, so the partner
lane is computed as `(tid % 32) ^ d` and passed to `T.tvm_warp_shuffle`; bit-exactness against the
copy path is off the table as predicted, and the fp64 plus untouched-slots checks became the bar.

### 16.3 Item 5 — the sm_87 GEMV tile is K-blind, and the gap tracks N, not K

§9's open question asks why `out_proj`/`o_proj` (75%) and routed-expert down (76%) fall short of
the 88% the same schedule reaches at K=2048, and whether that is shape-driven or schedule-driven.
Reading the rule first ([gemv.py](3rdparty/tvm/python/tvm/s_tir/dlight/gpu/gemv.py)) changes what
the measurement needs to be:

**There is no K term in the tile choice at all.** The sm_87 branch sets `TS, TR = 32, 16` with
`TILE_S = 2` unconditionally, so one tile is applied to K=512 and K=4096 alike. K and the schedule
are therefore confounded in every number in §4.6 — nothing measured so far can separate them.

`TS × TILE_S = 64` output elements per block, so **block count is `N/64`**, and re-reading §4.6
against that makes the pattern look like N rather than K:

| kernel | N | blocks | blocks/SM | % of wall |
|---|---:|---:|---:|---:|
| `lm_head` | 248320 | 3880 | 242 | **100%** |
| attn `c_attn` | 9216 | 144 | 9.0 | **91%** |
| GDN `in_proj_qkv` | 8192 | 128 | 8.0 | **88%** |
| `out_proj`/`o_proj` | 2048 | 32 | 2.0 | 75% |
| shared expert gate_up | 1024 | 16 | 1.0 | 61% |
| MoE router (fp16) | 256 | 4 | 0.25 | 60% |

That is monotone in block count and *not* monotone in K — `in_proj_qkv` (K=2048) and `c_attn`
(K=2048) sit at 88–91% while `out_proj` (K=4096) sits at 75% with a quarter of the blocks. The
routed-expert MoE GEMVs fit too once the expert dimension is counted: gate_up gets 8×16 = 128
effective blocks and reaches 88%.

If that holds, the sm_87 override — which was tuned on the MoE GEMVs, where the top-8 expert
dimension multiplies block count by 8 — is **over-tiling the dense small-N kernels**, and
`out_proj`/`o_proj` at 8.1% of the decode budget is its largest victim.

Two instruments were added to settle it:
- `bench_moe_kernel.py` gained a K-sweep at fixed N=2048 and an N-sweep at fixed K=2048 — the
  first varies reduction depth alone, the second varies block count alone — plus achieved-bandwidth
  reporting against the 156 GB/s wall.
- `MLC_GEMV_TSTR="TS,TR,TILE_S"` in `gemv.py` overrides the sm_87 tile, so several schedules can be
  timed at one fixed shape. Inert when unset. `"16,32,1"` reproduces the generic CUDA schedule
  exactly, which is the A/B that matters; `"32,16,2"` reproduces production and doubles as a check
  that the hook is live.

#### The premise is wrong: K=4096 is the *best* K, not a deficit

| K (N=2048 fixed) | ms | GB/s | | N (K=2048 fixed) | ms | GB/s |
|---:|---:|---:|---|---:|---:|---:|
| 512 | 0.008 | 75.5 | | 512 | 0.011 | 54.5 |
| 1024 | 0.011 | 106.6 | | 1024 | 0.011 | 106.4 |
| 2048 | 0.018 | 131.9 | | 2048 | 0.018 | 131.8 |
| **4096** | 0.033 | **141.2** | | 4096 | 0.034 | 139.1 |
| 8192 | 0.077 | 123.6 | | | | |

Efficiency rises with work per launch along **both** axes. §9's question assumed K=2048 was the
good case and K=4096 the shortfall; at fixed N the opposite is true. `attn_o_proj` at its exact
production shape (N=2048, K=4096) measures **141.5 GB/s in isolation against 117.4 GB/s in the
§4.6 trace**.

#### No schedule beats the shipped one — item 5 should not be built

Same shapes, tile varied (% of the 156 GB/s wall):

| `TS,TR,TILE_S` | K=4096 | K=2048 | K=512 |
|---|---:|---:|---:|
| **`32,16,2` (production)** | **90.5%** | 84.9% | 43.5% |
| `16,32,1` (generic CUDA) | 90.3% | 83.0% | 39.6% |
| `32,16,1` | 90.4% | **85.4%** | 43.5% |
| `16,16,2` | 84.3% | 83.6% | 38.5% |
| `32,32,2` | 81.6% | 75.5% | 39.6% |
| `64,8,2` | 48.8% | 84.4% | **46.9%** |

The sm_87 tile is at or within 0.5% of the best tested schedule at every shape. Nothing improves
K=4096; the only tile that helps K=512 (`64,8,2`, +3.4 pts) costs 42 points at K=4096. **There is
no schedule win available, so §5 option 3 / §9 item 5 joins item 3 as refuted by measurement — do
not build it.** This comparison is the robust part of the section: every schedule sees the same
weights and the same cache behaviour, so the caveat below does not touch it.

#### Why the absolute numbers here run above §4.6, and why that is not a contradiction

The microbench times one weight tensor repeatedly, so anything near the 4 MB L2 gets reuse
production never sees — production streams 40 *different* `o_proj` tensors per token. The
inflation should then scale inversely with footprint, and it does exactly:

| kernel | weights | isolated | §4.6 traced | gap |
|---|---:|---:|---:|---:|
| `lm_head` | 286 MB (70× L2) | 167.4 GB/s | 156.1 | +3.3% (≈ the nsys overhead §4.6 notes) |
| `in_proj_qkv` | 8.4 MB (2× L2) | 148.4 | 137.8 | +7.7% |
| `out_proj`/`o_proj` | 4.7 MB (1.2× L2) | 141.5 | 117.4 | **+20.6%** |

A kernel 70× over L2 agrees with the trace to within nsys' own overhead; the one sitting just above
L2 is inflated most. **Trust §4.6 for production percentages and this bench only for A/B at a fixed
shape.** It also supplies the missing explanation: `o_proj` runs at 75% in production not because
of its K and not because of its tile, but because it streams from DRAM with no reuse while the
kernels it is being compared against are either far larger (and equally starved) or amortised
across 8 experts. Fixing that would mean changing *when* the weights are read, not how the GEMV is
scheduled.

**Instrument caveat worth carrying forward:** `bench_moe_kernel.py` should rotate over several
weight buffers before its absolute numbers are quoted against a trace again.

### 16.4 Item 1 — the option list was missing an option

The decision needs `bench_moe_kernel.py`, but reading the block first turns up
something the §9 write-up does not mention: **a per-token gemv dispatch for small batches already
exists and ships**, at [qwen3_5_moe_model.py:141-153](python/mlc_llm/model/qwen3_5_moe/qwen3_5_moe_model.py#L141-L153).
For `1 < num_tokens <= 5` the MoE block splits the batch and routes each token through the b=1
`dequantize_gemv` path, added for spec-decode verify. Its comment carries a measurement §9 does not
cite: *"Bench at B=24 group_gemm showed 0.058 ms/row flat at small batch vs gemv 0.008 ms/row →
~7× speedup per-token."*

So the choice is not only (a) Relax `If`, (b) a second decode entry point, (c) interactive-only.
There is (d): **widen the existing literal-batch split**, which needs no new kernel and no runtime
dispatch — only a literal `num_tokens`, which a fixed `max_batch_size` lib already has.

#### Measured — and the "~6×" understates it by 8×

ms per call, 35B-A3B MoE shapes (Ne=256, top_k=8), B = tokens × top_k. **`MLC_MOE_GEMM_V2=1` set**
— without it the bench silently measures the v1 fallback, which is the §8 trap in bench form, and
the two kernels differ by 3.5× at B=8, so the flag changes the conclusion:

| path | gate_up | down | **total** | vs gemv |
|---|---:|---:|---:|---:|
| **`dequantize_gemv` (b=1, ships today)** | 0.064 | 0.036 | **0.100** | — |
| `group_gemm` **v1** @ B=8 | 1.006 | 0.951 | 1.957 | 19.6× |
| `group_gemm` **v2** @ B=8 | 3.572 | 1.540 | **5.112** | **51×** |
| `group_gemm` v2 @ B=16 | 3.583 | 1.549 | 5.132 | 51× |
| `group_gemm` v2 @ B=32 | 3.602 | 1.563 | 5.165 | 52× |
| `group_gemm` v2 @ B=64 | 3.657 | 1.592 | 5.249 | 52× |

The gemv leg is validated against production: 0.064 ms/call here vs §4.6's 2.650 ms over 40 calls
= 0.066 ms/call.

**v2's cost is flat in B** (+2.6% from B=8 to B=64) because it is a dispatch-table kernel sized for
the full expert set: 256 experts × (1024×2048/2 + scales) ≈ 302 MB, which at 156 GB/s is 1.94 ms —
so 3.57 ms is ~54% of wall *for reading every expert*. At b=1 that is 32× the weight traffic the
top-8 actually needs. v1's persistent loop skips empty experts and so scales with B instead
(1.006 → 3.906 across B=8→64) — better at tiny B, worse asymptotically.

#### Decision: keep the specialization; if multi-sequence decode is wanted, take option (d)

Per-token gemv split costs `b × 0.100 ms`; v2 costs a flat ~5.15 ms:

| sequences b | per-token gemv split | `group_gemm` v2 | winner |
|---:|---:|---:|---|
| 1 | 0.100 | 5.112 | gemv **51×** |
| 2 | 0.200 | 5.132 | gemv 26× |
| 4 | 0.400 | 5.165 | gemv 13× |
| 8 | 0.800 | 5.249 | gemv **6.6×** |

Crossover is around **b ≈ 50 sequences**, which a 64 GB Orin running a 35B will not reach. So:

1. **§9 item 1's underlying question is settled: the b=1 MoE specialization is worth keeping**, and
   by a much wider margin than the source comment claims. The comment should be corrected from
   "~6×" to ~51× against v2 (~20× against v1) and dated.
2. **Option (a) — a Relax `If` for runtime dispatch — is the wrong shape of fix**, because there is
   no batch at which the dynamic path wins.
3. **If the 35B is to decode more than one sequence, option (d) is the cheap route**: widen the
   existing `1 < num_tokens <= 5` per-token split, which already ships for spec-decode verify. It
   duplicates weight traffic when two sequences share an expert, but gemv is already at 94.7% of
   wall, so the cost is exactly b× and nothing is lost to inefficiency.
4. The 35B stays single-sequence until someone wants (d); that is now a documented trade-off with
   numbers behind it rather than a pinned literal with a stale comment.

**Caveat:** the same L2-reuse limitation as §16.3 applies to absolute numbers, but the gemv leg
agrees with the trace to 3%, and a 51× ratio is not an L2 artifact.

### 16.5 Item 0c.1 built in TIR — and the probe's 2.79× does not survive the real geometry

`create_gated_delta_net_func_with_history_inplace_ksplit` now exists, behind
`MLC_QWEN35_GDN_KSPLIT=1|2|4|8` (compile-time, read during tracing, exactly like
`MLC_QWEN35_INPLACE_STATE`). `1` keeps §15's bit-exact kernel and is unchanged.

**The static profile reproduces the probe almost exactly**, which is the part of §16.2 that held up.
`ptxas -v` on the kernel TVM emits, `n_vh=32`:

| k_split | threads | floats/thread | registers | spill | barriers | regs/block | **blocks/SM, reg ceiling** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 128 | 255 | **192 B** | 0 | 32640 | **2** |
| 2 | 256 | 64 | 171 | 0 | 0 | 43776 | 1 |
| 4 | 512 | 32 | 117 | 0 | 0 | 59904 | 1 |
| 8 | 1024 | 16 | **62** | 0 | 0 | 63488 | 1 |

Zero spill from `k_split=2` up, and **zero barriers at every width** — the `tid = col*k_split + part`
layout keeps each reduction group inside one warp, so the butterfly is `log2(k_split)`
`__shfl_sync`es and no `__syncthreads` is emitted.

The last column is `65536 / regs_per_block` on sm_87's 65536 registers/SM — a **ceiling, not the
achieved value**, and the difference between the two is the whole story below: the *grid* can pin
blocks/SM lower than the registers allow. Achieved warps/SM, which is what matters:

| k_split | 0.8B (16 blocks / 16 SMs) | 35B (32 blocks / 16 SMs) |
|---:|---|---|
| 1 | 1 block/SM = **4 warps** (grid-bound; registers would allow 2) | 2 blocks/SM = **8 warps**, one wave |
| 2 | 1 block/SM = 8 warps | 1 block/SM = **8 warps, two waves** |
| 4 | 1 block/SM = 16 warps | 1 block/SM = 16 warps, two waves |
| 8 | 1 block/SM = 32 warps | 1 block/SM = 32 warps, two waves |

#### Measured on the kernels MLC actually compiles — and it is a lot less than 2.79×

[scripts/gdn_kernel_bench.py](scripts/gdn_kernel_bench.py), Orin, clocks already pinned
(`min_freq == max_freq == cur_freq == 1300.5 MHz`), batch=1, `max_history=64`, median of 3×20 calls.
ms per call:

| model (blocks) | seq_len | ks=1 | ks=2 | ks=4 | ks=8 | ks2 × | **ks4 ×** | ks8 × |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.8B** (16) | 128 | 1.0007 | 0.7437 | 0.5987 | 0.8441 | 1.35 | **1.67** | 1.19 |
| | 512 | 2.9242 | 2.0156 | 1.5035 | 1.6043 | 1.45 | **1.94** | 1.82 |
| | 2048 | 10.5991 | 7.1061 | 5.1574 | 4.6317 | 1.49 | 2.06 | **2.29** |
| **35B** (32) | 128 | 1.4310 | 1.4683 | 1.2366 | 1.6673 | **0.97** | 1.16 | 0.86 |
| | 512 | 3.6661 | 3.8301 | 3.0405 | 3.1775 | **0.96** | **1.21** | 1.15 |
| | 2048 | 12.5709 | 13.2908 | 10.3254 | 9.2003 | **0.95** | 1.22 | **1.37** |

**The 0.8B gets 1.94× where the probe said 2.79×, and the 35B gets 1.21× with `k_split=2` an
outright regression.**

#### Why: the probe's grid is the 0.8B's, and the 35B does not share it

To be fair to §16.2: `gdn_recurrence_probe.cu` launches `dim3 grid(n_kh, batch)` with `n_kh = 16`,
and §15.6's 95.6 ms trace **is the 0.8B** (`n_vh = 16`), so the probe matched the kernel it was
diagnosing. The trap is that the real kernel binds `blockIdx.x` to **`num_value_heads`**, not
`n_kh` — and on the 35B `n_vh = 32`, so it launches **32 blocks, not 16**. §16.2's ratio was never
measured at that geometry, and it does not transfer:

- **0.8B, 16 blocks on 16 SMs.** `k_split=1` gets 1 block/SM = **4 warps** however many registers
  ptxas leaves free — this is §16.2's grid-bound case, and it is real. Each doubling of `k_split`
  doubles warps/SM (4 → 8 → 16 → 32), and the measurements rise monotonically with it.
- **35B, 32 blocks on 16 SMs.** `k_split=1` fits **2 blocks/SM** — so the baseline already runs at
  **8 warps/SM**, twice what the probe measured, and the whole grid is resident in one wave. Every
  split width drops to 1 block/SM, so `k_split=2` buys **the same 8 warps/SM in two waves instead of
  one**, and pays for a shuffle per dot on top. That is the 0.95–0.97× regression, and it is a
  mechanism rather than noise: it is a strict loss on both axes.

`k_split=4` is the first width that actually raises the 35B's occupancy (16 warps vs 8), which is why
it is also the first width that wins there. And it **refutes §16.2's ranking of the two causes** on
that model: §16.2 concluded "most of the win is getting `state_local` off the register ceiling, not
occupancy", from the 16-block case. At 32 blocks the spill is *equally* gone at `k_split=2` and the
kernel is nonetheless **slower** — so where the grid already supplies 2 blocks/SM, occupancy is the
term that decides and the spill fix alone buys nothing. The causal story is right; its weighting is
geometry-dependent, which §16.2 had no way to see from one grid.

#### The 0.8B's own shortfall: the probe's *baseline* was 35% too slow, and the flush is secondary

`--max-history 1` reduces the flush from 63 positions to 1 while changing nothing else, which
isolates it. At seq_len=512:

| | ks=1 | ks=4 | ratio |
|---|---:|---:|---:|
| 0.8B, `max_history=64` | 2.9270 | 1.5185 | 1.93× |
| 0.8B, `max_history=1` | 2.5904 | 1.2296 | **2.11×** |
| 35B, `max_history=64` | 3.6959 | 3.0443 | 1.21× |
| 35B, `max_history=1` | 3.0336 | 2.4488 | **1.24×** |

The flush costs 0.34 ms on the 0.8B and 0.66 ms on the 35B — a clean 2×, matching its 2× state
volume, which is a good internal consistency check on the measurement.

**But it explains only about half the gap, and a prediction made before running this was refuted.**
For the probe's 2.79× to be right about the pure recurrence, the flush would have to be ~0.71 ms of
the 2.92; it is 0.34. Removing it entirely still only reaches 2.11×.

**What actually over-promised was the probe's baseline.** Comparing like with like — both flush-free:

| | probe (hand-written) | TIR | |
|---|---:|---:|---|
| `base` / `k_split=1` | 3.4900 | **2.5904** | TIR is **26% faster** |
| `ksplit4` / `k_split=4` | 1.2494 | 1.2296 | **1.6% apart — effectively identical** |

So the probe's *variant* was faithful; its reproduction of the *shipped* kernel was 35% slower than
what TVM actually emits. **The lesson generalises past this kernel: in a probe that A/Bs a
hand-written reproduction against a hand-written variant, the reproduction is the more dangerous
half, because getting it wrong inflates the ratio in the flattering direction and nothing in the
probe catches it.** Reproduce the baseline against the real kernel's own measured time before
trusting a ratio built on it — `gdn_func_history_inplace` was traced at 2.65 ms/call in §15.2 and
the probe's `base` read 3.49, which was visible without building anything.

#### End to end: 0.8B pp512 +25.0%, decode neutral, 108% of the traced estimate

`lib_gdnhist.so` vs `lib_ksplit4.so`, both benched in one session under `radix`, clocks pinned,
3 runs + warmup, `--unique-prompts` on by default at `radix`. Both were compiled with
`MLC_QWEN35_GDN_KSPLIT` set explicitly, so `lib_ksplit4.so` is the same configuration a default
build now produces:

| | pp512 tps | ttft | tg512 | tg1024 |
|---|---:|---:|---:|---:|
| `lib_gdnhist` (k_split=1) | 3911.80 | 130.9 ms | 89.20 | 90.58 |
| `lib_ksplit4` | **4888.03** | **105.8 ms** | 89.22 | 90.62 |
| | **+25.0%** | −25.1 ms | +0.02% | +0.04% |

**Decode is neutral to 0.04%**, which is the regression check that matters: the change touches only
`forward_with_history`, so `gdn_func_inplace` and the fused-prefill path must not move.

Scoring the prediction, per §15.2's rule — predicted from a measured kernel at the target shape,
renormalized against a trace of the actual A/B baseline:

- **Absolute, from the trace:** §15.2's `gdn_func_history_inplace` = 2654.7 µs × 36 calls over
  2 prefills × 18 GDN layers = **47.8 ms per pp512 prefill**, against a measured **130.9 ms** ttft —
  36.5% of ttft. (§15's 3934 tps is 130.1 ms, so the trace and today's baseline are the same
  configuration to 0.6%.)
- **Ratio, from the bench** at exactly that shape (0.8B, seq_len=512, `max_history=64`, batch=1):
  2.9242 → 1.5035 ms = 0.514. Applied to 47.8 ms: → 24.6 ms, a **23.2 ms** saving → predicted ttft
  107.7 ms, pp512 4754 (**+21.5%**).
- Measured: **25.1 ms** and **+25.0%**. **108% of the traced estimate** — the third in a row to land
  in band under this method (§14 was 107%).

**Taking the ratio from one instrument and the absolute from another is deliberate, and the two do
not agree on the absolute:** the bench reads 2.92 ms/call where the trace read 2.65 ms — 10% high,
plausibly synthetic inputs and no L2 warmth from neighbouring kernels. That is exactly why §15.2's
rule says to renormalize against the real baseline's trace: had the bench's absolute been used
throughout, the prediction would have inherited that 10%.

#### End to end: 35B pp512 +2.3%, and that is the honest size of it there

Same harness, same session, `radix`, `MLC_MOE_GEMM_V2=1` on both compiles (verified after the fact:
`nm -D --defined-only` counts 4 `group_gemm_v2`/`moe_dispatch_tables` symbols in each, per §8's trap):

| | pp512 tps | ttft | tg512 | tg1024 |
|---|---:|---:|---:|---:|
| `lib_gdnhist` (k_split=1) | 627.91 | 814.7 ms | 59.94 | 59.80 |
| `lib_ksplit4` | **642.08** | **797.3 ms** | 60.00 | 59.77 |
| | **+2.26%** | −17.4 ms | +0.10% | −0.05% |

**+2.3%, not +25%** — and that is what the 32-block geometry predicts, so it is a confirmation rather
than a disappointment. Checking it against the kernel bench, which is now the *only* instrument used
(no 35B trace of this pair exists): 30 GDN layers × 0.6256 ms saved per call at seq_len=512 =
**18.8 ms** predicted against **17.4 ms** measured, i.e. **93%**. Correcting for the bench's ~10%
optimism on absolutes (measured on the 0.8B above) gives ~17.1 ms and **102%**. Either reading lands
in band, which is a second, independent validation of `gdn_kernel_bench.py` as an instrument.

**It still ships on the 35B**, because the flag is one knob for both models, the 35B moves the right
way, and setting it to 1 to protect a +2.3% would cost the 0.8B its +25.0%.

#### Correctness — the bar changed as §16.2 said it must, and the split kernel is *more* accurate

`gdn_kernel_check.py --k-split N` (18 shapes: both head configs × 9 lengths, three of them wrapping).
Bit-exactness against the copy path is off the table by construction; what replaces it:

| bar | k_split=1 | k_split=2 | k_split=4 |
|---|---|---|---|
| output vs copy path | **0** (exact) | ≤4.8e-06 rel | ≤3.4e-06 rel |
| ring content vs copy path | **0** (exact) | ≤4.9e-06 rel | ≤3.8e-06 rel |
| **output vs fp64 reference** | ≤4.05e-06 | ≤2.72e-06 | **≤2.00e-06** |
| **every other ring slot** | **0** | **0** | **0** |

`--max-history 1` — the `disable` ring, where the skip guard fires for exactly one position instead
of 64 — passes at `k_split=4` too, so the guard is not width-sensitive.

Two things worth keeping. First, **the split kernels are closer to fp64 than the unsplit one**
(2.00e-06 vs 4.05e-06 at the worst shape) — splitting a 128-term fp32 dot into four 32-term partials
is a shallower reduction tree, so it rounds *less*. Second, **"every other ring slot byte-clean" stays
exactly 0**, which is the check that catches a ring misindex; reduction order cannot move a write into
the wrong slot, so dropping the other two exact bars costs no coverage there.

#### The state gates on `lib_ksplit4.so`

The kernel gate adjudicates the kernel; these adjudicate the lib. Unless noted, they run against
`dist/qwen3_5-0.8B-q0f16_fused/lib_ksplit4.so` — the 0.8B is the only model with an fp16 reference
that a token-level gate can be trusted against (§16.1's tier table). The 35B rows use §16.1's fp8
high-margin reference, which is the tier-2 check built for exactly this kind of change:

| gate | result |
|---|---|
| **35B** `high_margin_gate --check`, **`radix`** | ✅ **139/139** wide-margin at τ=2.0, near-ties 3/5 — identical to §16.1's baseline count |
| **35B** `high_margin_gate --check`, **`disable`** | ✅ **139/139** — the same wide-margin count as `radix`; near-ties 2/5 vs 3/5, which is the informational band where quantization legitimately differs |
| 0.8B `high_margin_gate --check`, **`radix`** | ✅ **361/361** wide-margin at τ=2.0 (376/376 at τ=1, 267/267 at τ=4, 64/64 at τ=8) |
| 0.8B `high_margin_gate --check`, **`disable`** | ✅ **361/361** — identical to `radix`, position for position |
| 0.8B `high_margin_gate --negative-control stale1` | ✅ **FAILS 342/361**, near-ties 0/39 — the same figure §16.1 calibrated, so the pass above is not vacuous |
| `prefix_cache_roundtrip` (`radix`) | ✅ **4/4 checks, 5/5 prompts each** — cold, exact-match reuse, fork-from-base, and fork + `PopN` rollback |
| `batch_decode_parity`, **`disable`** | ✅ **6/6** serial vs concurrent |
| `batch_decode_parity`, **`radix`** | ✅ **6/6**, and the §12 concurrency win holds at **2.66×** for 6 × 40 tokens |

The `disable` rows are the regression check: `disable` never enters `forward_with_history`, so a
lane-split-only change must leave it untouched, and it does.

#### Two harness notes

**ptxas spends 30–42 s on each split kernel, against 2–4 s unsplit** — the cost of fitting the state
into registers with zero spill under a 512- or 1024-thread launch bound. It is paid once per build
(TVM emits one instance of the PrimFunc), but it made the 18-shape gate a 12-minute run, because
`run_shape` recompiled per `seq_len`. `seq_len` is a *runtime* dimension, so the emitted code is
identical across the sweep: `gdn_kernel_check.py` now memoizes on `(kind, n_kh, n_vh, k_split)` and
the gate is **51 s at `k_split=1`, ~120 s at 2 or 4**.

`Executable` has no `time_evaluator`; it is on `.mod` (`mod["main"](*args)` once to warm up, then
`mod.mod.time_evaluator("main", dev, ...)`). `bench_moe_kernel.py` reaches it as `vm.module`.

### 16.6 Item 0d — `v_block`, and the first change in this document with an exact bar again

`create_gated_delta_net_func_with_history_inplace_ksplit` now takes **`v_block`**, the number of
value columns a block owns, behind `MLC_QWEN35_GDN_VBLOCK` (`0` = one block per head = the §16.5
grid). The grid becomes `(n_vh × V/v_block, batch)` and the block `v_block × k_split` threads.

**Verdict: +15.6% on the 0.8B, −8% on the 35B. It stays an opt-in knob, default `0`.** Both of this
item's predictions were wrong, and so was the expectation that it would be refuted outright:

- ❌ The item as first filed claimed **occupancy**. Wrong: warps/SM is set by registers/thread, which
  is set by the state slice `K/k_split`, and carving `V` up touches neither.
- ❌ The corrected version claimed the value was **unlocking `k_split=8`** — decoupled, `v_block=16,
  k_split=8` is 128 threads at 62 registers = 32 warps/SM, which no `v_block=V` form can express.
  Also wrong: `k_split=8` stays **worse than `k_split=4`** on the 0.8B in every form, so the
  occupancy it unlocks is not wanted. That half of the doubt recorded when filing the item was right.
- ✅ What pays *on the 0.8B* is **block granularity at fixed occupancy**. `k_split=4, v_block=32` is
  128 threads × 4 blocks/SM and `k_split=4, v_block=0` is 512 threads × 1 block/SM — **16 warps/SM
  either way** — and the small-block form is **15.6% faster**. Likely mechanism: 16 warps inside one
  block march through the recurrence in near-lockstep and stall on the FMA chain together, while
  four independent blocks decorrelate, so one can issue while another stalls.

Measured, clean single-process run, `max_history=64`, batch=1, median of 3×20:

| model | seq_len | ks1 | ks4 | ks4/v16 | ks4/v32 | ks8 | ks8/v16 | ks4 × | **ks4/v32 ×** | ks8 × |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.8B** | 512 | 2.9289 | 1.5239 | 1.3178 | **1.3193** | 1.6022 | 1.5624 | 1.92 | **2.22** | 1.83 |
| | 2048 | 10.6473 | 5.1566 | 4.5020 | **4.4711** | 4.6321 | 4.5509 | 2.06 | **2.38** | 2.30 |
| **35B** | 512 | 3.6681 | **3.0370** | 3.8663 | 4.0052 | 3.1765 | 3.1543 | **1.21** | **0.92** | 1.15 |
| | 2048 | 12.6698 | 10.3082 | 13.5295 | 14.0491 | **9.2017** | 9.6262 | 1.23 | **0.90** | **1.38** |

**The 35B regresses 8–10%, and it is §16.5's trap again with the sign flipped.** The two models want
opposite grids: the 0.8B wants small blocks, the 35B wants large ones. The mechanism that fits is
**redundant broadcast traffic against available slack**. Carving `V` into 4 chunks makes every head's
`k`/`q`/`gate`/`beta` be read by 4 blocks instead of 1. The 0.8B is latency-starved — `k_split=1` sits
at 4 warps/SM — so it has bandwidth to spare and buys decorrelation with it. The 35B has twice the
value heads, twice the blocks, was never grid-starved (2 blocks/SM at `k_split=1`), and has
correspondingly less slack, so the extra traffic dominates whatever decorrelation it gains.

**So `MLC_QWEN35_GDN_VBLOCK` keeps its default of `0` (inert) and ships as an opt-in.** No lib
rebuild and no re-gate are required, because nothing about the default configuration changes. A
model-aware default keyed on `n_vh` was considered and rejected: it would be a rule fitted through
**two** points, which is precisely the extrapolation that made §16.2 over-promise. The 35B is the
primary target and `0` is right for it; `MLC_QWEN35_GDN_VBLOCK=32` is worth 15.6% on the 0.8B for
anyone iterating there.

Two smaller things the sweep settles: the 0.8B's benefit **saturates by `v_block=32`** (`16` and `32`
are within 0.1%), so there is no reason to go narrower; and on the 35B at `seq_len=2048`,
`k_split=8` at `v_block=0` remains the best configuration measured (**1.38×**), which §16.5 already
flagged and this run confirms at a second `v_block`.

**It is bit-exact, and the gate now proves that rather than asserting it.** Re-gridding changes no
reduction order, so `gdn_kernel_check.py --v-block N` adds a check the §16.5 bars cannot make:
`vb_exact`, a direct comparison of output *and* ring against `v_block=V` at the same `k_split`,
required to be **exactly 0**. At `k_split=8, v_block=16` it is `0.0e+00` on every shape, both head
configs, including the wrapping ones — so this is the first history-path change since §15 held to an
exact bar rather than a tolerance.

### 16.7 The 35B prefill trace — the recurrence is 11%, the MoE is 64%, and §9's priorities are wrong

Traced after §16.5/§16.6 (`lib_ksplit4.so`, `radix`, prompt-len 512, nsys `--cuda-graph-trace=node`).
The profiled pp512 prefill step is **754.0 ms, 99.1% kernel-busy — only 0.9% idle**, so there is no
launch-overhead story here at all:

| kernel | calls | ms | % of prefill |
|---|---:|---:|---:|
| `dequantize_group_gemm_v2` | 40 | **275.71** | **36.6%** |
| `dequantize_group_gemm_v21` | 40 | **120.21** | **15.9%** |
| `gdn_func_history_inplace_ksplit` | 30 | 83.88 | 11.1% |
| `fused_dequantize1_NT_matmul8_2` (GDN `in_proj_qkvzab`) | 30 | 46.10 | 6.1% |
| `fused_multiply9_sum3` (MoE combine) | 40 | 41.25 | 5.5% |
| `scatter_output` | 40 | 22.37 | 3.0% |
| `take` | 40 | 20.84 | 2.8% |
| `fused_dequantize2_NT_matmul9_2` | 40 | 19.09 | 2.5% |
| `conv1d_history_inplace` | 30 | 11.25 | 1.5% |

**Roll-ups: the MoE expert GEMM pair is 395.9 ms = 52.5% of prefill, and MoE machinery as a whole —
the pair plus combine, `scatter_output` and `take` — is ~480 ms = 64%.** The GDN recurrence that
§9 item 0c has called "the biggest prefill item, by 5×" through four sections is **83.9 ms, 11.1%,
and the third-largest item**.

**This retires item 0c.2's priority on the 35B.** §15.6's framing was measured on the 0.8B, where
the recurrence *was* dominant; §16.5 then cut it further. Amdahl on 11.1%: even a **perfect**
chunked reformulation — the full ~15× to matmul efficiency — is worth at most
`1/(1 − 0.111) = 1.125×`, **+12.5% prefill on the 35B**, for the largest and riskiest piece of work
in this document. (The 0.8B is a different story: it is dense, has no MoE, and its recurrence is
~23% of ttft after §16.5, so 0c.2 is still worth ~+27% there. But the 0.8B is the iteration vehicle,
not the target.)

**The real target is `dequantize_group_gemm_v2`, and §16.4 already described why it should be
beatable — it just drew the conclusion for the wrong phase.** §16.4 established that v2 is a
dispatch-table kernel *sized for the full 256-expert set*, so it reads ~302 MB per call and its cost
is **flat in batch** (+2.6% from B=8 to B=64). That made it 51× worse than the b=1 gemv at decode,
which is where §16.4 stopped. But prefill runs it at **B = 512 tokens × top-8 = 4096 rows**, and
flat-in-batch is exactly the property you want there — which is why it is the right *family* for
prefill even though it is wrong for decode. The question §16.4 never asked is whether this
*implementation* is efficient at B=4096, and the trace says probably not: 275.71 ms over 40 calls is
**6.9 ms/call against a ~1.94 ms weight-bandwidth floor** (302 MB at 156 GB/s), i.e. **~3.6× off the
roofline**, with the arithmetic (4096 × 2048 × 1024 × 2 = 17.2 GFLOP/call) needing only ~3.2 ms even
at fp32 CUDA-core peak and far less on tensor cores.

**Next measurement, before any building:** establish whether v2 at B=4096 is bandwidth-bound,
compute-bound or schedule-bound, and whether it uses tensor cores at all. `bench_moe_kernel.py`
already sweeps this kernel — it needs a large-B mode. ⚠️ Its L2-reuse instrument bug (§16.3) matters
much less at B=4096 than at B=8, but check it before quoting absolutes.

> ⚠️ **Harness failure worth recording, because it cost a discarded table.** A split kernel spends
> 30–42 s per config in ptxas, so a sweep looks idle for many minutes. A run was wrongly declared
> dead — **`ps -C python` does not find it**, because the process shows as `timeout NNNN python ...`
> and the executable-name match misses it — and a second sweep was started on top of the first, plus
> a third in the foreground. Three benchmark processes shared the GPU and the resulting timings were
> thrown away. Use `pgrep -af gdn_kernel_bench`, and treat "no output yet" as "still compiling"
> rather than "died". The correctness results in this section are unaffected: `vb_exact` compares
> tensor contents, which contention cannot perturb.

### 16.8 Item 0e measured — v2 is CTA-bound, and 27–50% of its CTAs are padding

§16.7 asked three questions before any building: at prefill's `B = 4096`, is
`dequantize_group_gemm_v2` bandwidth-, compute- or schedule-bound, and does it use tensor cores at
all. `bench_moe_kernel.py` gained a prefill-scale B sweep (`gate_up_b512` … `gate_up_b8192`, and the
same for `down`), roofline reporting against both walls, and a replay of v2's dispatch-table
construction so the CTA count is known for every point. **Answers: tensor cores yes, schedule-bound,
and the specific schedule fault is that 27–50% of the CTAs at the production shape do a full tile of
work and throw the result away.**

#### Tensor cores: yes, and they are nowhere near the limit

The emitted CUDA for `dequantize_group_gemm_v2_kernel` contains `nvcuda::wmma::mma_sync` over
`fragment<matrix_a/matrix_b/accumulator, 16,16,16, half>`, 256 threads per CTA — the hand-tensorize
in `_schedule_v2()` does what it says. So the question is settled, and settled in the direction that
*removes* a hypothesis: at B=4096 the kernel reaches **2.34 TFLOP/s, 5.5% of the ~42.6 TFLOP/s fp16
tensor ceiling**. It is not compute-bound. Nor is it bandwidth-bound: the compulsory traffic (all 256
experts' weights and scales, plus activations) is **44.6 GB/s = 28.6% of the 156 GB/s wall**.

Also worth naming: it runs at **43.9% of the 5.33 TFLOP/s *CUDA-core fp32* ceiling**. A wmma kernel
whose throughput is a large fraction of the non-tensor-core wall is telling you the tensor cores are
waiting on scalar work — here, the int4 dequant that feeds them.

#### The B sweep, and a two-parameter model that fits all of it

`MLC_MOE_GEMM_V2=1`, medians over `--repeats 3 --number 20`. `real`/`pad` are CTAs with a live
`te[bx]` and with the sentinel `te[bx] = -1`:

| shape | B | CTAs (real + pad) | median ms | µs/CTA |
|---|---:|---|---:|---:|
| `gate_up` N=1024 K=2048 | 8 | 2056 (64 + 1992) | 3.574 | 1.738 |
| | 512 | 2304 (2048 + 256) | 4.270 | 1.853 |
| | 2048 | 3072 (2048 + 1024) | 5.549 | 1.806 |
| | **4096** | **4096 (2048 + 2048)** | **7.338** | **1.791** |
| | 8192 | 6144 (4096 + 2048) | 11.024 | 1.794 |
| `down` N=2048 K=512 | 8 | 4112 (128 + 3984) | 1.541 | 0.375 |
| | 512 | 4608 (4096 + 512) | 1.861 | 0.404 |
| | 2048 | 6144 (4096 + 2048) | 2.424 | 0.395 |
| | **4096** | **8192 (4096 + 4096)** | **3.211** | **0.392** |
| | 8192 | 12288 (8192 + 4096) | 4.870 | 0.396 |

**µs/CTA is constant to ±3% while the useful work per call moves 512×** (0.03 → 34.4 GFLOP). Fitting
`t = n_real·c_real + n_pad·c_pad` by least squares over all 12 points (the ten above plus the two
random-routing points below) predicts every one of them to **≤1.6%, median 0.31%**:

| | c_real | c_pad | pad / real |
|---|---:|---:|---:|
| `gate_up` (K=2048) | 1.837 µs | 1.732 µs | **94.3%** |
| `down` (K=512) | 0.406 µs | 0.375 µs | **92.3%** |

Two things fall out. **A padding CTA costs 92–94% of a real one** — which is what the source says it
should: the sentinel path guards only the `X` reads (`m_offset + i < row_end`) and the final store,
while the `W_shared` dequant loop and the whole wmma reduction run unconditionally on `e_safe = 0`.
And **c_real scales with K, not with rows**: 4.52× for a 4× K ratio, while a CTA holding 2 live rows
costs what one holding 16 does. The per-CTA constant is the `K` loop; **which part of that loop it
is — the `BLK_N × K` dequant, the wmma reduction, or the shared traffic between them — is not
separated by these measurements**, and the `BLK_M` sweep below shows why the obvious reading is
wrong.

This also corrects §16.4's *reason*, while leaving its decision intact. v2's cost is flat from B=8 to
B=64 not because "the full 302 MB expert set is read every call" — at B=8 only 8 experts are touched
and the measured traffic is **2.7 GB/s, 1.7% of the wall** — but because the grid
`UPPER = (ceildiv(B, BLK_M) + Ne) · tiles_per_n` is **97% padding** at that batch and barely moves.
The b=1 conclusion (keep the gemv specialization, 51×) is unaffected; only the mechanism was wrong.

#### How much padding is real routing's problem, not the bench's

`spread=True` divides B exactly evenly, which at B=4096 over Ne=256 lands 16 rows on every expert —
one `BLK_M=16` tile each, zero rounding waste, and therefore the *maximum* padding share. That is a
flattering assumption, so the same shape was re-run under `spread="random"` (each row picks an expert
uniformly). Identical kernel, identical launch, only the indptr differs:

| routing | CTAs (real + pad) | pad share | median ms |
|---|---|---:|---:|
| `gate_up` even-spread | 4096 (2048 + 2048) | **50.0%** | 7.338 |
| `gate_up` uniform-random | 4096 (2992 + 1104) | **27.0%** | 7.393 |
| `down` even-spread | 8192 (4096 + 4096) | **50.0%** | 3.211 |
| `down` uniform-random | 8192 (5984 + 2208) | **27.0%** | 3.250 |

The total CTA count is fixed by B and Ne regardless of routing, and `c_pad ≈ c_real`, so **the
runtime barely moves (+0.7% / +1.2%) while the padding share halves.** That is the model's sharpest
prediction and it holds. It is also the L2 control §16.3 asked for: if the sweep's absolutes were an
addressing artifact, this would have moved them.

#### What a fix is worth — and the counterintuitive part

Giving the sentinel CTAs an early exit (guard the `W_shared` load and the compute block on
`e_v >= 0`, as the store already is) recovers `n_pad · c_pad`:

| routing | pair (v2 + v21) | after | kernel | **pp512 at 52.5% of prefill** |
|---|---:|---:|---:|---:|
| even-spread | 10.549 ms | 5.466 ms | 1.93× | **+33.9%** |
| uniform-random | 10.643 ms | 7.903 ms | 1.35× | **+15.6%** |

**The win grows as the router gets more balanced**, because a balanced router puts every expert's row
count near a clean multiple of `BLK_M` and pushes the padding share toward its 50% ceiling. Qwen3.5's
router is load-balance-trained, so production sits somewhere in **[27%, 50%]** and the honest range
is **+16% to +34% pp512**. Either end beats item 0c.2's +12.5% ceiling for a fraction of the risk,
and unlike 0c.2 this **changes no arithmetic** — the padding CTAs' results are already discarded by
the store predicate, so a correct early exit is bit-exact by construction.

#### Two things this did not establish

1. **Where the per-CTA 1.84 µs actually goes** — dequant ALU, shared-memory traffic, or global
   load latency. `ncu` is installed but **the GPU performance counters are not accessible to this
   user** (`ERROR: The user does not have permission to access NVIDIA GPU Performance Counters`), so
   the within-CTA breakdown is unmeasured. It does not block the early exit, but it is what decides
   whether the *remaining* 2.6× off the bandwidth roofline is worth chasing after it.
2. **The production routing histogram.** The [27%, 50%] range above is bracketed, not measured; a
   dump of a real prefill's indptr would collapse it to a number.

⚠️ **Absolutes: prefer the trace.** The bench reads 7.338 ms/call at B=4096 where §16.7's nsys trace
reads 6.9 ms — 6% apart, the usual gap between an isolated microbench and the same kernel inside a
CUDA graph. Every conclusion above is a **ratio** measured within one harness, which is why the gap
does not matter here; do not quote the bench's milliseconds as production numbers.

### 16.9 Item 0f, two attempts that failed, and what they rule out

Both cautions in item 0f's entry turned out to be load-bearing, and a third route that looked free
from §16.8's cost model is a 1.6–2.6× **regression**. Nothing here changes §16.8's measurements; it
narrows the fix that can act on them. Neither attempt is committed — `moe_matmul.py` carries only the
`MLC_MOE_GEMM_V2_BLKM` A/B knob, default `16`, which is the shipping value and therefore inert.

#### (a) The guard cannot go around the blocks — `compute_at` fails

Wrapping the four loop nests of `_gemm_v2_func` in `if e_v >= 0:` — the direct reading of item 0f —
parses and blockizes fine, and `_coop(x_shared)` survives it. The next primitive does not:

```
sch.compute_at(w_shared, k_o_o, preserve_unit_loops=True)
  -> tvm.error.InternalError: unordered_map::at
```

An `IfThenElse` between the sblock and its target loop breaks the scope bookkeeping `compute_at`
relies on. So the guard has to be applied to the *scheduled* function, not the source one — wrap the
CTA body after `_schedule_v2()` returns. That route was scoped and not attempted.

**The API archaeology is done, so start here.** The stmt/expr nodes are in **`tvm.tirx`**, not
`tvm.s_tir` (which holds only `Schedule`/`ScheduleState`/`TensorIntrin`/dlight) and not `tvm.tir`
(which does not exist in this fork). `tvm.tirx` has `PrimFunc`, `For`, `SeqStmt`, `IfThenElse`, and
`tvm.tirx.stmt_functor` provides **`ir_transform`**, `post_order_visit`, `pre_order_visit`,
`substitute` — enough for the wrap. **The one unknown left is `thread_extent` hoisting**: the thread
bindings end up inside the conditional, which is legal CUDA because the sentinel is CTA-uniform, but
is what TVM's lowering normally objects to. That is a build-and-see, not an argument.

#### (b) Widening `BLK_M` is a regression, and the reason is a loop order

§16.8 measured `cost = CTAs × c(K)` with `c` flat in how many rows a CTA holds. Read naively that
says: hold more rows per CTA, launch fewer, pay the same each. `MLC_MOE_GEMM_V2_BLKM`, at
`gate_up` B=4096 under uniform-random routing:

| BLK_M | CTAs (real + pad) | median ms | µs/CTA | vs BLK_M=16 |
|---:|---|---:|---:|---:|
| **16** | 4096 (2992 + 1104) | **7.393** | 1.805 | — |
| 32 | 3072 (2048 + 1024) | 11.588 | 3.772 | **0.64×** |
| 64 | 2560 (2048 + 512) | 19.191 | 7.497 | **0.39×** |

**µs/CTA is proportional to BLK_M** (2.09×, 1.99×), so the CTA count falls and the per-CTA cost rises
by more. The cause is in `_schedule_v2()`: `sch.reorder(i_o, j_o, k_o_o, k_o_i)` puts `i_o` — the
`BLK_M / MICRO` loop — **outermost**, and both cooperative loads are `compute_at(k_o_o)`, i.e. inside
it. So the entire K loop, dequant included, re-runs once per `i_o`. Widening `BLK_M` multiplies the
weight dequant instead of amortizing it.

This is also the measurement §16.8 could not make. `c` scaling with `BLK_M` means the per-CTA
constant is **not** the `BLK_N × K` dequant alone — that term is independent of `BLK_M` and would
have stayed flat. It is the whole k-loop body, repeated `i_o` times. §16.8's attribution of the
constant to the dequant has been corrected there.

**So BLK_M widening is not dead — it is blocked on the loop order.** Hoisting the shared loads above
`i_o` (or sinking `i_o` inside `k_o_o`) makes one dequant serve `BLK_M / 16` accumulator fragments,
which is both the fix for this regression and the standard register-blocking that would lift the
5.5%-of-tensor-ceiling number. It is a larger change than 0f's early exit and should follow it.

#### What is still true, and what to do next

The early exit remains the best-understood lever: **bit-exact, 27–50% of CTAs, +16% to +34% pp512**.
It now needs the post-schedule wrap in (a) rather than the source-level guard. If that proves
expensive, the fallback with the same payoff and no conditional is to stop *launching* the padding
CTAs — compact the dispatch table with an exclusive scan over `ceildiv(count_e, BLK_M)` — which needs
a data-dependent grid extent and so a host round-trip per call, and should be costed before it is
built.

⚠️ **One instrument fix went in with this.** `v2_grid()` in `bench_moe_kernel.py` mirrored `BLK_M` as
a literal `16`, so the moment the A/B knob existed it reported CTA counts for a grid the kernel was
not launching (4096 CTAs at every BLK_M). It now reads the same env var the kernel does. The first
BLK_M table produced this way was wrong in its CTA and µs/CTA columns and was re-measured; the
medians were unaffected.

### 16.10 Item 0f built — the guard is a loop extent, not a branch, and it is bit-exact

The early exit exists, behind `MLC_MOE_GEMM_V2_SKIPPAD=1`, **default `0` pending the end-to-end
numbers**. It is bit-exact on every shape and routing tested and worth **1.28×–1.73× on the kernel**.

#### Two refusals, and the formulation that gets past both

§16.9 recorded the first: a source-level `if e_v >= 0:` dies in `sch.compute_at`. Applying the guard
*after* `_schedule_v2()` — the fix §16.9 recommended — gets further and then hits a second, more
interesting refusal. The mutation applies cleanly, lowering starts, and:

```
src/s_tir/transform/thread_storage_sync.cc:119
  Check failed: condition_counter() == 0 (1 vs. 0) : Cannot insert syncs inside condition
```

The cooperative loads carry `__syncthreads()`, and `ThreadSync` will not place a barrier inside an
`IfThenElse` — a correct rule in general, since a divergent barrier hangs, and unnecessary here
because `te[bx]` is CTA-uniform. The pass has no way to know that, and there is no annotation for it.

**So the guard is not a conditional at all.** The `k_o_o` reduction loop's extent is rewritten from
`K / BLK_K` to `Select(e_v >= 0, K / BLK_K, 0)`. A padding CTA runs it zero times. There is no
`IfThenElse`, so `ThreadSync` is satisfied; the extent is CTA-uniform, so every thread agrees on the
trip count and the barriers inside are either all taken or all skipped. The rewrite is ~40 lines in
`_guard_padding_ctas()`, and it asserts that exactly one serial loop of that extent exists — if the
schedule ever grows a second, it fails loudly instead of silently halving the win.

Node classes are `tvm.tirx` (§16.9); `tvm.tirx.stmt_functor.ir_transform` plus `post_order_visit` to
pick up the `e_v` Bind is the whole API surface needed.

#### Bit-exact, on the bar §16.6 set

`scripts/moe_gemm_check.py` builds the kernel twice in one process and requires `np.array_equal` —
**not** a tolerance. The claim is that padding CTAs never produced anything observable, so anything
but exact equality means a live tile was skipped.

| shape | B | routing | exact | before → after | |
|---|---:|---|---|---:|---:|
| `gate_up` | 8 | even | ✅ | — | |
| `gate_up` | 4096 | even | ✅ | 7.341 → 4.234 ms | **1.73×** |
| `gate_up` | 4096 | random | ✅ | 7.382 → 5.516 ms | **1.34×** |
| `gate_up` | 777 | random | ✅ | — | |
| `down` | 8 | even | ✅ | — | |
| `down` | 4096 | even | ✅ | 3.210 → 1.924 ms | **1.67×** |
| `down` | 4096 | random | ✅ | 3.245 → 2.530 ms | **1.28×** |
| `down` | 777 | random | ✅ | — | |

`B=777` is in there because it is a multiple of neither `BLK_M` nor `Ne`: tiles are partially filled
and the store predicate is doing real work on most experts' last tile. That is the case a wrong guard
would break, and it is exact.

#### A skipped CTA costs 20% of a full one, not 0

§16.8 predicted 1.93× / 1.35× for a *free* exit. The random-routing results land almost exactly there
(1.34 measured vs 1.35 predicted; 1.28 vs 1.34) but the even-spread ones fall short (1.73 vs 1.93).
Solving the two-cost model on the new points explains it — one consistent residual across both
shapes:

| | c_real after | c_skip | c_skip / c_pad before |
|---|---:|---:|---:|
| `gate_up` | 1.713 µs | 0.355 µs | **20.5%** |
| `down` | 0.395 µs | 0.074 µs | **19.8%** |

A skipped CTA still fills its accumulator, still stores it to `O_tile`, and still runs the
predicated-off `O_tile → global` store loop; only the k-loop disappears. 80% of the padding cost is
gone, not 100%, and the shortfall shows up most where padding is most of the grid. **Recovering the
last 20% means zeroing those trailing loops too**, which is the same trick on loops of extent 1 and 2
— extents that are not unique in this kernel, so it needs a targeted match rather than the
extent-equality search used here. Worth ~7% more on the pair; not done.

#### End-to-end: estimated, not yet measured

Pair at B=4096: **10.63 → 8.05 ms (1.32×) under uniform-random routing**, 10.55 → 6.16 ms (1.71×)
under even spread. Amdahl at §16.7's 52.5% of pp512 gives **+14.6% to +28.0% prefill**.

⚠️ **Both of those are projections from a microbench, and the default stays `0` until pp512 is
actually measured.** That is the whole failure mode §9's traps list is about. The remaining work is:
rebuild the 35B lib with `MLC_MOE_GEMM_V2_SKIPPAD=1`, A/B pp512, and run the state gate — which
**cannot** change (the kernel is bit-exact, so the gate is a regression check on the plumbing, not a
judgement on the arithmetic). Flip the default on those numbers, not on these.

### 16.11 Item 0f measured end-to-end — +19.4% pp512, and the default is flipped

`MLC_MOE_GEMM_V2_SKIPPAD` now defaults to **`1`**. Lib: `lib_skippad.so`, compiled with
`MLC_MOE_GEMM_V2=1` (4 v2 symbols confirmed by `nm`), `radix`, prompt-len 512, 3 runs after a warmup.

| | `lib_ksplit4` (baseline) | `lib_skippad` | |
|---|---:|---:|---:|
| **pp512** | 644.26 tps | **769.18 tps** | **+19.4%** |
| ttft | 794.7 ms | **665.6 ms** | −16.2% |
| tg512 | 60.16 tps | 60.09 tps | −0.1% (noise) |

Run-to-run spread was 0.05 tps on the new lib (769.15 / 769.18 / 769.20), so the +19.4% is not a
sampling artifact. **Decode is untouched**, as expected — at b=1 the MoE goes through the gemv path
(§16.4), not this kernel, so the only thing that could have moved is prefill.

#### The gate: identical to the baseline, which is a stronger statement than "passes"

`high_margin_gate.py` against `high_margin_ref_35b_fp8.json`, both prefix-cache modes:

| mode | τ=1.0 | τ=2.0 | τ=4.0 | τ=8.0 | near-ties | |
|---|---|---|---|---|---|---|
| `radix` | 142 / 0 (100%) | **139/139** | 99 / 0 | 17 / 0 | 3/5 | PASS |
| `disable` | 142 / **1** (99.30%) | **139/139** | 99 / 0 | 17 / 0 | 2/5 | PASS |

`disable` has one mismatch at τ=1.0 that `radix` does not, and **passing was not enough to accept
that** — a bit-exact kernel should reproduce the baseline's mismatch counts exactly, so an extra one
would have contradicted the whole premise. Re-ran the baseline `lib_ksplit4` through the identical
`disable` check as a control:

```
CONTROL lib_ksplit4 disable:  1.0 → 142 / 1 / 99.30%   2.0 → 139/139   near-ties 2/5
```

**Every column matches, including the near-tie count.** The τ=1.0 mismatch is pre-existing and
belongs to q4 quantization vs the fp8 reference, not to this change. Bit-exactness now has end-to-end
evidence, not just the microbench's.

#### The end-to-end number back-fills §16.8's open question

§16.10 projected **+14.6% to +28.0%**, the range being how balanced the production router is.
Measured **+19.4%** sits inside it, and inverting the arithmetic turns the projection into a
measurement of the thing that was unknown:

- Amdahl at §16.7's 52.5% ⇒ the kernel pair got **1.448×** in production
  (microbench brackets: 1.32× at 27% padding, 1.71× at 50%)
- feeding that back through §16.10's `c_pad/c_real = 0.933` and `c_skip/c_pad = 0.201` ⇒
  **~40% of v2's CTAs at production prefill are padding**

That is between uniform-random's 27% and a perfectly balanced router's 50%, which is exactly where a
load-balance-trained router should sit. ⚠️ **It is inferred, not observed** — it leans on §16.7's
52.5% and on the microbench constants. A direct indptr histogram would still be worth having, and it
is now the cheap way to check this inference rather than to size the work.

#### Where the 35B stands

| | before today | now |
|---|---:|---:|
| **pp512** (`radix`) | 642 | **769** |
| decode tg512 | 60.00 | 60.09 |

Prefill is now **2.17×** what it was at the start of 2026-07-25c (355). The remaining known headroom
in this kernel is the ~20% of a full CTA that a skipped one still costs (§16.10) — worth ~7% more on
the pair, ~3% end-to-end — and then the register-blocking that §16.9's `BLK_M` regression pointed at,
which is the larger prize and still unquantified.

---

## 17. Session 2026-07-26d — the win was in the k-step, and the benchmark was picking winners

§16.11 left three uncosted candidates and no queued item. All are now settled. **Neither of the two
ranked highest was the win**: the +14.1% came from `BLK_K`, a parameter nobody had swept. Then
§17.7–§17.10 went further and produced two corrections to this very section — read them before
acting on §17.1, and read **§17.9 before trusting any pp number in this document**.

| | `lib_skippad` (§16.11) | **`lib_blkk64`** | |
|---|---:|---:|---:|
| **pp512** (`radix`) | 767.20 tps | **875.37 tps** | **+14.1%** |
| ttft | 667.6 ms | **584.9 ms** | −12.4% |
| tg512 | 59.92 tps | 59.97 tps | +0.1% (noise) |

Run-to-run spread on the new lib was 0.13% (874.50 / 875.63 / 875.37), so the gain is not sampling.
Decode is untouched, as expected — at b=1 the MoE goes through the gemv path (§16.4), not this kernel.

⚠️ **These are filler-prompt numbers.** §17.9 found the bench prompt is one sentence repeated (11
distinct tokens per 512), which on a MoE concentrates routing. On real prose the same two libs read
**838.06** for `lib_blkk64`; the +14.1% itself is routing-independent and stands, but the absolute
figures are ~4% optimistic.

**How this section reads end to end, since it corrects itself twice:**

| | claim | status |
|---|---|---|
| §17.1 | `BLK_M` refuted at every shape | **partly retracted by §17.8** — right for pp512, wrong about the hoist |
| §17.2 | whole-body padding guard | landed, small (3–7% kernel) |
| §17.3 | `BLK_K` 32 → 64 | **landed, +14.1% pp512** — the session's win |
| §17.7 | real CTAs at 85–87% of the wall | on balanced routing this kernel is done |
| §17.8 | retraction + cost model | predicted 0h shape-split |
| §17.9 | the bench prompt picks winners | **the largest finding; changes how to bench → filed as item 0i, the next task** |
| §17.10 | 0h built and measured | works, but no compile-time `BLK_M` is Pareto — parked |

### 17.1 Candidate 1 (register-blocking `BLK_M`) — refuted, and not for the reason §16.9 gave

§16.9 diagnosed the `BLK_M` regression as `i_o` sitting outside the k-loop, so every extra
row-fragment re-runs the whole `BLK_N x K` dequant, and §16.11 promoted "hoist the shared loads above
`i_o`" to **"the largest remaining prize in the MoE"**. That diagnosis is correct and the conclusion
drawn from it is not, because **at the target shape there is no CTA count to save.**

pp512 is 512 tokens x top-8 = **4096 rows over 256 experts = exactly 16 rows per expert**, and
`BLK_M` is already 16. So `ceildiv(count_e, BLK_M)` is 1 at `BLK_M` = 16, 32 *and* 64: widening it
cuts the real CTA count from 256 to 256 to 256, while multiplying each CTA's dequant by 2x and 4x.
The hoist cannot help because there is nothing for it to amortize.

The obvious rejoinder is that the real prefill chunk is 2048 tokens (`prefill_chunk_size=2048`), i.e.
B=16384, where the count *does* halve. Measured there too — it still never wins:

| shape | routing | B | BLK_M=16 | BLK_M=32 | BLK_M=64 |
|---|---|---:|---:|---:|---:|
| gate_up | even | 4096 | 3.214 ms | 6.258 (0.51x) | 12.219 (0.26x) |
| gate_up | random | 4096 | 3.960 ms | 6.236 (0.63x) | 12.229 (0.32x) |
| gate_up | even | 16384 | 10.397 ms | 10.443 (**1.00x**) | 12.823 (0.81x) |
| gate_up | random | 16384 | 11.280 ms | 12.645 (0.89x) | 15.815 (0.71x) |
| down | even | 16384 | 5.248 ms | 5.208 (**1.01x**) | 5.272 (1.00x) |
| down | random | 16384 | 5.817 ms | 6.357 (0.92x) | 7.412 (0.78x) |

At B=16384 the halved CTA count is **exactly cancelled** by the doubled per-CTA cost — 1.00x and
1.01x on balanced routing, and a loss on ragged routing. `BLK_M` is bit-exact across all values (the
reduction over K is split identically), which is what made the sweep cheap; it stays at 16 and
`MLC_MOE_GEMM_V2_BLKM` stays a diagnostic.

> ⚠️ **This section originally concluded "so the hoist would buy the right to break even, at every
> shape this model actually runs." That inference is wrong and §17.8 retracts it.** The doubled
> per-CTA cost *is* the thing the hoist removes, so "halved count × doubled cost = 1.00x" says
> nothing about the hoisted kernel. The measurements above stand — they are all *un*-hoisted — but
> what they support is narrower than what was claimed. Read §17.8 before acting on this section.

### 17.2 Candidate 2 (the last 20% of a skipped CTA) — built, bit-exact, and much smaller than billed

§16.10 measured a k_o_o-skipped CTA at 20% of a full one and attributed the residue to the
accumulator fill, the accumulator -> `O_tile` store and the predicated-off global store, projecting
~7% more on the pair and ~3% end-to-end. It is built — and the attribution was wrong.

The mechanism generalises §16.10's rather than extending it. Instead of hunting the trailing store
loops by extent (they are extents 1 and 2, not unique, which is why §16.10 called it "a targeted
match"), the CTA body is wrapped in a unit loop annotated `moe_pad_guard`, and the *same*
`Select`-on-extent rewrite zeroes that. One match, and it skips **everything** rather than 80% of it.
The uniformity and bit-exactness arguments are unchanged — `e_v` is CTA-uniform, and nothing a
skipped CTA writes leaves shared memory.

It is worth 3–7% on the kernel, not the projected 7% on the pair:

| shape | routing | no guard | `koo` (§16.10) | **whole body** |
|---|---|---:|---:|---:|
| gate_up | even | 5.145 ms | 3.240 (1.59x) | **3.084 (1.67x)** |
| gate_up | random | 5.162 ms | 4.212 (1.23x) | **3.933 (1.31x)** |
| down | even | 2.691 ms | 1.533 (1.76x) | **1.481 (1.82x)** |
| down | random | 2.698 ms | 2.021 (1.33x) | **1.963 (1.37x)** |

Inverting the three-way for the per-CTA cost (using §16.10's `c_pad/c_real = 0.933`) says where the
20% actually went. After the k_o_o guard alone, `c_skip/c_real` is **4.8–7.4% on gate_up and
16.7–16.8% on down** — not a uniform 20% — and the whole-body guard moves it to 3.9–6.9% and
13.2–13.8%. It removes about **one sixth** of the residue. The rest is not the store tail at all: it
is CTA launch and dispatch-table read, which scales with nothing the kernel does, hence its being a
4% share of a big `gate_up` CTA and a 17% share of a small `down` one.

**The padding lane is now closed.** At the ~40% production padding §16.11 inferred, what remains of it
is ~4% of `gate_up` and ~9% of `down`, essentially all launch overhead, and the only way to remove
that is to stop launching those CTAs — which needs a host round-trip per call (§16.10). Not worth it.

### 17.3 What actually paid: `BLK_K` 32 -> 64, and it was never swept

`BLK_K` sets the k-step of the cooperative fetch, so it also sets **how many bytes of each `W` row
are pulled per step: `BLK_K` int4 values = `BLK_K/2` bytes.** At the original 32 that is **16 bytes —
half a 32-byte sector.** Every row-chunk read pulled a sector it half-used (the other half being
consumed on the *next* k iteration), and the k-loop paid a `__syncthreads()` pair per 16 bytes/row.
64 makes a row-chunk exactly one sector and halves the barrier count.

Bit-exact at every value, and 64 is the optimum on both shapes:

| shape | routing | BLK_K=32 | **BLK_K=64** | BLK_K=128 |
|---|---|---:|---:|---:|
| gate_up | even | 4.060 ms | **3.082 (1.32x)** | 4.461 (0.91x) |
| gate_up | random | 5.480 ms | **3.935 (1.39x)** | 5.917 (0.93x) |
| down | even | 1.917 ms | **1.501 (1.28x)** | 2.387 (0.80x) |
| down | random | 2.501 ms | **1.967 (1.27x)** | 3.084 (0.81x) |

128 regresses (0.80x–0.93x): shared memory grows linearly with `BLK_K` and occupancy falls off. The
default is now 64, with `MLC_MOE_GEMM_V2_BLKK` to A/B and an automatic halving fallback if `BLK_K`
does not divide `K` (the schedule splits `k` by `BLK_K // MICRO`).

**The prediction, made the way §5's estimation lesson prescribes.** Kernel pair at the target shape
(B=4096, both routings bracketing production): `lib_skippad`'s config 6.016/8.037 ms -> `lib_blkk64`'s
4.565/5.896 ms = **1.32x–1.36x**. Renormalising §16.7's 52.5% pair share through §16.11's +19.4%
puts the pair at **43.3%** of prefill now, so Amdahl predicts **+12.3%**. Measured **+14.1%** — two
points high, which back-solves to the pair being ~48% of prefill rather than 43.3%. Sixth consecutive
estimate to land using "measure at the target shape, renormalise against the actual A/B baseline".

Attribution between the two changes, from the same microbench: **`BLK_K` ~1.27x, whole-body guard
~1.05x**, compounding to ~1.34x.

### 17.4 A fragility fixed on the way

`BLK_K=64` initially failed to build the `down` shape: item 0f located the `k_o_o` loop by matching
`extent == K // BLK_K`, and at K=512 that extent is 8 — and so is another loop in the schedule. The
assertion §16.10 added caught it and refused to run rather than guessing, which is the only reason
this was a two-minute fix instead of a silent halving of the win.

Both guards are now located by **loop annotation** (`moe_koo_guard`, `moe_pad_guard`) applied in the
schedule, not by extent. The assertions remain, one per marker. Generalisable: *identify a loop by a
tag you attached, never by a property that happens to be unique at today's constants.*

### 17.5 One negative result worth not repeating

The generated CUDA shows each thread issuing **four identical `W_q[...]` loads** and unpacking only 4
of the 8 nibbles in a `uint32`, so thread pairs fetch the same word. Widening the W fetch to VEC=8
(one whole word per thread) is the obvious fix and **cannot be expressed**: the dequant's intermediate
is a `uint32` vector, and TVM stops at `Ramp of more than 4 lanes is not allowed` — a 128-bit ceiling.
The duplicate fetches share an address, so they cost L1 requests rather than DRAM traffic, which is
consistent with `BLK_K` (a sector-granularity fix) paying and this not being reachable.

### 17.6 Gates

| gate | result |
|---|---|
| `scripts/moe_gemm_check.py`, 8 cases, `np.array_equal` at BLK_K=64 | ✅ **8/8 bit-exact** |
| `scripts/moe_blkm_check.py`, BLK_M 16/32/64 x 2 shapes x 2 routings x B=4096/16384 | ✅ **exact at every value** |
| `scripts/moe_skippad_ab.py`, three-way 0/koo/1 | ✅ **all bit-exact vs no-guard** |
| 35B `high_margin_gate.py` vs fp8 ref, **radix** | ✅ 1.0 → 142/**0** (100%), 2.0 → **139/139**, near-ties 3/5 — **identical to `lib_skippad`** |
| 35B `high_margin_gate.py` vs fp8 ref, **disable** | ✅ 1.0 → 142/**1** (99.30%), 2.0 → **139/139**, near-ties 2/5 — **identical to `lib_skippad`** |

Both modes reproduce §16.11's columns exactly, including the pre-existing τ=1.0 `disable` mismatch
and both near-tie counts. As in §16.11, *identical* is the claim, not *passing* — a bit-exact kernel
that changed a mismatch count would have contradicted its own premise. The 0.8B is not rebuilt: it
has no MoE and never reaches this kernel.

### 17.7 The real CTAs, roofline'd — the question the handoff left

Every "% of wall" the workplan quotes for `dequantize_group_gemm_v2` comes from §16.8, which measured
it **before** the padding CTAs were skipped and **before** `BLK_K`. [scripts/moe_gemm_roofline.py](scripts/moe_gemm_roofline.py)
re-measures the CTAs that do real work, on their own.

It does not assume a padding cost. §16.10's `c_pad/c_real = 0.933` was measured on *unguarded*
padding CTAs and §17.2 showed it does not carry to guarded ones, so both coefficients are **fit by
least squares** over a sweep of (B, routing), which moves `n_real` and `n_pad` semi-independently.
The two-parameter model holds: residual median **1.0–2.9%**, max 12.4% (worst case `gate_up` at
BLK_K=64). `c_real · n_real` is then the real CTAs' time, and their obliged bytes are known exactly.

At B=4096 — pp512's 512 tokens x top-8 — against the 156 GB/s achievable wall:

| shape | routing | BLK_K=32 | **BLK_K=64** | % of the 42.6 TFLOP/s tensor ceiling |
|---|---|---:|---:|---:|
| gate_up | balanced | 57.0% | **85.5%** | 16.5% |
| down | balanced | 67.7% | **87.1%** | 15.9% |
| gate_up | ragged | 39.7% | 59.5% | 16.5% |
| down | ragged | 47.1% | 60.6% | 15.9% |

**Three things fall out.**

1. **`BLK_K` was a memory-efficiency fix and the roofline says so.** 57.0% → 85.5% and 67.7% → 87.1%
   is the whole of §17.3's win, arriving exactly where the half-sector diagnosis predicted it would.
2. **On balanced routing the real CTAs are done.** 85–87% against §5's tier-1 band of 88–100%. There
   is no bandwidth story left in this kernel at that routing, and compute was never close (16%).
3. **The ragged rows are 25 points lower on *identical* unique bytes.** Both routings hit all 256
   experts, so DRAM must supply the same 327.2 MB either way — but ragged routing needs 2944 CTAs
   instead of 2048, because `ceildiv(count_e, 16)` fragments. The extra CTAs re-read weights that L2
   can serve, so they cost time without costing DRAM traffic. **This is the remaining headroom, and
   it is tile fragmentation, not bandwidth.**

Note that "issued GB/s" is constant across routings by construction (issued bytes and time are both
proportional to `n_real` at fixed `BLK_M`), so it carries no information — only the *unique* column
does. Stated here because the printed table shows both.

### 17.8 Retraction: §17.1's conclusion about the hoist does not follow from §17.1's measurements

§17.1 measured `BLK_M > 16` losing at every shape and concluded that hoisting the shared loads above
`i_o` "would buy the right to break even, at every shape this model actually runs." **That is wrong,
and it is the same error this document has now made five sessions running: reasoning about a
configuration from a measurement taken under a different one.** Every `BLK_M` number in §17.1 is
un-hoisted, and the doubled per-CTA cost that cancels the halved CTA count *is precisely what the
hoist removes*. "1.00x" is what a cancelled win looks like, not evidence that the win is not there.

The right instrument is a cost model. §17.7 established that time tracks CTA count at fixed `BLK_M`;
extending that to *issued bytes per CTA* — with `W + Scale` issued `BLK_M/16` times when un-hoisted,
once when hoisted — reproduces §17.1's own measurements:

| | predicted | measured | err |
|---|---:|---:|---:|
| B=4096 even, BLK_M=32 | 2.00× slower | 1.96× | 2.0% |
| B=4096 even, BLK_M=64 | 4.00× slower | 3.85× | 4.0% |
| B=16384 even, BLK_M=32 | 1.00× slower | 1.00× | 0.0% |
| B=16384 even, BLK_M=64 | 1.00× slower | 1.23× | 19% |

Three of four within 4%. (The `BLK_M=64` outlier is shared memory: `X_tile` scales with `BLK_M`, and
at 64 the CTA needs ~45 KB of the 48 KB budget.) The **same model, with the hoist**, predicts:

| shape | BLK_M=32 + hoist |
|---|---:|
| B=4096 balanced — pp512's best case | **0.76× (a loss)** |
| B=4096 at ~40% padding — production pp512 (§16.11's inference) | **0.91× (a loss)** |
| B=4096 ragged | 1.09× |
| **B=16384 balanced — the 2048-token prefill chunk** | **1.51×** |
| **B=16384 ragged** | **1.37×** |

**So the corrected verdict is shape-split, not a refutation.** At pp512 — the headline benchmark —
`BLK_M` loses *even with the hoist*, because 16 rows/expert already fills a `BLK_M=16` tile and
widening it only inflates the X and O traffic. §17.1's conclusion is right for the number everyone
quotes. But `prefill_chunk_size` is **2048**, so every prompt longer than one chunk runs at B=16384,
where the model says the hoist is worth **~1.4–1.5× on this kernel** — and that is a workload the
pp512 benchmark cannot see at all.

**What this changes for the open list:** item 0h is **not refuted, it is shape-split** (entry
rewritten). It is also no longer a candidate for a *default* — `BLK_M` is a compile-time constant, so
taking the B=16384 win would mean regressing pp512 unless the kernel is specialised per chunk size.

> ✅ **Built and measured in §17.10.** The shape-split verdict holds and the direction predictions
> land within 8%. Two things this section did not anticipate: the crossover is at **pp ≈ 450**, not
> at the chunk boundary, so `BLK_M` loses on *short* prompts rather than merely failing to win at
> pp512; and the synthetic routings this section's model was calibrated on are themselves
> unrepresentative (§17.9), which is why the *end-to-end* numbers, not these, are the ones to quote.

**And the meta-lesson, now five for five.** §16.2's probe grid, §15.6's model, item 0d's occupancy
arithmetic, §9's priority order, and now this. Every one was an extrapolation across conditions, and
every one was cheap to check. The check that would have caught this one is two lines of arithmetic
over issued bytes — less work than the sweep that produced the wrong conclusion.

### 17.9 The bench prompt was choosing the winner — 11 distinct tokens per 512

Digging into item 0h surfaced a harness fault of the same family as §14.1, and it is worth
reading before any other MoE number in this document.

`scratch_mlc_tg_sweep.py`'s prompt is `PROMPT_FILLER = "The quick brown fox jumps over the lazy
dog. " * 200` — **one sentence repeated**. A 512-token prompt built from it contains **11 distinct
tokens (2.1%)**; 512 tokens of real prose contain **219 (42.8%)**, a 20× difference in diversity.

On a dense model that is harmless: prefill cost does not depend on *which* tokens arrive. **On a MoE
it decides the answer**, because the router keys on hidden states, so a low-diversity prompt
concentrates routing onto far fewer experts — and expert concentration is exactly what sets this
kernel's tile count (§17.7). Measured on the shipped `lib_blkk64`:

| | filler prompt | real prose | filler overstates by |
|---|---:|---:|---:|
| pp512 | 874.49 | **838.06** | **+4.3%** |
| pp2048 | 953.26 | **945.81** | +0.8% |

So **every pp512 figure in this document is a filler number and is ~4% optimistic.** They remain
comparable *to each other* — the bias is common to all of them — so no earlier conclusion about a
kernel change is overturned by this. But the absolute pp512 headline is not what a real request sees.

**It is worse than a scale factor for anything routing-dependent.** The first end-to-end A/B of item
0h, run on the filler, read **+5.5% at pp512**; the same A/B on prose reads **+1.8%**. The filler
inflated the gain 3×. And in the other direction, the microbench's synthetic routings (`even` and
uniform `random`) predicted 0h would *lose* 25% at pp512 — the wrong **sign**, because neither
synthetic routing resembles what a real router does.

`--prompt-file` is added to the harness for this, with [scripts/make_prose_corpus.py](scripts/make_prose_corpus.py)
to build the corpus reproducibly. It uses `zlib.crc32` rather than `hash()` to pick each run's window,
because `hash()` on `str` is salted per process and would have silently given two libs different
prompts — §14.1's bug class, one layer down.

> **Rule going forward: a MoE A/B whose mechanism touches tile counts, expert counts or routing must
> be run with `--prompt-file`.** The filler is fine for decode, for dense kernels, and for anything
> whose cost is routing-independent — `BLK_K` (§17.3) is in that category, which is why its win
> reproduced on both synthetic routings and does not need re-measuring.

**`--prompt-file` is a mitigation, not the fix.** One prose corpus is still one sample of a routing
distribution nobody has measured, and the microbench's synthetic routings remain wrong regardless of
what the end-to-end harness does. The fix is **item 0i** — dump the real expert histogram and drive
both instruments from it. It is the next task.

### 17.10 Item 0h measured end-to-end — the hoist works, and no compile-time `BLK_M` is Pareto

The hoist was built (`MLC_MOE_GEMM_V2_HOIST=1`): `sch.reorder(j_o, k_o_o, k_o_i, i_o)` puts the
row-fragment loop *inside* the k-loop that `_coop` attaches the shared loads to, so a CTA runs the
`BLK_N x K` weight dequant once instead of `BLK_M/MICRO` times. At `BLK_M=16` `i_o` has extent 1 and
the reorder is inert — measured **1.00× on all 8 microbench cells and bit-exact**, which is the
check that the reorder itself is sound rather than merely fast.

**Kernel level, all 48 configs bit-exact.** §17.8's predictions land:

| | predicted (§17.8) | measured |
|---|---:|---:|
| B=4096 even, M=32 | 0.76× | 0.79× |
| B=4096 ragged, M=32 | 1.09× | 1.01× |
| B=16384 even, M=32 | 1.51× | 1.44× |
| B=16384 ragged, M=32 | 1.37× | 1.32× |

`BLK_M=64` + hoist, which the model was not asked about, reaches **2.12× / 1.87×** at B=16384.

**End-to-end on real prose (§17.9), which is the measurement that decides it:**

| pp | `lib_blkk64` (M=16, shipped) | `lib_hoist32` (M=32+hoist) | `lib_hoist64` (M=64+hoist) |
|---:|---:|---:|---:|
| 128 | 552.08 | 521.85 (−5.5%) | 494.89 (**−10.4%**) |
| 256 | 707.45 | 688.79 (−2.6%) | 665.23 (−6.0%) |
| 512 | 838.06 | 848.73 (+1.3%) | 853.14 (+1.8%) |
| 2048 | 945.81 | 1019.31 (+7.8%) | 1064.55 (**+12.6%**) |

Decode neutral throughout (59.2–60.2). `lib_hoist64` passes the 35B state gate **identically to
`lib_blkk64` in both prefix-cache modes** — same τ columns, same near-tie counts — so the
bit-exactness claim holds end-to-end as well as in the microbench.

**Neither width is Pareto.** Both lose on short prompts and win on long ones, crossing over near
**pp ≈ 450**. `BLK_M=32` is not the safe middle it looked like — it is merely a smaller version of
the same trade (−5.5% short, +7.8% long). Widening `BLK_M` inflates the X and O traffic a tile must
move whether or not the rows are real, and at 128 tokens there are only 4 rows per expert to
amortize it over.

**Therefore: defaults are unchanged — `BLK_M=16`, `MLC_MOE_GEMM_V2_HOIST=0`.** The shipped kernel is
byte-identical to what §17.6 gated. Turning the hoist on by itself buys nothing at `BLK_M=16`, and
changing a gated kernel for no measured benefit is not worth the risk.

**What would make it shippable, now fully costed rather than guessed:** specialise on `B` at runtime
— emit both `(BLK_M=16)` and `(BLK_M=64)` dispatch+GEMM pairs and branch on `x.shape[0]` around
`B ≈ 3600` (pp450 × top-8). `LowBatchGemvSpecialize` is the in-tree precedent for a batch-conditioned
Relax branch. Worth **+12.6% on ≥2048-token prefill with no short-prompt cost**, on a model whose
context window is 262144 and whose `prefill_chunk_size` is 2048 — so long prompts run the winning
shape for all but their last chunk. The cost is two kernel variants in the binary (the dispatch table
is `BLK_M`-dependent, so it cannot be shared) and a branch on a symbolic shape.

---

## 18. Session 2026-07-26e — the routing nobody had measured, and what it overturns

Item 0i was the whole point of this session, and it did what §17.9 predicted it would: it moved
numbers. It also produced a **retraction of §17.7**, the section that declared the MoE GEMM finished.

| | claim | status |
|---|---|---|
| §18.1 | the real expert histogram | **measured** — 171/256 experts hit at pp512, busiest takes 396 of 4096 rows |
| §18.2 | both synthetic routings judged against it | wrong at B=4096, `random` is fine at B=16384 |
| §18.3 | item 0h re-measured on real routing | **wins at every measured shape**, up to 1.56× on the GEMM pair |
| §18.4 | **§17.7 retracted** | the kernel is at **45% of the wall**, not 85–87%. The lane is re-opened |
| §18.5 | item 0f re-measured | 1.35× on prose, and the filler overstated it |
| §18.6 | item 0j — dispatch tile order | ❌ **refuted**, 0.99–1.01× across 10 configs, bit-exact |
| §18.7 | item 0k — skip all-padding row fragments | built, bit-exact; **only pays at `BLK_M=32`** |
| §18.9 | the frontier, 4 libs × 3 lengths, one clock state | **monotone, nothing Pareto**; best pp2048 **1063 tps** |
| §18.10 | "defaults unchanged" checked by diffing generated CUDA | **was false when written**; now true and proved |
| §18.11 | where the lane stands | re-opened; next lever is **fuller tiles, not bigger** |
| §18.12 | VL re-gate | a **VL-only** `BLASDispatch` compile break; worked around |
| §18.13–14 | VL gate, margin-scored | ✅ **PASS** — the one divergence is a 0.05-nat near-tie |
| §18.15 | the 35B vs the original, measured | **2.29×** pp512 / **2.36×** pp2048 shipped; 2.66× best |

### 18.1 Item 0i — the real expert histogram

[scripts/moe_expert_histogram.py](scripts/moe_expert_histogram.py) puts a forward hook on each of the
40 text-stack routers under the existing fp8 HF path (§6.1) and bincounts the top-8 assignments. No
MLC instrumentation, no compile. Prompts come from the bench harness's own `build_prompt`, so the
routing measured is the routing the §17.10 pp numbers ran under.

Two implementation notes that cost time and are worth not repeating. The routers are at
**`model.layers.N.mlp.gate`** — *not* `language_model.layers...` as §9's item entry assumed; the
multimodal prefix does not appear when `AutoModelForCausalLM` unwraps to the text model, and
`model.config` is then a bare `Qwen3_5MoeTextConfig` with no `.text_config`. Both were discovered by
an `AttributeError` **after an 11-minute weight load**, which is why the script now has a `--dry-run`
that builds the model on the meta device and checks the layout in seconds. There is also no MTP
router in this instantiation at all (the excluded-group count came back empty), so the "exclude layer
40" warning in item 0i's entry was guarding against something that is not there.

**pp512 (B = 512 × top-8 = 4096), median over 40 layers, 3 corpus windows:**

| | experts hit | busiest expert | tiles @16 | @32 | @64 | padding @16 |
|---|---:|---:|---:|---:|---:|---:|
| **real prose** | **171** / 256 | **396 rows** | **362** | **246** | **191** | **29.3%** |
| synthetic `even` | 256 | 16 | 256 | 256 | 256 | 0% |
| synthetic `random` | 256 | 29 | 368 | 256 | 256 | 30.4% |
| bench filler | 137 | **512** (saturated) | 342 | 222 | 168 | 25.0% |

**pp2048 (B = 16384):**

| | experts hit | busiest expert | tiles @16 | @32 | @64 | padding @16 |
|---|---:|---:|---:|---:|---:|---:|
| **real prose** | **217** / 256 | 1660 rows | **1145** | **646** | **404** | **10.6%** |
| synthetic `random` | 256 | 88 | 1146 | 635 | 379 | 10.6% |
| synthetic `even` | 256 | 64 | 1024 | 512 | 256 | 0% |

**Three things fall out.**

1. **Real routing is heavily skewed.** The busiest expert takes 396 of 4096 rows — 9.7% of a
   256-expert layer's traffic, **25× uniform**. Neither synthetic routing has anything like it.
2. **§16.11's padding share was inferred and is too high.** It reasoned ~40% from an end-to-end
   ratio; measured is **29.3% at pp512 and 10.6% at pp2048** (range across layers 25.4–34.9% and
   9.7–11.5%).
3. **§17.9's filler bias now has a mechanism, and the arithmetic checks out.** The filler yields 342
   tiles against prose's 362 — **5.5% fewer** — and §17.9 measured pp512 reading **4.3% faster** on
   filler. The routing histogram predicts the tps bias almost exactly. Its busiest expert absorbs
   *all 512 tokens*; prose's takes 396.

**Layer depth barely matters, and that is the useful part.** Expert-hit count varies 50% across
depth (247 at layer 0, 158 by layer 36) but tile count does not: 349–380 at pp512, 1139–1152 at
pp2048. Concentrating rows onto fewer experts makes longer runs (fewer partial tiles) but leaves
fewer experts contributing one, and the two nearly cancel. **So a single representative routing is
legitimate for tile-count work** — the microbench does not need per-layer specialisation, it needed a
*real* distribution. Layer 0 is the one outlier: it routes near-uniformly (247 hit), as if the router
has not yet specialised on anything.

Corpus-window spread is small (176.6 / 185.4 / 178.4 experts hit at pp512), so the numbers are not an
artifact of one window — though they remain one corpus, and one corpus is not English.

**Decode is a different regime and is not this kernel's problem.** Eight decode steps touch 32
distinct experts (41 after a 2048-token prefill), so a decode *sequence* streams ~1/8 of the expert
weights. That is a weight-residency fact, not a tile-count one: at b=1 the MoE goes through the gemv
path (§16.4), not this GEMM.

### 18.2 What the two instruments were actually getting wrong

The failure is **shape-specific, not universal**, which is why it went unnoticed so long.

At **B=16384** synthetic `random` is a good proxy — 1146 tiles against the real 1145, and within 6%
even at `BLK_M=64`. That is why §17.10's pp2048 microbench prediction matched its end-to-end result.

At **B=4096** it is not. Real routing cuts tiles as `BLK_M` widens — 362 → 246 → 191 — while both
synthetics **saturate**: `even` sits flat at 256 and `random` goes 368 → 256 → 256. The synthetics are
blind to the `BLK_M=64` reduction entirely, and that missing term is precisely what made them predict
item 0h's **wrong sign** at the headline benchmark shape.

The mechanism is expert *count*, not raggedness. 4096 rows over 256 experts is 16 rows each — exactly
one `BLK_M=16` tile, nothing to save. 4096 rows over the **171** experts real routing hits is ~24
each, which spills into a second tile at width 16 and fits one at width 32. Uniform routings cannot
produce that because they hit every expert by construction.

### 18.3 Item 0h re-measured on real routing — it wins at every shape measured

All 24 configs bit-exact. `H=1` is the item-0h hoist.

| | shipped M=16 | M=32/H=1 | M=64/H=1 |
|---|---:|---:|---:|
| **pp512** `gate_up` | 3.771 ms | 3.614 (1.04×) | 3.647 (1.03×) |
| **pp512** `down` | 1.916 ms | 1.889 (1.01×) | 1.918 (1.00×) |
| **pp512 pair** | **5.687** | **5.503 (1.033×)** | 5.565 (1.022×) |
| **pp2048** `gate_up` | 11.235 ms | 8.511 (1.32×) | **7.065 (1.59×)** |
| **pp2048** `down` | 5.815 ms | 4.594 (1.27×) | **3.857 (1.51×)** |
| **pp2048 pair** | **17.050** | 13.105 (1.30×) | **10.922 (1.56×)** |

Without the hoist every width still loses (0.41×–0.90×), which is §17.1's measurement reproduced and
its diagnosis confirmed.

**The sign is now right.** The synthetic microbench predicted a **25% loss** at pp512; real routing
says a small win, and the end-to-end prose A/B in §17.10 measured +1.3% / +1.8%. Cross-checking:
1.033× on a GEMM pair that is 52.5% of prefill predicts **+1.7%** end-to-end against §17.10's measured
**+1.3%**. The instrument and the end-to-end number now agree, which is the thing item 0i was for.

At pp2048 the kernel gain (1.56×) is much larger than the end-to-end one (§17.10's +12.6%), and that
is expected rather than contradictory: the 52.5% MoE share was traced at pp512, and full attention
grows quadratically, so the MoE is a smaller slice of a 2048-token prefill.

### 18.4 Retraction: §17.7's "no bandwidth story left" was measured on a routing that never happens

§17.7 concluded the real CTAs sit at **85–87% of the 156 GB/s wall**, inside §5's tier-1 band, and
that "there is no bandwidth story left in this kernel." Both figures are `even`-routing figures.
Re-run with the measured `indptr` at B=4096:

| `gate_up`, B=4096 | n_real CTAs | unique MB | **% of the wall** | issued/unique |
|---|---:|---:|---:|---:|
| synthetic `even` (§17.7's number) | 2048 | 327.2 | **86.5%** | 1.36× |
| synthetic `random` | 2944 | 327.2 | 60.2% | 1.95× |
| **real prose (median layer)** | 2912 | 200.9 | **45.3%** | **2.96×** |
| real prose (min / max layer) | 2744 / 3144 | 200.9 / 287.0 | 39.7% / 49.5% | 2.96× / 2.38× |

`down` is the same story: 86.8% → **46.2%**.

**The mechanism is that real routing moves both terms the wrong way at once.** Only ~171 of 256
experts are hit, so unique DRAM traffic falls to 61% of what `even` requires — but fragmentation
*raises* tile count 42%, so the kernel issues **2.9× the bytes DRAM actually supplies** and L2 absorbs
the difference. §17.7 saw a 1.36× ratio and concluded the kernel was bandwidth-saturated. It is not:
it is at **~45% of the wall**, and the gap is tile fragmentation.

**So the MoE lane is re-opened, and §18.3 is the first evidence of what that is worth** — widening
`BLK_M` attacks exactly this term, which is why it buys 1.56× at pp2048 rather than the ~1.0× the
`even`-calibrated model expected.

The two-parameter fit still holds under real routing (`c_skip/c_real` = 34.0% for `gate_up`, 17.6%
for `down`; residual median 3.8%/1.2%).

**And the meta-lesson is now six for six.** §16.2's probe grid, §15.6's model, item 0d's occupancy
arithmetic, §9's priority order, §17.1's hoist conclusion, and now §17.7's roofline. Every one was a
number measured under conditions that did not hold where it was applied. §17.9 named the disease and
this section is another case of it — the roofline was *re-measured* in §17.7, carefully, on the wrong
routing.

### 18.5 Item 0f re-measured on real routing

The padding-CTA skip is worth **less** than the filler said and still a lot:

| B=4096 `gate_up` | unskipped | skipped | gain |
|---|---:|---:|---:|
| real prose | 5.090 ms | 3.757 | **1.35×** |
| bench filler | 5.041 ms | 3.477 | 1.45× |
| decode (B=64) | 2.500 ms | 0.437 | 5.73× |

Bit-exact throughout. The shipped default is unchanged and remains correct.

### 18.6 Item 0j — reorder the dispatch table's tiles for L2. Refuted.

§18.4's 2.9× issued/unique is L2 traffic, and how much L2 can absorb depends on the order CTAs visit
tiles. Within one expert's private CTA range the order is free:

* **m-major (shipped)** `off = tmi*tiles_per_n + tni` — consecutive CTAs share `X_tile` and sweep the
  expert's whole weight set, then sweep it again for the next row-tile. Each weight slice is
  re-fetched `nb` times at a reuse distance of the expert's entire footprint (1 MB for `gate_up`),
  against a 4 MB L2 with several experts in flight.
* **n-major** `off = tmi + tni*nb` — consecutive CTAs share the *weight* slice, cutting the reuse
  distance to one slice.

Both assign the identical set of `(e, m, n)` triples to disjoint outputs, so it is bit-exact by
construction — and it measured so, in all 10 configs.

**It buys nothing: 0.99×–1.01× across `BLK_M` ∈ {16,64} × hoist ∈ {0,1} × both shapes at B=16384,
and 0.99× at B=4096.** L2 is evidently already capturing the reuse the m-major order leaves on the
table, so the 2.9× amplification is not costing what its size suggests. Kept as
`MLC_MOE_GEMM_V2_TILEORDER=n`, default `m` (unchanged), as a documented dead end.

That is a useful negative: it means §18.4's headroom is **not** a cache-ordering problem, and the
tile-count lever (§18.3) is the one that works.

### 18.7 Item 0k — skip the row fragments that hold no real rows. Width-dependent, and it changes the ranking.

§18.4 says the kernel's problem is tile fragmentation, and §18.1 says why there is so much of it:
a real pp512 prefill puts **~24 rows on each hit expert**, so at `BLK_M=64` a tile carries 24 real
rows and 40 padding ones. `X_shared` is already predicated on `row_end`, so padding rows cost no DRAM
traffic — but the wmma reduction still runs all `BLK_M/MICRO` row fragments and the ones past the
real rows reduce nothing but zeros. **That is the mechanism behind §17.10's short-prompt loss**, and
removing it looked like it should make a wide tile safe everywhere.

Built as `MLC_MOE_GEMM_V2_SKIPROWS=1`: rewrite the row-fragment loop's *extent* to
`min(ceildiv(row_end - m_offset, MICRO), BLK_M/MICRO)`. Mechanism and justification are item 0f's,
one level down — an `if` is impossible because the cooperative loads carry `__syncthreads()` and
ThreadSync refuses a barrier inside a condition, the extent is CTA-uniform, and a skipped fragment's
global store is predicated off by `m_offset + i < row_end` regardless of what its accumulator holds.
**All 12 A/B cells bit-exact.**

Two implementation notes. The annotation lands on **two** loops after scheduling — the accumulator
init nest and the compute nest — because `blockize` leaves one and `reverse_compute_at` re-materialises
the other; both must be shortened and both are safe. And `m_offset`/`row_end` have to be found **by
name over the whole body**, not by walking the CTA block's leading let-chain: the schedule re-nests
those bindings, and the let-statement node is not exported under a stable name from `tvm.tirx`.

| H=1, real routing | pp512 (B=4096) | pp2048 (B=16384) |
|---|---:|---:|
| M=16 | 0.91× / 0.95× | 0.95× / 0.97× |
| **M=32** | **1.06× / 1.06×** | **1.03× / 1.01×** |
| M=64 | 0.91× / 0.91× | 0.91× / 0.92× |

(`gate_up` / `down`.)

**It only pays at `BLK_M=32`, and it costs at both 16 and 64.** At 16 the guard should be inert — the
loop has extent 1 — and instead it loses 5–9%, which is the tell: replacing a constant extent with a
runtime expression is not free, it blocks the unroll. At 64 the fragments skipped do not pay for that
same loss. At 32 they just do.

**Where that leaves the ranking.** Both surviving configs now beat the shipped `BLK_M=16` at *both*
measured shapes, which nothing did before this session:

| GEMM pair vs shipped M=16 | pp512 | pp2048 |
|---|---:|---:|
| **M=32 / hoist / skiprows** | **1.098×** | 1.323× |
| M=64 / hoist | 1.022× | **1.558×** |

Neither dominates the other, but the interesting one is `M=32+skiprows`: §17.10 measured plain
`M=32+hoist` at **−5.5% end-to-end on pp128**, and padding-row compute is exactly what a 128-token
prefill has most of. Whether the row skip removes that loss is not a kernel question — it needs the
end-to-end A/B, which is §18.8.

### 18.8 End-to-end, on prose, one session, one clock state

`lib_m32rows` = `BLK_M=32` + hoist + row-skip, compiled with
`MLC_MOE_GEMM_V2_BLKM=32 MLC_MOE_GEMM_V2_HOIST=1 MLC_MOE_GEMM_V2_SKIPROWS=1`. Benched against the
shipped `lib_blkk64` on `--prompt-file` prose, `--prefix-cache-mode radix`, 3 runs after 1 warmup.

⚠️ **`jetson_clocks` was not set for these runs** (no passwordless sudo in this session), so absolute
figures sit slightly under §17.10's. The A/B is unaffected — every leg here ran under the same clock
state, and `lib_blkk64` at pp2048 reproduces §17.10's number to 0.1% (945.00 vs 945.81), which is the
check that the two sessions are comparable at all.

| pp | `lib_blkk64` (shipped) | `lib_m32rows` | Δ |
|---:|---:|---:|---:|
| 128 | 576.33 | 564.16 | **−2.1%** |
| 512 | 827.39 | **850.39** | **+2.8%** |
| 2048 | 945.00 | **995.41** | **+5.3%** |

Decode neutral throughout (59.1–60.2 tps on both).

**The row skip does what §18.7 predicted at the short end and the opposite at the long end.** §17.10
measured plain `BLK_M=32`+hoist at −5.5% / +1.3% / +7.8%; adding the row skip moves that to
−2.1% / +2.8% / **+5.3%**. So it more than halves the short-prompt loss and nearly doubles the pp512
gain — and it gives back 2.5 points at pp2048.

That shape is consistent with the mechanism rather than with noise. The guard's cost (a runtime loop
extent, which §18.7 isolated at 5–9% where it skips nothing) is paid by **every** CTA; its benefit
accrues only to **partial** tiles. Padding share at `BLK_M=32` is 47.9% at pp512 and 20.8% at pp2048
(§18.1), so the trade gets worse exactly as the prompt gets longer. The row skip and `BLK_M` widening
pull in *opposite* directions with prompt length, which is why neither alone is Pareto.

### 18.9 The whole family, measured in one session — and it is monotone

§18.8's cross-session comparison against §17.10 was suggestive, not evidence, so all four libs were
re-benched back to back under one clock state. `lib_hoist32` / `lib_hoist64` are §17.10's legs
(`BLK_M` 32 / 64 + hoist, no row skip).

| pp | `lib_blkk64` (shipped) | `lib_m32rows` | `lib_hoist32` | `lib_hoist64` |
|---:|---:|---:|---:|---:|
| 128 | **576.33** | 564.16 (−2.1%) | 548.11 (−4.9%) | 524.67 (−9.0%) |
| 512 | 827.39 | **850.39 (+2.8%)** | 833.71 (+0.8%) | 826.86 (−0.1%) |
| 2048 | 945.00 | 995.41 (+5.3%) | 1020.29 (+8.0%) | **1063.48 (+12.5%)** |

Decode neutral on all four (59.1–60.2). §17.10's figures reproduce within 1.5 points at every cell
(−5.5/+1.3/+7.8 vs −4.9/+0.8/+8.0 for `hoist32`; −10.4/+1.8/+12.6 vs −9.0/−0.1/+12.5 for `hoist64`),
which is what makes the two sessions comparable.

**The family is monotone in exactly one parameter — how much padded row-space a CTA carries.** Read
left to right, every step trades short-prompt throughput for long-prompt throughput, with no
exception in 12 cells. `lib_m32rows` is a strictly milder version of `lib_hoist32`, not a different
trade: the row skip moves it *back toward* the shipped kernel at pp128 (−4.9% → −2.1%) and *also*
back at pp2048 (+8.0% → +5.3%). It buys its short-prompt safety with long-prompt gain, which is the
same currency `BLK_M` spends in the other direction.

**So the row skip does not rescue a wide tile, it interpolates.** §18.7 hoped it would remove the
short-prompt loss and make a wide `BLK_M` Pareto. It does not: it slides along the same frontier.
That frontier is real and nothing measured this session crosses it.

**In absolute latency the trade is more favourable than the percentages suggest**, because the
short-prompt loss is small in ms and the long-prompt gain is large:

| ttft, ms | shipped | `m32rows` | `hoist32` | `hoist64` |
|---:|---:|---:|---:|---:|
| pp128 | 222.1 | 226.9 (+4.8) | 233.5 (+11.4) | 244.0 (+21.9) |
| pp512 | 608.5 | 586.1 (−22.4) | 597.2 (−11.3) | 598.5 (−10.0) |
| pp2048 | 2167.2 | 2057.4 (−109.8) | 2007.3 (−159.9) | **1925.8 (−241.4)** |

Every non-shipped lib pays single-digit milliseconds at pp128 to save tens or hundreds at pp512 and
pp2048. On a model with a 262144-token context chunked at 2048, that is a trade worth making — but it
is a *workload* judgement, not a measurement, so **the defaults are left unchanged** (`BLK_M=16`,
`HOIST=0`, `SKIPROWS=0`, `TILEORDER=m`) and the four libs are kept as the evidence for making it
deliberately. §17.10 declined the same choice for the same reason; the difference is that the
frontier is now mapped rather than guessed at.

### 18.10 Gates, and a claim that turned out not to be true until it was fixed

| check | result |
|---|---|
| v2 GEMM bit-exactness across **`SKIPROWS` 0/1** × `BLK_M` 16/32/64 × 2 shapes × B 4096/16384, real routing (§18.7) | ✅ **12/12 exact** |
| v2 GEMM bit-exactness across **`TILEORDER` m/n** × `BLK_M` 16/64 × hoist 0/1 × 2 shapes (§18.6) | ✅ **10/10 exact** |
| item 0f's `SKIPPAD` gate re-run under `SKIPROWS=1`, `BLK_M=64` | ✅ 8/8 exact |
| **35B state gate, `lib_m32rows`, `radix`** | ✅ **139/139 wide-margin positions** (near-ties 3/5) |
| **35B state gate, `lib_m32rows`, `disable`** | ✅ **139/139 wide-margin positions** (near-ties 2/5) |
| **generated CUDA for the shipped default, before vs after this session** | ✅ **identical** — see below |

**The "defaults are unchanged" claim was false when first written, and diffing the generated
CUDA is what caught it.** [scripts/moe_dump_cuda.py](scripts/moe_dump_cuda.py) emits the device
source so the claim can be checked rather than argued. Two real findings:

1. **An unconditional `sch.annotate` is not free.** Tagging the row-fragment loop for item 0k — even
   with the guard disabled and nothing reading the annotation — stopped TVM eliminating the unit
   loop, so the *shipped* `BLK_M=16` build gained a `for (a0_0 = 0; a0_0 < 1; ++a0_0)` wrapped round
   the entire CTA body. nvcc would delete it and the timing delta was 0.3% (inside noise), which is
   exactly why timing could not have caught this. The annotate is now under `if SKIPROWS`.
2. **Item 0j's parameterised index left dead locals in the dispatch kernel.** Rewriting the tile
   offset as strides so one loop nest serves both orders emitted two unused `int`s per iteration.
   Rewritten as a second whole prim_func under `if TILE_N_MAJOR`, so the shipped nest is verbatim.

**Diffing generated code needs a control.** TVM's CSE numbering is **not stable run to run** — two
dumps of the *identical* source differ by ~14 lines of pure `cse_vN` renaming, which is *more* than
the 8-line delta the real comparison showed. Normalising `cse_v[0-9]+` and confirming the same-source
control diffs empty is what makes the result mean anything. Without that step the honest reading of
the raw diff would have been "it changed", and the honest reading after over-correcting would have
been "it didn't" — neither supported.

**Generalised: a claim of the form "this change is inert when disabled" is checkable, and cheap to
check.** Four sessions of this document have asserted some version of it from exactness gates plus
timing. Exactness gates compare *outputs* and cannot see a scheduling change; timing cannot resolve
sub-1% effects. The artifact to compare is the emitted code.

### 18.11 Where this leaves the MoE lane

**The lane is open again, and §18.4 is why.** §17.7 closed it on the strength of an 85–87% roofline
that turns out to be an artifact of `even` routing; the real figure is **45%**, and the gap is tile
fragmentation rather than bandwidth. Everything measured this session is consistent with that single
diagnosis:

* widening `BLK_M` (fewer, fuller tiles) **works** — up to 1.56× on the GEMM pair at pp2048 (§18.3);
* skipping empty row fragments **works where padding dominates** — +2.8% end-to-end at pp512 (§18.7);
* reordering tiles for L2 **does nothing** (§18.6), so the 2.9× issued/unique is not a locality
  problem the kernel can fix by scheduling.

**What is now known that was not:**

| | before §18 | after §18 |
|---|---|---|
| experts hit at pp512 | assumed 256 | **171** |
| padding share at `BLK_M=16` | inferred ~40% (§16.11) | **29.3%** measured (10.6% at pp2048) |
| % of the bandwidth wall | 85–87% (§17.7) | **45%** |
| best known pp2048 | 945 tps | **1063 tps** (`lib_hoist64`, +12.5%) |
| whether any `BLK_M` is Pareto | "no" (§17.10), from one A/B pair | **no**, from a mapped 4×3 frontier |

> ⚠️ **Retracted by §19.3.** Everything from here to the end of this section rests on "what stops a
> wide tile winning everywhere is padding-row compute". Item 0l built the fix this paragraph asks
> for, it works as a mechanism, and it recovered **2 points of a 31-point gap** — so the premise is
> wrong. Read §19.3 and item 0n before acting on any of it. The last paragraph, on mixed tiles,
> stands: it was never about padding-row compute.

**The next lever is to stop a wide tile paying for its own padding — filed as item 0l, and it is
the first MoE idea in three sessions that is not a point on the frontier §18.9 just mapped.**

The reasoning is short. Weight traffic is proportional to tile count, `BLK_M=64` cuts tiles 1.90× at
pp512 and 2.84× at pp2048, and that is exactly why it wins 1.56× at pp2048. What stops it winning
everywhere is padding-row compute — 24 real rows in a 64-row tile. Item 0k removed that compute and
*still* lost at `BLK_M=64` (0.91×), but §18.7 isolated why: at `BLK_M=16`, where 0k's guard is
logically inert, it **still costs 5–9%**. The loss is not the skipping, it is that a runtime loop
extent blocks the unroll — and 0k pays that on every CTA.

At `BLK_M=64` there are only **four** possible active-fragment counts. Specialising them statically
keeps the unroll and still skips the empty work. **The first thing to measure is the pp128 leg**,
because that is the cell that has killed every wide tile so far.

One idea recorded as probably dead so it is not re-derived: letting a single CTA cover rows from two
experts would eliminate the remainder entirely, and the dispatch table already carries a per-CTA row
offset — but **the weight tile is per-expert**, so a mixed tile needs two `BLK_N × K` weight loads,
doubling the dominant cost to save one tile.

**And item 0h's runtime branch is now clearly the wrong shape of fix.** §17.10 costed it as
"emit both `BLK_M=16` and `BLK_M=64` pairs, branch on `x.shape[0]` near 3600". §18.9's frontier says
a branch buys the *upper envelope* of that table — roughly 576 / 850 / 1063 — which is worth having,
but it doubles the kernel count in the binary to buy at most +12.5% at one end. The in-tree
`LowBatchGemvSpecialize` precedent is also weaker than it looked: it branches inside **one** PrimFunc,
where the launch config is the max over both bodies, so the narrow-tile path would inherit the wide
path's shared-memory footprint and lose occupancy — the exact cost the branch exists to avoid.
A Relax-level `If` avoids that, and there is **no `relax.If` anywhere in mlc_llm** to copy from.

### 18.12 The VL re-gate — the checkpoint was never the blocker, and neither is a download

§9's entry was corrected once already (the claim that a multi-GB download was needed was wrong —
`Qwen/Qwen3.5-0.8B` *is* the VL checkpoint). The build was then described as "a local
`convert_weight` + `gen_config` + `compile`... the rebuild itself should be uneventful." It was not.

**`convert_weight` and `gen_config` work exactly as predicted.** `--model-type qwen3_5_vl` produces
**383 MLC params** against the text-only build's 284 — the ~99 extra are the vision tower, so the
loader's `HF_VISUAL_PREFIX` lines up as §9 said — and `mtp.*` is correctly reported unused.

**`compile` dies in `BLASDispatch`:**

```
tvm.error.InternalError: Check failed: (tensor_sinfo) is false:
    Expect TensorStructInfo, but received: relax.ShapeStructInfo
  ... mlc_llm/compiler_pass/blas_dispatch.py:40, in FuseOpsByPattern/RunCodegen
```

**Why no other build has ever hit this:** `cublas_gemm` auto-enables only for **unquantized**
weights — `_cublas_gemm` in [compiler_flags.py:103](python/mlc_llm/interface/compiler_flags.py#L103)
returns False unless the quantization is `q0f16`/`q0bf16`/`q0f32` or fp8. Every 35B build in this
document is `q4f16_1`, so `BLASDispatch` was never in their pipeline at all. The VL build is the
first `q0f16` compile since the box moved to CUDA 13.2 / LLVM 18.

**Workaround, and it is a workaround:** `--opt "flashinfer=1;cublas_gemm=0;cudagraph=1"` compiles
cleanly. That is legitimate for the re-gate, whose question is whether the *state path* still
produces the right tokens — cuBLAS dispatch is a throughput choice, not a correctness one — but it
means the VL lib gated here is not the lib a default `--opt O2` would produce.

**It is VL-specific, and that was worth one recompile to establish.** The text-only
`dist/qwen3_5-0.8B-q0f16` config, through the *same* default pipeline with cuBLAS enabled, compiles
cleanly. So this is not "no unquantized model compiles on this box" — `BLASDispatch` is fine on the
text stack and something in the **vision tower's** graph hands the cuBLAS pattern matcher a
`ShapeStructInfo` where it expects a tensor. `qwen3_5_vl_model.py`'s `image_embed` and
`vision/qwen3_vl_vit.py` are where to look; the likely shape is an op whose argument is a `ShapeExpr`
(a reshape target or a `strided_slice` bound) sitting inside a matmul pattern's match region.

**Scope note for the re-gate that follows:** it runs on the `cublas_gemm=0` lib. That answers the
question the re-gate exists to answer — do the five inherited state-path changes (§11, §13, §14,
§15, §16.5) still produce the right tokens on the VL path — because cuBLAS dispatch is a throughput
choice. It does **not** clear a default-`O2` VL build, which remains broken.

### 18.13 The VL re-gate ran — and the result is "not cleared", not "passed" or "broken"

With `pillow` and `torchvision==0.26.0+cu130` installed (both missing since the re-bootstrap; the
matching `+cu130` build exists on PyPI and imports cleanly against `torch 2.11.0+cu130`), both legs
run for the first time since `f667b07e` in May.

| prompt | tokens matched |
|---|---|
| 1. "Describe this image in one short sentence." | **12/29** |
| 2. "What animal is shown in the image?" | 50/50 |
| 3. "What is the dominant color of the animal in this image?" | 50/50 |
| 4. "Is this a domestic pet or a wild animal?" | 50/50 |
| 5. "Give a one-word answer: …facial expression?" | 5/5 |
| **aggregate** | **167/184 (90.8%)** — bar is 96%, so the harness prints **FAIL** |

**Do not read that as a regression, and do not read it as a pass.** Four of five prompts are
**token-identical, 155/155**. The whole deficit is one divergence, and decoding it shows what it is:

```
first diff at step 12:  MLC = 13 ('.')   HF = 11 (',')
reference: "...walks through a snowy forest, its thick fur and distinctive markings..."
```

MLC ends the sentence where HF continues it — on a prompt that asked for **one short sentence**, at a
point where the clause is already grammatically complete. That is the exact signature of a near-tie,
and **§16.1 is the section that established raw match counts cannot distinguish a near-tie from a
regression.** The `vl5` harness predates that lesson (May 2026) and scores an unweighted count against
a 96% bar, which is the instrument §16.1 replaced for the text path.

**It is also not comparable to `f667b07e`'s 176/180.** The reference was rebuilt tonight under
transformers 5.14.1, and it is a *different reference*: prompt 1 is now 29 tokens where it was 25, and
the total is 184 where it was 180. The old number was never going to reproduce.

### 18.14 The VL gate gets a margin, and the answer is PASS

Rather than leave §18.13 unresolved, `--greedy-parity-vl5` was given the missing instrument. Two
changes to [validate.py](validate.py):

1. `--reference-vl5` now captures the **per-position top1−top2 margin** (via `generate(...,
   output_logits=True)`), the same quantity `high_margin_gate.py` records, on the same scale
   (`tau = 2.0` nats ≈ top-1 is 7.4× top-2).
2. The gate scores the **first divergence** against the reference's margin at that position.

**The first-divergence framing is not a shortcut, it is forced by this driver.** `high_margin_gate.py`
teacher-forces, so every position is independently comparable. `--greedy-parity-vl5` decodes
**free-running**, feeding MLC its own tokens — so the moment MLC picks differently it is conditioned on
a different prefix and *every later position is unscoreable, not wrong*. Counting them, as the 96% bar
does, charges one flip for the entire tail. Positions before the first divergence are genuine
agreements; the first divergence is the only thing that carries information.

**Result:**

```
prompt 1/5: 12/29   first diff @12, ref margin 0.05 -> NEAR-TIE
prompt 2/5: 50/50   prompt 3/5: 50/50   prompt 4/5: 50/50   prompt 5/5: 5/5
raw count bar = 176/184 (96%): FAIL     <- informational only
MARGIN VERDICT at tau=2.0: 0 prompt(s) diverged at a wide-margin position: PASS
```

**The reference's own margin at the divergence is 0.05 nats — top-1 was 1.05× top-2.** The `,`-vs-`.`
choice was very nearly a coin flip *in the HF reference itself*, so an MLC build landing on the other
side of it says nothing about the state path. Prompt 1 is low-margin throughout (only 8/29 positions
clear tau, median 1.06) because "describe this image in one short sentence" admits many acceptable
continuations — §16.1 made the same observation when it built a deliberately high-margin prompt set
for the text path.

**So the VL path is cleared**, and the five inherited state-path changes (§11, §13, §14, §15, §16.5)
are gated on it for the first time since `f667b07e`. The raw 96% count is retained in the output but
marked informational; it is the instrument §16.1 retired, and it produced a FAIL on a model that is
behaving correctly.

**Still outstanding for the VL path**, and neither is about correctness: the `BLASDispatch` compile
break (§18.12) means this is a `cublas_gemm=0` lib, and no VL *performance* number has ever been taken.

### 18.15 The 35B against the original, measured — not quoted

Every "overall" figure in this document chains ratios taken on the filler prompt across five sessions
and different clock states. This is the whole span measured directly: the **original**
`dist/qwen3_6-35B-A3B-q4f16_1/lib.so` (2026-07-24, before §14/§15/§16/§17) against tonight's libs, on
prose, `radix`, one clock state, 3 runs after 1 warmup.

| | original (2026-07-24) | shipped `lib_blkk64` | best measured tonight |
|---|---:|---:|---:|
| **pp512** | 361.59 | 827.39 (**2.29×**) | 850.39 — `m32rows` (**2.35×**) |
| **pp2048** | 399.83 | 945.00 (**2.36×**) | 1063.48 — `hoist64` (**2.66×**) |
| **tg512** (after pp512) | 54.10 | 59.99 (+10.9%) | 60.01 (+10.9%) |
| ttft pp2048 | 5122.5 ms | 2167.2 ms | **1925.8 ms** |

**The prefill work is a 2.3–2.4× on the shipped default and up to 2.7× on the frontier, and decode
picked up ~11% along the way** — decode was never the target of any of it, so that is fusion and
state-path work paying off incidentally.

Two things worth noting against the headline table at the top of this document. The **355 tps** it
records for "start of 2026-07-25c" is a *filler* number, and the original lib measures **361.59** on
prose — so in this one case the filler was not optimistic and the two agree within 2%. And the
overall multiplier there (+147%) is the pp512 chain; measured end-to-end it is **+129%** for the
shipped lib. The chain was close, but it was a chain.

---

## 19. Session 2026-07-27 — the last two leads: 0l refutes the premise it was built on, 0m is two ops

Both items §18 left open are closed, and **neither changes a default**. Item **0l** works exactly as
designed and disproves the hypothesis it was designed to exploit. Item **0m** turns out to be two
ops, and the fix is a pattern check.

### 19.1 Item 0l — the guard is a predicate, not a loop extent

§18.11's diagnosis of item 0k was that the loss is not the skipping but the *dynamic extent*: 0k
replaced a compile-time `BLK_M/MICRO` trip count with `ceildiv(real rows, MICRO)`, and at `BLK_M=16`,
where that expression can only ever be 1, it still cost 5–9%.

Built as `MLC_MOE_GEMM_V2_ROWSPEC=1`, and simpler than the filed plan. Rather than emit four
statically-unrolled bodies and select among them (10 fragment bodies at `BLK_M=64`), keep the loop
exactly as it is — constant extent, constant fragment indices — and predicate its body:

```
for i_o in range(BLK_M // MICRO):        # unchanged, still a compile-time constant
    if i_o * MICRO < row_end - m_offset: # CTA-uniform
        <wmma fragment>
```

**A predicate is admissible here for the reason item 0f's was not.** §16.9 established that
ThreadSync refuses `Cannot insert syncs inside condition`, which is why 0f had to move its guard into
a loop *extent*. Under the hoist the fragment loop sits *below* `k_o_o`, where the cooperative loads
and their barriers live, so the predicated region is barrier-free. Without the hoist `i_o` is the
outermost loop and encloses them — so `ROWSPEC` asserts `HOIST=1` rather than failing in a pass 200
lines downstream. The annotation lands on two loops (`a0_0_init` and `a0_0` — the accumulator init
nest and the compute nest), and both are safe to predicate: a skipped fragment's global store is
predicated off by `m_offset + i < row_end` regardless of what its accumulator holds. Same
bit-exactness argument as 0f and 0k, one level down.

### 19.2 The kernel A/B — and the `BLK_M=16` control that turns §18.11 into a measurement

[scripts/moe_rowspec_ab.py](scripts/moe_rowspec_ab.py) runs three legs off identical inputs in one
process — `base` (neither guard), `skiprows` (0k), `rowspec` (0l) — at `HOIST=1`, against the real
routing in `tuning/expert_hist_35b.npz`. **24/24 cells bit-exact against `base`** (12 real-routing,
12 synthetic).

| vs `base` at the same `BLK_M`, gate_up / down | pp512, B=4096 | pp2048, B=16384 |
|---|---:|---:|
| M=16 `skiprows` (0k) | 0.91× / 0.95× | 0.96× / 0.97× |
| **M=16 `rowspec` (0l)** | **1.00× / 1.00×** | **1.00× / 1.00×** |
| M=32 `skiprows` | 1.06× / 1.07× | 1.03× / 1.01× |
| M=32 `rowspec` | 1.01× / 1.02× | 1.01× / 1.04× |
| M=64 `skiprows` | 0.90× / 0.91× | 0.91× / 0.92× |
| **M=64 `rowspec`** | **1.03× / 1.00×** | **1.00× / 0.98×** |

**The `BLK_M=16` row is the result.** There the guard is logically inert under *both* mechanisms, and
0k costs 5–9% while 0l costs nothing at all, to three digits. §18.11 attributed 0k's loss to the
runtime extent blocking the unroll; that is now measured rather than inferred, and the mechanism 0l
was filed to build does what it was supposed to do. At `BLK_M=64` it converts 0k's 0.91× into
1.00–1.03×.

One thing it does *not* do is beat 0k at `BLK_M=32` (1.01–1.04× against 1.03–1.07×). Left unexplained
rather than guessed at; it does not bear on the conclusion below, and §18's traps were all created by
explaining a number without measuring it.

### 19.3 …and that is exactly why item 0l is refuted

The filed prediction was that `BLK_M=64` + hoist + static specialisation would beat `BLK_M=16` **at
every prompt length**, crossing the frontier §18.9 mapped instead of sliding along it. The stated
refutation criterion was the short-prompt leg, "the first thing to measure, not the last".

At the pp128 scale — B=1024, both synthetic routings, since the histogram covers only 512 and 2048:

| B=1024, gate_up / down | vs `base` M=64 | **vs shipped M=16** |
|---|---:|---:|
| M=64 + hoist (`base`) | — | 0.69× / 0.60× |
| M=64 + hoist + `rowspec` | 1.03× / 0.99× | **0.71× / 0.60×** |

**Removing every padding fragment's reduction closes 2 points of a 31-point gap on `gate_up` and
nothing at all on `down`.** So the hypothesis four sections have carried — that padding-row wmma
compute is why a wide tile loses at short prompts (§17.10, §18.7, §18.11) — is **wrong**. That compute
is real and worth 0–3%; it is not the cost.

The remaining cost has to be **per-CTA rather than per-tile**, because at B=1024 the tile count is
*identical* at both widths: no expert holds more than 64 rows, so every hit expert gets exactly one
m-tile either way. Same CTAs, same weight traffic, same live rows — and a 30–40% gap. Item **0n** is
that question, and it is a measurement rather than another kernel variant.

### 19.4 End-to-end, one session, one clock state

`lib_rowspec64` = `BLK_M=64` + hoist + `ROWSPEC`. Benched against the shipped `lib_blkk64` on
`--prompt-file` prose, `--prefix-cache-mode radix`, 3 runs after 1 warmup, both legs back to back.

| pp | `lib_blkk64` (shipped) | `lib_rowspec64` | Δ | (§18.9's `lib_hoist64`) |
|---:|---:|---:|---:|---:|
| 128 | **550.51** | 500.17 | **−9.1%** | −9.0% |
| 512 | 836.56 | **855.89** | **+2.3%** | −0.1% |
| 2048 | 946.32 | **1061.53** | **+12.2%** | +12.5% |

Decode neutral throughout (59.0–60.1 tps on both legs at every length).

⚠️ `jetson_clocks` needs an interactive sudo, which this session did not have, so absolute figures sit
~1–3% under §18.9's. **The cross-session control holds**: `lib_blkk64` at pp2048 reproduces §18.9's
945.00 to **0.14%** (946.32), which is what makes the two sessions comparable at all.

**Read against `lib_hoist64`, item 0l buys +2.4 points at pp512 and nothing at either end.** That is
the same shape the kernel A/B predicted (1.03× at B=4096, 1.00×/0.98× at B=16384) and it is not
enough to move the trade: **pp128 is −9.1%, so `BLK_M=64` is still on the wrong side of §18.9's
frontier and the defaults are unchanged** (`BLK_M=16`, `HOIST=0`, `SKIPROWS=0`, `ROWSPEC=0`,
`TILEORDER=m`). `ROWSPEC` ships off, exactly as `SKIPROWS` and `HOIST` do.

### 19.5 Item 0m — it is two ops, and they are the patch merger

§18.12 had it diagnosed to one pass. The localisation it proposed — bisect by `entry_functions` —
**does not work**, and that is worth recording: `RunCodegen`'s `entry_functions` filter does not stop
it serialising every `Codegen`-annotated function in the module, so **all 17 entry functions fail with
the identical error**, `softmax_with_temperature` included. What localises it is asking the *fused*
module a different question: which lifted function has a non-tensor parameter?

**Mechanism, end to end.** When `FuseOpsByPattern` lifts a matched region into a composite function,
any symbolic variable the region uses but that none of its parameters *defines* is appended as a
`tir_vars: R.Shape([...])` parameter
([fuse_ops.cc:567](3rdparty/tvm/src/relax/transform/fuse_ops.cc#L567)). The BYOC JSON serializer then
walks **every** parameter and requires a `TensorStructInfo`
([codegen_json.h:289](3rdparty/tvm/src/relax/backend/contrib/codegen_json/codegen_json.h#L289)). That
`ICHECK` is the crash.

A variable is *defined* by a parameter only when it appears as a bare `tir.Var` in that parameter's
shape. `R.Tensor((seq_len, 2048))` defines `seq_len`; `R.Tensor((num_patches // 4, 3072))` defines
nothing. The second form is the patch merger —
[qwen3_vl_vit.py:200-202](python/mlc_llm/model/vision/qwen3_vl_vit.py#L200) reshapes `(n, hidden)` to
`(n // merge_sq, merge_sq * hidden)` before `linear_fc1`. Exactly two regions are affected:

| lifted function | what it is | appended param |
|---|---|---|
| `fused_relax_permute_dims_relax_matmul_relax_add_relax_nn_gelu_cublas` | `visual.merger.linear_fc1` + bias + gelu | `tir_vars: R.Shape([num_patches])` |
| `fused_relax_permute_dims_relax_matmul_relax_add4_cublas` | `visual.merger.linear_fc2` + bias | `tir_vars: R.Shape([num_patches])` |

Both are called from `image_embed`. Nothing in the text stack produces that shape, which is why the
text-only `q0f16` build has always compiled — §18.12 was right that it is VL-specific, and this is
why.

### 19.6 The fix — decline the match, do not switch the pass off

[blas_dispatch.py](python/mlc_llm/compiler_pass/blas_dispatch.py) now wraps every cuBLAS/hipBLAS
fusion pattern's `check` with a predicate answering "would this match need a `tir_vars` parameter?",
comparing the symbolic vars the region *uses* against the ones its parameters *define*
(`relax.analysis.tir_vars_in_struct_info` vs `definable_tir_vars_in_struct_info`). `ShapeExpr` and
`PrimValue` arguments are inlined rather than parameterised
([fuse_ops.cc:644](3rdparty/tvm/src/relax/transform/fuse_ops.cc#L644)), so they are excluded from the
*defining* side — the conservative direction, since counting them would let a match through that then
kills the compile.

Declining is the supported way for a BYOC pattern to say "not this one": the two merger matmuls fall
back to the generated kernel and every other offload is kept. That is strictly narrower than the
`cublas_gemm=0` workaround, which switched the pass off for the whole model.

**Measured rather than argued** — `FuseOpsByPattern` run twice on the same module, raw patterns then
guarded, diffing the set of `Codegen`-annotated functions:

| model | offloads, raw | guarded | declined |
|---|---:|---:|---|
| `qwen3_5-0.8B-vl-q0f16` | 22 | **20** | exactly the two merger matmuls above |
| `qwen3_5-0.8B-q0f16` (text) | 13 | **13** | none — the guard is inert off the vision tower |

`dist/qwen3_5-0.8B-vl-q0f16/lib_cublas.so` is the first VL lib built at **default `--opt`** with
`cublas_gemm=1`. The hipBLAS branch gets the identical guard: same serializer, same failure mode.

### 19.7 The VL gate on the cuBLAS lib — 184/184, and §18.14's near-tie flips to agreement

```
prompt 1/5: 29/29   2/5: 50/50   3/5: 50/50   4/5: 50/50   5/5: 5/5
AGGREGATE 184/184 (100.0%)   raw count bar: PASS   MARGIN VERDICT at tau=2.0: PASS
```

§18.14's lib diverged on prompt 1 at step 12 (12/29), at a position whose reference margin is
**0.05 nats**; this one does not. **That is not evidence that cuBLAS is more correct.** A 1.05×
top1/top2 gap is a coin flip and all that changed is the accumulation order of 20 matmuls. What it
does confirm is §18.14's reading of that divergence as a numerical near-tie rather than a state-path
fault — a perturbation this small moves it. The VL path is now gated on the lib a default `--opt`
produces, which is the gap §18.12's scope note left open.

### 19.8 Where this leaves the MoE lane

§18.11 called 0l "the first MoE idea in three sessions that is not a point on the frontier §18.9
mapped". It turned out to be a point on it after all — but it moved the *diagnosis*, and that is what
the next item has to be built on:

| | before §19 | after §19 |
|---|---|---|
| why 0k lost 5–9% where its guard is inert | inferred: a runtime extent blocks the unroll | **measured**: the predicated form costs 1.00× |
| why a wide `BLK_M` loses at short prompts | padding-row wmma compute (§17.10, §18.7, §18.11) | **not that** — removing it recovers 0–3 of 31–40 points |
| what the wide tile's short-prompt cost is | — | per-CTA, not per-tile, and **unidentified** |
| best known pp2048 | 1063.48 (`hoist64`, §18.9) | 1061.53 (`rowspec64`) — the same number, and neither ships |

**The open question is now sharp enough to be worth one measurement rather than a build**, and it is
filed as item 0n. At B=1024 the two widths launch the same CTAs over the same rows; everything that
differs is per-CTA. Diffing the emitted CUDA (`scripts/moe_dump_cuda.py` at `BLK_M` 16 vs
64+hoist+`ROWSPEC`) already names three surviving `BLK_M`-scaled terms, none of which 0k or 0l
touched — all of them sit *above* the fragment loop the two items guarded:

| per-CTA term, from the emitted source | `BLK_M=16` | `BLK_M=64` |
|---|---:|---:|
| dynamic shared memory (`X_tile` 72-half rows + `W_tile`; `O_tile` aliases into `W_tile`) | 10368 halves, **20.25 kB** | 13824 halves, **27.0 kB** |
| `X_shared` cooperative-store fragments per k-step (predicated on `row_end`, but *stored* regardless) | 1 | **4** |
| `load_matrix_sync` of X fragments per k-step (`A_mat` sits at `k_o_i`, outside the guarded loop) | 1 | **4** |

Shared memory alone is 1.33×, which on a 164 kB/SM budget is 8 resident CTAs against 6 — real, but
probably not 30–40% on its own. The other two rows are 4× *unconditional* work at 64, and they are
the reason predicating the reduction bought so little: **0l skipped the multiply, not the operand
traffic feeding it.** That is where item 0n should start, and it suggests the fix — if any — is a
narrower `BLK_N` at `BLK_M=64` (trading the same shared budget the other way) or predicating `A_mat`,
not a wider tile.

**And the pattern behind this session, which is the same one §16's closing note named.** Both leads
came in as confident mechanisms: 0l's "the win should survive" rested on a cost model nobody had
measured at the short-prompt shape, and 0m's "bisect by `entry_functions`" rested on reading a pass
signature rather than running it. Both were wrong in the same way — an inference from the right
neighbourhood, never checked against the thing itself. The half-day 0l cost bought a **retraction of
four sections' worth of shared assumption**, which is worth more than the +2.3% at pp512 it also
produced.

---

## 20. Session 2026-07-27b — item 0o: the first VL number, and cuBLAS is net-negative on it

### 20.1 The measurement that had never been taken

`validate.py --perf-vl5` (new mode) is the `--greedy-parity-vl5` driver minus the reference
comparison, plus `dev.sync()` and a clock around each VM call. It takes item 0o's three separable
numbers. Cat fixture, 2520 patches → 630 image tokens, `seq_len=652`, 64 decode steps, n=20 timed
iterations after 5 warmup, medians:

| | `lib_cublas.so` (default `--opt`, 20 offloads) | `lib.so` (`cublas_gemm=0`) |
|---|---:|---:|
| `image_embed` | 337.28 ms | **297.34 ms** |
| `prefill` (652 tok) | **149.37 ms** (4365 tok/s) | 161.91 ms (4027 tok/s) |
| `decode` | 11.36 ms (88.1 tok/s) | 11.37 ms (88.0 tok/s) |
| **ttft** = embed + prefill | 486.65 ms | **459.25 ms** |

**Three things fall out immediately, and the first two were assumptions nobody had checked.**

1. **`image_embed` is not cheap.** It is **69% of ttft** and **2.26× the prefill of the entire
   652-token sequence**. The parity driver calls it *inside* the per-prompt loop under a comment
   reading "same image across prompts; could cache but cheap" — so every `--greedy-parity-vl5` run
   has been paying it **five times**, ~1.35 s of pure waste per gate.
2. **The default `--opt` build is the slower one.** §19.6 retired `cublas_gemm=0` as a "workaround";
   the first performance number says that workaround build is **5.6% better on ttft**. Neither lib
   is Pareto — cuBLAS is +8.4% on prefill and −11.8% on the tower.
3. **`radix` vs `disable` is neutral here** — prefill 1.000×, decode 1.003×, on both libs. The
   history path costs the VL model nothing, which is the §15 fusion doing its job.

*Control.* The cuBLAS leg re-run immediately after the plain leg reproduces `image_embed` to
**0.11%** (337.28 → 337.66) and prefill to **0.05%**, so the A/B is not ordering or thermal drift.

**Item 0o's split-by-call instruction is what made this readable.** A whole-model delta would have
been −5.6% and would not have attributed itself; split, it is two opposite-signed effects that
nearly cancel.

### 20.2 Why cuBLAS loses the tower — an accounting that closes to 0.8%

`nsys`, 13 `image_embed` calls, tower only (the perf harness's prefill/decode kernels swamp it
otherwise). Per iteration, 12 vision layers:

| | `lib.so` (plain) | `lib_cublas.so` |
|---|---:|---:|
| QK^T + scale | `fused_NT_matmul19_multiply22` **113.61** | `ampere_sgemm_128x128_tn` 43.55 + `multiply22` 117.34 = **160.89** |
| FFN / projection GEMMs + epilogues | 30.54 | **27.50** |
| softmax, P@V, transposes, norms | 154.75 | 154.71 |
| **total kernel time** | **298.86** | **343.10** |

The trace sums land on the wall clock (298.86 vs 297.34 measured; 343.10 vs 337.28), so the tower is
kernel-bound with no gaps, and **+44.24 ms is the whole regression**.

**The mechanism.** [qwen3_vl_vit.py:151-170](python/mlc_llm/model/vision/qwen3_vl_vit.py#L151-L170)
runs vision attention in fp32 deliberately — fp16 "collapses tower parity (max diff 2.03 / rel 39%)"
— so `matmul(q32, k_t)` is an **fp32** GEMM immediately followed by `multiply(attn_scores, scaling)`.
Those two fuse into one kernel. Offloading the matmul to cuBLAS **breaks that fusion**, and the
tensor it was fusing over is `(12, 2520, 2520)` fp32 = **305 MB per layer**:

| | predicted | measured |
|---|---:|---:|
| extra DRAM traffic from the broken fusion | 610 MB/layer (write, then read+write) | — |
| …at the 156 GB/s wall | **3.91 ms/layer** | **3.94 ms/layer** |
| × 12 layers | 46.9 ms | **47.28 ms** |

**0.8%.** The regression is exactly the round trip, and nothing else.

**And cuBLAS is not bad at the GEMM — it is 2.6× better at it.** 9.75 GFLOP in 3.63 ms is
**2.69 TFLOP/s** against the generated kernel's **1.03**. It simply cannot win: the GEMM it improves
takes 3.6 ms and the fusion it destroys costs 3.9 ms, every layer. Meanwhile the *other* giant
matmul, `matmul(attn_probs, v32)`, was never offloaded in either leg (84.4 ms in both) — its
`astype` epilogue kept it fused, which is the same effect working in our favour by accident.

**Why the FFN goes the other way.** cuBLAS picks `ampere_fp16_s16816gemm_*` there — **tensor-core**
kernels — and their epilogues (`gelu_tanh`, `add16`) are over `(2520, 3072)` fp16 = 15 MB, not
305 MB. Breaking a fusion is cheap when the tensor is 20× smaller, and tensor cores pay for it:
**−3.04 ms/iteration in cuBLAS's favour.** The rule that falls out is *offload when it buys tensor
cores*, and fp32 is where it does not.

### 20.3 The fix — decline the fp32 offload, keep the fp16 ones

Same "decline the match" mechanism §19.6 built, one more predicate:
[blas_dispatch.py](python/mlc_llm/compiler_pass/blas_dispatch.py) `_region_is_fp32` returns True when
any tensor in the matched region is fp32, and the wrapped `check` declines it. Gated by
**`MLC_BLAS_SKIP_FP32`, default `1`**.

`dist/qwen3_5-0.8B-vl-q0f16/lib_nofp32blas.so`, measured identically:

| | `lib_cublas` (was default) | `lib.so` | **`lib_nofp32blas`** |
|---|---:|---:|---:|
| `image_embed` | 337.28 | 297.34 | **293.33** |
| `prefill` | **149.37** | 161.91 | 149.87 |
| `decode` | 11.36 | 11.37 | 11.38 |
| **ttft** | 486.65 | 459.25 | **443.20** |

**Pareto, and it beats both existing libs on the tower.** ttft is **−8.9%** against the shipped
default and **−3.5%** against the `cublas_gemm=0` build. `image_embed` at 293.33 is better than
*either* prior lib because it keeps the fp16 FFN offloads the plain build gives up while declining
the fp32 one the cuBLAS build takes. Predicted 295.86 / ttft 445.3 before building; measured
293.33 / 443.20 — **0.9% and 0.5%**.

*The trace confirms the guard did exactly one thing.* `ampere_sgemm_128x128_tn` (43.55) and the
standalone `multiply22_kernel` (117.34) are **gone**, `fused_NT_matmul8_multiply22_kernel` (113.53)
is **back**, and both `ampere_fp16_s16816gemm_*` (9.33, 6.04), `gelu_tanh` (7.67) and `add16` (4.43)
are **unchanged from the cuBLAS leg** to within 0.02 ms.

*The knob is provably the only change.* `lib_ctrl_skip0.so`, compiled from the **same edited source**
with `MLC_BLAS_SKIP_FP32=0`, reproduces the pre-change `lib_cublas.so` to the digit —
`image_embed` **337.28 vs 337.28**, prefill 149.25 vs 149.37 (0.08%), ttft 486.53 vs 486.65 (0.02%).
So the edit is inert at `0` and the entire win comes from the guard, not from anything else the
recompile touched.

*Gate.* `--greedy-parity-vl5` on `lib_nofp32blas.so`: **184/184 exact, margin verdict PASS at
τ=2.0** — identical to `lib_cublas.so`'s §19.7 result, not merely passing the bar.

**Scope, stated rather than implied.** `_cublas_gemm`
([compiler_flags.py:103](python/mlc_llm/interface/compiler_flags.py#L103)) only enables the pass for
`q0f16`/`q0bf16`/`q0f32`/fp8, and the VL `q0f16` build is the only one of those in this project — the
35B and 0.8B text models are `q4f16_1`, so the pass is off for them and **this guard is unreachable
there**. No 35B or text-model number in this document can move. The one case it would change
unmeasured is a **`q0f32`** model, where it would decline cuBLAS wholesale; `MLC_BLAS_SKIP_FP32=0`
restores the old behaviour.

**The honest caveat on the predicate.** The real mechanism is "the displaced fusion writes a tensor
bigger than the GEMM reads", and fp32 is a *proxy* for that, not the thing. The precise test is not
computable in a pattern check here — the tower's shapes are symbolic in `num_patches`, and
`12·s² > 4·(12·s·64 + 12·64·s)` is unprovable without a bound on `s`, so an analyzer-based version
would decline nothing and fix nothing. The proxy is exact on every configuration that exists here.

### 20.4 What this leaves

**Item 0o is closed, and it answers its own follow-up question backwards.** The entry asked whether
the two merger matmuls §19.6 declined are "worth recovering by giving them a bare-`tir.Var` shape,
which would make them eligible again". They are **not** — the measurement says the marginal cuBLAS
offload on this tower is a *liability* wherever it displaces a fusion over a large tensor, and the
merger matmuls sit right after a reshape at `num_patches // 4`. Recovering them is the wrong
direction; §19.6's decline was accidentally the right call for a second reason.

**The next lever in the tower is not cuBLAS at all — it is the two kernels cuBLAS never touched.**
At `lib_nofp32blas`, `image_embed`'s 293 ms is dominated by three kernels that are all the same shape
of problem:

| kernel | ms/iter | what it is | note |
|---|---:|---|---|
| `fused_NT_matmul8_multiply22` | 113.53 | QK^T + scale, fp32 | **1.03 TFLOP/s** — cuBLAS showed 2.69 is reachable |
| `fused_matmul14_cast17` | 84.33 | P@V + cast, fp32 | never offloaded, never measured |
| `softmax` | 48.42 | over `(12, 2520, 2520)` fp32 | 305 MB in, 305 MB out |

That is **84% of the tower in three kernels**, all of them fp32 and all of them moving the 305 MB
score tensor. The prize is not a better GEMM — it is **not materializing the score matrix at all**,
which is what flash attention is for. Filed as **item 0p**.

**Landed on the way: the gate stops running the tower five times.** `image_embed` sat inside
`--greedy-parity-vl5`'s per-prompt loop for a fixture that never changes (§20.1). Hoisted; the gate
still scores **184/184 exact, margin PASS**, and now runs in **13.2 s**, saving 4 × `image_embed`
(**1.17 s** on `lib_nofp32blas`, 1.35 s on `lib_cublas`).

### 20.5 Item 0p, part 1 — the wall was wrong, and the fusion was worth nothing

**First, an instrument.** `scripts/vit_attn_bench.py` rebuilds the attention block from
[qwen3_vl_vit.py](python/mlc_llm/model/vision/qwen3_vl_vit.py)'s ops, runs the same passes and the
same dlight schedules, and times each generated PrimFunc alone — because iterating on a kernel
through a 3.5-minute whole-model compile is the wrong loop. It emits the same four PrimFuncs the
real compile does and **reproduces §20.2's per-layer numbers to 1.3%**, which is the bar that makes
anything built on it trustworthy:

| | microbench | §20.2, traced | delta |
|---|---:|---:|---|
| QK^T + scale | 9.59 | 9.46 | +1.4% |
| P@V + cast | 7.04 | 7.03 | +0.1% |
| softmax | 3.97 | 4.04 | −1.7% |

**⚠️ The bandwidth wall used in §20.2 was wrong for this access pattern: it is 184.8 GB/s, not
156.** A static-shape softmax came in at 3.35 ms for 610 MB — 182 GB/s, which the 156 figure says is
impossible. A device-to-device copy of exactly the score tensor (304.8 MB) settles it at **3.30 ms,
184.8 GB/s**. Both numbers are real: 156 GB/s is what the *MoE GEMM's strided expert-weight* access
achieves and is correct there; the tower streams contiguously and gets 184.8. This is the fifth time
this document has been bitten by a figure quoted out of the shape it was measured on, and the rule
from §15.2 applies unchanged — **renormalize against a measurement at the target shape.** Re-based:

| | measured | bound @184.8 GB/s | off | effective |
|---|---:|---:|---:|---:|
| QK^T + scale | 9.46 | 1.83 (compute) | **4.9×** | 35.9 GB/s |
| softmax | 4.04 → **3.35** static | 3.30 (BW) | **1.0×** | 182.0 GB/s |
| P@V + cast | 7.03 | 1.83 (compute) | **3.8×** | 45.1 GB/s |

**So softmax is finished.** At 98.5% of a measured copy of the same bytes there is nothing in it —
item 0p's step (a) is answered, and the only way to improve it is to stop producing its input.

**Hypothesis 1, symbolic shapes: largely refuted.** `--static` pins `seq_len` to a literal instead of
`num_patches`. Worth **6.5%** overall (QK^T 9.59 → 8.93, softmax 3.97 → 3.35, P@V unmoved) — real,
but not a 4× gap. dlight tiling blind is not the problem.

**Hypothesis 2, prescaling: worth nothing, and that is the whole point.**
`matmul(q, k^T) · c ≡ matmul(q · c, k^T)`, which moves the scale off a 305 MB tensor onto a 7.7 MB
one. Measured, it buys **nothing**: `NT_matmul` alone is **8.93 ms**, to the digit what
`fused_NT_matmul_multiply` costs. The multiply was already free inside the fusion.

**That null result is the finding.** If the fusion buys nothing, then the fusion is not worth
protecting — and protecting it is exactly what §20.3's guard does. The QK matmul becomes a *bare*
GEMM with nothing to displace, so cuBLAS can have it at the 3.63 ms §20.2 already measured, against
the generated kernel's 8.93.

### 20.6 …which retires §20.3's guard one section after it landed

`MLC_QWEN35_VL_PRESCALE_Q` (default **`1`**) moves the scale onto `q`;
`MLC_BLAS_SKIP_FP32` therefore drops to default **`0`**. Measured with no environment variables set
at all — `lib_vl2.so`, plain `--opt`, which is the configuration that actually ships:

| | `lib_cublas` (§20 start) | `lib_nofp32blas` (§20.3) | **`lib_vl2`** (§20.6) |
|---|---:|---:|---:|
| `image_embed` | 337.28 | 293.33 | **223.33** |
| `prefill` | 149.37 | 149.87 | 149.54 |
| `decode` | 11.36 | 11.38 | 11.38 |
| **ttft** | 486.65 | 443.20 | **372.88** |

**`image_embed` −33.8% and ttft −23.4% against where this session started**, decode and prefill
untouched. Gate: **184/184 exact, margin PASS** — prescaling changes fp32 rounding by ~1 ulp and the
gate is unmoved. `lib_prescale.so` (built with both variables set explicitly) and `lib_vl2.so` agree
to **0.07%**, which is the check that the defaults are what they claim to be.

*The trace closes the accounting.* `ampere_sgemm_128x128_tn` is back at **43.55 ms/iter — identical
to §20.2's cuBLAS leg** — while *both* the 113.53 ms fused kernel and the 117.34 ms standalone
`multiply22` are gone. Kernel delta 70.0 ms against a measured `image_embed` drop of 70.15 ms: **0.2%**.

**§20.3 was right about the mechanism and wrong about the fix, and it is worth being precise about
which.** Its diagnosis — cuBLAS cannot pay back a fusion it destroys over a 305 MB tensor — is
confirmed, and its 3.91-vs-3.94 ms/layer prediction still stands. But it treated the fusion as
something to protect, when the fusion was worth **0.00 ms**. Declining the offload bought 3.0 ms/iter
and gave up 70. *One graph-level identity beat the compiler-pass fix by 23×*, and the two knobs are
now coupled: set `MLC_QWEN35_VL_PRESCALE_Q=0` and `MLC_BLAS_SKIP_FP32` is worth `1` again, because
that configuration pays §20.2's 47 ms.

**The pattern, and it is the one §16 and §19 both closed on.** §20.3 measured a real mechanism and
then optimised *around* it instead of asking whether the thing it was protecting had any value. The
question that cracked it — "what does the fusion actually buy?" — cost one microbench run and
returned **zero**. Both sessions' biggest wins came from measuring a premise nobody had priced,
not from a cleverer kernel.

### 20.7 Where the tower stands now

`image_embed` is 223 ms, and the three-kernel picture has changed shape:

| kernel | ms/iter | ms/layer | bound | off | status |
|---|---:|---:|---:|---:|---|
| `fused_matmul9_cast17` (P@V + cast) | 84.34 | 7.03 | 1.83 | **3.8×** | ⬅️ now the largest, and untouched |
| `softmax` | 48.57 | 4.05 | 3.30 | 1.0× | at the wall, closed |
| `ampere_sgemm` (QK^T) | 43.55 | 3.63 | 1.83 | 2.0× | cuBLAS; 51% of fp32 peak |

**P@V is now the biggest single kernel in the tower** and is the one thing here nobody has attacked.
It is `matmul(attn_probs, v32)` with an `astype` epilogue, and that epilogue is why cuBLAS never took
it — the same fusion question §20.5 just answered for QK^T, and it has *not* been asked here. The
cast writes fp16 (3.87 MB) against reading a 305 MB fp32 `attn_probs`, so the fusion is protecting
almost nothing, and the identity is even simpler: there is no algebra to rearrange, just a cast that
could move. **That is the next measurement, and the microbench answers it without a compile.**

Beyond it, the tower's floor is set by materializing the score matrix at all: 305 MB written by QK^T,
read+written by softmax, read by P@V. Flash attention removes all three — the compute floor is
2 × 1.83 = 3.67 ms/layer against today's 14.7 — but it is a real build, in fp32, with no HF kernel to
copy (the reference runs eager attention). Price P@V first; it may be most of the remaining gap for a
fraction of the risk.
