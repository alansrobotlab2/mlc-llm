# Workplan: Qwen3.6-35B-A3B on JetPack 7.2 / CUDA 13.2

**Sessions:** 2026-07-24 (Stage 0/1), 2026-07-25a (kernel attribution, re-scope, options 1 and 2
landed, concurrent serving fixed), 2026-07-25b (conv-state fusion, §9 item 3 refuted, everything
committed), 2026-07-25c (history-path conv fusion — §14), 2026-07-25d (history-path *recurrent*
fusion — §15)
**Status:** **35B-A3B tg512 54.13 → 60.20 (+11.2%)** from four landed changes: the GDN
input-projection merge (§10), the in-place recurrent state (§11), concurrent serving on hybrid
models (§12), and the in-place conv state (§13). **Prefill on the *default* `prefix_cache_mode=radix`
was never measured until 2026-07-25c, and it was getting barely half the headline number** — §14
and §15 close that gap almost completely:

| pp512, `radix` (the default) | start of 2026-07-25c | now | vs `disable` |
|---|---:|---:|---:|
| 35B-A3B | 355 | **629 (+77%)** | 646 — **1.03×** |
| 0.8B | 1469 | **3934 (+168%)** | 4070 — **1.04×** |

**Everything through §14 is committed; §15 is not yet — see §9.** **Next session: start at §9.**
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
| `lib_gdnhist.so` | **the current build** — everything in `lib_histconv` plus the §15 history-path recurrent fusion. Bench and gate against this |
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
| [scripts/gdn_kernel_check.py](scripts/gdn_kernel_check.py) | **new (§15)** — the recurrent analogue of `conv1d_kernel_check`. Gates both GDN in-place kernels against the *copy-path kernel* (so the bar is bit-exactness, not a tolerance) plus an fp64 reference of the recurrence, across 9 seq_lens × 2 head configs including 5 that wrap the ring. Separates "ring misindexed" from "output wrong" — the off-by-one control leaves `out_bit` at 0 while `state_err` hits 130. No model, no weights, no engine |
| [scripts/conv1d_kernel_check.py](scripts/conv1d_kernel_check.py) | **new (§13), extended (§14)** — now gates the history variant too, including ring-wrap shapes. Numerical unit gate for the fused conv1d: kernel vs fp64 across 12 shapes and both conv widths. Separates "wrong" from "rounded differently", which no token-diff can do on the 35B. Needs no model, no weights, no engine; runs in seconds |
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
python scratch_mlc_tg_sweep.py \
  --model-dir dist/qwen3_6-35B-A3B-q4f16_1 \
  --model-lib dist/qwen3_6-35B-A3B-q4f16_1/lib.so \
  --pp 512 --tg 512,1024,2048,4096,8192 --runs 3 --warmup 1 --json-out tuning/<name>.json

# correctness (q0f16 only — q4 vs fp16 is not a valid gate)
python validate.py --reference-only --model Qwen/Qwen3.5-0.8B --device cuda:0 \
    --cache reference_outputs.pt
python validate.py --greedy-parity --model Qwen/Qwen3.5-0.8B --device cuda:0 \
    --mlc-model-dir dist/qwen3_5-0.8B-q0f16 \
    --mlc-lib dist/qwen3_5-0.8B-q0f16/lib.so --cache reference_outputs.pt

# kernel unit gates — no model, no engine, seconds. Run these FIRST on any state change.
python scripts/gdn_kernel_check.py                     # §15, recurrent; add --seq-lens to narrow
python scripts/conv1d_kernel_check.py                  # §13/§14, conv

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

All work from both 2026-07-25 sessions is now in git on branch `qwen3_5`:

| commit | what |
|---|---|
| `e3b099e8` | §10 in_proj merge + §11 in-place recurrent state |
| `838c2d1b` | §12 concurrent serving on hybrid models |
| `fce533aa` | §6.1 fp8 software dequant + six gates/probes + fp8 reference cache |
| `54777593` | §1–§12 workplan, qwen3_5.md corrections, tuning data |
| `6749a630` | `analyze_decode_trace` geometry verification |
| `3ff691f5` | §13 conv-state fusion + `conv1d_kernel_check.py` |
| `c2fd8691` | §13 workplan, §8 build warning, §9 item 3 refutation |

✅ **The TVM submodule commit IS pushed — this section said otherwise for three sessions and was
wrong.** `3rdparty/tvm` points at `4624d97` (branch `qwen35-inplace-rnn-state` on the
`alansrobotlab2/relax` fork), carrying the three `vm.builtin.rnn_state_*` accessors that §11, §13,
§14 and §15 all depend on. Verified 2026-07-25d: `git ls-remote origin qwen35-inplace-rnn-state`
returns `4624d972…`, identical to local `HEAD`.

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

> **Item IDs are stable, not sequential.** They are referenced from §12–§15 and from the Done
> sections above, so closed items keep their number rather than being renumbered away. Ordering
> below is by measured expected value as of 2026-07-25d.

#### Open

> As of 2026-07-26 the open list is **one item plus one blocked precondition**. Items 0b, 1 and 5
> closed this session (§16); 0c's step 1 is done and measured, leaving only step 2.

**0c. The GDN recurrence is parallelism-starved — the biggest prefill item, by 5×.**
§15.6 measured it: `gdn_func_history_inplace` is **95.6 ms against 19.3 ms for the next kernel**,
running at **202 GFLOP/s, ~3.8% of sm_87 fp32 peak**, because it launches `batch × n_vh` blocks of
`V` threads (**2048 threads on the 0.8B**, on a 16-SM GPU) and each walks the sequence
sequentially. §15 said there was no bandwidth left to reclaim; §16.2 shows that was DRAM-only —
the kernel spills 47 of its 128 state rows to local memory and re-reads them every position. Two
steps, in order:
  1. ✅ **Done, §16.2 — and it is worth ~2.8×, so do it before considering step 2.** The
     hypothesis was half right. Registers hit the 255 ceiling but permit 2 blocks/SM; the *grid*
     (16 blocks on 16 SMs) is what pins occupancy at 1. Splitting `K` across two **blocks** is not
     buildable — the reduction is inside a thread, so it would need a global barrier per position
     — but splitting it across **lanes** (`tid = v*2 + half`, one `__shfl_xor_sync`, no
     `__syncthreads`) is, and it measures **2.24–2.44×**; a 4-way split reaches **2.79×**. Most of
     that is getting `state_local` off the register ceiling, not occupancy: 4 accumulators alone
     buy only 1.27×. Measured with [gdn_recurrence_probe.cu](scripts/gdn_recurrence_probe.cu);
     **not yet built in TIR**.
  2. **Then scope the chunked linear-attention formulation** — matmuls over a chunk of C positions
     instead of a scalar loop, as in `../flash-linear-attention/fla/layers/gated_deltanet.py` and
     vLLM's `qwen3_next.py`. Substantially bigger than anything in §10–§15: it **changes the
     arithmetic**, so bit-exactness is off the table and item 0b is a prerequisite for judging it
     on the 35B, and it needs its own chunk-state intermediate. Note it does **nothing for
     decode** — at `seq_len=1` the chunked form degenerates, and `gdn_func_inplace` is already
     only 2.1% of the whole-run budget.

**The VL path has not been re-gated — and it is blocked on an artifact, not on work.** Checked
2026-07-26: there is **no VL checkpoint in the HF cache and no VL build in `dist/`**, so this needs
a multi-GB download before any of it can start. The package is registered
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
| `python/mlc_llm/model/qwen35/qwen35_model.py` | **§15** `create_gated_delta_net_func_with_history_inplace` + the recurrent half of `forward_with_history` behind `state_io`. **§14** `create_causal_conv1d_func_with_history_inplace` + `state_io` threaded through `forward_with_history`; the per-model hoist block factored into `_maybe_hoist_state_io`. **§10–13:** `in_proj_qkvzab` + `_in_proj()` helper; **§11** `create_gated_delta_net_func_inplace`, `_GDNStateIO`, `_hoist_gdn_state_io`, `MLC_QWEN35_INPLACE_STATE` toggle; **§13** `create_causal_conv1d_func_inplace` + `conv_storages` on `_GDNStateIO` |
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
| `python/mlc_llm/support/auto_target.py` | **§14.6 new** — `MLC_NVCC_OPTIONS` / `MLC_DUMP_CUDA` hooks + nvcc phase timing |
| `scratch_mlc_tg_sweep.py` | **§14.1 new** — `--prefix-cache-mode` and per-run prompt salting; without both, the default config was unmeasurable |
| `scripts/greedy_snapshot.py` | **§14** — `--prefix-cache-mode`; it hardcoded `disable`, where a history-path change is inert and the gate passes vacuously |
| `scripts/profile_decode_35b.py` | **§14** — `--prefix-cache-mode` + prompt salting so a radix trace contains prefill |
| `qwen3_5.md` | §14.5 + 5 stale-item corrections |
| `workplan-cuda-13.md` | **new** — this file |
| `.gitignore` | exception for `reference_outputs_35b.pt` **and `reference_outputs_35b_fp8.pt`** — the fp8 name was still ignored, so §7's "commit this" silently had not happened |

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

## 16. Session 2026-07-26 — the gates get teeth, and three diagnoses (in progress)

Four §9 items were open at the start of this session: **0b** (a deterministic 35B state gate),
**0c** (the parallelism-starved GDN recurrence), **1** (should the 35B decode more than one
sequence), and **5** (the tier-2 GEMV retune). This section covers all four. Measurements that
were still running when it was written are marked **[pending]**.

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
`gdn_func_history_inplace` is 95.6 ms of the 35B pp512 budget (§15.6); at 2.79× that lane would
fall to ~34 ms, but the flush is excluded and Amdahl applies to the rest of prefill, so treat ~2.8×
as an upper bound on the kernel and *not* as a prefill prediction. Per §15.2's rule, the estimate
to publish should be renormalized against a trace of the actual A/B baseline before any lib is
built.

**Status: not built in TIR.** This is a measured design decision, not a landed change — the TIR
kernel would need the lane-split layout, the `__shfl_xor_sync` reduction, and a re-gate through
`gdn_kernel_check.py` (bit-exactness against the copy path is **off the table**, since the
reduction order changes; the fp64 check becomes the bar).

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
