# Qwen3-Next / Qwen3.5 / Qwen3.6 in MLC-LLM — Technical Reference

This is the living technical document for bringing up the Qwen3-Next family of hybrid (GatedDeltaNet + GQA) models in MLC-LLM. Living = update as decisions land or assumptions change. Date-stamped progress goes in [`worklog.md`](./worklog.md).

---

## 1. Goal & Scope

End goal: **Qwen3.6-35B-A3B** (35B-param hybrid MoE, ~3B activated) running end-to-end on MLC-LLM with greedy-decode parity vs. HuggingFace transformers.

De-risk path: validate **Qwen3.5-0.8B** first. It is the smallest member of the family, dense (no MoE), with symmetric linear-attention heads and standard RoPE. Anything that breaks here is in the GatedDeltaNet / hybrid stack, isolated from MoE complexity.

Strict ordering — nothing on 35B until 0.8B passes the parity bar in §11.

---

## 2. Quick Start: Compile & Run Qwen3.6-35B-A3B on Orin

End-to-end recipe for the shipped lib at [dist/qwen3_6-35B-A3B-q4f16_1/](dist/qwen3_6-35B-A3B-q4f16_1/). Tested on Orin AGX (sm_87, MAXN). ~25 min from a **working tree that already has the toolchain built** — dominated by the 72 GB HF download. From a *bare* clone add the toolchain bootstrap in §2.1.1 (submodules + TVM + mlc-llm C++), which is the dominant cost on a fresh machine.

### 2.1 Prereqs

- Orin AGX (or any sm ≥ 87 CUDA device with ≥ 24 GB VRAM for q4f16_1).
- `MAXN` power profile (`sudo nvpmodel -m 0 && sudo jetson_clocks`) for benchmark stability.
- ~110 GB free disk: 72 GB HF snapshot + 19 GB converted params + ~10 GB of build trees.
- The locally-built TVM + mlc-llm C++ runtime, sourced via `.envrc.local` (§2.1.1). Note this is the **`PYTHONPATH` + `TVM_LIBRARY_PATH` + `MLC_LIBRARY_PATH`** layout, *not* an installed editable wheel — the repo `python/` tree is imported directly.
- **No HF auth needed.** `Qwen/Qwen3.6-35B-A3B` is public (apache-2.0) and downloads unauthenticated; earlier drafts of this doc called it gated, which was wrong. Setting `HF_TOKEN` only buys higher rate limits.

Platform this recipe is validated on, re-bootstrapped 2026-07-24 after the JetPack upgrade:

| | was (2026-04 phases) | now |
|---|---|---|
| OS | Ubuntu 22.04 | **Ubuntu 24.04.4 LTS** |
| JetPack | 6.2.2 | **7.2-b187** (L4T R39.2.0) |
| CUDA | 12.6 | **13.2** (nvcc V13.2.78) |
| LLVM | `llvm-15-dev` | **llvm-18** (18.1.3, the apt default on 24.04) |
| GCC | — | 13.3.0 |
| Python | — | 3.12.3 |
| GPU arch | sm_87 | sm_87 (unchanged) |

Deltas the upgrade forced on the build config — all three are in §2.1.1:
1. `USE_LLVM` now points at `llvm-config-18`. Still `--link-shared`: Ubuntu ships LLVM without `libPolly.a`, so static linking fails on 24.04 exactly as it did on 22.04.
2. `USE_GTEST OFF` is now **required**. 24.04's GTest 1.14 exports `GTest::GTest` without `IMPORTED_LOCATION`, and TVM's `CMakeLists.txt:402` hard-errors on it (`Neither GTest::GTest nor GTest::gtest targets defined IMPORTED_LOCATION`).
3. CUDA 13 emits a wall of `__VECTOR_TYPE_DEPRECATED__` warnings on `double4`/`float4` (`use double4_16a or double4_32a`). Noise, not errors — TVM does not build with `-Werror`.

#### 2.1.1 Bootstrap from a bare clone

Only needed once per machine (or after a JetPack bump). Everything below is run from the repo root.

```bash
# 0. apt deps (24.04). llvm-18-dev supplies llvm-config-18 + headers.
sudo apt install -y llvm-18-dev cmake build-essential

# 1. Submodules — tvm is our fork (github.com/alansrobotlab2/relax), pinned by
#    the gitlink. --depth 1 is safe: git still checks out the recorded commit.
git submodule update --init --recursive --depth 1
git submodule status --recursive | grep -E "^[-+]" || echo "all at recorded commits"

# 2. Rust — tokenizers-cpp is a Rust crate and mlc-llm's cmake hard-errors
#    ("Cargo is not found!") without it. rustup is user-local, no sudo.
#    24.04's apt rustc is 1.75; rustup stable (1.97 here) is the safe choice.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --profile minimal
export PATH="$HOME/.cargo/bin:$PATH"

# 3. venv + python deps.
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install "huggingface_hub[cli]" ninja \
    "apache-tvm-ffi==0.1.10" ml_dtypes safetensors tqdm requests shortuuid \
    prompt_toolkit fastapi uvicorn openai pandas tiktoken sentencepiece \
    "transformers>=5.6" accelerate numpy cloudpickle psutil typing_extensions pytest

# torch is REQUIRED, but only as a tensor reader: loader/utils.py
# `load_safetensor_shard` does safetensors.safe_open(framework="pt",
# device="cpu"). The CPU wheel is enough and avoids multi-GB CUDA wheels.
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Four dependency traps, all of which abort the pipeline in non-obvious ways:

- **`apache-tvm-ffi` must be pinned to the vendored version — `0.1.10`.** Unpinned, pip takes the newest on PyPI (0.1.12 today), but `3rdparty/tvm/3rdparty/tvm-ffi` is pinned at tag **v0.1.10** (`1fed0ae`). Both the wheel and our build emit `libtvm_ffi.so` with the *same SONAME*, so ld.so loads only the first — the wheel's — while `libtvm.so` was compiled against v0.1.10 headers. The result is not a clean import error but a `std::terminate` during `import tvm`:
  ```
  terminate called after throwing an instance of 'tvm::ffi::Error'
    what():  TypeAttr `__ffi_repr__` is already registered for type index 131.
  ```
  Re-derive the right pin after any TVM bump with
  `git -C 3rdparty/tvm/3rdparty/tvm-ffi describe --tags` (needs `git fetch --tags`; `--depth 1` clones carry none).
- **`pytest` is a runtime dep, not a test dep.** `USE_RPC` is ON in TVM's default config, and `tvm.rpc.testing` → `tvm.testing` → `import pytest` at module import.
- **`torch` is required** even for a pure-safetensors checkpoint — the earlier claim in this doc that it was optional was wrong. CPU-only is sufficient for convert/compile; `validate.py`'s HF reference leg needs the CUDA build from §2.3.1.
- **`accelerate` is required by the parity harness.** transformers 5.x needs it for `device_map`, so `validate.py --reference-only` hard-fails without it. Easy to miss because nothing in the convert/compile/serve path touches it.

TVM build config — write to `3rdparty/tvm/build/config.cmake`:

```cmake
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(USE_CUDA ON)
set(USE_CUBLAS ON)
set(USE_THRUST ON)
set(USE_CUTLASS ON)
set(USE_CUDNN OFF)
set(USE_CURAND OFF)
set(USE_NCCL OFF)
set(USE_NVTX OFF)
set(USE_LLVM "/usr/bin/llvm-config-18 --link-shared")
set(USE_FLASHINFER OFF)   # FlashInfer is JIT'd at mlc_llm-compile time, not linked into TVM
set(USE_GTEST OFF)        # 24.04 GTest 1.14 has no IMPORTED_LOCATION -> configure error
set(CMAKE_CUDA_ARCHITECTURES 87)
```

```bash
# 3. Build TVM (630 ninja edits; the long pole on a fresh machine).
cd 3rdparty/tvm/build
PATH=/usr/local/cuda/bin:$PWD/../../../.venv/bin:$PATH \
  cmake .. -G Ninja -DCMAKE_MAKE_PROGRAM=$PWD/../../../.venv/bin/ninja
PATH=/usr/local/cuda/bin:$PATH ../../../.venv/bin/ninja -j 10
cd -

# 4. Build the mlc-llm C++ runtime against that TVM.
cmake -B build -G Ninja \
    -DTVM_SOURCE_DIR=3rdparty/tvm \
    -DCMAKE_CUDA_ARCHITECTURES=87 \
    -DUSE_CUDA=ON -DUSE_CUTLASS=ON -DUSE_THRUST=ON
ninja -C build
```

Then `source .envrc.local`, which wires the whole thing together:

```bash
export MLC_LLM_HOME=/home/alfie/mlc-llm
export PYTHONPATH="$MLC_LLM_HOME/python:$MLC_LLM_HOME/3rdparty/tvm/python:$PYTHONPATH"
export TVM_LIBRARY_PATH="$MLC_LLM_HOME/3rdparty/tvm/build"   # libtvm.so + libtvm_runtime.so
export MLC_LIBRARY_PATH="$MLC_LLM_HOME/build"                # libmlc_llm.so + libmlc_llm_module.so
export PATH="$MLC_LLM_HOME/.venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
```

`nvcc` must be on `PATH` at *compile* time, not just at build time — the `flashinfer=1` opt JIT-compiles FlashInfer's paged decode/prefill kernels through `tvm.relax.backend.cuda.flashinfer`, which shells out to nvcc.

Carried over from the 22.04 build and still true on 24.04:
- `libtvm.so` in `3rdparty/tvm/build/` is **not** rebuilt by `ninja -C build` — the mlc-llm build has its own copy under `build/tvm/`, while the Python `tvm` package loads from `3rdparty/tvm/build/`. After patching TVM C++, rerun `ninja` in `3rdparty/tvm/build/` separately or the new symbols stay invisible from Python.

**Verification status (2026-07-24): green end-to-end.** The whole chain was rebuilt from a bare clone on 24.04 / JetPack 7.2 / CUDA 13.2 / LLVM 18 and runs: TVM (629 ninja edits, 0 errors) → mlc-llm C++ (213 edits, 0 errors) → convert_weight → gen_config → compile → generation. **No source changes were needed for CUDA 13 or LLVM 18** — the feared Thrust/CUB and CUTLASS API drift did not materialize; the only CUDA-13 output is a wall of `double4`/`float4` deprecation warnings. Every fix was environmental (the config.cmake edits above plus the three dependency traps).

Reference timings on this box (12-core Orin AGX, `-j 10`): TVM ~50 min, mlc-llm C++ ~13 min, plus a 71.9 GB download and ~40 min of model steps.

### 2.2 Download weights

```bash
# 71.9 GB bf16 across 26 shards (1045 tensors). ~35 min unauthenticated on a
# fast link; hf_transfer is not available for aarch64 so this is plain HTTP.
SNAP=$(.venv/bin/hf download Qwen/Qwen3.6-35B-A3B --max-workers 8 | tail -1)

# The command prints the snapshot dir as its last line. To recover it later:
# SNAP=~/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/<rev>/
# (rev 995ad96eacd98c81ed38be0c5b274b04031597b0 as of 2026-07-24)
```

Verify before spending 5 min on conversion — a truncated shard set fails deep inside `convert_weight`:

```bash
.venv/bin/python -c "
import json; i=json.load(open('$SNAP/model.safetensors.index.json'))
print(i['metadata']['total_size']/1e9, 'GB /', len(i['weight_map']), 'tensors')"
# -> 71.903645408 GB / 1045 tensors
```

`hf download` on hub 1.x drops `--local-dir-use-symlinks` (it is cache-only by default), so the older form of this command in prior revisions of this doc no longer parses.

### 2.3 Convert + gen-config + compile

```bash
source .envrc.local   # required — see §2.1.1

.venv/bin/python -m mlc_llm convert_weight "$SNAP" \
    --quantization q4f16_1 \
    -o dist/qwen3_6-35B-A3B-q4f16_1                                # ~11 min, no GPU; 19 GB output

.venv/bin/python -m mlc_llm gen_config "$SNAP" \
    --quantization q4f16_1 --conv-template qwen3_5 \
    -o dist/qwen3_6-35B-A3B-q4f16_1                                # 8 sec; auto-picks model_type=qwen3_5_moe

MLC_MOE_GEMM_V2=1 .venv/bin/python -m mlc_llm compile dist/qwen3_6-35B-A3B-q4f16_1 \
    --device cuda \
    --opt "flashinfer=1;cublas_gemm=1;cudagraph=1;cutlass=1" \
    -o dist/qwen3_6-35B-A3B-q4f16_1/lib.so                         # ~14 min, 200 MB sm_87 lib
```

Measured on the 24.04 / JetPack 7.2 re-bootstrap (2026-07-24):

| step | time | output |
|---|---|---|
| `convert_weight` | 11 min | 18.187 GB, 186 shards, **4.345 bits/param**, 35,951,822,704 params (peak RAM 3.7 GB, 66.97 GB streamed from disk) |
| `gen_config` | 8 sec | `model_type=qwen3_5_moe`, `conv_template=qwen3_5`, ctx 262144, `active_vocab_size` 248320 → 248077 |
| `compile` (FI on) | 14 min | `lib.so`, 199,630,112 B |
| `compile` (FI off) | 13 min | `lib_nofi.so`, 197,474,024 B |

`convert_weight` is quiet for its first ~10 min — it is CPU-bound building the
quantization plan over 1045 source tensors before any progress bar appears. Same
for `compile`'s "Exporting the model to TVM compiler". Neither is hung.

Two flags matter here:
- **`MLC_MOE_GEMM_V2=1`** (env var, Phase 9b) — opts the int4 MoE GEMM into the
  dispatch-table + hand-tensorized wmma m16n8k16 kernel. **2.52× pp512** vs
  the persistent-loop v1 (Stage 9.2 baseline 207.95 → 523.67 tps). Decode
  (b=1) is unaffected — still routes through `dequantize_gemv` shortcut.
- **`flashinfer=1`** (compile opt) — links FlashInfer's paged-decode +
  paged-prefill kernels. **+21 % tg512** at pp=512 KV depth (44.88 → 54.35).
  FlashInfer compiles cleanly on Orin sm_87 since the Phase 6 ABI fix; the
  earlier "FlashInfer cache lacks sm_87" claim is stale (was a JIT issue
  fixed in vendored TVM). `--model-lib` is still recommended at runtime to
  bypass the JIT cache lookup.

  **`flashinfer=1` needs a CUDA-enabled torch** — see §2.3.1. It is not a
  pure compile-time flag.

- **`cublas_gemm=1` is a no-op at `q4f16_1`** and is silently dropped. The
  effective opt string echoed back by the compiler is
  `flashinfer=1;cublas_gemm=0;faster_transformer=0;cudagraph=1;cutlass=1`.
  `OptimizationFlags._cublas_gemm` only honors the flag for
  `q0f16 / q0bf16 / q0f32` or an `e4m3`/`e5m2` quantization, so every
  4-bit build ignores it. Harmless to keep for copy-paste symmetry with the
  q0f16 recipes, but do not credit it for any of the throughput.

- **`cutlass=1` is inert on sm_87** and can be dropped. Earlier revisions of this
  doc called it a "long-standing Orin-tuned flag"; it does nothing here.
  [op/extern.py:43-47](python/mlc_llm/op/extern.py#L43-L47) gates CUTLASS to
  `sm_90a`/`sm_100a`, and `CUTLASS.cmake:59,65` gates the CUDA sources the same
  way, so `tvm_cutlass_objs` comes out empty. Like `cublas_gemm=1` it is harmless
  to keep, but do not credit it for throughput. (`faster_transformer` is
  hard-disabled outright at `extern.py:48`.) **`cudagraph=1` is the real one** —
  §4.7 of [workplan-cuda-13.md](workplan-cuda-13.md) measures only 131 of ~460
  decode launches escaping the graph.

Combined, this lib hits **pp512 = 561.16 / tg512 = 54.35** on the Orin AGX
MAXN bench — past every Phase 9 gate including the 450-tps "parity to
llama.cpp" stretch. (Those figures are the 2026-04 JetPack 6.2.2 numbers. The
JetPack 7.2 / CUDA 13.2 rebuild **has** now been re-benched and is a wash —
see §14.5.)

#### 2.3.1 FlashInfer needs CUDA torch (JetPack 7.2 gap)

`--opt flashinfer=1` does not link a prebuilt kernel library. TVM JITs the
paged decode/prefill kernels through
[tvm/relax/backend/cuda/flashinfer.py](3rdparty/tvm/python/tvm/relax/backend/cuda/flashinfer.py),
which does `from flashinfer.jit import gen_customize_batch_{prefill,decode}_module`.
The `flashinfer-python` package calls `torch.cuda.get_device_properties()` at
**import** time, so a CPU-only torch fails the compile with
`AssertionError: Torch not compiled with CUDA enabled`.

This never came up on 22.04 because JetPack 6's system torch was CUDA-enabled and
visible through the venv's `--system-site-packages`. On JetPack 7.2 there is no
Jetson torch wheel yet — jetson-ai-lab has no `jp7` index (`jp7/cu130` 404s). The
`sbsa/cu130` index works:

```bash
.venv/bin/pip install \
    --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130/+simple \
    --extra-index-url https://pypi.org/simple \
    "torch==2.11.0"
.venv/bin/pip install flashinfer-python     # 0.6.15.post1 as of 2026-07-24
```

That torch reports `arch_list = [sm_80, sm_90, sm_100, sm_110, sm_120]` and warns
that sm_87 has no matching SASS — **which does not matter here**. Torch is used
only to import flashinfer and query device properties; the kernels themselves are
emitted as CUDA source and compiled for sm_87 by nvcc. `torch.cuda.is_available()`
is True and `get_device_properties(0)` correctly returns `Orin sm_87`.

Note this torch cannot *run* GPU ops on an Orin. Nothing in the
convert/compile/serve path needs it to — MLC never calls torch at runtime, and
`convert_weight` uses it only as a CPU tensor reader. But `validate.py`'s
HuggingFace-reference parity harness **does** need working GPU torch, so
re-running §11 parity on JetPack 7.2 is blocked until a real sm_87 wheel exists.

Verify FlashInfer actually linked rather than silently falling back —
`create_flashinfer_paged_kv_cache` swallows `NotImplementedError` and returns `[]`:

```bash
nm -D --defined-only dist/qwen3_6-35B-A3B-q4f16_1/lib.so \
    | grep -ciE "flashinfer|batch_prefill|batch_decode"    # 58  (FI-off lib: 5)
du -sh ~/.cache/flashinfer                                 # 6.2M of JIT'd kernels
```
The size delta alone is not proof: the FI lib is only 2.1 MB larger than the FI-off
lib here (199,630,112 vs 197,474,024 B), not the ~5 MB the 2026-04 build showed.

A FlashInfer-off build of the same dir is kept alongside `lib.so` for
apples-to-apples regression checks. On the 2026-07-24 rebuild that is
`lib_nofi.so` (197,474,024 B), produced by re-running the §2.3 compile with
`flashinfer=0` and a different `-o`. The 2026-04 equivalents
(`lib_phase9b_v2.so`, `lib_phase9_cta1024_pre_v2.so`) were build artifacts and
did not survive the machine rebuild — `dist/` is gitignored.

**Always pass `--model-lib`** when more than one `.so` sits in the model dir. A
harness that globs `*.so` and takes `[0]` can silently pick the FlashInfer-off
lib and deliver ~82 % of headline tg (this exact bug is documented in §14.1).

### 2.4 Run

Interactive chat:

```bash
source .envrc.local && .venv/bin/python -m mlc_llm chat \
    dist/qwen3_6-35B-A3B-q4f16_1 --device cuda:0 \
    --model-lib dist/qwen3_6-35B-A3B-q4f16_1/lib.so
```

`--model-lib` is required — without it the JIT cache lookup re-resolves to the FlashInfer path and segfaults. Engine API:

```python
from mlc_llm import MLCEngine
engine = MLCEngine(
    "dist/qwen3_6-35B-A3B-q4f16_1",
    model_lib="dist/qwen3_6-35B-A3B-q4f16_1/lib.so",
    mode="interactive", device="cuda:0",
)
```

### 2.5 Bench (TG=512 steady-state, MAXN locked)

```bash
source .envrc.local && .venv/bin/python bench_mlc.py \
    --model dist/qwen3_6-35B-A3B-q4f16_1 \
    --pp 128 --tg 512 --runs 3 --warmup 1
```

### 2.6 Known CLI quirks

- There is no `mlc_llm` shell script — nothing is pip-installed, `mlc_llm` is imported off `PYTHONPATH`. Always invoke as `.venv/bin/python -m mlc_llm <cmd>`, and always after `source .envrc.local`.
- `convert_weight` and `gen_config` reject HF repo IDs; pass the local snapshot path (`$SNAP` above).
- `gen_config` prints argparse errors to stdout but the actual error to stderr — when redirecting, keep them separate (`> out 2> err`) or you'll see an empty `Error` block.
- The `qwen3_5` conv template has **thinking enabled** — the assistant prefix opens a `<think>` block, so short `max_tokens` returns reasoning rather than an answer. Use `qwen3_5_nothink` for terse replies, or budget enough tokens to close the block.

### 2.7 Optional variants in dist/

`dist/` is gitignored, so this table describes what the recipes *produce*, not
what is necessarily on disk. After the 2026-07-24 rebuild only the default dir
exists; the kvint8 / kvfp8 / mtp-draft variants need their own convert+compile
passes to be recreated.

| Build | Use case | Notes |
|---|---|---|
| [dist/qwen3_6-35B-A3B-q4f16_1/](dist/qwen3_6-35B-A3B-q4f16_1/) | **default**, max throughput | Phase 9b v2 (MoE dispatch + wmma m16n8k16) + FlashInfer (paged-decode/prefill linked). pp512 = 561.16 / tg512 = 54.35 on Orin AGX (JetPack 6.2.2 numbers; not re-benched after the 7.2 rebuild) |
| [dist/qwen3_6-35B-A3B-q4f16_1_tir/](dist/qwen3_6-35B-A3B-q4f16_1_tir/) | apples-to-apples vs int8 | fp16 KV, FlashInfer hard-disabled in compile flags |
| [dist/qwen3_6-35B-A3B-q4f16_1_kvint8/](dist/qwen3_6-35B-A3B-q4f16_1_kvint8/) | capacity-bound (~2× context) | int8 KV; throughput-neutral but byte-divergent from fp16 (parity 2/5 EXACT, semantic drift only) |
| [dist/qwen3_6-35B-A3B-q4f16_1_kvfp8/](dist/qwen3_6-35B-A3B-q4f16_1_kvfp8/) | reference for sm ≥ 89 port | fp8 KV; -25% on Orin (software fp8 dequant), preserved for Blackwell port |
| [dist/qwen3_6-35B-A3B-q4f16_1-mtp-draft/](dist/qwen3_6-35B-A3B-q4f16_1-mtp-draft/) | spec-decode draft (not default) | EAGLE-style 1-layer MTP head, 0.71 GB; loses to target_only by 17% on Orin |

### 2.8 Compile & Run Qwen3.5-0.8B (dense)

Same toolchain as §2.3, smaller model. The dense variant has no MoE so `MLC_MOE_GEMM_V2` is irrelevant; the rest of the speedup flags (FlashInfer, cudagraph, cutlass, cublas_gemm) all apply. Quant of choice is **`q4f16_g16e`** — group=16, embed/final_fc included, dlight-tuned for sm_87. Compile takes ~3.5 min and the resulting lib is ~39 MB (with FlashInfer kernels linked).

**Headline (2026-04-30, Orin AGX MAXN, this lib):** TG=512 → **134.82 tps**, 1.345× over llama.cpp Q4_K_XL (100.3 tps). Depth-flat: only -4% drift across 16× decode depth. Full sweep:

| tg   | llama.cpp Q4_K_XL (pure tg) | MLC q4f16_g16e + FI | ratio |
|---:|---:|---:|---:|
|  512 | 100.3 | **134.82** | **1.345×** |
| 1024 | 100.1 | **134.29** | **1.341×** |
| 2048 |  99.7 | **133.54** | **1.340×** |
| 4096 |  98.0 | **132.17** | **1.349×** |
| 8192 |  96.5 | **129.59** | **1.343×** |

Median of 3 runs each (1 warmup), pp=512 prefill + tg=N decode, run-to-run variance ≤ 0.05%. Detailed analysis in §14.2.

```bash
# 1. Weights (1.5 GB bf16, ~30 sec on a fast link)
hf download Qwen/Qwen3.5-0.8B
SNAP=~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/<rev>/

# 2. Convert weights (no GPU; 559 MB output)
.venv/bin/python -m mlc_llm convert_weight "$SNAP" \
    --quantization q4f16_g16e \
    -o dist/qwen3_5-0.8B-q4f16_g16e

# 3. Generate engine config (3 sec; auto-picks model_type=qwen3_5)
.venv/bin/python -m mlc_llm gen_config "$SNAP" \
    --quantization q4f16_g16e --conv-template qwen3_5 \
    -o dist/qwen3_5-0.8B-q4f16_g16e

# 4. Compile with all speedups (~3.5 min)
.venv/bin/python -m mlc_llm compile dist/qwen3_5-0.8B-q4f16_g16e \
    --device cuda \
    --opt "flashinfer=1;cublas_gemm=1;cudagraph=1;cutlass=1" \
    -o dist/qwen3_5-0.8B-q4f16_g16e/lib.so
```

Recompile-only path (weights already converted and gen_config already run — the params and `mlc-chat-config.json` stay valid across runtime ABI bumps):

```bash
source .envrc.local && .venv/bin/python -m mlc_llm compile \
    dist/qwen3_5-0.8B-q4f16_g16e --device cuda \
    --opt "flashinfer=1;cublas_gemm=1;cudagraph=1;cutlass=1" \
    -o dist/qwen3_5-0.8B-q4f16_g16e/lib.so
```

Bench (TG=512 steady-state, MAXN locked):

```bash
source .envrc.local && .venv/bin/python bench_mlc.py \
    --model-dir dist/qwen3_5-0.8B-q4f16_g16e --device cuda:0 \
    --pp 512 --tg 512 --runs 3 --warmup 1
```

Run (chat / engine API): identical to §2.4, swap model dir / lib path. `--model-lib` is still recommended at runtime.

---

## 3. Family Map (as of 2026-04)

| Model | Released | Total / Active params | Hybrid layers | Linear heads (K/V) | MoE | mRoPE |
|---|---|---|---|---|---|---|
| Qwen3-Next-80B-A3B | 2025-09 | 80B / 3B | 48 (`[L,L,L,F]×12`) | 16 / 32 | 512 experts, 10 active + 1 shared | yes |
| **Qwen3.5-0.8B** | 2026-03 | 0.8B / dense | 24 (`[L,L,L,F]×6`) | 16 / 16 | dense MLP | no |
| Qwen3.5-2B | 2026-03 | 2B / dense | 28 (`[L,L,L,F]×7`) | 16 / 16 | dense MLP | no |
| Qwen3.5-4B | 2026-03 | 4B / dense | 32 (`[L,L,L,F]×8`) | 16 / 16 | dense MLP | no |
| Qwen3.5-9B | 2026-03 | 9B / dense | 36 (`[L,L,L,F]×9`) | 16 / 16 | dense MLP | no |
| Qwen3.5-27B | 2026-03 | 27B / dense | 40 (`[L,L,L,F]×10`) | 16 / 32 | dense MLP | no |
| **Qwen3.6-27B** | 2026-04 | 27B / dense | 40 (`[L,L,L,F]×10`) | 16 / 32 | dense MLP | yes |
| **Qwen3.6-35B-A3B** | 2026-04 | 35B / ~3B | 40 (`[L,L,L,F]×10`) | 16 / 32 | 256 experts, 8 active + 1 shared | yes |

All share `model_type: qwen3_5` (or `qwen3_5_moe` for the MoE variants); the original Qwen3-Next still uses `model_type: qwen3_next`. Layer counts and head ratios for the 4B/9B/27B dense variants are projected from the published family pattern (`full_attention_interval=4`, scaling depth ~ `√(params)`); confirm against `config.json` once each is brought up.

Canonical HF repos: `Qwen/Qwen3-Next-80B-A3B-Instruct`, `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-{2B,4B,9B,27B}`, `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.6-35B-A3B`.

The 4B/9B/27B dense variants and Qwen3.6-27B are not yet brought up — they are listed here so the gap analysis (§8) can be applied to them when the time comes. They reuse the dense `qwen3_5` module (no MoE fork needed). Qwen3.5-27B and Qwen3.6-27B switch to asymmetric linear heads (16/32), which exercises the same kernel path the 35B-A3B already validates.

---

## 4. Architecture Summary

### 4.1 Hybrid layer pattern

Every fourth layer is full softmax attention; the other three are GatedDeltaNet linear attention. Indexed by `full_attention_interval=4`:

- linear at indices 0, 1, 2 → full at index 3 → linear at 4, 5, 6 → full at 7 → …
- 0.8B: 24 layers → 18 linear, 6 full
- 35B-A3B: 40 layers → 30 linear, 10 full
- 80B-A3B: 48 layers → 36 linear, 12 full

### 4.2 Full-attention layer

Standard GQA, with two notable additions vs. plain Qwen3:
- **Output gate** (`attn_output_gate: true`): the Q projection emits `2 × num_heads × head_dim` floats; half are queries, half are gate values. The attention output is multiplied element-wise by `sigmoid(gate)` before `o_proj`.
- **Partial RoPE** (`partial_rotary_factor: 0.25`): RoPE is applied to only the first 25% of `head_dim`. With `head_dim=256`, that's 64 rotated dims, 192 untouched.
- Per-head Q and K RMSNorm (no bias).
- Head dim 256 across all variants.

### 4.3 GatedDeltaNet linear-attention layer

A delta-rule SSM with gating. Per-token computation, per `value_head`:

```
S_t = g_t · S_{t-1} + β_t · k_t · (v_t - S_{t-1} · k_t)         # state update (delta rule)
o_t = S_t · q_t                                                  # output
```

with:
- `q, k`: `(num_key_heads, key_head_dim)` post-Conv1d, post-SiLU, post-L2-norm
- `v`: `(num_value_heads, value_head_dim)` post-Conv1d, post-SiLU
- `β = sigmoid(b)` (per head)
- `g = exp(-exp(A_log) · softplus(a + dt_bias))` (per head, fp32)
- `S`: `(num_value_heads, key_head_dim, value_head_dim)` recurrent state, **fp32**
- For asymmetric heads (`num_value_heads > num_key_heads`), Q/K are repeated `num_value_heads // num_key_heads` times.

Sub-block layout (HF / vLLM names):
- `in_proj_qkvz` — fused Q+K+V+Z projection (Z is the post-recurrence output gate stream)
- `in_proj_ba` — fused β + α projection
- `conv1d` — depthwise causal convolution, kernel size 4, applied to QKV after `in_proj`
- SiLU after Conv1d
- L2-norm on Q and K
- `A_log`, `dt_bias` — bare tensors (no `.weight` suffix)
- `FusedRMSNormGated` — multiplies by `SiLU(Z)` after the recurrence
- `out_proj` — final output projection

The released 0.8B and 35B-A3B checkpoints both ship the four-projection layout (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`) — confirmed in §8 by `safetensors.index.json`. The loader at [python/mlc_llm/model/qwen35/qwen35_loader.py](python/mlc_llm/model/qwen35/qwen35_loader.py) maps these directly; no fused-`qkvz`/`ba` consolidation in the released configs.

### 4.4 RMSNorm quirk

Qwen3.5 uses `output = norm(x) · (1 + weight)` with weight initialized to 0. TVM `nn.RMSNorm` uses `output = norm(x) · weight`. The loader handles this by adding `1.0` to all standard RMSNorm weights at load time. The gated norm inside GatedDeltaNet (`linear_attn.norm`) does **not** get the `+1.0`.

Affected: `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, top-level `model.norm`.

### 4.5 MoE block (35B-A3B, 80B-A3B) — for Stage 5

- `num_experts=256`, `num_experts_per_tok=8`, plus 1 shared expert
- `moe_intermediate_size=512`, `shared_expert_intermediate_size=512`
- `decoder_sparse_step=1` (every layer is MoE in the MoE variants)
- Routing: softmax over experts, top-k, normalize the chosen probs (`norm_topk_prob=true`)
- Output = `Σ p_i · expert_i(x) + shared_expert(x)`

### 4.6 mRoPE (35B-A3B, 80B-A3B)

`mrope_section: [11, 11, 10]` — head_dim is split into three rotation sub-bands rotated against three position axes (text + spatial). For text-only input, all three sub-bands rotate against the same flattened position, collapsing to standard 1D RoPE. **Stage 6 confirmed** that the existing `RopeMode.NORMAL` in [python/mlc_llm/nn/kv_cache.py](python/mlc_llm/nn/kv_cache.py) produces output identical to the HF mRoPE path on text-only input (4/5 prompts EXACT match; the one diverging step was at the rank-1-vs-rank-2 noise floor). Multimodal (image+text) input still requires explicit `mrope_section` handling — not on the project critical path.

### 4.7 MTP (Multi-Token Prediction) head

Present in the 3.5/3.6 checkpoints (`mtp_num_hidden_layers: 1`). **Phase 4B shipped** as a draft module for spec decode in two flavors:

- [python/mlc_llm/model/qwen35_mtp_draft/](python/mlc_llm/model/qwen35_mtp_draft/) — 0.8B draft (243-line model, 111-line loader). γ=4 lands 120.5 tps with byte-identical parity to target_only on Orin.
- [python/mlc_llm/model/qwen3_5_moe_mtp_draft/](python/mlc_llm/model/qwen3_5_moe_mtp_draft/) — 35B-A3B draft (240-line model, 137-line loader). γ=1 lands 40.8 tps; loses to target_only by 17 % on Orin (BW-bound regime), but the head is correct (96 % step-1 accept rate) and the wiring is robust.

Critical wiring detail (Phase 4B retro): the EAGLE fc head expects `cat([inputs_embeds, hidden_states])`, **not** `cat([h_norm, e_norm])` as the original 0.8B port had. With the wrong order the first half of `fc.weight` (trained for embeddings) reads hidden states and produces 0 % accept. Fix is at [qwen3_5_moe_mtp_draft_model.py:120](python/mlc_llm/model/qwen3_5_moe_mtp_draft/qwen3_5_moe_mtp_draft_model.py#L120) and [qwen35_mtp_draft_model.py:119](python/mlc_llm/model/qwen35_mtp_draft/qwen35_mtp_draft_model.py#L119).

The MTP layer reuses the model's PagedKVCache via `num_attention_layers + config.mtp_num_hidden_layers` — see `create_paged_kv_cache` at [qwen35_model.py:1270](python/mlc_llm/model/qwen35/qwen35_model.py#L1270).

---

## 5. State Layouts

### 5.1 Recurrent state (linear-attention layers only)

Per layer, allocated by `RNNState.create` in [python/mlc_llm/model/qwen35/qwen35_model.py](python/mlc_llm/model/qwen35/qwen35_model.py) (`create_rnn_state`):
- `state_id=0`: recurrent state `S`, shape `(num_value_heads, key_head_dim, value_head_dim)`, dtype **fp32**
- `state_id=1`: Conv1d ring buffer, shape `(kernel_size - 1, qkv_dim)` = `(3, qkv_dim)`, dtype = model dtype

For 0.8B: `state_id=0` is `(16, 128, 128)` fp32 = 1 MB per layer × 18 layers = 18 MB recurrent state per sequence.

### 5.2 Paged KV cache (full-attention layers only)

Standard `PagedKVCache.create_generic` with `attn_kind="mha"`:
- `num_hidden_layers = num_attention_layers + mtp_num_hidden_layers` (the full layers + one slot per MTP layer — 6+1 for 0.8B, 10+1 for 35B-A3B)
- `qk_head_dim = v_head_dim = 256`
- `rope_mode = RopeMode.NORMAL`, `rotary_dim = head_dim · partial_rotary_factor = 64`

`kHybrid` KVStateKind handling lives in the runtime (kept stateful per-layer-type); our model code consumes both objects and the cache layer dispatches by layer index.

### 5.3 KV-cache dtype split (Phase 5 + Phase 6)

The KV pages can be stored in a different dtype from the model activations. Plumbing lives one layer below the model code (model `create_paged_kv_cache` still passes only `dtype`):

- **Python**: `dtype_kv` flows through [python/mlc_llm/nn/kv_cache.py](python/mlc_llm/nn/kv_cache.py) `create_generic` → [python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py](python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py) → [3rdparty/tvm/python/tvm/relax/frontend/nn/llm/kv_cache.py](3rdparty/tvm/python/tvm/relax/frontend/nn/llm/kv_cache.py) `TIRPagedKVCache`. Trailing `rx.StringImm(dtype_kv)` arg into the runtime constructor.
- **C++ runtime** ([3rdparty/tvm/src/runtime/vm/paged_kv_cache.cc](3rdparty/tvm/src/runtime/vm/paged_kv_cache.cc)): page buffer allocated as `dtype_kv`, temp Q/K/V/O still in `dtype`. New `std::vector<Tensor> scales_;` field allocated parallel to `pages_` (full-size fp32 for MHA layers, `{1}` placeholder for linear-attn). Threaded through 5 kernel call sites and 3 MHA virtuals on `PagedPrefillFunc` / `PagedDecodeFunc` / `PagedPrefillTreeMaskFunc` ([attn_backend.h](3rdparty/tvm/src/runtime/vm/attn_backend.h)). FlashInfer overrides ignore scales (no int8/fp8 KV support).
- **TIR kernels**: [_page_kernels.py](3rdparty/tvm/python/tvm/relax/frontend/nn/llm/_page_kernels.py) (per-token quant on write, scales memcpy in copy/compact), [_decode_kernels.py](3rdparty/tvm/python/tvm/relax/frontend/nn/llm/_decode_kernels.py) and [_prefill_kernels.py](3rdparty/tvm/python/tvm/relax/frontend/nn/llm/_prefill_kernels.py) (`T.cast(int8, fp16) * scale` on K/V loads), [tree_attn.py](3rdparty/tvm/python/tvm/relax/frontend/nn/llm/tree_attn.py) (symmetric for spec verify). Scales tensor is **always passed** for signature uniformity; gated on `dtype_kv == "int8"` (Python-time branch). fp16/bf16 paths are byte-identical to before.

Shipped variants in dist/:
- fp16 KV (default): throughput-optimal at short ctx, KV-bound at long ctx
- int8 KV: throughput-neutral, ~2× capacity, 2/5 EXACT parity (semantic drift only)
- fp8 KV: −25 % at tg8192 on Orin (software dequant); preserved as reference for sm ≥ 89 ports

Bench numbers in §14.3.

---

## 6. Reference Implementations

Read-only; do not vendor.

- **HuggingFace transformers** (use ≥ 4.57; earlier releases had a feature-dim bug in `torch_chunk_gated_delta_rule`, HF #40963):
  - `src/transformers/models/qwen3_next/modeling_qwen3_next.py`
  - `src/transformers/models/qwen3_next/modular_qwen3_next.py`
  - `src/transformers/models/qwen3_next/configuration_qwen3_next.py`
- **vLLM**:
  - `vllm/model_executor/models/qwen3_next.py` (mostly wiring)
  - `vllm/model_executor/layers/mamba/gdn_linear_attn.py` (the actual `GatedDeltaNetAttention` layer — has the projection-name details `in_proj_qkvz`, `in_proj_ba`)
- **flash-linear-attention** (canonical kernel reference for cross-checking math):
  - `fla/layers/gated_deltanet.py`
  - `fla/ops/gated_delta_rule/`
- **NVlabs GatedDeltaNet** (ICLR 2025 paper code): https://github.com/NVlabs/GatedDeltaNet
- **vLLM blog**: https://blog.vllm.ai/2025/09/11/qwen3-next.html

---

## 7. Existing MLC Implementation Inventory

[python/mlc_llm/model/qwen35/](python/mlc_llm/model/qwen35/) was added in PR #3449 (Oct 2025) and is the foundation. Phase 4B added the two MTP draft modules; Phase 5 forked qwen3_5_moe.

### 7.1 [qwen35_model.py](python/mlc_llm/model/qwen35/qwen35_model.py) (1419 lines)

| Lines | Component | Notes |
|---|---|---|
| 30–144 | `Qwen35Config` | Reads HF config, handles VLM nesting (`text_config`, `rope_parameters`), exposes `layer_types()`, carries MTP + spec-decode fields |
| 146–150 | `Qwen35Embedding` | Tied lm_head via `lm_head_forward` |
| 152–169 | `Qwen35MLP` | Standard `gate_up_proj` + `down_proj`, SiLU |
| 171–246 | `Qwen35Attention` | GQA + sigmoid output gate. `c_attn` is `2·h_q + 2·h_kv` heads (Q+gate+K+V fused). Per-token small-batch dispatch (Phase 4B B.5/B.6) |
| 248–370 | `create_gated_delta_net_func` | TIR kernel, thread-per-V-column, fp32 state, supports prefill (loop over t) and decode. Phase 4 v6 register-cached state (`state_local` sblock) — 5 GMEM passes collapsed to 2 |
| 372–485 | `create_gated_delta_net_func_with_history` | History-mode variant for spec-decode verify; flushes per-position state into RNNState history slots so rejected tokens can roll back without state corruption |
| 487–882 | `Qwen35GatedDeltaNet` | Sub-projections, Conv1d, L2-norm, gate/beta. `forward` (decode/prefill) and `forward_with_history` (spec verify). Per-token small-batch verify dispatch in 5 sites |
| 884–948 | `Qwen35DecoderLayer` | Dispatches between full and linear by layer type |
| 950–960 | `_Qwen35MTPDecoderLayer` | Wraps a normal decoder layer with the EAGLE-style fused embedding+hidden input |
| 962–1017 | `Qwen35MTPHead` | EAGLE fc head + one decoder layer + norm. **Concat order is `cat([inputs_embeds, hidden_states])`** (Phase 4B critical fix) |
| 1019–1061 | `Qwen35Model` | Decoder stack + final norm |
| 1063–1244 | `Qwen35LMHeadModel` | Top-level + `prefill` / `decode` / `batch_*` / `batch_verify_to_last_hidden_states` (γ-specialized verify entries from Phase 4B) |
| 1246–1268 | `create_rnn_state` | RNNState init: state_id 0 (S, fp32) + state_id 1 (conv buffer, model dtype) + history slots for spec verify |
| 1270–1300 | `create_paged_kv_cache` | Allocates `num_attention_layers + mtp_num_hidden_layers` slots (full layers + MTP layer) |

### 7.2 [qwen35_loader.py](python/mlc_llm/model/qwen35/qwen35_loader.py) (223 lines)

- `hf = "model.language_model"` hard-coded prefix (line 51) — both 0.8B and 35B-A3B are VLMs (architectures: `Qwen3_5MoeForConditionalGeneration` for 35B, equivalent VLM wrap on 0.8B), so the prefix is correct as-is. Vision and MTP weights drop silently via the `named_parameters` walk.
- Fuses HF `q_proj/k_proj/v_proj` → MLC `c_attn`.
- Maps `in_proj_qkv`, `A_log`, `dt_bias` (no `.weight`), `conv1d.weight` → `conv1d_weight`.
- Fuses `gate_proj/up_proj` → `gate_up_proj`.
- Adds `+1.0` to the standard RMSNorm weights; leaves gated `linear_attn.norm` alone.

### 7.3 Stage-5 / Phase-4B siblings

- [python/mlc_llm/model/qwen3_5_moe/](python/mlc_llm/model/qwen3_5_moe/) — Stage-5 MoE fork (703-line model + loader). `Qwen35MoEConfig` extends `Qwen35Config` with MoE + mRoPE fields; `Qwen35MoESparseMoeBlock` mirrors `Qwen2MoeSparseMoeBlock` (router → softmax-topk → cumsum/get_indices → MixtralExperts → moe_sum) plus a sigmoid-gated dense `shared_expert`. Reuses qwen35's GDN + attention path via direct import — zero duplication.
- [python/mlc_llm/model/qwen35_mtp_draft/](python/mlc_llm/model/qwen35_mtp_draft/) — 0.8B EAGLE draft module (Phase 4B).
- [python/mlc_llm/model/qwen3_5_moe_mtp_draft/](python/mlc_llm/model/qwen3_5_moe_mtp_draft/) — 35B-A3B EAGLE draft module (Phase 4B).

### 7.4 Registration

[python/mlc_llm/model/model.py:414-485](python/mlc_llm/model/model.py#L414-L485) registers five entries:
- `qwen3_5` / `qwen3_5_text` → `Qwen35LMHeadModel`
- `qwen35_mtp_draft` → `Qwen35MTPDraftLM`
- `qwen3_5_moe` / `qwen3_5_moe_text` → `Qwen35MoEForCausalLM`
- `qwen3_5_moe_mtp_draft` → `Qwen35MoEMTPDraftLM`

[python/mlc_llm/conversation_template/qwen3_5.py](python/mlc_llm/conversation_template/qwen3_5.py) registers the conversation template (reused by both dense and MoE — chat format is identical).

No model preset for `qwen3_5` in [python/mlc_llm/model/model_preset.py](python/mlc_llm/model/model_preset.py); deferred indefinitely as the `convert_weight + gen_config + compile` flow from §2 is the canonical path.

---

## 8. Gap Table (current vs. target)

Updated 2026-04-25 after inspecting the actual `Qwen/Qwen3.5-0.8B` `config.json` and `model.safetensors.index.json`.

| Capability | qwen35 today | Qwen3.5-0.8B needs | Qwen3.6-35B-A3B needs |
|---|---|---|---|
| Hybrid layer dispatch | ✅ | ✅ | ✅ |
| GatedDeltaNet TIR kernel | ✅ | ✅ | ✅ (verify asymmetric head path) |
| Conv1d state | ✅ | ✅ | ✅ |
| Dual KV state (Paged + RNN) | ✅ | ✅ | ✅ |
| Output-gated GQA | ✅ (hardcoded) | ✅ (config has `attn_output_gate=true`) | ✅ |
| Partial RoPE | ✅ | ✅ (`partial_rotary_factor=0.25`) | ✅ |
| RMSNorm `+1.0` quirk | ✅ | ✅ | ✅ |
| HF prefix `model.language_model.*` | ✅ hardcoded VLM | ✅ **confirmed: 0.8B IS a VLM**, prefix is correct | ✅ (3.6 is multimodal) |
| Tied embeddings (`tie_word_embeddings`) | ✅ supports both | ✅ true | ✅ (probably) |
| `in_proj_qkv/z/a/b` (4 separate weights, not fused `qkvz`/`ba`) | ✅ matches | ✅ confirmed in `safetensors.index.json` | ⚠️ verify for 35B |
| Symmetric linear heads (16/16) | ✅ | ✅ (16/16) | — needs 16/32 |
| Asymmetric head reshape/repeat in kernel callers | partial (`heads_per_group` in kernel) | n/a | needs verification |
| Dense MLP | ✅ | ✅ | needs MoE swap |
| MoE block (256 experts + 1 shared) | ❌ | n/a | needed |
| mRoPE (`mrope_section`) | ❌ | ⚠️ **0.8B has `mrope_section=[11,11,10]` too**, but for text-only inference this reduces to standard 1D RoPE — `RopeMode.NORMAL` should be correct. Verify at Stage 3. | needed (real multimodal) |
| MTP head | ❌ skipped | skip (`mtp_num_hidden_layers=1` in config) | skip in v1 |
| Model preset | ❌ | nice-to-have | nice-to-have |
| Validation harness | ✅ (`validate.py` v1, Stage 0) | ✅ done | reuses 0.8B harness |

### Confirmed 0.8B HF config values

```
hidden_size:           1024
num_hidden_layers:     24
num_attention_heads:   8
num_key_value_heads:   2          (GQA 4:1)
head_dim:              256
vocab_size:            248320
max_position:          262144     (256K context)
tie_word_embeddings:   true
attn_output_gate:      true
partial_rotary_factor: 0.25       (rotary_dim = 64)
rope_theta:            10000000
mrope_section:         [11, 11, 10]   (sums to 32; ×2 = 64 = rotary_dim)
mrope_interleaved:     true
mtp_num_hidden_layers: 1          (skipped)
linear_key_head_dim:   128
linear_value_head_dim: 128
linear_num_key_heads:  16
linear_num_value_heads:16        (symmetric on 0.8B)
linear_conv_kernel_dim: 4
full_attention_interval: 4
```

### Confirmed 0.8B HF weight names (per layer i)

Linear-attention layer (i ∈ {0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22}):
```
model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight   # fused Q+K+V
model.language_model.layers.{i}.linear_attn.in_proj_z.weight     # output gate stream
model.language_model.layers.{i}.linear_attn.in_proj_a.weight     # decay control
model.language_model.layers.{i}.linear_attn.in_proj_b.weight     # update rate β
model.language_model.layers.{i}.linear_attn.out_proj.weight
model.language_model.layers.{i}.linear_attn.conv1d.weight
model.language_model.layers.{i}.linear_attn.norm.weight          # gated RMSNorm (no +1)
model.language_model.layers.{i}.linear_attn.A_log                # NO .weight suffix
model.language_model.layers.{i}.linear_attn.dt_bias              # NO .weight suffix
```

Full-attention layer (i ∈ {3, 7, 11, 15, 19, 23}):
```
model.language_model.layers.{i}.self_attn.q_proj.weight   # 2× Q dim (Q + gate, layout per-head [Q_d, gate_d])
model.language_model.layers.{i}.self_attn.k_proj.weight
model.language_model.layers.{i}.self_attn.v_proj.weight
model.language_model.layers.{i}.self_attn.o_proj.weight
model.language_model.layers.{i}.self_attn.q_norm.weight   # +1 needed
model.language_model.layers.{i}.self_attn.k_norm.weight   # +1 needed
```

All layers also have:
```
model.language_model.layers.{i}.input_layernorm.weight             # +1 needed
model.language_model.layers.{i}.post_attention_layernorm.weight    # +1 needed
model.language_model.layers.{i}.mlp.gate_proj.weight
model.language_model.layers.{i}.mlp.up_proj.weight
model.language_model.layers.{i}.mlp.down_proj.weight
```

Top-level:
```
model.language_model.embed_tokens.weight
model.language_model.norm.weight       # +1 needed
model.visual.*                         # ignored
mtp.*                                  # ignored
```

The existing loader handles all of this correctly (explicit mappings for `in_proj_qkv`/`A_log`/`dt_bias`/`conv1d.weight`, catch-all for `in_proj_z/a/b/out_proj/norm`, RMSNorm `+1` predicate covers exactly the right weights). **No loader changes needed for 0.8B** based on static inspection — the prediction in the original gap table that "loader prefix likely needs fix" was wrong. The loader is correct as-is.

### Confirmed 35B-A3B HF config values (2026-04-25, fetched from `Qwen/Qwen3.6-35B-A3B`)

```
architectures:                Qwen3_5MoeForConditionalGeneration  (multimodal — text under model.language_model.*)
model_type (top):             qwen3_5_moe
model_type (text_config):     qwen3_5_moe_text
hidden_size:                  2048
num_hidden_layers:            40
num_attention_heads:          16
num_key_value_heads:          2          (GQA 8:1)
head_dim:                     256
vocab_size:                   248320
max_position_embeddings:      262144
tie_word_embeddings:          false      (lm_head is a separate weight at top level)
attn_output_gate:             true
partial_rotary_factor:        0.25       (rotary_dim = 64)
rope_theta:                   10000000
mrope_section:                [11, 11, 10]
mrope_interleaved:            true
mtp_num_hidden_layers:        1          (skipped)
linear_key_head_dim:          128
linear_value_head_dim:        128
linear_num_key_heads:         16
linear_num_value_heads:       32         (asymmetric — 2× value heads vs 0.8B's 16/16)
linear_conv_kernel_dim:       4
full_attention_interval:      4          (also explicit `layer_types` array — same pattern)
mamba_ssm_dtype:              float32
moe_intermediate_size:        512
shared_expert_intermediate_size: 512
num_experts:                  256
num_experts_per_tok:          8          (top-8 routing)
decoder_sparse_step:          (not set)  → default 1 in Qwen35MoEConfig (every layer is MoE)
norm_topk_prob:               (not set)  → default true
dtype:                        bfloat16
```

### Confirmed 35B-A3B HF weight names (per layer i)

Linear-attn layers (30 total — same names as 0.8B, but `A_log`/`dt_bias`/`norm` shapes are `[32]`/`[32]`/`[128]` for 32 value heads):
```
model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight   shape=[8192, 2048]   (16·128 + 16·128 + 32·128)
model.language_model.layers.{i}.linear_attn.in_proj_z.weight     shape=[4096, 2048]   (32·128)
model.language_model.layers.{i}.linear_attn.in_proj_a.weight     shape=[32, 2048]
model.language_model.layers.{i}.linear_attn.in_proj_b.weight     shape=[32, 2048]
model.language_model.layers.{i}.linear_attn.out_proj.weight      shape=[2048, 4096]
model.language_model.layers.{i}.linear_attn.conv1d.weight        shape=[8192, 1, 4]
model.language_model.layers.{i}.linear_attn.norm.weight          shape=[128]
model.language_model.layers.{i}.linear_attn.A_log                shape=[32]
model.language_model.layers.{i}.linear_attn.dt_bias              shape=[32]
```

Full-attn layers (10 total at indices 3,7,11,15,19,23,27,31,35,39):
```
model.language_model.layers.{i}.self_attn.q_proj.weight   shape=[8192, 2048]   (2·16·256 with attn_output_gate)
model.language_model.layers.{i}.self_attn.k_proj.weight   shape=[512, 2048]    (2·256)
model.language_model.layers.{i}.self_attn.v_proj.weight   shape=[512, 2048]
model.language_model.layers.{i}.self_attn.o_proj.weight   shape=[2048, 4096]   (16·256)
model.language_model.layers.{i}.self_attn.q_norm.weight   shape=[256]   (+1 needed)
model.language_model.layers.{i}.self_attn.k_norm.weight   shape=[256]   (+1 needed)
```

MoE FFN — every layer (HF pre-stacks experts; no `.weight` suffix on the stacked tensors):
```
model.language_model.layers.{i}.mlp.gate.weight                            shape=[256, 2048]      (router)
model.language_model.layers.{i}.mlp.experts.gate_up_proj                   shape=[256, 1024, 2048]   (pre-stacked, fused gate+up)
model.language_model.layers.{i}.mlp.experts.down_proj                      shape=[256, 2048, 512]
model.language_model.layers.{i}.mlp.shared_expert.gate_proj.weight         shape=[512, 2048]
model.language_model.layers.{i}.mlp.shared_expert.up_proj.weight           shape=[512, 2048]
model.language_model.layers.{i}.mlp.shared_expert.down_proj.weight         shape=[2048, 512]
model.language_model.layers.{i}.mlp.shared_expert_gate.weight              shape=[1, 2048]   (sigmoid gate)
```

Top-level (note `lm_head` is NOT under `model.language_model.`):
```
model.language_model.embed_tokens.weight  shape=[248320, 2048]
model.language_model.norm.weight          shape=[2048]   (+1 needed)
lm_head.weight                            shape=[248320, 2048]   (untied)
model.visual.*                            ignored
mtp.*                                     ignored
```

[python/mlc_llm/model/qwen3_5_moe/qwen3_5_moe_loader.py](python/mlc_llm/model/qwen3_5_moe/qwen3_5_moe_loader.py) handles every line above. The pre-stacked experts pass through directly to `MixtralExperts.weight` since both are `[num_experts, out_features, in_features]` — no `np.stack` or `np.concatenate` needed for the routed experts (the shared expert still needs gate+up fusion, exactly as in qwen2_moe).

---

## 9. Known Pitfalls (from upstream issues + our own scars)

1. **Recurrence dtype** — `mamba_ssm_dtype: float32` in the official config. Keep `A_log.exp()`, `softplus(a + dt_bias)`, and `S` in fp32 even when the model dtype is fp16/bf16. Drift in recurrence is silent and accumulates over long sequences.
2. **State aliasing** — never alias the read and write of recurrent state in the same op. SGLang #20791: a flashinfer `gated_delta_rule_decode_pretranspose` regression was traced to no-buffer scheduling aliasing the in/out state. RNNState's `get`/`set` already returns a fresh tensor — do not optimize that away.
3. **Speculative decode + recurrent state rollback (Phase 4B)** — vLLM #39273 originally argued spec decode rollback on rejected tokens corrupts GDN state. **Solved here** via the history-mode forward path: [create_gated_delta_net_func_with_history](python/mlc_llm/model/qwen35/qwen35_model.py#L372) emits a full per-position state history, [forward_with_history](python/mlc_llm/model/qwen35/qwen35_model.py#L618) writes it via `state.set_with_history(...)`, and on rejection the engine snaps back to the per-position slot. Shipped on both 0.8B (γ=4 byte-identical) and 35B-A3B (γ=1 correct). The dispatch picks history-mode automatically for spec verify; non-spec decode keeps the cheaper non-history path.
4. **Marlin tile sizes** at high TP — vLLM #35924: `MIN_THREAD_N=64` broke `in_proj_ba` whose output is only `num_v_heads` (very narrow). Watch when adding TP > 1 quantized paths.
5. **HF parity reference** — must be transformers ≥ 4.57. Earlier `torch_chunk_gated_delta_rule` had a feature-dim mismatch (HF #40963). Pin in `validate.py`.
6. **Don't lose `attn_output_gate`** — easy to drop when subclassing a vanilla Qwen3 attention. The full layer's `c_attn` width is `(2·h_q + 2·h_kv)·d`, not `(h_q + 2·h_kv)·d`. Hardcoded in `Qwen35Attention`; both 0.8B and 35B-A3B set this true so no config switch is needed.
7. **MTP head concat order (Phase 4B)** — the EAGLE fc head expects `cat([inputs_embeds, hidden_states])`, **not** the reverse. With the wrong order the first half of `fc.weight` (trained for embeddings) reads hidden states → 0 % accept. The original 0.8B port shipped reversed; both drafts are now correct ([qwen35_mtp_draft_model.py:119](python/mlc_llm/model/qwen35_mtp_draft/qwen35_mtp_draft_model.py#L119), [qwen3_5_moe_mtp_draft_model.py:120](python/mlc_llm/model/qwen3_5_moe_mtp_draft/qwen3_5_moe_mtp_draft_model.py#L120)).
8. **Hybrid + FlashInfer RNN-state init (Phase 6 latent fix)** — [cpp/serve/function_table.cc:245-275](cpp/serve/function_table.cc#L245-L275) had RNN-state setup nested inside `if (sliding_window || !flashinfer_defined)`, so hybrid+FlashInfer left `create_rnn_state_func_` null. Bug was masked for months because Phase 5's fp8 lib raised `NotImplementedError` from the FlashInfer dispatch (caught → empty), short-circuiting before the null deref. Phase 6's regression case (fp16, dtype_kv == dtype) included FlashInfer and exposed it. Fix hoists RNN-state setup to its own branch on `kv_state_kind == kHybrid`.
9. **MoE per-token vs batched dispatch (Phase 4B)** — [group_quantization.py:800-811](python/mlc_llm/quantization/group_quantization.py#L800-L811) routes `if indptr.ndim == 2: dequantize_gemv else: dequantize_group_gemm`. The MoE block's `if num_tokens == 1:` resolves to gemv only when `num_tokens` is a literal — for spec-verify with symbolic `seq_len`, it routes to `dequantize_group_gemm` regardless of actual seq_len. group_gemm is ~6× slower than gemv on Orin at small batch. Phase 4B added γ-specialized verify entries (`batch_verify_to_last_hidden_states_g{1,2,3,4}`) to keep the literal-seq path live in 5 sites.
10. **FLA on Blackwell** — fla-org #607 is a backward-pass bug; doesn't affect inference. Listed for context only.

---

## 10. Phase History (everything has shipped)

Correctness phase (the original [.claude/plans/ok-we-re-going-to-squishy-harbor.md](.claude/plans/ok-we-re-going-to-squishy-harbor.md)):

| Stage | Status | Result |
|---|---|---|
| 0 — doc + worklog | ✅ | this file + [worklog.md](worklog.md) |
| 1 — `validate.py` PyTorch reference | ✅ | greedy + per-layer hidden-state dump cached to `reference_outputs.pt` |
| 2 — bring up qwen35 against 0.8B checkpoint | ✅ | compiled and ran first try; the predicted prefix-detection failure didn't materialize (0.8B is a VLM) |
| 3 — per-layer numerical parity | ✅ (skipped) | passed straight to Stage 4 once the model loaded coherently |
| 4 — end-to-end greedy parity (50 × 5 prompts) | ✅ | 50/50 on all 5 prompts, 0.8B |
| 5 — fork qwen3_5_moe | ✅ | [python/mlc_llm/model/qwen3_5_moe/](python/mlc_llm/model/qwen3_5_moe/) |
| 6 — validate 35B-A3B end-to-end | ✅ | 4/5 EXACT, 5th diverged at fp16 noise floor (rank-1-vs-rank-2 logit gap 0.19) |

Perf phase ([.claude/plans/phase2-perf.md](.claude/plans/phase2-perf.md), [phase2c](.claude/plans/phase2c-perf-after-profile.md), [phase2d](.claude/plans/phase2d-ft-hybrid-quant.md), [phase4-perf](.claude/plans/phase4-perf.md)):

| Phase | Status | Result |
|---|---|---|
| 1 — first-pass kernel triage | ✅ | MoE `dequantize_gemv` reachable (CTA grid + static `num_tokens=1`); parallel topk_softmax kernel; 35B-A3B 10.12 → 47.88 tps |
| 2 — sm_87 dlight tuning + GDN register-cached state | ✅ | gdn_func 40.4 → 17.3 µs/call; 35B v6 = 52.62 tps tg64 (1.789× llama.cpp Q4_K_S) |
| 2C — post-profile cleanup | ✅ | `attn_o_proj` and MoE `gate_up` confirmed Pareto-optimal (14-config tile sweep); no further tile gains |
| 2D — FT hybrid quant | ❌ closed | CUDA-graph exclusion eats kernel gains |
| 3 — B-ext spec decode | ❌ dead | 5.2 % token agreement; abandoned in favor of MTP |
| 4B — MTP spec decode | ✅ | EAGLE-style draft for both models. Critical fix: `cat([e, h])` order (was reversed). 0.8B γ=4 = 120.5 tps byte-identical; 35B γ=1 = 40.8 tps correct (loses 17 % to target_only on Orin's BW-bound regime — wins are expected on BW-rich hardware) |
| 5 — fp8 KV cache | ⚠️ shipped, off by default | All plumbing lands; software fp8 dequant on sm_87 costs −25 % at tg8192. Lib at [dist/qwen3_6-35B-A3B-q4f16_1_kvfp8/](dist/qwen3_6-35B-A3B-q4f16_1_kvfp8/) preserved as reference for sm ≥ 89 |
| 6 — int8 KV cache | ✅ | Throughput-neutral on Orin (`cvt.rn.f16.s8` is single-SASS since Pascal). 3/4 land-criteria pass; parity 2/5 EXACT (semantic drift, not catastrophic). Lib at [dist/qwen3_6-35B-A3B-q4f16_1_kvint8/](dist/qwen3_6-35B-A3B-q4f16_1_kvint8/) opt-in for capacity-bound deployments. Latent function_table.cc bug surfaced and fixed (see §9.8) |

Date-stamped detail in [worklog.md](worklog.md).

---

## 11. Acceptance Bars (all met)

Per CLAUDE.md, fp16 throughout (SSM math fp32 internally):

| Stage | Bar | 0.8B result | 35B-A3B result |
|---|---|---|---|
| Embeddings | `atol ≤ 1e-4` (fp32), `atol ≤ 1e-3` (fp16) | ✅ | ✅ |
| Per-layer output, full attention | `rtol = 1e-3`, `atol = 1e-3` (fp16) | ✅ | ✅ |
| Per-layer output, linear attention | `rtol = 2e-3`, `atol = 2e-3` (fp16) | ✅ | ✅ |
| Recurrent state `S` (fp32) | `atol ≤ 1e-4` after Conv1d, `atol ≤ 1e-3` after first recurrence | ✅ | ✅ |
| Greedy decode, 50 tokens | ≥ 48/50 per prompt, 5 fixed prompts | 50/50 × 5 | 50/50 × 4 + 33/50 × 1 (fp16-noise flip at step 34, logit gap 0.19) |

To re-run regressions on the 0.8B (which also smoke-tests the shared qwen35 path used by 35B-A3B):

```bash
source .envrc.local && \
.venv/bin/python validate.py --greedy-parity \
    --model Qwen/Qwen3.5-0.8B \
    --mlc-model-dir dist/qwen3_5-0.8B-q0f16 \
    --device cuda:1
```

Failure mode (preserved for future regressions): dump per-tensor numpy on both sides, diff with `np.testing.assert_allclose`, log `np.abs(a-b).max()` and `argmax(abs(a-b))`. Validate GatedDeltaNet sub-steps in order: post-Conv1d → post-SiLU → post-L2-norm Q/K → β/g values → S after step 1 → output.

---

## 12. Out of Scope (still)

What never shipped and is not on the immediate roadmap:

- **Tensor parallel (TP > 1)** — `Qwen35MoEDecoderLayer` has no `_set_tp` (qwen35's dense base also has none). Would mirror qwen2_moe's pattern. Not needed for single-Orin or single-Blackwell deployment.
- **Real multimodal** — `model.visual.*` weights are dropped silently by the loader; mRoPE is wired only for the text-only collapse (§4.6). Vision tower would need its own module.
- **Engine γ=1 fast path** — replace one b=2 batched verify with two sequential single-token decodes. Math says +14 % over current spec on the 35B-Orin but still doesn't beat target_only. Lateral move; shelved (worklog 2026-04-28 cont. 14).
- **Custom CUDA** — TIR-only convention held throughout. Phase 5/6 added vendored TVM C++ changes (paged_kv_cache.cc, attn_backend.h, codegen_cuda.cc fp8 helpers), but no hand-written CUDA kernels.

What was originally listed as out-of-scope and shipped anyway: kernel perf (Phase 1-2), MTP head (4B), speculative decoding (4B), q4f16_1 quantization (default 35B lib), int8/fp8 KV (5+6).

---

## 13. Repo Conventions to Honor

- **MoE variants get their own module.** `qwen3` / `qwen3_moe`, `qwen2` / `qwen2_moe`, `mistral` / `mixtral`, `deepseek` / `deepseek_v2`. Honored: dense at [python/mlc_llm/model/qwen35/](python/mlc_llm/model/qwen35/), MoE at [python/mlc_llm/model/qwen3_5_moe/](python/mlc_llm/model/qwen3_5_moe/).
- **MTP draft modules also get their own module** (Phase 4B convention): [python/mlc_llm/model/qwen35_mtp_draft/](python/mlc_llm/model/qwen35_mtp_draft/), [python/mlc_llm/model/qwen3_5_moe_mtp_draft/](python/mlc_llm/model/qwen3_5_moe_mtp_draft/).
- Loader files alongside model files; quantization configured declaratively in `model.py` via `make_quantization_functions(...)` — no per-model quantization file unless the model needs `BlockScaleQuantize` or similar.
- Conversation templates in [python/mlc_llm/conversation_template/qwen3_5.py](python/mlc_llm/conversation_template/qwen3_5.py), registered via `ConvTemplateRegistry.register_conv_template`. Reused by both dense and MoE — chat format is identical.
- **`tirx` (not `tir`)** — the repo migrated to `tirx` namespace in PR #3462. Confirm via `from tvm.script import tirx as T` at the top of every model/kernel file.
- **KV-cache dtype split**: when adding new quant paths, plumb through [python/mlc_llm/nn/kv_cache.py](python/mlc_llm/nn/kv_cache.py), [dispatch_kv_cache_creation.py](python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py), and the C++ runtime — see §5.3. Existing model `create_paged_kv_cache` does NOT need changes.

---

## 14. Benchmarks: Unsloth Q4_K_S (llama.cpp) vs MLC compiled

All numbers Orin AGX (sm_87), MAXN power profile, batch=1, no concurrency. Same precision class on both sides: 4-bit weights + fp16 activations (Unsloth `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` ≈ 19.45 GB, MLC `q4f16_1` ≈ 19 GB at 4.345 bits/param). Bench harness: [bench_compare.py](bench_compare.py) drives `llama-bench` (`-p N` for prefill, `-d N -n tg` for decode-at-depth) and `bench_mlc.py` at the same context lengths.

> **Bandwidth: use 156.2 GB/s, not 204.8.** The 204.8 GB/s figure quoted throughout this section is
> the spec sheet (256-bit LPDDR5 @ 3200 MT/s). A native sm_87 read-dominated kernel (128-bit
> vectorized, 2 GiB buffer, far past the 4 MB L2) reaches **156.2 GB/s = 76.3% of spec** —
> [scripts/bw_probe.cu](scripts/bw_probe.cu). Every roofline computed against 204.8 overstates
> available headroom by ~31%. Corrected 35B roofline: 2,946,429,568 active params/token at 4.345
> bits = **1.600 GB/token** → **97.5 tps** achievable-BW roofline (128.0 against spec). Measured
> 54.13 tps = 86.6 GB/s = **55.5% of the achievable wall**, not the 42% a spec-peak comparison
> implies. Reproduce with [scripts/active_params.py](scripts/active_params.py). Note this is a
> *weight* roofline only — §4.6 of [workplan-cuda-13.md](workplan-cuda-13.md) shows another
> 245 MiB/token of recurrent-state copy traffic on top.

### 14.1 Qwen3.6-35B-A3B — bench chart

**Lib:** [dist/qwen3_6-35B-A3B-q4f16_1/lib.so](dist/qwen3_6-35B-A3B-q4f16_1/lib.so) — Phase 9b v2 GEMM + FlashInfer + cudagraph + cutlass, 202 MB.
**Protocol:** `scratch_mlc_tg_sweep.py --pp 512 --tg N --runs 3 --warmup 1`, `prefix_cache_mode="disable"`, `mode="interactive"`. llama.cpp: `llama-bench -pg 512,N -fa 1 -r 3` (pure-tg backed out of the blended `-pg` measurement).
**Headline (tg=512):** **MLC pp=561.5 tps · tg=54.46 tps · 1.927× over llama.cpp Q4_K_XL · 1.866× over llama.cpp Q4_K_S.**

| tg     | MLC q4f16_1 v2+FI | llama.cpp Q4_K_S¹ | llama.cpp Q4_K_XL² | ratio (vs Q4_K_XL) |
|---:|---:|---:|---:|---:|
|  512   | **54.46**         | 29.19              | 28.26              | **1.927×**         |
| 1024   | **54.30**         | 29.30              | 28.21              | **1.925×**         |
| 2048   | **54.07**         | 29.31              | 28.13              | **1.922×**         |
| 4096   | **53.69**         | 29.04              | 28.07              | **1.913×**         |
| 8192   | **53.00**         | 28.48              | 27.86              | **1.902×**         |
| **Δ tg512→tg8192** | **−2.7 %** | **−2.4 %**        | **−1.4 %**        | flat              |

¹ llama.cpp Q4_K_S from the 2026-04-29 sweep against `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`.
² llama.cpp Q4_K_XL (2026-04-30, this session) against `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (20.81 GiB, staged at [models/qwen3.6-35b-a3b/](../models/qwen3.6-35b-a3b/)). Pure tg backed out of `-pg 512,N` blended via `tg_tps = tg / (total/blended − pp/pp_tps)`, with `pp_tps = 632.12 ± 2.46` from a clean warm-cache standalone `pp512` measurement (the original sweep's `pp512` row had cold-cache contamination, ± 318.78). Raw [tuning/lcpp_tg_sweep_35b_20260430_201146.md](tuning/lcpp_tg_sweep_35b_20260430_201146.md). Q4_K_XL is slightly slower than Q4_K_S as expected — XL bumps select tensors to higher bit-widths for a ~+1% perplexity recovery at a ~3% throughput tax.

pp_tps locked at **561.5 ± 0.4** across all 15 reps. Decode is dead flat across 16× depth. Both stacks are weight-BW bound (LPDDR5 ≈ 204 GB/s shared); the earlier "MLC crosses below at long ctx" regression in the historical table is **closed** by FlashInfer linking + Phase 9b v2 GEMM. **The `--model-lib` flag is mandatory for this dist dir** — without it the harness's `glob("*.so")[0]` picks `lib_phase9b_v2.so` (FlashInfer-OFF, 197 MB) and silently delivers ~82% of headline tg. (Patched in [scratch_mlc_tg_sweep.py](scratch_mlc_tg_sweep.py): now prefers `lib.so` and fails loudly when multiple variants exist with no canonical name.)

#### History (for reference; superseded by the shipping chart above)

Initial baseline (2026-04-27, pre-perf-work):

| ctx | llama.cpp Q4_K_S | MLC q4f16_1 | ratio |
|---:|---:|---:|---:|
| 128  | 29.59 | 10.12 | 0.34× |
| 4096 | 28.49 |  8.36 | 0.29× |

Phase 6 fp16 KV TIR (FlashInfer-off, pre-9b):

| pp / tg     | llama.cpp Q4_K_S | MLC q4f16_1 | ratio | note |
|---:|---:|---:|---:|---|
| 128 / 64    | 29.59 | 54.41 | 1.84× | short-ctx win |
| 4096 / 256  | 28.5  | 24.45 | 0.86× | KV-fallback crossover |
| 8192 / 256  | ~28   | 15.88 | 0.57× | structural BW saturation |

Perf-progression (tg512 unless noted):

| version | tg_tps | vs llama.cpp Q4_K_S |
|---|---:|---:|
| Initial baseline (2026-04-27) | 10.12 | 0.34× |
| MoE dispatch fix (gemv reachable) | 44.85 | 1.52× |
| Parallel topk_softmax | 47.88 | 1.629× |
| sm_87 dlight GEMV tuning (v5) | 51.37 | 1.745× |
| gdn_func register-cached state (v6) | 52.62 | 1.789× |
| Phase 9b v2 GEMM (FI-off) | 44.86 | 1.51× |
| **Phase 9b v2 + FlashInfer (shipped)** | **54.46** | **1.866×** |

Phase 9b also lifted **pp512 from 207.95 → 561.5 tps (2.70×)** — the bigger headline of that session.

### 14.2 Qwen3.5-0.8B — bench chart

**Lib:** [dist/qwen3_5-0.8B-q4f16_g16e/lib.so](dist/qwen3_5-0.8B-q4f16_g16e/lib.so) — q4f16_g16e (group=16, embed/final_fc included) + FlashInfer + cudagraph + cutlass + cublas_gemm, 39 MB. Compile recipe in §2.8.
**Protocol:** identical to §14.1 — `scratch_mlc_tg_sweep.py --pp 512 --tg N --runs 3 --warmup 1`; llama.cpp `llama-bench -pg 512,N -fa 1 -r 3` (pure-tg backed out of blended).
**Headline (tg=512):** **MLC pp=2870 tps · tg=134.82 tps · 1.345× over llama.cpp Q4_K_XL.**

| tg     | MLC q4f16_g16e + FI | llama.cpp Q4_K_XL¹ | ratio        |
|---:|---:|---:|---:|
|  512   | **134.82**          | 100.3              | **1.345×**   |
| 1024   | **134.29**          | 100.1              | **1.341×**   |
| 2048   | **133.54**          |  99.7              | **1.340×**   |
| 4096   | **132.17**          |  98.0              | **1.349×**   |
| 8192   | **129.59**          |  96.5              | **1.343×**   |
| **Δ tg512→tg8192** | **−4 %**  | **−4 %**          | flat         |

¹ llama.cpp pure-tg backed out of `-pg 512,N` blended via `tg_tps = tg / (total/blended − pp/pp_tps)` (`pp_tps`=4538). Raw [tuning/lcpp_tg_sweep_0.8b_20260430_165824.md](tuning/lcpp_tg_sweep_0.8b_20260430_165824.md).

Run-to-run variance ≤ 0.05% (tg=4096: three identical samples 132.17 / 132.17 / 132.19). Both stacks weight-BW bound — 522 MiB weights / 204 GB/s ≈ 391 tps theoretical, llama.cpp lands at 25% efficiency, MLC at 34%. The +34% MLC win is the kernel/quant gap (group=16 + dlight + cudagraph + cutlass), not an attention-side win — FlashInfer's role here is keeping the curve depth-flat, not adding raw throughput.

#### History (for reference)

Initial baseline (2026-04-27, q4f16_1 lib, no dlight, no FlashInfer):

| ctx  | llama.cpp Q4_K_S | MLC q4f16_1 | ratio |
|---:|---:|---:|---:|
| 128  | 107.99 | 131.62 | 1.22× |
| 4096 | 102.58 |  63.27 | 0.62× |

This-session lib-config progression (all at pp=512 / tg=512):

| build (this session) | tg_tps | ratio vs Q4_K_XL | note |
|---|---:|---:|---|
| q4f16_2 default compile (no opts) | 99.33 | 0.99× | parity with llama.cpp; misleading first read |
| q4f16_g16e, `flashinfer=0;cudagraph=1;cutlass=1;faster_transformer=1` | 112.82 | 1.13× | dlight-tuned, no FI |
| **q4f16_g16e + `flashinfer=1` (shipped)** | **134.82** | **1.345×** | headline |
| q0f16-mtp + draft (γ=4 spec, 2026-04-28 cont. 12) | 120.5 | 1.20× vs Q4_K_S | superseded by FI build |

The FlashInfer-on `q4f16_g16e` build is now the recommended 0.8B target on Orin; dominates the prior γ=4 spec-decode result on target-only throughput while staying byte-identical to the reference.

### 14.3 KV-cache dtype variants (35B-A3B)

Phase 5 (fp8) and Phase 6 (int8) shipped the dtype-split refactor. fp8 is structurally a loss on Orin (software dequant); int8 is throughput-neutral but byte-divergent from fp16. Apples-to-apples (TIR kv_cache for both, FlashInfer disabled in fp16 lib for fair compare):

| pp / tg | fp16 TIR (tps) | int8 (tps) | int8 vs fp16 |
|---:|---:|---:|---:|
| 128 / 64   | 54.41 | 51.87 | -4.7 % |
| 512 / 256  | 46.34 | 45.43 | -2.0 % |
| 4096 / 256 | 24.45 | 24.06 | -1.6 % |
| 8192 / 256 | 15.88 | 15.63 | -1.6 % |

Parity (5 prompts × 50 tokens, temp=0.0): int8 = 2/5 EXACT, semantic drift only — outputs match for the first 100-155 chars then diverge by 1-2 tokens. Use the int8 lib only for capacity-bound deployments (~2× context for the same VRAM).

### 14.4 Perf protocol (pin this when re-benching)

- TG = 512 steady-state, MAXN locked (`sudo nvpmodel -m 0 && sudo jetson_clocks`), no concurrent processes.
- **`jetson_clocks` is worth ~1.4% and is not optional.** `nvpmodel -m 0` alone leaves the
  `nvhost_podgov` governor in charge; it does reach the 1300.5 MHz ceiling under load, so a
  frequency readout looks identical, but throughput is not: 35B tg512 measured **53.13 unpinned vs
  54.13 pinned**. Coarse frequency sampling hides the loss — do not skip the second command because
  the clocks "look right".
- Run 3 reps + 1 warmup, report median.
- llama.cpp side: `llama-bench -m <gguf> -p <ctx> -n <tg> -r 3 -ngl 99` (or `-d <ctx> -n <tg>` for decode-at-depth).
- MLC side: `bench_mlc.py --pp <ctx> --tg <tg> --runs 3 --warmup 1` with `EngineConfig(prefix_cache_mode="disable")` — see [bench_harness_gotchas.md](.claude/projects/-home-alfie-mlc-llm/memory/bench_harness_gotchas.md) for the three "no GPU activity" deadlock modes.
- Always pass `--model-lib` to MLC's chat/engine — the JIT cache otherwise picks the FlashInfer path and segfaults on sm_87.
- llama.cpp build: `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release` (substitute `120` for Blackwell).

### 14.5 JetPack 7.2 / CUDA 13.2 re-bench (2026-07-24) — parity

Everything above §14.5 was measured on Ubuntu 22.04 / JetPack 6.2.2 / **CUDA 12.6** / LLVM 15. The
box was re-bootstrapped onto Ubuntu 24.04 / JetPack 7.2 / **CUDA 13.2** / LLVM 18 (§2.1.1) and the
whole §14.4 protocol re-run. **The toolchain move is a wash — do not expect a CUDA 13 dividend.**

**Qwen3.6-35B-A3B** (`lib.so`, MoE GEMM v2 + FlashInfer):

| tg | 12.6 tg_tps | **13.2 tg_tps** | delta | 13.2 pp_tps |
|---:|---:|---:|---:|---:|
| 512 | 54.46 | **54.13** | −0.6% | 566.33 |
| 1024 | 54.30 | **54.00** | −0.6% | 566.32 |
| 2048 | 54.07 | **53.83** | −0.4% | 566.32 |
| 4096 | 53.69 | **53.34** | −0.7% | 565.59 |
| 8192 | 53.00 | **52.68** | −0.6% | 565.65 |

pp512 561.5 → **566.33 (+0.9%)**. **Qwen3.5-0.8B** (`q4f16_g16e` + FlashInfer): tg512 134.82 →
**133.47** (−1.0%), flat to **128.16** at tg8192 (was 129.59); pp512 2870 → **2889 (+0.7%)**.

Run-to-run spread across engine loads is ~0.5%, so the decode delta is at the edge of noise.
Prefill marginally up, decode marginally down, depth-flat behaviour preserved. nvcc 13.2 also
reproduces 12.6 codegen on the MoE microbench to within 0.1% (gate_up 1.002 vs 1.0016 ms, down
0.948 vs 0.9495 ms) — but compare only against `baseline_moe_v0_cta1024.json`; `baseline_moe.json`
is the CTA_COUNT=64 config and looks like a 2–3.6× regression that is not one.

Raw: `tuning/mlc_tg_sweep_35b_cuda13_20260724_210926.json`,
`tuning/mlc_tg_sweep_0.8b_cuda13_20260724_213627.json`, `tuning/moe_kernel_cuda13_20260724.json`.

Full per-kernel decode attribution on this stack — which kernels are at the memory wall and which
are not — is in **[workplan-cuda-13.md](workplan-cuda-13.md) §4.6**, reproducible via
[scripts/analyze_decode_trace.py](scripts/analyze_decode_trace.py). Headline: four kernels (37% of
the token budget) are at 88–100% of the 156.2 GB/s wall and are finished; six more (30%) sit at
44–76% and are where the remaining headroom is; GPU idle is only 5.3%.

---

## 15. Open Items

- ~~**Re-bench on JetPack 7.2.**~~ **Done 2026-07-24 — see §14.5.** Verdict: parity, prefill +0.9%, decode −0.6%.
- ~~**Re-run §11 parity on JetPack 7.2 — currently blocked.**~~ **Not blocked; 0.8B passed.** The
  worry was that the `sbsa/cu130` torch has no sm_87 SASS, but **every native kernel runs correctly
  via PTX JIT** and it is a valid correctness oracle — do not "fix" this. `q0f16` greedy parity vs
  HF fp16 is **5/5 prompts × 50/50 tokens**, which also proves the CUDA 13.2 build is numerically
  exact. One dependency gap: `accelerate` is missing from the §2.1.1 pip list and transformers 5.x
  needs it for `device_map` — `validate.py --reference-only` hard-fails without it (`pip install
  accelerate`, 1.14.0 here).
  Do **not** gate on `q4f16_g16e` vs HF fp16 (1/5): that is a 4-bit-vs-fp16 comparison where every
  divergence is a genuine near-tie ("Paris." vs "Paris,", `n <= 1` vs `n == 0`, both reaching 42),
  all outputs coherent, and the high-margin Fibonacci prompt is 50/50. Use `q0f16` for correctness.
- **The 35B-A3B has no parity gate on this hardware.** `--greedy-parity` needs the 72 GB fp16 HF
  reference resident and this box has 61 GB; the recipe at [worklog.md:3226](worklog.md#L3226) was
  run on a Blackwell machine, and `reference_outputs*.pt` is gitignored and was lost in the
  re-bootstrap. Either regenerate the 3.5 kB cache off-box and commit it, or accept coherence-only
  checks on Orin. This blocks any claim that a 35B optimization is correctness-preserving.
- **Bring up Qwen3.5-{4B, 9B, 27B} and Qwen3.6-27B.** 4B/9B reuse the dense `qwen3_5` module unchanged. 27B and 3.6-27B are the first dense models with asymmetric linear heads (16/32) — exercises the kernel path the 35B-A3B already validates.
- **Fresh apples-to-apples bench numbers post-Phase-6 for the 0.8B** — the §14.2 table is the 2026-04-27 baseline. Re-bench with the shipped lib to capture the +21 % from the dlight TX patch (worklog 2026-04-28 cont. 12: 99.5 → 120.5 tps γ=4).
- **Long-context (≥4K) crossover vs llama.cpp.** MLC's TIR PagedKVCache reads at lower effective BW than llama.cpp's `q8_0` KV at long sequence (§14.1). FlashInfer JIT path on sm_87 is the most likely fix lane — bounded probe described in the convo recap.
- **Real multimodal (mRoPE + vision tower).** Text-only collapse works; full multimodal needs `RopeMode` extension (or inline RoPE) + a vision module — non-trivial lift, only when needed.
- **Blackwell port + bench.** The 35B + spec-decode infrastructure is already ready for BW-rich hardware; on Orin spec loses to target_only by 17 % (BW-bound), but the math says it should win on hardware with a wider BW budget. Verify by porting + benching.
