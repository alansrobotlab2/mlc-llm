#!/usr/bin/env python3
"""moe_dump_cuda.py — emit the v2 MoE kernels' generated CUDA, for byte-diffing.

Any change to `moe_matmul.py` that is *supposed* to leave the shipped configuration alone
should leave the generated source alone too. Timing evidence cannot show that (a 0.3%
delta is indistinguishable from noise), and the exactness gates cannot either — they
compare outputs, not code. This dumps the text so a change can be shown inert:

    python scripts/moe_dump_cuda.py --out /tmp/after.cu
    git stash push -- python/mlc_llm/op/moe_matmul.py
    python scripts/moe_dump_cuda.py --out /tmp/before.cu
    git stash pop
    for f in before after; do sed -E 's/cse_v[0-9]+/cse_vN/g' /tmp/$f.cu > /tmp/$f.norm.cu; done
    diff /tmp/before.norm.cu /tmp/after.norm.cu

**Normalize `cse_vN` before diffing.** TVM's common-subexpression numbering is not stable
run to run — two dumps of the *identical* source differ by ~14 lines of pure renaming — so
a raw diff reports changes that are not changes. Always dump twice from one source first
and confirm the control diff is empty after normalizing; that is what makes the real
comparison meaningful.

Defaults to the shipped configuration, which is the one the claim is about.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
os.environ.setdefault("MLC_MOE_GEMM_V2", "1")
os.environ.setdefault("MLC_MOE_GEMM_V2_SKIPPAD", "1")

import tvm  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from moe_gemm_check import SHAPES  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=4096)
    cli = p.parse_args()

    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)

    # Rebuild the same module `build()` makes, but keep the compiled artifact so its
    # device source can be read out.
    from tvm import relax
    from tvm.s_tir import dlight as dl
    from tvm.relax.frontend import nn
    from tvm.relax.frontend.nn import spec
    import mlc_llm.op.moe_matmul as mm
    from moe_gemm_check import GROUP_SIZE, NE

    chunks = []
    for name, (N, K) in SHAPES.items():
        class _Mod(nn.Module):
            def forward(self, x, w, scale, indptr):
                return mm.dequantize_group_gemm(
                    x, w, scale, indptr, quantize_dtype="int4",
                    indptr_dtype="int32", group_size=GROUP_SIZE,
                )

        mod, _ = _Mod().export_tvm(spec={"forward": {
            "x": spec.Tensor([cli.batch, K], "float16"),
            "w": spec.Tensor([NE, N, K // 8], "uint32"),
            "scale": spec.Tensor([NE, N, K // GROUP_SIZE], "float16"),
            "indptr": spec.Tensor([NE + 1], "int32")}})
        from mlc_llm.compiler_pass.fuse_dequantize_transpose import FuseDequantizeTranspose
        from mlc_llm.compiler_pass.fuse_transpose_matmul import FuseTransposeMatmul
        from mlc_llm.compiler_pass.fuse_dequantize_matmul_ewise import FuseDequantizeMatmulEwise
        from mlc_llm.compiler_pass.low_batch_specialization import LowBatchGemvSpecialize
        with target:
            mod = FuseDequantizeTranspose()(mod)
            mod = FuseTransposeMatmul()(mod)
            mod = relax.transform.LegalizeOps()(mod)
            mod = relax.transform.AnnotateTIROpPattern()(mod)
            mod = relax.transform.FoldConstant()(mod)
            mod = relax.transform.FuseOps()(mod)
            mod = relax.transform.FuseTIR()(mod)
            mod = FuseDequantizeMatmulEwise()(mod)
            mod = LowBatchGemvSpecialize()(mod)
            mod = dl.ApplyDefaultSchedule(
                dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(), dl.gpu.Fallback())(mod)
        built = relax.build(mod, target=target)
        # Walk the import tree to the device module; the accessor names differ
        # between tvm_ffi versions, so find the CUDA source by searching.
        def _device_source(mod, depth=0):
            if depth > 4:
                return None
            try:
                src = mod.inspect_source()
            except Exception:
                src = None
            if src and "__global__" in src:
                return src
            subs = getattr(mod, "imports", [])
            for sub in (subs() if callable(subs) else subs):
                got = _device_source(sub, depth + 1)
                if got:
                    return got
            return None

        src = _device_source(built.mod)
        if src is None:
            raise SystemExit("could not locate the generated CUDA source in the module tree")
        chunks.append(f"// ==================== {name} N={N} K={K} ====================\n{src}")

    with open(cli.out, "w") as fh:
        fh.write("\n".join(chunks))
    total = sum(len(c) for c in chunks)
    print(f"wrote {cli.out}: {total} chars across {len(chunks)} shapes")


if __name__ == "__main__":
    main()
