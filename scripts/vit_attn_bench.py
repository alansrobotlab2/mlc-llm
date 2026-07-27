#!/usr/bin/env python3
"""vit_attn_bench.py — the Qwen3.5-VL vision tower's attention block, alone.

Workplan item 0p. §20.2 measured the tower's three big kernels end to end; against the
**measured** 184.8 GB/s wall (§20.5 -- *not* the 156 GB/s used elsewhere in the
workplan, which is the MoE's strided-access figure) two of them are 4-5x off their
bound and the third is already on it:

    QK^T + scale   9.46 ms/layer   35.9 GB/s effective   4.9x off (compute bound)
    softmax        4.04 ms/layer  182.0 GB/s effective   1.0x off  <- at the wall
    P&V + cast     7.03 ms/layer   45.1 GB/s effective   3.8x off (compute bound)

Two matmuls that are neither bandwidth- nor compute-bound are a *schedule* problem,
and iterating on one through a 3.5-minute whole-model compile is the wrong loop. This
rebuilds just the attention block from `qwen3_vl_vit.py`'s ops, runs the same passes
and the same dlight schedules, and times each generated PrimFunc on its own.

Two hypotheses it was built to test, and what they returned:

  `--static`     is the gap because `seq_len` is symbolic (`num_patches`) so dlight
                 tiles blind?  **Largely no** -- worth 6.5%, not 4x.
  `--prescale`   `matmul(q,k^T)*c == matmul(q*c, k^T)`. Moves the scale off a 305 MB
                 tensor onto a 7.7 MB one. **Worth nothing on its own** (the fused
                 kernel was already getting the multiply free) -- and that null
                 result is the point: it proves the fusion buys nothing, so the
                 matmul can be handed to cuBLAS for free. That is the 70 ms
                 (§20.5), and this bench is how it was found before a compile.

    python scripts/vit_attn_bench.py                 # symbolic, as compiled today
    python scripts/vit_attn_bench.py --static --prescale
    python scripts/vit_attn_bench.py --dump-cuda /tmp/attn.cu

Fidelity bar: the symbolic leg must reproduce §20.2's per-layer numbers. It does, to
**1.3%**. If it ever stops, this instrument is measuring something else and nothing
built on it is safe -- the same trap §17.9 and §18.2 record for the other two benches
in this repo.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import tvm  # noqa: E402
import tvm.tirx  # noqa: E402
from tvm import relax  # noqa: E402
from tvm.relax.frontend import nn  # noqa: E402
from tvm.relax.frontend.nn import op  # noqa: E402
from tvm.relax.frontend.nn import spec  # noqa: E402
from tvm.s_tir import dlight as dl  # noqa: E402

# The cat fixture at the tower's native patch size, and the tower's own config.
SEQ = 2520
HEADS = 12
HEAD_DIM = 64
LAYERS = 12
SCALING = HEAD_DIM ** -0.5


class VisionAttnMath(nn.Module):
    """qwen3_vl_vit.py:161-171, verbatim in ops.

    Starts after the rotary application and ends before the out-projection, which is
    exactly the span §20.2's three kernels cover. fp32 throughout is not a choice --
    fp16 collapses tower parity (max diff 2.03 / rel 39%), see the source comment.
    """

    prescale = False
    pv_cast = True

    def forward(self, q, k, v):
        q32 = op.astype(q, "float32")
        k32 = op.astype(k, "float32")
        v32 = op.astype(v, "float32")
        k_t = op.permute_dims(k32, axes=[0, 2, 1])
        scale = nn.Tensor.from_const(np.array(SCALING, dtype="float32"))
        if self.prescale:
            # matmul(q, k^T) * c  ==  matmul(q * c, k^T). Algebraically identical,
            # but the scale moves off a (h, s, s) = 305 MB tensor and onto a
            # (h, s, d) = 7.7 MB one -- 40x less data touched for the same result.
            q32 = op.multiply(q32, scale)
            attn_scores = op.matmul(q32, k_t)
        else:
            attn_scores = op.matmul(q32, k_t)
            attn_scores = op.multiply(attn_scores, scale)
        attn_probs = op.softmax(attn_scores, axis=-1)
        attn_out = op.matmul(attn_probs, v32)
        if not self.pv_cast:
            # Item 0q: price the `astype` epilogue by removing it. Dropping the cast
            # leaves `matmul` bare, so the delta against the fused `matmul_cast1` is
            # exactly what the fusion buys -- the same question §20.5 asked of the QK
            # matmul's `multiply`, where the answer was 0.00 ms.
            #
            # NOT an equivalent graph (the output stays fp32), and not a candidate
            # change on its own. It is a measurement of the fusion's value; the real
            # change, if this comes back ~free, is to move the cast later so cuBLAS
            # can take the matmul.
            return attn_out
        return op.astype(attn_out, "float16")


def build(target, seq, static: bool, prescale: bool = False, pv_cast: bool = True,
          cublas: bool = False):
    from mlc_llm.compiler_pass.fuse_transpose_matmul import FuseTransposeMatmul

    s = seq if static else "num_patches"
    m = VisionAttnMath()
    m.prescale = prescale
    m.pv_cast = pv_cast
    mod, _ = m.export_tvm(
        spec={
            "forward": {
                "q": spec.Tensor([HEADS, s, HEAD_DIM], "float16"),
                "k": spec.Tensor([HEADS, s, HEAD_DIM], "float16"),
                "v": spec.Tensor([HEADS, s, HEAD_DIM], "float16"),
            }
        }
    )
    with target:
        if cublas:
            # Same position as the real pipeline (pipeline.py:126, before FuseOps).
            from mlc_llm.compiler_pass.blas_dispatch import BLASDispatch
            mod = BLASDispatch(target)(mod)
        mod = FuseTransposeMatmul()(mod)
        mod = relax.transform.LegalizeOps()(mod)
        mod = relax.transform.AnnotateTIROpPattern()(mod)
        mod = relax.transform.FoldConstant()(mod)
        mod = relax.transform.FuseOps()(mod)
        mod = relax.transform.FuseTIR()(mod)
        mod = dl.ApplyDefaultSchedule(
            dl.gpu.Matmul(), dl.gpu.GEMV(), dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(), dl.gpu.Fallback(),
        )(mod)
    return mod


def _alloc_shape(dev, shape, dtype):
    arr = (np.random.rand(*shape).astype(dtype) - 0.5) * 0.1
    return tvm.runtime.tensor(arr, device=dev)


def _alloc(dev, buf, seq):
    """Random device tensor for a PrimFunc buffer, binding the symbolic dim to `seq`."""
    shape = [int(x) if isinstance(x, tvm.tirx.IntImm) else seq for x in buf.shape]
    dtype = buf.dtype
    if dtype.startswith("float"):
        # Small values keep softmax's exp in range; magnitude is irrelevant to timing.
        arr = (np.random.rand(*shape).astype(dtype) - 0.5) * 0.1
    else:
        arr = np.zeros(shape, dtype=dtype)
    return tvm.runtime.tensor(arr, device=dev)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=int, default=SEQ)
    p.add_argument("--static", action="store_true", help="pin seq_len to a literal")
    p.add_argument("--prescale", action="store_true", help="scale q before the matmul, not the scores after")
    p.add_argument("--no-pv-cast", action="store_true", help="item 0q: drop the astype epilogue on P@V to price its fusion")
    p.add_argument("--cublas", action="store_true", help="run BLASDispatch, then time end-to-end on the VM")
    p.add_argument("--dump-cuda", default=None)
    p.add_argument("--repeat", type=int, default=20)
    cli = p.parse_args()

    dev = tvm.cuda(0)
    target = tvm.target.Target.from_device(dev)
    mod = build(target, cli.seq, cli.static, cli.prescale, not cli.no_pv_cast, cli.cublas)

    prim = {gv.name_hint: f for gv, f in mod.functions.items()
            if isinstance(f, tvm.tirx.PrimFunc)}
    leg = ("static" if cli.static else "symbolic") + (" +prescale" if cli.prescale else "") + (" -pvcast" if cli.no_pv_cast else "") + (" +cublas" if cli.cublas else "")
    print(f"[vit-attn] seq={cli.seq}  {leg}  target={target.kind.name}")
    print(f"[vit-attn] {len(prim)} PrimFuncs: {', '.join(sorted(prim))}")
    print()

    if cli.dump_cuda:
        built = relax.build(mod, target=target)

        def _src(m, depth=0):
            if depth > 4:
                return None
            try:
                s = m.inspect_source()
            except Exception:
                s = None
            if s and "__global__" in s:
                return s
            subs = getattr(m, "imports", [])
            for sub in (subs() if callable(subs) else subs):
                got = _src(sub, depth + 1)
                if got:
                    return got
            return None

        src = _src(built.mod)
        if src is None:
            raise SystemExit("could not locate generated CUDA in the module tree")
        with open(cli.dump_cuda, "w") as fh:
            fh.write(src)
        print(f"[vit-attn] CUDA -> {cli.dump_cuda} ({len(src)} bytes)")
        print()

    total = 0.0
    print(f"{'kernel':42s} {'ms/layer':>9s} {'x12':>8s}")
    print("-" * 62)
    for name in sorted(prim):
        f = prim[name]
        try:
            built = tvm.compile(f.with_attr("global_symbol", name), target=target)
        except Exception as e:  # noqa: BLE001
            print(f"{name:42s}   build failed: {str(e)[:40]}")
            continue
        args = [_alloc(dev, bf, cli.seq) for bf in f.buffer_map.values()]
        ev = built.mod.time_evaluator(name, dev, number=1, repeat=cli.repeat)
        ms = ev(*args).median * 1e3
        total += ms
        print(f"{name:42s} {ms:9.2f} {ms * LAYERS:8.1f}")
    print("-" * 62)
    print(f"{'TOTAL':42s} {total:9.2f} {total * LAYERS:8.1f} ms/image_embed")
    print()
    # End-to-end on the VM. PrimFunc-by-PrimFunc timing cannot see a cuBLAS offload
    # (it is a Codegen extern, not a PrimFunc), so this is the only comparable number
    # across the --cublas legs.
    ex = relax.build(mod, target=target)
    vm = relax.VirtualMachine(ex, device=dev)
    args = [_alloc_shape(dev, [HEADS, cli.seq, HEAD_DIM], "float16") for _ in range(3)]
    for _ in range(3):
        vm["forward"](*args)
    dev.sync()
    vms = vm.module.time_evaluator("forward", dev, number=1, repeat=cli.repeat)(
        *args).median * 1e3
    print(f"{'end-to-end (VM)':42s} {vms:9.2f} {vms * LAYERS:8.1f} ms/image_embed")
    print()
    print("§20.2 measured, for comparison:  QK^T 9.46 + softmax 4.04 + P@V 7.03 "
          f"= 20.53 ms/layer, 246.3 ms/iter")


if __name__ == "__main__":
    main()
