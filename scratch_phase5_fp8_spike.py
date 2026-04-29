"""Phase 5.1 spike: prove fp8_e4m3fn -> fp16 cast + scale lowers cleanly on sm_87.

Mimics the decode kernel page-load pattern in miniature:
  pages : float8_e4m3fn[N, D]
  scale : float32[1]
  out   : float16[N, D]
  out[i, j] = T.cast(pages[i, j], "float16") * T.cast(scale[0], "float16")

If TVM's CUDA codegen emits valid PTX for sm_87 and the kernel produces values
matching a host-side reference, Phase 5 has a green light.
"""
import ml_dtypes
import numpy as np
import tvm
from tvm.script import tirx as T


N = 64
D = 128


@tvm.script.ir_module
class FP8DequantSpike:
    @T.prim_func
    def main(
        pages_handle: T.handle,
        scale_handle: T.handle,
        out_handle: T.handle,
    ):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        pages = T.match_buffer(pages_handle, (N, D), "float8_e4m3fn")
        scale = T.match_buffer(scale_handle, (1,), "float32")
        out = T.match_buffer(out_handle, (N, D), "float16")
        for bx in T.thread_binding(N, thread="blockIdx.x"):
            for tx in T.thread_binding(D, thread="threadIdx.x"):
                with T.sblock("compute"):
                    out[bx, tx] = T.cast(pages[bx, tx], "float16") * T.cast(
                        scale[0], "float16"
                    )


def main():
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_87"})
    print(f"target: {target}")
    print(f"compiling FP8DequantSpike...")

    mod = FP8DequantSpike
    try:
        rt_mod = tvm.tirx.build(mod, target=target)
    except Exception as e:
        print(f"!!! COMPILE FAILED: {e}")
        raise

    print("compile OK; rt_mod type =", type(rt_mod).__name__)
    print("imports:", list(rt_mod.imports))
    for sub in rt_mod.imports:
        print(f"  sub kind: {sub.kind}")
        try:
            src = sub.inspect_source()
            print("--- generated CUDA source ---")
            print(src)
            print("--- end CUDA source ---")
        except Exception as e:
            print(f"(inspect_source failed: {e})")

    # Build inputs.
    rng = np.random.default_rng(42)
    pages_fp32 = rng.normal(0, 0.5, size=(N, D)).astype(np.float32)
    pages_fp32 = np.clip(pages_fp32, -448.0, 448.0)  # e4m3fn max
    pages_fp8_np = pages_fp32.astype(ml_dtypes.float8_e4m3fn)
    scale_fp32 = np.array([0.125], dtype=np.float32)

    dev = tvm.cuda(0)
    pages_tvm = tvm.runtime.tensor(pages_fp8_np, device=dev)
    scale_tvm = tvm.runtime.tensor(scale_fp32, device=dev)
    out_tvm = tvm.runtime.tensor(np.zeros((N, D), np.float16), device=dev)

    print("running kernel...")
    rt_mod["main"](pages_tvm, scale_tvm, out_tvm)
    dev.sync()

    # Reference: pages cast to fp8 (lossy quant), back to fp32, * scale.
    pages_round_fp32 = pages_fp8_np.astype(np.float32)
    ref = (pages_round_fp32 * scale_fp32[0]).astype(np.float16).astype(np.float32)
    got = out_tvm.numpy().astype(np.float32)

    abs_err = np.max(np.abs(got - ref))
    rel_err = np.max(np.abs(got - ref) / (np.abs(ref) + 1e-9))
    print(f"max abs err: {abs_err:.6e}")
    print(f"max rel err: {rel_err:.6e}")
    print(f"ref[0,:8]: {ref[0, :8]}")
    print(f"got[0,:8]: {got[0, :8]}")
    assert abs_err < 1e-3, f"abs err too large: {abs_err}"
    print("\n*** 5.1 SPIKE PASS: fp8 -> fp16 cast lowers cleanly on sm_87 ***")


if __name__ == "__main__":
    main()
