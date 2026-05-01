"""Minimal test: can we build a *scalar* fp16 matmul to CUDA via tirx.build?

Confirms that the spike's BuildCUDA failure is specific to the dlight-tensorized
body, not a general problem with our build path.
"""
from __future__ import annotations
import numpy as np
import tvm
from tvm import s_tir
from tvm.s_tir import dlight as dl
from tvm.script import tirx as T


M, N, K = 256, 256, 256

@T.prim_func(private=True)
def matmul_unscheduled(
    X: T.Buffer((M, K), "float16"),
    W: T.Buffer((N, K), "float16"),
    O: T.Buffer((M, N), "float16"),
):
    T.func_attr({"tirx.noalias": True})
    for i, j, k in T.grid(M, N, K):
        with T.sblock("compute"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                O[vi, vj] = T.float16(0.0)
            O[vi, vj] = O[vi, vj] + X[vi, vk] * W[vj, vk]


def main():
    target = tvm.target.Target({
        "kind": "cuda",
        "arch": "sm_87",
        "max_threads_per_block": 1024,
        "max_shared_memory_per_block": 49152,
        "thread_warp_size": 32,
    })
    dev = tvm.cuda(0)

    print("=== Test 1: from_expr + global_symbol, no dlight ===")
    f = matmul_unscheduled.with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(f)
    sch = s_tir.Schedule(mod)
    blk = sch.get_sblock("compute")
    i, j, k = sch.get_loops(blk)
    sch.bind(i, "blockIdx.x")
    sch.bind(j, "threadIdx.x")
    try:
        rt = tvm.tirx.build(sch.mod["main"], target=target)
        print("  BUILD OK (manual schedule, from_expr)")
    except Exception as e:
        print(f"  BUILD FAILED: {type(e).__name__}: {str(e).splitlines()[-1][:160]}")

    print("\n=== Test 2: from_expr + global_symbol, dlight Fallback ===")
    f = matmul_unscheduled.with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(f)
    with target:
        mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
    try:
        rt = tvm.tirx.build(mod["main"], target=target)
        print("  BUILD OK")
    except Exception as e:
        print(f"  BUILD FAILED: {type(e).__name__}: {str(e).splitlines()[-1][:160]}")

    print("\n=== Test 3: from_expr + global_symbol, dlight Matmul tensorize ===")
    f = matmul_unscheduled.with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(f)
    with target:
        mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(mod)
    body = str(mod["main"])
    print(f"  has wmma: {'wmma' in body}; has mma_sync: {'mma_sync' in body}")
    try:
        rt = tvm.tirx.build(mod["main"], target=target)
        print("  BUILD OK")
    except Exception as e:
        print(f"  BUILD FAILED: {type(e).__name__}: {str(e).splitlines()[-1][:160]}")


if __name__ == "__main__":
    main()
