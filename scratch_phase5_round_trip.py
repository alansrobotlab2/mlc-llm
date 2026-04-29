"""Phase 5.4 round-trip: append (fp16 K/V -> fp8 pages) + debug_get_kv (fp8 -> fp16).

Build the patched kernels standalone (no model), drive them with toy inputs,
verify the output recovers the input within fp8 quantization noise.
"""
import ml_dtypes
import numpy as np
import tvm

from tvm.relax.frontend.nn.llm._page_kernels import (
    _kv_cache_transpose_append,
    _kv_cache_debug_get_kv,
)


def main():
    # Toy shapes matching Qwen3.6-35B-A3B's full-attn layer geometry.
    num_layers = 1
    num_kv_heads = 2
    head_dim = 256
    page_size = 16
    num_pages = 4
    ntoken = 8
    dtype = "float16"
    dtype_kv = "float8_e4m3fn"

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_87"})

    # Build append kernel.
    append_fn = _kv_cache_transpose_append(num_kv_heads, head_dim, dtype, page_size=page_size, dtype_kv=dtype_kv)
    print("append: built")

    # Build debug-get kernel.
    debug_get_fn = _kv_cache_debug_get_kv(num_layers, num_kv_heads, head_dim, dtype, dtype_kv=dtype_kv)
    print("debug_get: built")

    rt_append = tvm.tirx.build(append_fn, target=target)
    rt_get = tvm.tirx.build(debug_get_fn, target=target)
    print("compile OK")

    # Inputs (fp16 K/V data).
    rng = np.random.default_rng(7)
    k_fp16 = rng.normal(0, 1.0, size=(ntoken, num_kv_heads, head_dim)).astype(np.float16)
    v_fp16 = rng.normal(0, 1.0, size=(ntoken, num_kv_heads, head_dim)).astype(np.float16)
    # Clip to fp8 range to keep the test sane.
    k_fp16 = np.clip(k_fp16, -448, 448).astype(np.float16)
    v_fp16 = np.clip(v_fp16, -448, 448).astype(np.float16)
    position_map = np.arange(ntoken, dtype=np.int32)  # token i -> position i

    dev = tvm.cuda(0)
    pages_fp8 = np.zeros((num_pages, 2, num_kv_heads, page_size, head_dim), dtype=ml_dtypes.float8_e4m3fn)
    pages_tvm = tvm.runtime.tensor(pages_fp8, device=dev)
    k_tvm = tvm.runtime.tensor(k_fp16, device=dev)
    v_tvm = tvm.runtime.tensor(v_fp16, device=dev)
    pos_tvm = tvm.runtime.tensor(position_map, device=dev)

    rt_append["tir_kv_cache_transpose_append"](pages_tvm, k_tvm, v_tvm, pos_tvm)
    dev.sync()

    # Read back via debug-get.
    pos_back = np.arange(ntoken, dtype=np.int32)
    pos_back_tvm = tvm.runtime.tensor(pos_back, device=dev)
    k_out = tvm.runtime.tensor(np.zeros((num_layers, ntoken, num_kv_heads, head_dim), np.float16), device=dev)
    v_out = tvm.runtime.tensor(np.zeros((num_layers, ntoken, num_kv_heads, head_dim), np.float16), device=dev)
    rt_get["tir_kv_cache_debug_get_kv"](pages_tvm, pos_back_tvm, k_out, v_out, 0)
    dev.sync()

    k_back = k_out.numpy()[0]
    v_back = v_out.numpy()[0]

    # Reference: ml_dtypes round-trip (fp16 -> fp8 -> fp16).
    k_ref = k_fp16.astype(ml_dtypes.float8_e4m3fn).astype(np.float16)
    v_ref = v_fp16.astype(ml_dtypes.float8_e4m3fn).astype(np.float16)

    k_err = np.max(np.abs(k_back.astype(np.float32) - k_ref.astype(np.float32)))
    v_err = np.max(np.abs(v_back.astype(np.float32) - v_ref.astype(np.float32)))
    print(f"K round-trip max abs err vs ml_dtypes ref: {k_err:.4e}")
    print(f"V round-trip max abs err vs ml_dtypes ref: {v_err:.4e}")

    # Compare to original fp16 — should be close (within fp8 quant noise).
    k_quant_err = np.max(np.abs(k_back.astype(np.float32) - k_fp16.astype(np.float32)))
    v_quant_err = np.max(np.abs(v_back.astype(np.float32) - v_fp16.astype(np.float32)))
    print(f"K quant err (fp16 vs fp8-rounded): {k_quant_err:.4e}")
    print(f"V quant err (fp16 vs fp8-rounded): {v_quant_err:.4e}")
    print(f"K original sample: {k_fp16[0, 0, :8]}")
    print(f"K back sample:     {k_back[0, 0, :8]}")
    print(f"K ml_dtypes ref:   {k_ref[0, 0, :8]}")

    # Acceptance: kernel reproduces ml_dtypes round-trip exactly.
    assert k_err < 1e-3, f"K kernel disagrees with ml_dtypes round-trip: {k_err}"
    assert v_err < 1e-3, f"V kernel disagrees with ml_dtypes round-trip: {v_err}"
    print("\n*** 5.4 ROUND-TRIP PASS: append+debug_get parity with ml_dtypes ***")


if __name__ == "__main__":
    main()
