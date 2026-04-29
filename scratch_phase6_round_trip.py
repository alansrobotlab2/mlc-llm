"""Phase 6.2 round-trip: append (fp16 K/V -> int8 pages + fp32 scales) + debug_get_kv
(int8+scale -> fp16). Standalone kernel test, no model.

Verifies the kernel matches a Python reference of per-token symmetric quant:
    scale = max(|x|) / 127
    quant = round(x / scale)
    dequant = quant * scale
"""
import numpy as np
import tvm

from tvm.relax.frontend.nn.llm._page_kernels import (
    _kv_cache_transpose_append,
    _kv_cache_debug_get_kv,
)


def py_quant_dequant_per_token(x_fp16):
    """Reference per-token symmetric quant + dequant. Mirrors the TIR kernel.

    Args:
        x_fp16: ndarray, shape (ntoken, num_heads, head_dim), dtype np.float16
    Returns:
        x_back: ndarray, same shape, dtype np.float16 (round-trip through int8)
        scales_per_token: ndarray, shape (ntoken, num_heads), dtype np.float32
    """
    x_fp32 = x_fp16.astype(np.float32)
    max_abs = np.max(np.abs(x_fp32), axis=-1)  # (ntoken, num_heads)
    scale = np.where(max_abs > 0, max_abs / 127.0, np.float32(1.0)).astype(np.float32)
    quant = np.round(x_fp32 / scale[..., None]).clip(-128, 127).astype(np.int8)
    dequant = quant.astype(np.float32) * scale[..., None]
    return dequant.astype(np.float16), scale, quant


def main():
    # Toy shapes — Qwen3.6-35B-A3B's full-attn layer geometry.
    num_layers = 1
    num_kv_heads = 2
    head_dim = 256
    page_size = 16
    num_pages = 4
    ntoken = 8
    dtype = "float16"
    dtype_kv = "int8"

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_87"})

    # Build kernels.
    append_fn = _kv_cache_transpose_append(
        num_kv_heads, head_dim, dtype, page_size=page_size, dtype_kv=dtype_kv
    )
    debug_get_fn = _kv_cache_debug_get_kv(
        num_layers, num_kv_heads, head_dim, dtype, dtype_kv=dtype_kv
    )
    print("[1/3] kernels built (append, debug_get)")

    rt_append = tvm.tirx.build(append_fn, target=target)
    rt_get = tvm.tirx.build(debug_get_fn, target=target)
    print("[2/3] kernels compiled")

    # Inputs.
    rng = np.random.default_rng(7)
    k_fp16 = rng.normal(0, 1.0, size=(ntoken, num_kv_heads, head_dim)).astype(np.float16)
    v_fp16 = rng.normal(0, 1.0, size=(ntoken, num_kv_heads, head_dim)).astype(np.float16)
    position_map = np.arange(ntoken, dtype=np.int32)  # token i -> position i

    dev = tvm.cuda(0)
    pages_init = np.zeros(
        (num_pages, 2, num_kv_heads, page_size, head_dim), dtype=np.int8
    )
    scales_init = np.zeros(
        (num_pages, 2, num_kv_heads, page_size), dtype=np.float32
    )
    pages_tvm = tvm.runtime.tensor(pages_init, device=dev)
    scales_tvm = tvm.runtime.tensor(scales_init, device=dev)
    k_tvm = tvm.runtime.tensor(k_fp16, device=dev)
    v_tvm = tvm.runtime.tensor(v_fp16, device=dev)
    pos_tvm = tvm.runtime.tensor(position_map, device=dev)

    # Append: fp16 K/V -> int8 pages + fp32 scales.
    rt_append["tir_kv_cache_transpose_append"](
        pages_tvm, scales_tvm, k_tvm, v_tvm, pos_tvm
    )
    dev.sync()

    # Read back the per-token scales the kernel wrote, compare to py reference.
    py_k_back, py_k_scale, py_k_quant = py_quant_dequant_per_token(k_fp16)
    py_v_back, py_v_scale, py_v_quant = py_quant_dequant_per_token(v_fp16)

    pages_dev = pages_tvm.numpy()  # (num_pages, 2, num_kv_heads, page_size, head_dim)
    scales_dev = scales_tvm.numpy()  # (num_pages, 2, num_kv_heads, page_size)

    # Reshape kernel output to (ntoken, kv_idx, head, head_dim) by walking position_map.
    kernel_quant_k = np.empty((ntoken, num_kv_heads, head_dim), dtype=np.int8)
    kernel_quant_v = np.empty((ntoken, num_kv_heads, head_dim), dtype=np.int8)
    kernel_scale_k = np.empty((ntoken, num_kv_heads), dtype=np.float32)
    kernel_scale_v = np.empty((ntoken, num_kv_heads), dtype=np.float32)
    for t in range(ntoken):
        pos = position_map[t]
        page_idx = pos // page_size
        page_off = pos % page_size
        kernel_quant_k[t] = pages_dev[page_idx, 0, :, page_off, :]
        kernel_quant_v[t] = pages_dev[page_idx, 1, :, page_off, :]
        kernel_scale_k[t] = scales_dev[page_idx, 0, :, page_off]
        kernel_scale_v[t] = scales_dev[page_idx, 1, :, page_off]

    scale_err_k = np.max(np.abs(kernel_scale_k - py_k_scale))
    scale_err_v = np.max(np.abs(kernel_scale_v - py_v_scale))
    quant_diff_k = np.max(np.abs(kernel_quant_k.astype(np.int32) - py_k_quant.astype(np.int32)))
    quant_diff_v = np.max(np.abs(kernel_quant_v.astype(np.int32) - py_v_quant.astype(np.int32)))

    print(f"  K scale max abs err vs py-ref: {scale_err_k:.4e}")
    print(f"  V scale max abs err vs py-ref: {scale_err_v:.4e}")
    print(f"  K quant max int diff vs py-ref: {quant_diff_k}")
    print(f"  V quant max int diff vs py-ref: {quant_diff_v}")

    # debug_get: int8 pages + scales -> fp16.
    pos_back_tvm = tvm.runtime.tensor(np.arange(ntoken, dtype=np.int32), device=dev)
    k_out = tvm.runtime.tensor(np.zeros((num_layers, ntoken, num_kv_heads, head_dim), np.float16), device=dev)
    v_out = tvm.runtime.tensor(np.zeros((num_layers, ntoken, num_kv_heads, head_dim), np.float16), device=dev)
    rt_get["tir_kv_cache_debug_get_kv"](
        pages_tvm, scales_tvm, pos_back_tvm, k_out, v_out, 0
    )
    dev.sync()

    k_back = k_out.numpy()[0]
    v_back = v_out.numpy()[0]

    k_err = np.max(np.abs(k_back.astype(np.float32) - py_k_back.astype(np.float32)))
    v_err = np.max(np.abs(v_back.astype(np.float32) - py_v_back.astype(np.float32)))
    print(f"[3/3] round-trip vs Python reference:")
    print(f"  K dequant max abs err: {k_err:.4e}")
    print(f"  V dequant max abs err: {v_err:.4e}")

    # Quant noise vs original fp16 — should be max_abs/127 per token, ~max-quant-noise scale.
    k_quant_err = np.max(np.abs(k_back.astype(np.float32) - k_fp16.astype(np.float32)))
    v_quant_err = np.max(np.abs(v_back.astype(np.float32) - v_fp16.astype(np.float32)))
    bound = float(np.max(py_k_scale))
    print(f"  K quant noise vs fp16 original: {k_quant_err:.4e}  (bound: max_abs/127 ~= {bound:.4e})")
    print(f"  V quant noise vs fp16 original: {v_quant_err:.4e}")
    print(f"  K original [0,0,:8]: {k_fp16[0, 0, :8]}")
    print(f"  K back     [0,0,:8]: {k_back[0, 0, :8]}")
    print(f"  K py-ref   [0,0,:8]: {py_k_back[0, 0, :8]}")

    assert scale_err_k < 1e-5, f"K scales mismatch: {scale_err_k}"
    assert scale_err_v < 1e-5, f"V scales mismatch: {scale_err_v}"
    assert quant_diff_k <= 1, f"K quant diff > 1: {quant_diff_k}"
    assert quant_diff_v <= 1, f"V quant diff > 1: {quant_diff_v}"
    assert k_err < 1e-3, f"K dequant disagrees with py-ref: {k_err}"
    assert v_err < 1e-3, f"V dequant disagrees with py-ref: {v_err}"
    print("\n*** 6.2 ROUND-TRIP PASS: append+debug_get matches Python reference ***")


if __name__ == "__main__":
    main()
