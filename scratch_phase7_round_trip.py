"""Phase 7 round-trip: append (fp16 K/V -> mxfp4 packed pages + per-block fp32
scales) + debug_get_kv (mxfp4 -> fp16). Standalone kernel test, no model.

Verifies the kernel matches a Python reference:
    per (token, K-or-V, head, block_of_32):
        scale = max(|x|) / 6
        nibble = E2M1_quant(x / scale)            ([0, 0.5, 1, 1.5, 2, 3, 4, 6])
        byte   = (nibble_hi << 4) | (nibble_lo & 0xF)
"""
import numpy as np
import tvm

from tvm.relax.frontend.nn.llm._page_kernels import (
    _kv_cache_transpose_append,
    _kv_cache_debug_get_kv,
)


BLOCK = 32
FP4_MAGS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
FP4_MIDPOINTS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
FP4_MAX = 6.0


def py_quant_dequant_per_block(x_fp16):
    """Per-block-32 mxfp4 quant + dequant.

    Args:
        x_fp16: (ntoken, num_heads, head_dim) fp16
    Returns:
        x_back  : (ntoken, num_heads, head_dim) fp16
        scales  : (ntoken, num_heads, head_dim/32) fp32
        packed  : (ntoken, num_heads, head_dim/2) int8 (two nibbles/byte)
    """
    x = x_fp16.astype(np.float32)
    nt, nh, hd = x.shape
    nb = hd // BLOCK
    x_blk = x.reshape(nt, nh, nb, BLOCK)
    max_abs = np.max(np.abs(x_blk), axis=-1)  # (nt, nh, nb)
    scale = np.where(max_abs > 0, max_abs / FP4_MAX, np.float32(1.0)).astype(np.float32)
    abs_norm = np.abs(x_blk) / scale[..., None]
    abs_norm_clipped = np.clip(abs_norm, 0.0, FP4_MAX)
    mag_idx = np.searchsorted(FP4_MIDPOINTS, abs_norm_clipped, side="right").astype(np.uint8)
    sign_bit = (x_blk < 0).astype(np.uint8)
    nibbles = ((sign_bit << 3) | mag_idx).reshape(nt, nh, hd)
    nib_lo = nibbles[..., 0::2]
    nib_hi = nibbles[..., 1::2]
    packed = ((nib_hi.astype(np.int16) << 4) | (nib_lo.astype(np.int16) & 0xF)).astype(np.int8)
    mag = FP4_MAGS[mag_idx]
    signed = np.where(sign_bit == 1, -mag, mag)
    dequant = (signed * scale[..., None]).reshape(nt, nh, hd).astype(np.float16)
    return dequant, scale, packed


def main():
    num_layers = 1
    num_kv_heads = 2
    head_dim = 256
    page_size = 16
    num_pages = 4
    ntoken = 8
    dtype = "float16"
    dtype_kv = "mxfp4"

    nb = head_dim // BLOCK

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_87"})

    append_fn = _kv_cache_transpose_append(
        num_kv_heads, head_dim, dtype, page_size=page_size, dtype_kv=dtype_kv
    )
    debug_get_fn = _kv_cache_debug_get_kv(
        num_layers, num_kv_heads, head_dim, dtype, dtype_kv=dtype_kv
    )
    print("[1/3] kernels built")

    rt_append = tvm.tirx.build(append_fn, target=target)
    rt_get = tvm.tirx.build(debug_get_fn, target=target)
    print("[2/3] kernels compiled")

    rng = np.random.default_rng(7)
    k_fp16 = rng.normal(0, 1.0, size=(ntoken, num_kv_heads, head_dim)).astype(np.float16)
    v_fp16 = rng.normal(0, 1.0, size=(ntoken, num_kv_heads, head_dim)).astype(np.float16)
    position_map = np.arange(ntoken, dtype=np.int32)

    dev = tvm.cuda(0)
    pages_init = np.zeros((num_pages, 2, num_kv_heads, page_size, head_dim // 2), dtype=np.int8)
    scales_init = np.zeros((num_pages, 2, num_kv_heads, page_size, nb), dtype=np.float32)
    pages_tvm = tvm.runtime.tensor(pages_init, device=dev)
    scales_tvm = tvm.runtime.tensor(scales_init, device=dev)
    k_tvm = tvm.runtime.tensor(k_fp16, device=dev)
    v_tvm = tvm.runtime.tensor(v_fp16, device=dev)
    pos_tvm = tvm.runtime.tensor(position_map, device=dev)

    rt_append["tir_kv_cache_transpose_append"](
        pages_tvm, scales_tvm, k_tvm, v_tvm, pos_tvm
    )
    dev.sync()

    py_k_back, py_k_scale, py_k_packed = py_quant_dequant_per_block(k_fp16)
    py_v_back, py_v_scale, py_v_packed = py_quant_dequant_per_block(v_fp16)

    pages_dev = pages_tvm.numpy()
    scales_dev = scales_tvm.numpy()

    # Pull each token's slot back to (ntoken, num_kv_heads, ...) for direct compare.
    kernel_pack_k = np.empty((ntoken, num_kv_heads, head_dim // 2), dtype=np.int8)
    kernel_pack_v = np.empty((ntoken, num_kv_heads, head_dim // 2), dtype=np.int8)
    kernel_scale_k = np.empty((ntoken, num_kv_heads, nb), dtype=np.float32)
    kernel_scale_v = np.empty((ntoken, num_kv_heads, nb), dtype=np.float32)
    for t in range(ntoken):
        pos = position_map[t]
        page_idx = pos // page_size
        page_off = pos % page_size
        kernel_pack_k[t] = pages_dev[page_idx, 0, :, page_off, :]
        kernel_pack_v[t] = pages_dev[page_idx, 1, :, page_off, :]
        kernel_scale_k[t] = scales_dev[page_idx, 0, :, page_off, :]
        kernel_scale_v[t] = scales_dev[page_idx, 1, :, page_off, :]

    scale_err_k = np.max(np.abs(kernel_scale_k - py_k_scale))
    scale_err_v = np.max(np.abs(kernel_scale_v - py_v_scale))
    pack_match_k = np.array_equal(kernel_pack_k, py_k_packed)
    pack_match_v = np.array_equal(kernel_pack_v, py_v_packed)
    print(f"  K scale max abs err: {scale_err_k:.4e}")
    print(f"  V scale max abs err: {scale_err_v:.4e}")
    print(f"  K packed bytes match: {pack_match_k}")
    print(f"  V packed bytes match: {pack_match_v}")
    if not pack_match_k:
        bad = np.argwhere(kernel_pack_k != py_k_packed)
        print(f"    {len(bad)} K bytes differ; first 4: {bad[:4]}")
    if not pack_match_v:
        bad = np.argwhere(kernel_pack_v != py_v_packed)
        print(f"    {len(bad)} V bytes differ; first 4: {bad[:4]}")

    # debug_get_kv round-trip
    pos_back_tvm = tvm.runtime.tensor(np.arange(ntoken, dtype=np.int32), device=dev)
    k_out = tvm.runtime.tensor(np.zeros((num_layers, ntoken, num_kv_heads, head_dim), np.float16), device=dev)
    v_out = tvm.runtime.tensor(np.zeros((num_layers, ntoken, num_kv_heads, head_dim), np.float16), device=dev)
    rt_get["tir_kv_cache_debug_get_kv"](
        pages_tvm, scales_tvm, pos_back_tvm, k_out, v_out, 0
    )
    dev.sync()

    k_back = k_out.numpy()[0]
    v_back = v_out.numpy()[0]

    k_pyref_err = np.max(np.abs(k_back.astype(np.float32) - py_k_back.astype(np.float32)))
    v_pyref_err = np.max(np.abs(v_back.astype(np.float32) - py_v_back.astype(np.float32)))
    k_orig_err = np.max(np.abs(k_back.astype(np.float32) - k_fp16.astype(np.float32)))
    v_orig_err = np.max(np.abs(v_back.astype(np.float32) - v_fp16.astype(np.float32)))
    bound = float(np.max(py_k_scale)) * 0.5
    print(f"[3/3] round-trip:")
    print(f"  K dequant vs py-ref max abs err : {k_pyref_err:.4e}")
    print(f"  V dequant vs py-ref max abs err : {v_pyref_err:.4e}")
    print(f"  K vs original (4-bit noise)     : {k_orig_err:.4e}  (per-block bound ~= {bound:.4e})")
    print(f"  V vs original (4-bit noise)     : {v_orig_err:.4e}")
    print(f"  K original [0,0,:8]: {k_fp16[0, 0, :8]}")
    print(f"  K back     [0,0,:8]: {k_back[0, 0, :8]}")
    print(f"  K py-ref   [0,0,:8]: {py_k_back[0, 0, :8]}")

    assert scale_err_k < 1e-5, f"K scales mismatch: {scale_err_k}"
    assert scale_err_v < 1e-5, f"V scales mismatch: {scale_err_v}"
    assert pack_match_k, "K packed bytes diverge from py-ref"
    assert pack_match_v, "V packed bytes diverge from py-ref"
    assert k_pyref_err < 1e-3, f"K dequant disagrees with py-ref: {k_pyref_err}"
    assert v_pyref_err < 1e-3, f"V dequant disagrees with py-ref: {v_pyref_err}"
    print("\n*** PHASE 7 ROUND-TRIP PASS — append+debug_get matches numpy ref ***")


if __name__ == "__main__":
    main()
