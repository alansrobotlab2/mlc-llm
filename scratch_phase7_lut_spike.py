"""Phase 7 LUT spike — fp4 (E2M1) round-trip with per-block-32 scales.

De-risks the only piece without a Phase 6 analog before touching real kernels:
- 16-entry E2M1 grid expressed in TIR via if_then_else chain
- Per-block-32 max-abs reduction
- Nibble pack into int8 (two fp4 values per byte: low=even idx, high=odd idx)
- Round-trip dequant matches numpy reference

Storage (flat for the spike):
    pages : int8  shape (head_dim // 2,)
    scales: fp32  shape (head_dim // 32,)

E2M1 grid: [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] (sign in bit 3).
"""
import numpy as np
import tvm
from tvm.script import tirx as T


HEAD_DIM = 256
BLOCK = 32

# ----- numpy reference -----------------------------------------------------

FP4_MAGS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
FP4_MIDPOINTS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
FP4_MAX = 6.0


def py_quant_dequant_per_block(x_fp16):
    """Per-block-32 symmetric quant to E2M1, then dequant. Matches the TIR kernel.

    Args:
        x_fp16: ndarray, shape (head_dim,), dtype np.float16
    Returns:
        x_back   : ndarray (head_dim,)   fp16, after round-trip
        scales   : ndarray (head_dim/32,) fp32
        nibbles  : ndarray (head_dim,)    uint8 — per-element 4-bit values
        packed   : ndarray (head_dim/2,)  int8  — two nibbles per byte
    """
    x = x_fp16.astype(np.float32)
    n = x.shape[0]
    nb = n // BLOCK
    x_blk = x.reshape(nb, BLOCK)
    max_abs = np.max(np.abs(x_blk), axis=-1)
    scale = np.where(max_abs > 0, max_abs / FP4_MAX, np.float32(1.0)).astype(np.float32)
    # Quant: find nearest grid point in magnitude.
    abs_norm = np.abs(x_blk) / scale[:, None]
    abs_norm_clipped = np.clip(abs_norm, 0.0, FP4_MAX)
    # side='right' matches the TIR kernel's strict `<` chain: at exact midpoint
    # values, both round UP to the larger grid point (avoids fp tie-break drift).
    mag_idx = np.searchsorted(FP4_MIDPOINTS, abs_norm_clipped, side="right")  # 0..7
    sign_bit = (x_blk < 0).astype(np.uint8)
    nibbles = ((sign_bit << 3) | mag_idx.astype(np.uint8)).astype(np.uint8).reshape(n)
    # Pack: byte i holds (nibbles[2i] in low, nibbles[2i+1] in high).
    nibbles_lo = nibbles[0::2]
    nibbles_hi = nibbles[1::2]
    packed = ((nibbles_hi.astype(np.int16) << 4) | (nibbles_lo.astype(np.int16) & 0xF)).astype(np.int8)
    # Dequant.
    mag = FP4_MAGS[mag_idx]
    signed = np.where(sign_bit == 1, -mag, mag)
    dequant = signed * scale[:, None]
    return dequant.reshape(n).astype(np.float16), scale, nibbles, packed


# ----- TIR kernel ----------------------------------------------------------

def make_pack_kernel():
    """Per-block-32 max-abs + E2M1 quant + nibble pack. One thread does one
    head_dim row. Mirrors the eventual real kernel's serial-over-blocks shape."""
    head_dim = HEAD_DIM
    block = BLOCK
    nb = head_dim // block

    @T.prim_func
    def fp4_pack(
        var_x: T.handle,
        var_pages: T.handle,
        var_scales: T.handle,
    ):
        T.func_attr({"tirx.noalias": True, "tirx.is_scheduled": True})
        x = T.match_buffer(var_x, (head_dim,), "float16")
        pages = T.match_buffer(var_pages, (head_dim // 2,), "int8")
        scales = T.match_buffer(var_scales, (nb,), "float32")
        # 1 block, 1 thread — toy spike.
        for _bx in T.thread_binding(1, thread="blockIdx.x"):
            for _tx in T.thread_binding(1, thread="threadIdx.x"):
                with T.sblock("fp4_pack_root"):
                    T.reads()
                    T.writes()
                    max_abs = T.sblock_alloc_buffer((1,), "float32", scope="local")
                    scale = T.sblock_alloc_buffer((1,), "float32", scope="local")
                    for b in T.serial(nb):
                        max_abs[0] = T.float32(0)
                        for f in T.serial(block):
                            max_abs[0] = T.max(
                                max_abs[0],
                                T.abs(T.cast(x[b * block + f], "float32")),
                            )
                        scale[0] = T.if_then_else(
                            max_abs[0] > T.float32(0),
                            max_abs[0] / T.float32(6.0),
                            T.float32(1.0),
                        )
                        scales[b] = scale[0]
                        # Pack pairs of consecutive elements into one byte.
                        for fp in T.serial(block // 2):
                            f_lo = b * block + 2 * fp
                            f_hi = b * block + 2 * fp + 1
                            x_lo = T.cast(x[f_lo], "float32")
                            x_hi = T.cast(x[f_hi], "float32")
                            ax_lo = T.abs(x_lo) / scale[0]
                            ax_hi = T.abs(x_hi) / scale[0]
                            mag_lo: T.int32 = T.if_then_else(
                                ax_lo < T.float32(0.25), 0,
                                T.if_then_else(ax_lo < T.float32(0.75), 1,
                                T.if_then_else(ax_lo < T.float32(1.25), 2,
                                T.if_then_else(ax_lo < T.float32(1.75), 3,
                                T.if_then_else(ax_lo < T.float32(2.5), 4,
                                T.if_then_else(ax_lo < T.float32(3.5), 5,
                                T.if_then_else(ax_lo < T.float32(5.0), 6, 7)))))))
                            mag_hi: T.int32 = T.if_then_else(
                                ax_hi < T.float32(0.25), 0,
                                T.if_then_else(ax_hi < T.float32(0.75), 1,
                                T.if_then_else(ax_hi < T.float32(1.25), 2,
                                T.if_then_else(ax_hi < T.float32(1.75), 3,
                                T.if_then_else(ax_hi < T.float32(2.5), 4,
                                T.if_then_else(ax_hi < T.float32(3.5), 5,
                                T.if_then_else(ax_hi < T.float32(5.0), 6, 7)))))))
                            sign_lo: T.int32 = T.if_then_else(x_lo < T.float32(0), 1, 0)
                            sign_hi: T.int32 = T.if_then_else(x_hi < T.float32(0), 1, 0)
                            nibble_lo: T.int32 = (sign_lo << 3) | mag_lo
                            nibble_hi: T.int32 = (sign_hi << 3) | mag_hi
                            pages[b * (block // 2) + fp] = T.cast(
                                (nibble_hi << 4) | (nibble_lo & 0xF),
                                "int8",
                            )

    return fp4_pack


def make_unpack_kernel():
    """Read back: dequant E2M1 -> fp16."""
    head_dim = HEAD_DIM
    block = BLOCK
    nb = head_dim // block

    @T.prim_func
    def fp4_unpack(
        var_pages: T.handle,
        var_scales: T.handle,
        var_out: T.handle,
    ):
        T.func_attr({"tirx.noalias": True, "tirx.is_scheduled": True})
        pages = T.match_buffer(var_pages, (head_dim // 2,), "int8")
        scales = T.match_buffer(var_scales, (nb,), "float32")
        out = T.match_buffer(var_out, (head_dim,), "float16")
        for _bx in T.thread_binding(1, thread="blockIdx.x"):
            for _tx in T.thread_binding(1, thread="threadIdx.x"):
                with T.sblock("fp4_unpack_root"):
                    T.reads()
                    T.writes()
                    for f in T.serial(head_dim):
                        byte_i = f // 2
                        byte_val: T.int32 = T.cast(pages[byte_i], "int32") & 0xFF
                        nibble: T.int32 = T.if_then_else(
                            f % 2 == 0,
                            byte_val & 0xF,
                            (byte_val >> 4) & 0xF,
                        )
                        sign_bit: T.int32 = (nibble >> 3) & 1
                        mag_idx: T.int32 = nibble & 0x7
                        mag_f32: T.float32 = T.if_then_else(
                            mag_idx == 0, T.float32(0.0),
                            T.if_then_else(mag_idx == 1, T.float32(0.5),
                            T.if_then_else(mag_idx == 2, T.float32(1.0),
                            T.if_then_else(mag_idx == 3, T.float32(1.5),
                            T.if_then_else(mag_idx == 4, T.float32(2.0),
                            T.if_then_else(mag_idx == 5, T.float32(3.0),
                            T.if_then_else(mag_idx == 6, T.float32(4.0),
                                                          T.float32(6.0))))))))
                        signed_val: T.float32 = T.if_then_else(
                            sign_bit == 1, -mag_f32, mag_f32
                        )
                        out[f] = T.cast(signed_val * scales[f // block], "float16")

    return fp4_unpack


# ----- main ----------------------------------------------------------------

def main():
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_87"})

    pack_fn = make_pack_kernel()
    unpack_fn = make_unpack_kernel()
    print("[1/3] kernels defined")

    rt_pack = tvm.tirx.build(pack_fn, target=target)
    rt_unpack = tvm.tirx.build(unpack_fn, target=target)
    print("[2/3] kernels compiled")

    rng = np.random.default_rng(7)
    x_fp16 = rng.normal(0, 1.0, size=(HEAD_DIM,)).astype(np.float16)

    # Numpy reference.
    py_back, py_scale, py_nibbles, py_packed = py_quant_dequant_per_block(x_fp16)

    dev = tvm.cuda(0)
    x_tvm = tvm.runtime.tensor(x_fp16, device=dev)
    pages_tvm = tvm.runtime.tensor(np.zeros((HEAD_DIM // 2,), np.int8), device=dev)
    scales_tvm = tvm.runtime.tensor(np.zeros((HEAD_DIM // BLOCK,), np.float32), device=dev)
    out_tvm = tvm.runtime.tensor(np.zeros((HEAD_DIM,), np.float16), device=dev)

    rt_pack["fp4_pack"](x_tvm, pages_tvm, scales_tvm)
    dev.sync()

    pages_dev = pages_tvm.numpy()
    scales_dev = scales_tvm.numpy()

    scale_err = np.max(np.abs(scales_dev - py_scale))
    pack_match = np.array_equal(pages_dev, py_packed)
    print(f"  scales max abs err vs py-ref: {scale_err:.4e}")
    print(f"  packed bytes exact match    : {pack_match}")
    if not pack_match:
        diff_idx = np.where(pages_dev != py_packed)[0]
        print(f"    {len(diff_idx)} bytes differ; first 8 indices: {diff_idx[:8]}")
        for i in diff_idx[:4]:
            print(f"    idx {i}: dev={pages_dev[i]:#x} py={py_packed[i]:#x}")

    rt_unpack["fp4_unpack"](pages_tvm, scales_tvm, out_tvm)
    dev.sync()

    dev_back = out_tvm.numpy()
    err_vs_pyref = np.max(np.abs(dev_back.astype(np.float32) - py_back.astype(np.float32)))
    err_vs_orig = np.max(np.abs(dev_back.astype(np.float32) - x_fp16.astype(np.float32)))
    bound = float(np.max(py_scale)) * 0.5  # max quant noise per block ~= scale/2
    print(f"[3/3] round-trip:")
    print(f"  TIR vs py-ref dequant max abs err : {err_vs_pyref:.4e}")
    print(f"  TIR back vs original (4-bit noise): {err_vs_orig:.4e}  (per-block bound ~= {bound:.4e})")
    print(f"  x      [:8]: {x_fp16[:8]}")
    print(f"  py_back[:8]: {py_back[:8]}")
    print(f"  dev    [:8]: {dev_back[:8]}")
    print(f"  scales (8 blocks): {scales_dev}")

    assert scale_err < 1e-5, f"scales mismatch: {scale_err}"
    assert pack_match, "packed bytes diverge from numpy ref"
    assert err_vs_pyref < 1e-3, f"TIR dequant disagrees with py-ref: {err_vs_pyref}"
    print("\n*** PHASE 7 LUT SPIKE PASS — fp4 round-trip in TIR matches numpy ref ***")


if __name__ == "__main__":
    main()
