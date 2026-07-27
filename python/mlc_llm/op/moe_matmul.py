"""Mixture of Experts operators"""

import os
from typing import Literal, Optional, Tuple  # noqa: UP035

from tvm import DataType, DataTypeCode, s_tir, tirx
from tvm.relax.frontend.nn import Tensor, op
from tvm.script import tirx as T

# mypy: disable-error-code="attr-defined,valid-type,name-defined"


def gemv(x: Tensor, w: Tensor, indptr: Tensor) -> Tensor:
    """GEMV for project-in (e1-e3) or project-out (e2) in MLP.

    Parameters
    ----------
    x : Tensor
        For project-in, the input tensor of shape (1, in_features); and for project-out, the input
        shape is (experts_per_tok, in_features), where `experts_per_tok` is the number of activated
        experts per token.

    w : Tensor
        The weight tensor of shape (local_experts, out_features, in_features), where `local_experts`
        is the total number of experts.

    indptr : Tensor
        The index pointer tensor of shape (1, experts_per_tok), where `experts_per_tok` is the
        number of activated experts per token.

    Returns
    -------
    out : Tensor
        The output tensor of shape (experts_per_tok, out_features), where `experts_per_tok` is the
        number of activated experts per token.
    """
    (local_experts, out_features, in_features), dtype = w.shape, w.dtype
    _, experts_per_tok = indptr.shape
    x_leading_dim, _ = x.shape

    def access_x(x, e, j):
        return x[0, j] if x_leading_dim == 1 else x[e, j]

    # NOTE: Currently it assumes x.dtype == w.dtype, but the constraint can be relaxed easily.
    assert w.shape == [local_experts, out_features, in_features] and w.dtype == dtype
    assert x.shape == [x_leading_dim, in_features] and x.dtype == dtype
    assert indptr.shape == [1, experts_per_tok] and indptr.dtype == "int32"
    assert x_leading_dim in [1, experts_per_tok]

    @T.prim_func(private=True)
    def _func(
        x: T.Buffer((x_leading_dim, in_features), dtype),
        w: T.Buffer((local_experts, out_features, in_features), dtype),
        indptr: T.Buffer((1, experts_per_tok), "int32"),
        o: T.Buffer((experts_per_tok, out_features), dtype),
    ):
        T.func_attr({"op_pattern": 4, "tirx.noalias": True})  # kOutEWiseFusable
        for e in T.thread_binding(experts_per_tok, thread="blockIdx.y"):
            with T.sblock("gemv_o"):
                e = T.axis.spatial(experts_per_tok, e)
                T.reads(x[:, :], w[indptr[0, e], :, :], indptr[0, e])
                T.writes(o[e, :])
                for i1, i2 in T.grid(out_features, in_features):
                    with T.sblock("gemv"):
                        i, j = T.axis.remap("SR", [i1, i2])
                        with T.init():
                            o[e, i] = T.cast(T.float16(0), dtype)
                        o[e, i] += access_x(x, e, j) * w[indptr[0, e], i, j]

    return op.tensor_ir_op(
        _func,
        "moe_gemv",
        args=[x, w, indptr],
        out=Tensor.placeholder([experts_per_tok, out_features], dtype),
    )


def dequantize_gemv(
    x: Tensor,
    w: Tensor,
    scale: Tensor,
    indptr: Tensor,
    quantize_dtype: str,
    group_size: int,
) -> Tensor:
    """GEMV for project-in (e1-e3) or project-out (e2) in MLP but the weight is quantized.
    It needs to be dequantized before the GEMV computation.

    Parameters
    ----------
    x : Tensor
        For project-in, the input tensor of shape (1, in_features); and for project-out, the input
        shape is (experts_per_tok, in_features), where `experts_per_tok` is the number of activated
        experts per token.

    w : Tensor
        The quantized weight tensor of shape (local_experts, out_features, in_features // n),
        where n is the number of elements per storage dtype, e.g. if the storage dtype is uint32,
        and the quantize dtype is int4, then n is 8.
        `local_experts` is the total number of experts including activated and non-active ones.

    scale : Tensor
        The scale tensor of shape (local_experts, out_features, in_features // group_size), where
        `local_experts` is the total number of experts including activated and non-active ones.

    indptr : Tensor
        The index pointer tensor of shape (1, experts_per_tok), where `experts_per_tok` is the
        number of activated experts per token.

    quantize_dtype : str
        The quantize dtype of the weight tensor, which is usually int3, int4 or fp8, etc.

    group_size : int
        The number of elements in each quantization group, e.g. 32 or 128.

    Returns
    -------
    out : Tensor
        The output tensor of shape (experts_per_tok, out_features), where `experts_per_tok` is the
        number of activated experts per token.
    """
    (x_leading_dim, in_features), model_dtype = x.shape, x.dtype
    (local_experts, out_features, _), storage_dtype = w.shape, w.dtype
    _, experts_per_tok = indptr.shape
    quantize_dtype_bits = DataType(quantize_dtype).bits
    num_elem_per_storage = DataType(storage_dtype).bits // quantize_dtype_bits
    num_group = (in_features + group_size - 1) // group_size
    num_storage = group_size // num_elem_per_storage * num_group

    def _dequantize(w, s, e, i, j):
        tir_bin_mask = tirx.const((2**quantize_dtype_bits) - 1, storage_dtype)
        tir_max_int = tirx.const((2 ** (quantize_dtype_bits - 1)) - 1, model_dtype)
        w = w[e, i, j // num_elem_per_storage]
        s = s[e, i, j // group_size]
        shift = (j % num_elem_per_storage * quantize_dtype_bits).astype(storage_dtype)
        w = tirx.bitwise_and(tirx.shift_right(w, shift), tir_bin_mask).astype(model_dtype)
        return (w - tir_max_int) * s

    def access_x(x, e, j):
        return x[0, j] if x_leading_dim == 1 else x[e, j]

    assert x.shape == [x_leading_dim, in_features] and x.dtype == model_dtype
    assert w.shape == [local_experts, out_features, num_storage] and w.dtype == storage_dtype
    assert scale.shape == [local_experts, out_features, num_group] and scale.dtype == model_dtype
    assert indptr.shape == [1, experts_per_tok] and indptr.dtype == "int32"
    assert x_leading_dim in [1, experts_per_tok]

    @T.prim_func(private=True)
    def _func(
        x: T.Buffer((x_leading_dim, in_features), model_dtype),
        w: T.Buffer((local_experts, out_features, num_storage), storage_dtype),
        scale: T.Buffer((local_experts, out_features, num_group), model_dtype),
        indptr: T.Buffer((1, experts_per_tok), "int32"),
        o: T.Buffer((experts_per_tok, out_features), model_dtype),
    ):
        T.func_attr({"op_pattern": 4, "tirx.noalias": True})  # kOutEWiseFusable
        for expert_id in T.thread_binding(experts_per_tok, thread="blockIdx.y"):
            with T.sblock("gemv_o"):
                e = T.axis.spatial(experts_per_tok, expert_id)
                y = T.sblock_alloc_buffer((out_features, in_features), model_dtype)
                for i1, i2 in T.grid(out_features, in_features):
                    with T.sblock("dequantize"):
                        i, j = T.axis.remap("SS", [i1, i2])
                        y[i, j] = _dequantize(w, scale, indptr[0, e], i, j)
                for i1, i2 in T.grid(out_features, in_features):
                    with T.sblock("gemv"):
                        i, j = T.axis.remap("SR", [i1, i2])
                        with T.init():
                            o[e, i] = T.cast(T.float16(0), model_dtype)
                        o[e, i] += access_x(x, e, j) * y[i, j]

    return op.tensor_ir_op(
        _func,
        "moe_dequantize_gemv",
        args=[x, w, scale, indptr],
        out=Tensor.placeholder([experts_per_tok, out_features], model_dtype),
    )


def dequantize_float8_gemv(
    x: Tensor,
    w: Tensor,
    scale: Optional[Tensor],
    indptr: Tensor,
    quantize_dtype: Literal["float8_e5m2", "float8_e4m3fn"],
) -> Tensor:
    """GEMV for project-in (e1-e3) or project-out (e2) in MLP but the weight is quantized in
    fp8 e5m2 or e4m3. It needs to be dequantized before the GEMV computation.

    Parameters
    ----------
    x : Tensor
        For project-in, the input tensor of shape (1, in_features); and for project-out, the input
        shape is (experts_per_tok, in_features), where `experts_per_tok` is the number of activated
        experts per token.

    w : Tensor
        The quantized weight tensor of shape (local_experts, out_features, in_features)

    scale : Optional[Tensor]
        The optional scale tensor of shape (1,)

    indptr : Tensor
        The index pointer tensor of shape (1, experts_per_tok), where `experts_per_tok` is the
        number of activated experts per token.

    quantize_dtype : Literal["float8_e5m2", "float8_e4m3fn"]
        The quantize dtype of the weight tensor, which is either float8_e5m2 or float8_e4m3fn.
    """
    (x_leading_dim, in_features), model_dtype = x.shape, x.dtype
    (local_experts, out_features, _), storage_dtype = w.shape, w.dtype
    _, experts_per_tok = indptr.shape
    quantize_dtype_bits = DataType(quantize_dtype).bits
    num_elem_per_storage = DataType(storage_dtype).bits // quantize_dtype_bits
    num_storage = tirx.ceildiv(in_features, num_elem_per_storage)

    def _dequantize(w, s, e, i, j):
        if num_elem_per_storage == 1:
            w = tirx.reinterpret(quantize_dtype, w[e, i, j])
        else:
            assert DataType(storage_dtype).type_code == DataTypeCode.UINT
            tir_bin_mask = tirx.const((2**quantize_dtype_bits) - 1, storage_dtype)
            w = w[e, i, j // num_elem_per_storage]
            shift = (j % num_elem_per_storage * quantize_dtype_bits).astype(storage_dtype)
            w = tirx.reinterpret(
                quantize_dtype,
                tirx.bitwise_and(tirx.shift_right(w, shift), tir_bin_mask).astype("uint8"),
            )
        w = w.astype(model_dtype)
        if s is not None:
            w = w * s[0]
        return w

    def access_x(x, e, j):
        return x[0, j] if x_leading_dim == 1 else x[e, j]

    @T.prim_func(private=True)
    def _func_with_scale(
        x: T.Buffer((x_leading_dim, in_features), model_dtype),
        w: T.Buffer((local_experts, out_features, num_storage), storage_dtype),
        scale: T.Buffer((1,), "float32"),
        indptr: T.Buffer((1, experts_per_tok), "int32"),
        o: T.Buffer((experts_per_tok, out_features), model_dtype),
    ):
        T.func_attr({"op_pattern": 4, "tirx.noalias": True})  # kOutEWiseFusable
        for expert_id in T.thread_binding(experts_per_tok, thread="blockIdx.y"):
            with T.sblock("gemv_o"):
                e = T.axis.spatial(experts_per_tok, expert_id)
                y = T.sblock_alloc_buffer((out_features, in_features), model_dtype)
                for i1, i2 in T.grid(out_features, in_features):
                    with T.sblock("dequantize"):
                        i, j = T.axis.remap("SS", [i1, i2])
                        y[i, j] = _dequantize(w, scale, indptr[0, e], i, j)
                for i1, i2 in T.grid(out_features, in_features):
                    with T.sblock("gemv"):
                        i, j = T.axis.remap("SR", [i1, i2])
                        with T.init():
                            o[e, i] = T.cast(T.float16(0), model_dtype)
                        o[e, i] += access_x(x, e, j) * y[i, j]

    @T.prim_func(private=True)
    def _func_without_scale(
        x: T.Buffer((x_leading_dim, in_features), model_dtype),
        w: T.Buffer((local_experts, out_features, num_storage), storage_dtype),
        indptr: T.Buffer((1, experts_per_tok), "int32"),
        o: T.Buffer((experts_per_tok, out_features), model_dtype),
    ):
        T.func_attr({"op_pattern": 4, "tirx.noalias": True})  # kOutEWiseFusable
        for expert_id in T.thread_binding(experts_per_tok, thread="blockIdx.y"):
            with T.sblock("gemv_o"):
                e = T.axis.spatial(experts_per_tok, expert_id)
                y = T.sblock_alloc_buffer((out_features, in_features), model_dtype)
                for i1, i2 in T.grid(out_features, in_features):
                    with T.sblock("dequantize"):
                        i, j = T.axis.remap("SS", [i1, i2])
                        y[i, j] = _dequantize(w, None, indptr[0, e], i, j)
                for i1, i2 in T.grid(out_features, in_features):
                    with T.sblock("gemv"):
                        i, j = T.axis.remap("SR", [i1, i2])
                        with T.init():
                            o[e, i] = T.cast(T.float16(0), model_dtype)
                        o[e, i] += access_x(x, e, j) * y[i, j]

    if scale is not None:
        return op.tensor_ir_op(
            _func_with_scale,
            "moe_dequantize_gemv",
            args=[x, w, scale, indptr],
            out=Tensor.placeholder([experts_per_tok, out_features], model_dtype),
        )
    return op.tensor_ir_op(
        _func_without_scale,
        "moe_dequantize_gemv",
        args=[x, w, indptr],
        out=Tensor.placeholder([experts_per_tok, out_features], model_dtype),
    )


def dequantize_block_scale_float8_gemv(
    x: Tensor,
    w: Tensor,
    w_scale: Tensor,
    expert_indices: Tensor,
    block_size: Tuple[int, int],  # noqa: UP006
    out_dtype: str,
) -> Tensor:
    """GEMV for project-in (e1-e3) or project-out (e2) in MLP but the weight is quantized in
    fp8 e5m2 or e4m3. It needs to be dequantized before the GEMV computation.

    Parameters
    ----------
    x : Tensor
        For project-in, the input tensor of shape (1, in_features); and for project-out, the input
        shape is (experts_per_tok, in_features), where `experts_per_tok` is the number of activated
        experts per token.

    w : Tensor
        The quantized weight tensor of shape (local_experts, out_features, in_features)

    w_scale : Tensor
        The scale tensor of shape
        (local_experts, out_features // block_size[0], in_features // block_size[1])

    indptr : Tensor
        The index pointer tensor of shape (1, experts_per_tok), where `experts_per_tok` is the
        number of activated experts per token.

    block_size : Tuple[int, int]
        The block size of the weight tensor.

    out_dtype : str
        The output dtype of the GEMV computation.
    """
    x_leading_dim, in_features = x.shape
    local_experts, out_features, k = w.shape
    _, experts_per_tok = expert_indices.shape
    model_dtype = x.dtype
    quantize_dtype = w.dtype

    assert out_features % block_size[0] == 0
    assert k % block_size[1] == 0

    def _dequantize(w, s, e, i, j):
        return w[e, i, j].astype(model_dtype) * s[e, i // block_size[0], j // block_size[1]].astype(
            model_dtype
        )

    def load_x(x, e, j):
        return x[0, j] if x_leading_dim == 1 else x[e, j]

    @T.prim_func(private=True)
    def _func(
        x: T.Buffer((x_leading_dim, in_features), model_dtype),
        w: T.Buffer((local_experts, out_features, k), quantize_dtype),
        w_scale: T.Buffer(
            (local_experts, out_features // block_size[0], k // block_size[1]),
            "float32",
        ),
        expert_indices: T.Buffer((1, experts_per_tok), "int32"),
        o: T.Buffer((experts_per_tok, out_features), out_dtype),
    ):
        T.func_attr({"op_pattern": 4, "tirx.noalias": True})  # kOutEWiseFusable
        for expert_id in T.thread_binding(experts_per_tok, thread="blockIdx.y"):
            with T.sblock("gemv_o"):
                e = T.axis.spatial(experts_per_tok, expert_id)
                y = T.sblock_alloc_buffer((out_features, in_features), model_dtype)
                for i1, i2 in T.grid(out_features, in_features):
                    with T.sblock("dequantize"):
                        i, j = T.axis.remap("SS", [i1, i2])
                        y[i, j] = _dequantize(w, w_scale, expert_indices[0, e], i, j)
                for i1, i2 in T.grid(out_features, in_features):
                    with T.sblock("gemv"):
                        i, j = T.axis.remap("SR", [i1, i2])
                        with T.init():
                            o[e, i] = T.cast(T.float16(0), out_dtype)
                        o[e, i] += (load_x(x, e, j) * y[i, j]).astype(out_dtype)

    return op.tensor_ir_op(
        _func,
        "moe_dequantize_gemv",
        args=[x, w, w_scale, expert_indices],
        out=Tensor.placeholder([experts_per_tok, out_features], out_dtype),
    )


def group_gemm(x: Tensor, w: Tensor, indptr: Tensor):
    """Group GEMM in MoE models.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape (batch_size, in_features), where `batch_size` could be dynamic shape.

    w : Tensor
        Weight tensor of shape (num_local_experts, out_features, in_features).
        `w[i, :, :]` is the weight matrix for the `i`-th local expert.

    indptr : Tensor
        Index pointer tensor of shape (num_local_experts + 1, ).
        `x[indptr[a] : indptr[a + 1]]` is the input for the `i`-th local expert.

    Returns
    -------
    out : Tensor
        Output tensor of shape (batch_size, out_features).
    """
    # NOTE: Currently it assumes x.dtype == w.dtype, but the constraint can be relaxed easily.
    (num_local_experts, out_features, in_features), dtype = w.shape, w.dtype

    assert x.shape[1:] == [in_features] and x.dtype == dtype
    assert indptr.shape == [num_local_experts + 1] and indptr.dtype == "int32"

    Ne, N, K = num_local_experts, out_features, in_features
    BLK_M, BLK_N, BLK_K = 8, 128, 32
    TX, TY, CTA_COUNT = 8, 32, 1024
    VEC_X, VEC_W, VEC_O, VEC_DOT = 1, 1, 1, 1
    UNROLL = 64
    STORAGE_ALIGN = False
    assert BLK_K % 8 == 0
    tiles_per_row = (N + BLK_N - 1) // BLK_N
    zero = tirx.const(0, dtype)

    @T.prim_func(private=True)
    def _func(
        var_x: T.handle,
        var_w: T.handle,
        var_indptr: T.handle,
        var_o: T.handle,
    ):
        T.func_attr({"tirx.is_scheduled": 1, "tirx.noalias": True})
        B = T.int32(is_size_var=True)
        X = T.match_buffer(var_x, (B, K), dtype)
        W = T.match_buffer(var_w, (Ne, N, K), dtype)
        indptr = T.match_buffer(var_indptr, (Ne + 1,), "int32")
        out = T.match_buffer(var_o, (B, N), dtype)

        for _bx in T.thread_binding(CTA_COUNT, thread="blockIdx.x"):
            with T.sblock("CTA"):
                bx = T.axis.spatial(CTA_COUNT, _bx)
                T.reads(indptr[:], X[:, :], W[:, :, :])
                T.writes(out[:, :])
                sum = T.sblock_alloc_buffer((2,), "int32", scope="local")
                row = T.sblock_alloc_buffer((2,), "int32", scope="local")
                cur_e = T.sblock_alloc_buffer((1,), "int32", scope="local")
                tile_id = T.sblock_alloc_buffer((1,), "int32", scope="local")
                sum[0] = 0
                sum[1] = T.ceildiv(indptr[1] - indptr[0], BLK_M) * tiles_per_row
                row[0] = 0
                row[1] = indptr[1] - indptr[0]
                cur_e[0] = 0
                tile_id[0] = bx
                while T.tvm_thread_invariant(cur_e[0] < Ne):
                    # move to the current group
                    while sum[1] <= tile_id[0] and cur_e[0] < Ne:
                        cur_e[0] += 1
                        if cur_e[0] < Ne:
                            e: T.int32 = cur_e[0]
                            delta: T.int32 = indptr[e + 1] - indptr[e]
                            sum[0] = sum[1]
                            sum[1] += T.ceildiv(delta, BLK_M) * tiles_per_row
                            row[0] = row[1]
                            row[1] += delta
                    # sync threads to make sure all threads have the same tile position
                    T.tvm_storage_sync("shared")
                    if T.tvm_thread_invariant(cur_e[0] < Ne):
                        # fetch current tile position
                        e: T.int32 = cur_e[0]
                        num_tiles: T.int32 = tile_id[0] - sum[0]
                        m_offset: T.int32 = BLK_M * T.floordiv(num_tiles, tiles_per_row) + row[0]
                        n_offset: T.int32 = BLK_N * T.floormod(num_tiles, tiles_per_row)
                        with T.sblock("gemm"):
                            T.reads(
                                row[1],
                                X[m_offset : m_offset + BLK_M, :],
                                W[e, n_offset : n_offset + BLK_N, :],
                            )
                            T.writes(
                                out[
                                    m_offset : m_offset + BLK_M,
                                    n_offset : n_offset + BLK_N,
                                ]
                            )
                            X_tile = T.sblock_alloc_buffer((BLK_M, K), dtype, scope="shared")
                            W_tile = T.sblock_alloc_buffer((BLK_N, K), dtype, scope="shared")
                            O_tile = T.sblock_alloc_buffer((BLK_M, BLK_N), dtype, scope="local")
                            for a0, a1 in T.grid(BLK_M, K):
                                with T.sblock("X_shared"):
                                    i, j = T.axis.remap("SS", [a0, a1])
                                    X_tile[i, j] = T.if_then_else(
                                        m_offset + i < row[1],
                                        X[m_offset + i, j],
                                        zero,
                                    )
                            for a0, a1 in T.grid(BLK_N, K):
                                with T.sblock("W_shared"):
                                    i, j = T.axis.remap("SS", [a0, a1])
                                    W_tile[i, j] = T.if_then_else(
                                        n_offset + i < N,
                                        W[e, n_offset + i, j],
                                        zero,
                                    )
                            for a0, a1, a2 in T.grid(BLK_M, BLK_N, K):
                                with T.sblock("compute"):
                                    i, j, k = T.axis.remap("SSR", [a0, a1, a2])
                                    with T.init():
                                        O_tile[i, j] = zero
                                    O_tile[i, j] += X_tile[i, k] * W_tile[j, k]
                            for a0, a1 in T.grid(BLK_M, BLK_N):
                                with T.sblock("store"):
                                    i, j = T.axis.remap("SS", [a0, a1])
                                    if m_offset + i < row[1] and n_offset + j < N:
                                        out[m_offset + i, n_offset + j] = O_tile[i, j]
                    # move to next tile
                    tile_id[0] += CTA_COUNT

    def _schedule():
        sch = s_tir.Schedule(_func)

        def _cooperative_fetch(block, vec_len):
            num_loops = len(sch.get_loops(block))
            sch.compute_at(block, ko, preserve_unit_loops=True)
            loops = sch.get_loops(block)[-num_loops:]
            ty, tx, _, vec = sch.split(
                sch.fuse(*loops),
                factors=[TY, TX, None, vec_len],
            )
            sch.vectorize(vec)
            sch.bind(ty, "threadIdx.y")
            sch.bind(tx, "threadIdx.x")
            if STORAGE_ALIGN:
                sch.storage_align(block, 0, axis=1, factor=8, offset=vec_len)
            return block

        main_block = sch.get_sblock("compute")
        x, y, k = sch.get_loops(main_block)
        ty, yi = sch.split(y, [TY, None])
        tx, xi, vec_c = sch.split(x, [TX, None, VEC_DOT])
        ko, ki = sch.split(k, factors=[None, BLK_K])
        sch.reorder(ty, tx, ko, ki, yi, xi, vec_c)
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec_c)
        if UNROLL > 0:
            sch.annotate(tx, ann_key="pragma_auto_unroll_max_step", ann_val=UNROLL)
            sch.annotate(tx, ann_key="pragma_unroll_explicit", ann_val=1)
        l2g = sch.get_sblock("store")
        sch.reverse_compute_at(l2g, tx, preserve_unit_loops=True)
        _, v = sch.split(sch.get_loops(l2g)[-1], [None, VEC_O])
        sch.vectorize(v)
        _cooperative_fetch(sch.get_sblock("X_shared"), vec_len=VEC_X)
        _cooperative_fetch(sch.get_sblock("W_shared"), vec_len=VEC_W)
        sch.decompose_reduction(main_block, ko)
        return sch.mod["main"]

    return op.tensor_ir_op(
        _schedule(),
        "group_gemm",
        args=[x, w, indptr],
        out=Tensor.placeholder([x.shape[0], out_features], dtype),
    )


def _dequantize_group_gemm_v2(
    x: Tensor,
    w: Tensor,
    scale: Tensor,
    indptr: Tensor,
    quantize_dtype: str,
    indptr_dtype: str,
    group_size: int,
) -> Tensor:
    """v2 path — dispatch tables + hand-tensorized wmma matmul.

    Phase 9b. Replaces the persistent-loop kernel with two prim_funcs:
    1. compute_moe_dispatch_tables: walks indptr, fills (tile_to_e, tile_to_m,
       tile_to_n) lookup tables sized to upper bound. Idle entries get
       sentinel tile_to_e = -1.
    2. dequantize_group_gemm_v2: each CTA reads its (e, m, n) from the
       lookup, dequantizes a BLK_N×K W slice into shared, and runs a
       hand-tensorized wmma.m16n8k16 matmul. Bounds-checked X reads + O
       writes guard against unaligned per-expert token counts.

    Selected via env var MLC_MOE_GEMM_V2=1; default stays on the v1
    persistent-loop kernel.
    """
    assert quantize_dtype == "int4", "v2 currently restricted to int4 quant (35B-A3B)"
    (_, in_features), model_dtype = x.shape, x.dtype
    (num_local_experts, out_features, _), storage_dtype = w.shape, w.dtype
    quantize_dtype_bits = DataType(quantize_dtype).bits
    num_elem_per_storage = DataType(storage_dtype).bits // quantize_dtype_bits
    num_group = (in_features + group_size - 1) // group_size
    num_storage = group_size // num_elem_per_storage * num_group

    Ne, N, K = num_local_experts, out_features, in_features
    # BLK_M must be a multiple of the wmma m16n8k16 tile's M=16; the schedule splits
    # it by MICRO=16 and the remainder becomes a serial loop over accumulators.
    #
    # Leave it at 16 -- but the reason is shape-dependent, so read §17.8 before changing
    # it. At pp512, 4096 rows over 256 experts is *exactly* 16 rows/expert, so
    # ceildiv(count_e, BLK_M) is 1 at 16, 32 and 64 alike: there is no CTA count to save
    # and widening only inflates X and O traffic. That holds with or without the load
    # hoist and is why the default stays here (measured 0.51x-0.63x at 32).
    #
    # It does NOT generalise to B=16384, the shape a 2048-token prefill chunk produces.
    # There the count does halve, and §17.1's 1.00x is the *un-hoisted* number: i_o sits
    # outside the k-loop, so each extra row-fragment re-runs the whole BLK_N x K dequant
    # and exactly cancels the saving. §17.8's cost model (validated to <=4%) predicts
    # 1.37x-1.51x there once the shared loads are hoisted above i_o. Unmeasured.
    #
    # MLC_MOE_GEMM_V2_BLKM stays as a diagnostic; bit-exact at every value, since it
    # does not touch the split over K.
    BLK_M = int(os.environ.get("MLC_MOE_GEMM_V2_BLKM", "16"))
    # Item 0h: put the row-fragment loop *inside* the k-loop the shared loads hang off,
    # so widening BLK_M amortizes one weight dequant over more rows instead of repeating
    # it. Inert at BLK_M=16 (i_o has extent 1). See _schedule_v2 and §17.8.
    HOIST = os.environ.get("MLC_MOE_GEMM_V2_HOIST", "0") == "1"
    # Item 0j: the order the dispatch table walks (m-tile, n-tile) within one expert.
    # See the comment at its use site in _dispatch_func. Bit-exact either way.
    TILE_N_MAJOR = os.environ.get("MLC_MOE_GEMM_V2_TILEORDER", "m") == "n"
    SKIPROWS = os.environ.get("MLC_MOE_GEMM_V2_SKIPROWS", "0") == "1"
    # Item 0l: same goal as 0k -- do no wmma work for row fragments that hold no real
    # rows -- but without making the loop extent dynamic. §18.7 isolated 0k's cost at
    # BLK_M=16, where its guard is logically inert (extent 1) and it *still* lost 5-9%:
    # replacing a constant extent with a runtime expression blocks the unroll, and 0k
    # pays that on every CTA. Here the extent stays the compile-time BLK_M/MICRO and the
    # per-fragment body is predicated instead, so every fragment index remains a
    # constant. See `_specialise_row_fragments`.
    ROWSPEC = os.environ.get("MLC_MOE_GEMM_V2_ROWSPEC", "0") == "1"
    assert not (SKIPROWS and ROWSPEC), (
        "MLC_MOE_GEMM_V2_SKIPROWS and _ROWSPEC are two mechanisms for the same thing; "
        "pick one"
    )
    # ROWSPEC predicates the fragment body, and a predicate is only legal where no
    # __syncthreads() lands inside it (§16.9). With the hoist the row-fragment loop sits
    # *below* k_o_o, where the cooperative loads and their barriers live, so its body is
    # barrier-free; without it, i_o is the outermost loop and wraps them, which is
    # exactly the case ThreadSync refuses. Item 0k's dynamic extent had no such
    # restriction -- an extent is not a condition -- which is why it could be swept over
    # both. Fail here rather than in a pass 200 lines downstream.
    assert not (ROWSPEC and not HOIST), (
        "MLC_MOE_GEMM_V2_ROWSPEC requires _HOIST=1: without the hoist the row-fragment "
        "loop encloses the cooperative loads' __syncthreads(), and ThreadSync refuses a "
        "barrier inside a condition"
    )
    assert BLK_M % 16 == 0, "BLK_M must be a multiple of the wmma M=16"
    BLK_N = 128
    # BLK_K sets the k-step of the cooperative fetch, and therefore how many bytes of
    # each W row are fetched per step: BLK_K int4 values = BLK_K/2 bytes. At the
    # original 32 that is 16 bytes — *half* a 32-byte sector — so every row-chunk read
    # pulled a sector it only half-used, and the k-loop paid a __syncthreads() pair per
    # 16 bytes/row. 64 makes a row-chunk exactly one sector and halves the barrier
    # count; measured bit-exact and 1.27x-1.39x on both 35B shapes. 128 regresses
    # (0.80x-0.93x) — shared memory grows linearly and occupancy falls off.
    BLK_K = int(os.environ.get("MLC_MOE_GEMM_V2_BLKK", "64"))
    assert BLK_K % 16 == 0, "BLK_K must be a multiple of the wmma K=16"
    while K % BLK_K:  # the schedule splits k by BLK_K // MICRO, so it has to divide K
        BLK_K //= 2
    assert BLK_K >= 16, f"K={K} is not a multiple of the wmma K=16"
    MICRO = 16
    # At BLK_M=16 the row-fragment loop already has extent 1, so there is nothing to
    # specialise and item 0f skips the only tile that could be empty (a padding CTA)
    # whole. Staying inert there matters: §18.10 found that an `sch.annotate` alone
    # stops TVM eliminating the unit loop, so an unconditional tag would change the
    # generated code of the shipped default.
    _rowspec_active = ROWSPEC and BLK_M // MICRO > 1
    tiles_per_n = (N + BLK_N - 1) // BLK_N
    assert N % BLK_N == 0, "v2 requires N % BLK_N == 0 (no col padding)"

    B = x.shape[0]
    upper = (tirx.ceildiv(B, BLK_M) + Ne) * tiles_per_n

    # ---------- prim_func 1: dispatch-table compute ----------
    # Triton-style parallel-over-experts dispatch. Each thread handles one
    # expert: it walks its own run of m-tiles in the 1D grid (with a +eid
    # slack offset that gives every expert at least 1 padding tile worth of
    # private indices), then fans m-tiles into n-tiles and writes (te,tm,tn).
    # Slack tiles between consecutive experts get sentinel te=-1.
    @T.prim_func(private=True)
    def _dispatch_func(
        indptr_buf: T.Buffer((Ne + 1,), indptr_dtype),
        var_te: T.handle,
        var_tm: T.handle,
        var_tn: T.handle,
    ):
        T.func_attr({"tirx.is_scheduled": 1, "tirx.noalias": True})
        UPPER = T.int32(is_size_var=True)
        te = T.match_buffer(var_te, (UPPER,), "int32")
        tm = T.match_buffer(var_tm, (UPPER,), "int32")
        tn = T.match_buffer(var_tn, (UPPER,), "int32")

        with T.sblock("root"):
            for eid in T.thread_binding(0, Ne, thread="threadIdx.x"):
                sb: T.int32 = T.ceildiv(indptr_buf[eid], BLK_M) + eid
                nb: T.int32 = T.ceildiv(indptr_buf[eid + 1] - indptr_buf[eid], BLK_M)
                sb_next: T.int32 = T.ceildiv(indptr_buf[eid + 1], BLK_M) + eid + 1
                for tmi in T.serial(nb):
                    for tni in T.serial(tiles_per_n):
                        te[(sb + tmi) * tiles_per_n + tni] = eid
                        tm[(sb + tmi) * tiles_per_n + tni] = indptr_buf[eid] + tmi * BLK_M
                        tn[(sb + tmi) * tiles_per_n + tni] = tni * BLK_N
                for sl in T.serial(sb_next - (sb + nb)):
                    for tni in T.serial(tiles_per_n):
                        te[(sb + nb + sl) * tiles_per_n + tni] = -1
                        tm[(sb + nb + sl) * tiles_per_n + tni] = 0
                        tn[(sb + nb + sl) * tiles_per_n + tni] = 0

    # Item 0j (refuted, §18.6): the same tiles handed out n-major instead of m-major, so
    # consecutive CTAs share a weight slice rather than an X_tile. Both orders cover
    # [0, nb*tiles_per_n) exactly and assign the identical set of (e, m, n) triples to
    # disjoint outputs, so it is bit-exact by construction — and measured so, 10/10.
    # It buys **nothing**: 0.99x-1.01x at B=4096 and B=16384 across BLK_M and hoist, so
    # §18.4's 2.9x issued/unique is not a cache-ordering problem. Kept as a flag because
    # the idea is an obvious one to re-propose; written as a second whole prim_func rather
    # than a parameterised index so the shipped path above stays byte-for-byte what §17.6
    # gated (verify with scripts/moe_dump_cuda.py).
    if TILE_N_MAJOR:

        @T.prim_func(private=True)
        def _dispatch_func(  # noqa: F811
            indptr_buf: T.Buffer((Ne + 1,), indptr_dtype),
            var_te: T.handle,
            var_tm: T.handle,
            var_tn: T.handle,
        ):
            T.func_attr({"tirx.is_scheduled": 1, "tirx.noalias": True})
            UPPER = T.int32(is_size_var=True)
            te = T.match_buffer(var_te, (UPPER,), "int32")
            tm = T.match_buffer(var_tm, (UPPER,), "int32")
            tn = T.match_buffer(var_tn, (UPPER,), "int32")

            with T.sblock("root"):
                for eid in T.thread_binding(0, Ne, thread="threadIdx.x"):
                    sb: T.int32 = T.ceildiv(indptr_buf[eid], BLK_M) + eid
                    nb: T.int32 = T.ceildiv(indptr_buf[eid + 1] - indptr_buf[eid], BLK_M)
                    sb_next: T.int32 = T.ceildiv(indptr_buf[eid + 1], BLK_M) + eid + 1
                    for tmi in T.serial(nb):
                        for tni in T.serial(tiles_per_n):
                            te[sb * tiles_per_n + tni * nb + tmi] = eid
                            tm[sb * tiles_per_n + tni * nb + tmi] = (
                                indptr_buf[eid] + tmi * BLK_M
                            )
                            tn[sb * tiles_per_n + tni * nb + tmi] = tni * BLK_N
                    for sl in T.serial(sb_next - (sb + nb)):
                        for tni in T.serial(tiles_per_n):
                            te[(sb + nb + sl) * tiles_per_n + tni] = -1
                            tm[(sb + nb + sl) * tiles_per_n + tni] = 0
                            tn[(sb + nb + sl) * tiles_per_n + tni] = 0

    te_t, tm_t, tn_t = op.tensor_ir_op(
        _dispatch_func,
        "moe_dispatch_tables",
        args=[indptr],
        out=(
            Tensor.placeholder([upper], "int32"),
            Tensor.placeholder([upper], "int32"),
            Tensor.placeholder([upper], "int32"),
        ),
    )

    # ---------- prim_func 2: dispatch-driven dequant + tensorized matmul ----------
    zero_f16 = T.float16(0.0)

    @T.prim_func(private=True)
    def _gemm_v2_func(
        var_x: T.handle,
        W_q: T.Buffer((Ne, N, num_storage), storage_dtype),
        Scale: T.Buffer((Ne, N, num_group), model_dtype),
        indptr_buf: T.Buffer((Ne + 1,), indptr_dtype),
        var_te: T.handle,
        var_tm: T.handle,
        var_tn: T.handle,
        var_o: T.handle,
    ):
        T.func_attr({"tirx.noalias": True})
        Bv = T.int32(is_size_var=True)
        UPPER = T.int32(is_size_var=True)
        X = T.match_buffer(var_x, (Bv, K), model_dtype)
        out = T.match_buffer(var_o, (Bv, N), model_dtype)
        te = T.match_buffer(var_te, (UPPER,), "int32")
        tm = T.match_buffer(var_tm, (UPPER,), "int32")
        tn = T.match_buffer(var_tn, (UPPER,), "int32")

        for _bx in T.thread_binding(UPPER, thread="blockIdx.x"):
            with T.sblock("CTA"):
                bx = T.axis.spatial(UPPER, _bx)
                T.reads(X[:, :], W_q[:, :, :], Scale[:, :, :], indptr_buf[:],
                        te[:], tm[:], tn[:])
                T.writes(out[:, :])

                X_tile = T.sblock_alloc_buffer((BLK_M, K), model_dtype, scope="shared.dyn")
                W_tile = T.sblock_alloc_buffer((BLK_N, K), model_dtype, scope="shared.dyn")
                O_tile = T.sblock_alloc_buffer((BLK_M, BLK_N), model_dtype, scope="shared.dyn")

                e_v = te[bx]
                m_offset = tm[bx]
                n_offset = tn[bx]
                row_end = T.if_then_else(e_v >= 0, indptr_buf[e_v + 1], 0)
                e_safe = T.if_then_else(e_v >= 0, e_v, 0)

                # Unit loop wrapping the whole CTA body. Inert as written; item 0f's
                # rewrite turns its extent into `Select(e_v >= 0, 1, 0)` so a padding
                # CTA skips *everything*, not just the reduction. See
                # `_guard_padding_ctas` for why this is a loop extent and not an `if`.
                for _guard in T.serial(1, annotations={"moe_pad_guard": 1}):
                    for a0, a1 in T.grid(BLK_M, K):
                        with T.sblock("X_shared"):
                            i, j = T.axis.remap("SS", [a0, a1])
                            X_tile[i, j] = T.if_then_else(
                                m_offset + i < row_end,
                                X[m_offset + i, j],
                                zero_f16,
                            )

                    for a0, a1 in T.grid(BLK_N, K):
                        with T.sblock("W_shared"):
                            i, j = T.axis.remap("SS", [a0, a1])
                            shift = T.Cast(storage_dtype, (j % num_elem_per_storage) * quantize_dtype_bits)
                            w_int = T.Cast(
                                model_dtype,
                                T.bitwise_and(
                                    T.shift_right(
                                        W_q[e_safe, n_offset + i, j // num_elem_per_storage],
                                        shift,
                                    ),
                                    T.Cast(storage_dtype, (1 << quantize_dtype_bits) - 1),
                                ),
                            )
                            W_tile[i, j] = (w_int - T.Cast(model_dtype, (1 << (quantize_dtype_bits - 1)) - 1)) * Scale[
                                e_safe, n_offset + i, j // group_size
                            ]

                    for a0, a1, a2 in T.grid(BLK_M, BLK_N, K):
                        with T.sblock("compute"):
                            i, j, k = T.axis.remap("SSR", [a0, a1, a2])
                            with T.init():
                                O_tile[i, j] = zero_f16
                            O_tile[i, j] = O_tile[i, j] + X_tile[i, k] * W_tile[j, k]

                    for a0, a1 in T.grid(BLK_M, BLK_N):
                        with T.sblock("store"):
                            i, j = T.axis.remap("SS", [a0, a1])
                            if m_offset + i < row_end:
                                out[m_offset + i, n_offset + j] = O_tile[i, j]

    # ---------- schedule: hand-tensorize wmma m16n8k16 ----------
    def _schedule_v2():
        from tvm.s_tir.tensor_intrin.cuda import get_wmma_intrin_group

        sch = s_tir.Schedule(_gemm_v2_func)

        TY = BLK_N // MICRO
        WARP = 32
        VEC = 4

        main_block = sch.get_sblock("compute")
        # The leading loop is the `moe_pad_guard` unit loop, not a compute axis.
        _guard, i, j, k = sch.get_loops(main_block)
        i_o, i_i = sch.split(i, factors=[None, MICRO])
        j_o, j_i = sch.split(j, factors=[None, MICRO])
        k_o, k_i = sch.split(k, factors=[None, MICRO])
        sch.reorder(i_o, j_o, k_o, i_i, j_i, k_i)

        block_inner = main_block
        block_outer = sch.blockize(i_i)

        k_o_o, k_o_i = sch.split(k_o, factors=[None, BLK_K // MICRO])
        # Item 0h (§17.8). `_coop` attaches the cooperative shared loads to k_o_o, so
        # whichever of i_o / k_o_o is outer decides how many times a CTA re-runs the
        # BLK_N x K weight dequant: i_o outer => BLK_M/MICRO times, k_o_o outer => once.
        # At the default BLK_M=16 i_o has extent 1 and the two are identical; the choice
        # only bites once BLK_M is widened, which is exactly what §17.1 measured without
        # it and §17.8 predicts with it.
        if HOIST:
            sch.reorder(j_o, k_o_o, k_o_i, i_o)
        else:
            sch.reorder(i_o, j_o, k_o_o, k_o_i)
        sch.bind(j_o, "threadIdx.y")
        # Item 0k: tag the row-fragment loop so `_guard_padding_rows` can shorten it to
        # the fragments that hold real rows.
        #
        # Only when the guard is actually wanted. An annotation is not free even if
        # nothing reads it: it stops TVM eliminating the unit loop, so at the shipped
        # BLK_M=16 an unconditional `annotate` emits `for (a0_0 = 0; a0_0 < 1; ++a0_0)`
        # around the entire CTA body. nvcc would delete that, but "the shipped kernel is
        # unchanged" is a claim worth being able to prove by diffing the generated CUDA
        # (scripts/moe_dump_cuda.py) rather than by arguing about the optimiser.
        if SKIPROWS or _rowspec_active:
            sch.annotate(i_o, "moe_row_guard", 1)
        # Tag k_o_o so item 0f can find it by name. It used to be located by matching
        # `extent == K // BLK_K`, which is not unique — at K=512, BLK_K=64 that extent
        # is 8 and so is another loop, and the rewrite refused to run at all.
        sch.annotate(k_o_o, "moe_koo_guard", 1)

        x_shared = sch.get_sblock("X_shared")
        w_shared = sch.get_sblock("W_shared")

        def _coop(blk):
            sch.compute_at(blk, k_o_o, preserve_unit_loops=True)
            loops = sch.get_loops(blk)[-2:]
            fused = sch.fuse(*loops)
            _, fy, fx, fv = sch.split(fused, factors=[None, TY, WARP, VEC])
            sch.bind(fy, "threadIdx.y")
            sch.bind(fx, "threadIdx.x")
            sch.vectorize(fv)
            sch.storage_align(blk, 0, axis=-2, factor=16, offset=8)

        _coop(x_shared)
        # NB: VEC is capped at 4 here and cannot be widened to cover a whole uint32.
        # At VEC=4 a thread unpacks 4 of the 8 nibbles in a word, so thread pairs fetch
        # the same word; VEC=8 would fix that but the dequant's intermediate is a uint32
        # vector, and `Ramp of more than 4 lanes is not allowed` (128-bit ceiling).
        # The duplicate fetches share an address, so they cost L1 requests, not DRAM.
        _coop(w_shared)

        A_mat = sch.cache_read(block_outer, 0, "wmma.matrix_a")
        B_mat = sch.cache_read(block_outer, 1, "wmma.matrix_b")
        sch.compute_at(A_mat, k_o_i)
        sch.compute_at(B_mat, k_o_i)

        # cache_write: compute writes into wmma.accumulator, auto-block stores
        # accumulator → O_tile (shared.dyn). The explicit "store" sblock copies
        # O_tile → out global with bounds-check predicate.
        acc_blk = sch.cache_write(block_outer, 0, "wmma.accumulator")
        sch.reverse_compute_at(acc_blk, j_o)

        si, sj = sch.get_loops(acc_blk)[-2:]
        si0, si1 = sch.split(si, factors=[None, MICRO])
        sj0, sj1 = sch.split(sj, factors=[None, MICRO])
        sch.reorder(si0, sj0, si1, sj1)

        store_block = sch.get_sblock("store")
        sch.reverse_compute_at(store_block, j_o, preserve_unit_loops=True)
        s_loops = sch.get_loops(store_block)[-2:]
        s_fused = sch.fuse(*s_loops)
        _, s_fx, s_fv = sch.split(s_fused, factors=[None, WARP, VEC])
        sch.bind(s_fx, "threadIdx.x")
        sch.vectorize(s_fv)

        block_init_c = sch.decompose_reduction(block_outer, k_o_o)
        block_init_c_inner = sch.get_child_blocks(block_init_c)[0]

        intrin_group = get_wmma_intrin_group(
            load_scope="shared.dyn", store_scope="shared.dyn",
            in_dtype="float16", out_dtype="float16", trans_b=True,
        )

        ai, aj = sch.get_loops(A_mat)[-2:]
        ai0, ai1 = sch.split(ai, factors=[None, MICRO])
        aj0, aj1 = sch.split(aj, factors=[None, MICRO])
        sch.reorder(ai0, aj0, ai1, aj1)
        sch.unroll(ai0)
        sch.unroll(aj0)
        sch.tensorize(ai1, intrin_group["load_a"])

        bi, bj = sch.get_loops(B_mat)[-2:]
        bi0, bi1 = sch.split(bi, factors=[None, MICRO])
        bj0, bj1 = sch.split(bj, factors=[None, MICRO])
        sch.reorder(bi0, bj0, bi1, bj1)
        sch.unroll(bi0)
        sch.unroll(bj0)
        sch.tensorize(bi1, intrin_group["load_b"])

        sch.tensorize(sch.get_loops(block_init_c_inner)[-2], intrin_group["init"])
        sch.tensorize(sch.get_loops(acc_blk)[-2], intrin_group["store"])
        sch.tensorize(sch.get_loops(block_inner)[-3], intrin_group["compute"])

        return sch.mod["main"]

    # ---------- item 0f: skip the dispatch table's padding CTAs ----------
    def _guard_padding_ctas(func, whole_body: bool = True):
        """Give the sentinel tiles' loops a zero trip count.

        v2's grid is `(ceildiv(B, BLK_M) + Ne) * tiles_per_n`; the `+ Ne` slack gives
        each expert a private index range without a prefix scan, and those slack CTAs
        carry `te[bx] = -1`. Un-guarded they load and dequantize a full BLK_N x K
        weight tile and run the whole wmma reduction before the store predicate
        discards it — §16.8 measured them at 92-94% of a real CTA's cost and 27-50%
        of the grid at prefill's B=4096.

        Two things this is *not*, both of which were tried first (§16.9):

        - not an `if` in the source prim_func: an IfThenElse between an sblock and its
          target loop breaks the scope bookkeeping, and `sch.compute_at` dies with
          `InternalError: unordered_map::at`.
        - not an IfThenElse around the scheduled CTA body either. The cooperative
          loads carry `__syncthreads()`, and ThreadSync refuses on principle:
          `Check failed: condition_counter() == 0 : Cannot insert syncs inside
          condition`. The barrier here is uniform across the CTA, but the pass cannot
          know that.

        Rewriting the *extent* of the k_o_o loop to `Select(e_v >= 0, K/BLK_K, 0)`
        sidesteps both. There is no conditional, so ThreadSync is satisfied; the
        extent is CTA-uniform, so every thread agrees on the trip count and the
        barriers inside the loop are either all executed or all skipped.

        Bit-exact by construction. A skipped CTA still fills its accumulator with
        zeros and still stores it to `O_tile`, and the global store is predicated on
        `row_end`, which is 0 exactly when `e_v < 0` — so nothing it writes was ever
        observable.

        `whole_body` extends the same trick to the `moe_pad_guard` unit loop that
        wraps the entire CTA body, which the k_o_o guard alone cannot reach: §16.10
        measured a k_o_o-skipped CTA at 20% of a full one, the residue being the
        accumulator fill, the accumulator -> O_tile store and the predicated-off
        global store loop. Zeroing the outer extent drops all three. The uniformity
        argument is unchanged — `e_v` is CTA-uniform either way — and so is
        bit-exactness, since none of the skipped writes leave shared memory.
        Set `MLC_MOE_GEMM_V2_SKIPPAD=koo` to A/B against the §16.10 mechanism.
        """
        # Locate `e_v`, bound by the first Bind in the CTA block, which is an
        # ancestor of every loop below and so is in scope at the extent.
        found = []
        tirx.stmt_functor.post_order_visit(
            func.body,
            lambda n: found.append(n.block.body.seq[0].var)
            if isinstance(n, tirx.SBlockRealize) and n.block.name_hint == "CTA"
            else None,
        )
        if len(found) != 1:
            raise RuntimeError(f"expected exactly one CTA sblock, found {len(found)}")
        e_v_var = found[0]

        hits, guards = [], []

        def _zero_extent(node):
            return tirx.For(
                node.loop_var, node.min,
                tirx.Select(e_v_var >= 0, node.extent, tirx.IntImm(node.extent.dtype, 0)),
                node.kind, node.body, node.thread_binding, node.annotations,
            )

        def _rewrite(node):
            if not isinstance(node, tirx.For) or node.thread_binding is not None:
                return None
            ann = node.annotations or {}
            if "moe_pad_guard" in ann:
                guards.append(node)
                return _zero_extent(node) if whole_body else None
            if "moe_koo_guard" in ann:
                hits.append(node)
                return _zero_extent(node)
            return None

        body = tirx.stmt_functor.ir_transform(func.body, None, _rewrite, ["tirx.For"])
        # Both markers must survive the schedule. Silently losing either one costs a
        # measured chunk of the win rather than producing a wrong answer, which is
        # exactly the kind of regression that hides — so fail instead.
        for tag, got in (("moe_pad_guard", guards), ("moe_koo_guard", hits)):
            if len(got) != 1:
                raise RuntimeError(
                    f"expected exactly one `{tag}` loop, found {len(got)}; "
                    "the schedule dropped or duplicated it"
                )
        return func.with_body(body)

    def _cta_row_bounds(func):
        """The CTA block's `m_offset` and `row_end` vars, found by name over the body.

        Both are bound near the top of the CTA block, but the schedule moves and re-nests
        those bindings, so walking the leading let-chain does not find them. Duck-typed on
        `.var` because the let-statement node is not exported under a stable name from
        `tvm.tirx`. Shared by items 0k and 0l, which need the same two vars.
        """
        want = {"m_offset", "row_end"}
        lets: dict = {}

        def _collect(n):
            var = getattr(n, "var", None)
            if var is not None and getattr(var, "name", None) in want:
                lets.setdefault(var.name, var)

        tirx.stmt_functor.post_order_visit(func.body, _collect)
        missing = want - set(lets)
        if missing:
            raise RuntimeError(
                f"CTA block does not bind {missing} (found {sorted(lets)}); "
                "source layout changed"
            )
        return lets["m_offset"], lets["row_end"]

    # ---------- item 0k: skip row fragments that hold no real rows ----------
    def _guard_padding_rows(func):
        """Shorten the row-fragment loop to `ceildiv(real rows in this tile, MICRO)`.

        A CTA covers `BLK_M` rows of one expert, but the expert's last tile is usually
        partial: item 0i measured a real pp512 prefill at ~23 rows per hit expert, so at
        `BLK_M=64` a tile carries 23 real rows and 41 padding ones. The `X_shared` load is
        already predicated on `row_end`, so those rows cost no DRAM traffic — but the wmma
        reduction still runs over all `BLK_M/MICRO` fragments and two of the four here are
        entirely zeros.

        ⚠️ **This docstring used to claim this is "the whole reason" widening `BLK_M` has a
        short-prompt cost. §19.3 refutes that.** Item 0l removed this compute without 0k's
        overhead and recovered 2 points of a 31-point gap at B=1024, so padding-fragment
        reduction is a real cost worth 0-3% and not the one that matters. What survives is
        per-CTA and sits *above* this loop -- the `X_shared` cooperative store and the
        `A_mat` fragment loads both run `BLK_M`-wide unconditionally. See item 0n.

        Mechanism is item 0f's, for item 0f's reason: an `if` cannot wrap these loops
        because the cooperative loads carry `__syncthreads()` and ThreadSync refuses to
        place a barrier inside a condition. The extent is CTA-uniform — it depends only on
        `tm[bx]` and `indptr` — so every thread agrees on the trip count.

        Bit-exact by construction: a skipped fragment's accumulator is never initialised
        and never stored, and the global store is predicated on `m_offset + i < row_end`,
        which is false for exactly those rows. Nothing skipped was ever observable, the
        same argument that gates item 0f.
        """
        m_off, row_end = _cta_row_bounds(func)

        hits = []

        def _rewrite(n):
            if not isinstance(n, tirx.For) or n.thread_binding is not None:
                return None
            if "moe_row_guard" not in (n.annotations or {}):
                return None
            hits.append(n)
            zero = tirx.IntImm(m_off.dtype, 0)
            rows = tirx.Max(row_end - m_off, zero)
            frags = tirx.floordiv(rows + (MICRO - 1), MICRO)
            return tirx.For(
                n.loop_var, n.min,
                tirx.Min(tirx.Cast(n.extent.dtype, frags), n.extent),
                n.kind, n.body, n.thread_binding, n.annotations,
            )

        body = tirx.stmt_functor.ir_transform(func.body, None, _rewrite, ["tirx.For"])
        # Two hits is the expected shape, not drift: `sch.blockize` leaves the row-fragment
        # loop in the compute nest, and `reverse_compute_at` of the accumulator -> O_tile
        # store re-materialises it in the store nest. Both must be shortened, and both are
        # safe to shorten for the same reason — a fragment that holds no real rows has its
        # global store predicated off regardless of what O_tile holds.
        if not 1 <= len(hits) <= 2:
            raise RuntimeError(
                f"expected 1 or 2 `moe_row_guard` loops, found {len(hits)}; "
                "the schedule dropped or duplicated it"
            )
        if os.environ.get("MLC_MOE_GEMM_V2_SKIPROWS_DEBUG"):
            for i, h in enumerate(hits):
                print(f"[0k] guard {i}: var={h.loop_var.name} extent={h.extent} "
                      f"body={str(h.body)[:120]!r}")
        return func.with_body(body)

    # ---------- item 0l: predicate the row fragments, keeping a static trip count ----------
    def _specialise_row_fragments(func):
        """Skip empty row fragments *without* making the loop's trip count dynamic.

        Item 0k (`_guard_padding_rows`) shortened the row-fragment loop's extent to
        `ceildiv(real rows, MICRO)`. It is correct and bit-exact, and it loses: §18.7
        measured 0.91x at `BLK_M=64`, and — the tell — 0.91-0.95x at `BLK_M=16`, where
        the guard can only ever produce extent 1 and is therefore doing nothing. The
        cost is not the skipping, it is that a runtime extent stops the fragment loop
        being unrolled with constant fragment indices, and every CTA pays it whether or
        not it has an empty fragment to skip.

        So keep the extent at the compile-time `BLK_M // MICRO` and predicate the body:

            for i_o in range(BLK_M // MICRO):      # unchanged, still constant
                if i_o * MICRO < row_end - m_offset:
                    <wmma fragment>

        Every fragment index stays a literal after unrolling, and an empty fragment
        costs a CTA-uniform branch instead of a full wmma reduction. The predicate is
        uniform — it reads only `tm[bx]` and `indptr` — so no warp diverges on it.

        Legality is the mirror image of item 0f's. 0f could not use a condition because
        the region it wanted to skip contains `__syncthreads()` and ThreadSync refuses
        `Cannot insert syncs inside condition`; it had to move the guard into a loop
        *extent*. Here, under the hoist, the fragment loop sits below `k_o_o` and the
        cooperative loads (with their barriers) sit above it, so the predicated region is
        barrier-free and a plain `IfThenElse` is admissible. The `assert` at the top of
        this function is what keeps that precondition true.

        Bit-exact by the same argument as 0k: a skipped fragment's accumulator is left
        untouched and its global store is predicated off by `m_offset + i < row_end`,
        which is false for exactly the rows a skipped fragment covers.
        """
        m_off, row_end = _cta_row_bounds(func)

        hits = []

        def _rewrite(n):
            if not isinstance(n, tirx.For) or n.thread_binding is not None:
                return None
            if "moe_row_guard" not in (n.annotations or {}):
                return None
            hits.append(n)
            # `n.loop_var` counts fragments, so its first row is `loop_var * MICRO`.
            # A fragment is live iff that row is still inside the expert's rows.
            live = n.loop_var * tirx.IntImm(n.loop_var.dtype, MICRO) < (row_end - m_off)
            return tirx.For(
                n.loop_var, n.min, n.extent, n.kind,
                tirx.IfThenElse(live, n.body, None),
                n.thread_binding, n.annotations,
            )

        body = tirx.stmt_functor.ir_transform(func.body, None, _rewrite, ["tirx.For"])
        # Same 1-or-2 shape as item 0k, for the same reason: `sch.blockize` leaves the
        # row-fragment loop in the compute nest and `reverse_compute_at` re-materialises
        # it in the accumulator -> O_tile store nest. Both are safe to predicate.
        if not 1 <= len(hits) <= 2:
            raise RuntimeError(
                f"expected 1 or 2 `moe_row_guard` loops, found {len(hits)}; "
                "the schedule dropped or duplicated it"
            )
        if os.environ.get("MLC_MOE_GEMM_V2_ROWSPEC_DEBUG"):
            for i, h in enumerate(hits):
                print(f"[0l] guard {i}: var={h.loop_var.name} extent={h.extent} "
                      f"body={str(h.body)[:120]!r}")
        return func.with_body(body)

    scheduled = _schedule_v2()
    if _rowspec_active:
        scheduled = _specialise_row_fragments(scheduled)
    # Item 0k, opt-in while it is being measured. Inert at BLK_M=16, where the row loop
    # has extent 1 and the only tile it can shorten is a padding CTA that item 0f already
    # skips whole.
    if SKIPROWS:
        scheduled = _guard_padding_rows(scheduled)
    # Default ON since §16.11: bit-exact on all 8 gate cases, +19.4% pp512 on the 35B
    # (644.26 -> 769.18 tps), decode neutral, and the state gate is *identical* to the
    # pre-change lib in both prefix-cache modes.
    #   `MLC_MOE_GEMM_V2_SKIPPAD=0`   restores the un-skipped kernel
    #   `MLC_MOE_GEMM_V2_SKIPPAD=koo` restores §16.10's k_o_o-only guard, which leaves
    #                                 a skipped CTA at 20% of a full one
    _skippad = os.environ.get("MLC_MOE_GEMM_V2_SKIPPAD", "1")
    if _skippad != "0":
        scheduled = _guard_padding_ctas(scheduled, whole_body=_skippad != "koo")

    return op.tensor_ir_op(
        scheduled,
        "dequantize_group_gemm_v2",
        args=[x, w, scale, indptr, te_t, tm_t, tn_t],
        out=Tensor.placeholder([x.shape[0], out_features], model_dtype),
    )


def dequantize_group_gemm(
    x: Tensor,
    w: Tensor,
    scale: Tensor,
    indptr: Tensor,
    quantize_dtype: str,
    indptr_dtype: str,
    group_size: int,
):
    """Group GEMM in MoE models but the weight is quantized.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape (batch_size, in_features), where `batch_size` could be dynamic shape.

    w : Tensor
        Weight tensor of shape (num_local_experts, out_features, in_features // n), where n is the
        number of elements per storage dtype, e.g. if the storage dtype is uint32, and the quantize
        dtype is int4, then n is 8.

    scale : Tensor
        The scale tensor of shape (num_local_experts, out_features, in_features // group_size).

    indptr : Tensor
        Index pointer tensor of shape (num_local_experts + 1, ). `x[indptr[a] : indptr[a + 1]]` is
        the input for the `i`-th local expert.

    group_size : int
        The number of elements in each quantization group, e.g. 32 or 128.

    quantize_dtype : str
        The quantize dtype of the weight tensor, which is usually int3, int4 or fp8, etc.

    indptr_dtype : str
        The dtype of the index pointer tensor, which can be int32 or int64.

    Returns
    -------
    out : Tensor
        Output tensor of shape (batch_size, out_features).
    """
    # Phase 9b — opt-in to dispatch-table + tensor-core kernel via env var.
    # Default stays on the v1 persistent-loop kernel until v2 ships.
    if os.environ.get("MLC_MOE_GEMM_V2", "0") == "1" and quantize_dtype == "int4":
        return _dequantize_group_gemm_v2(
            x, w, scale, indptr,
            quantize_dtype=quantize_dtype,
            indptr_dtype=indptr_dtype,
            group_size=group_size,
        )

    (_, in_features), model_dtype = x.shape, x.dtype
    (num_local_experts, out_features, _), storage_dtype = w.shape, w.dtype
    quantize_dtype_bits = DataType(quantize_dtype).bits
    num_elem_per_storage = DataType(storage_dtype).bits // quantize_dtype_bits
    num_group = (in_features + group_size - 1) // group_size
    num_storage = group_size // num_elem_per_storage * num_group

    def _dequantize(w, s, e, i, j):
        tir_bin_mask = tirx.const((1 << quantize_dtype_bits) - 1, storage_dtype)
        tir_max_int = tirx.const((2 ** (quantize_dtype_bits - 1)) - 1, model_dtype)
        w = w[e, i, j // num_elem_per_storage]
        s = s[e, i, j // group_size]
        shift = (j % num_elem_per_storage * quantize_dtype_bits).astype(storage_dtype)
        w = tirx.bitwise_and(tirx.shift_right(w, shift), tir_bin_mask).astype(model_dtype)
        return (w - tir_max_int) * s

    Ne, N, K = num_local_experts, out_features, in_features
    BLK_M, BLK_N, BLK_K = 8, 128, 32
    # CTA_COUNT history (Orin AGX, sm_87, 35B-A3B q4f16_1, top_k=8):
    #   1024 — Hopper default. Used pre-9e6c17ff.
    #     64 — 9e6c17ff (Phase 4) tuned for b=1 top-8 MoE GEMM at ~64-128
    #          tiles. Comment claimed "decode 2.0×/3.6×, prefill ~3% regression."
    #          The decode claim is now stale — qwen3_5_moe added a static
    #          `if num_tokens == 1: dequantize_gemv` shortcut so b=1 decode
    #          never reaches this kernel (see qwen3_5_moe_model.py:137-138).
    #   1024 (current) — restored 2026-04-30 in Phase 9 Stage 9.2. Bench
    #          measured pp512 +2.9% (202.0 → 207.9 tps), tg512 unchanged
    #          (decode shortcut keeps it off this kernel). The original
    #          "~3% prefill regression" claim was approximately right.
    #
    # Tile-constant ceiling on Orin: this hand-schedule sits near a local
    # optimum where shared-mem footprint (~10 KB / CTA at BLK_M=8/BLK_K=32)
    # gives ~9 active blocks / SM. Bigger tiles regress (BLK_M=16/BLK_K=64
    # bumps to ~26 KB / CTA → ~3 blocks / SM, −3.4% pp vs CTA=1024 baseline,
    # measured 2026-04-30). Bigger CTA grids don't help because Orin's 16 SMs
    # cap concurrent blocks at ~32-64 regardless of grid size — over-subscribed
    # CTAs queue serially through the persistent loop.
    #
    # The +40% prefill gap to llama.cpp is *not* reachable through tile-constant
    # tuning of this kernel. Real lever is tensor-core MMA (sm_87 supports
    # m16n8k16 fp16 MMA) or meta-schedule on an unscheduled variant. Phase 9
    # closed as partial; see worklog 2026-04-30 for the full bench tables.
    TX, TY, CTA_COUNT = 8, 32, 1024
    VEC_X, VEC_W, VEC_O, VEC_DOT = 1, 1, 1, 1
    UNROLL = 64
    STORAGE_ALIGN = False
    assert BLK_K % 8 == 0
    tiles_per_row = (N + BLK_N - 1) // BLK_N
    zero = tirx.const(0, model_dtype)
    if indptr_dtype == "int64":
        indptr = op.pad(indptr, [1, 0], "constant", 0)

    @T.prim_func(private=True)
    def _func(
        var_x: T.handle,
        w: T.Buffer((Ne, N, num_storage), storage_dtype),
        scale: T.Buffer((Ne, N, num_group), model_dtype),
        indptr: T.Buffer((Ne + 1,), indptr_dtype),
        var_o: T.handle,
    ):
        T.func_attr({"tirx.is_scheduled": 1, "tirx.noalias": True})
        B = T.int32(is_size_var=True)
        X = T.match_buffer(var_x, (B, K), model_dtype)
        out = T.match_buffer(var_o, (B, N), model_dtype)
        for _bx in T.thread_binding(CTA_COUNT, thread="blockIdx.x"):
            with T.sblock("CTA"):
                bx = T.axis.spatial(CTA_COUNT, _bx)
                T.reads(X[:, :], w[:, :, :], scale[:, :, :], indptr[:])
                T.writes(out[:, :])
                sum = T.sblock_alloc_buffer((2,), indptr_dtype, scope="local")
                row = T.sblock_alloc_buffer((2,), indptr_dtype, scope="local")
                cur_e = T.sblock_alloc_buffer((1,), indptr_dtype, scope="local")
                tile_id = T.sblock_alloc_buffer((1,), indptr_dtype, scope="local")
                sum[0] = 0
                sum[1] = T.ceildiv(indptr[1] - indptr[0], BLK_M) * tiles_per_row
                row[0] = 0
                row[1] = indptr[1] - indptr[0]
                cur_e[0] = 0
                tile_id[0] = bx
                while T.tvm_thread_invariant(cur_e[0] < Ne):
                    # move to the current group
                    while sum[1] <= tile_id[0] and cur_e[0] < Ne:
                        cur_e[0] += 1
                        if cur_e[0] < Ne:
                            e = cur_e[0]
                            delta = indptr[e + 1] - indptr[e]
                            sum[0] = sum[1]
                            sum[1] += T.ceildiv(delta, BLK_M) * tiles_per_row
                            row[0] = row[1]
                            row[1] += delta
                    # sync threads to make sure all threads have the same tile position
                    T.tvm_storage_sync("shared")
                    if T.tvm_thread_invariant(cur_e[0] < Ne):
                        # fetch current tile position
                        e = cur_e[0]
                        num_tiles = tile_id[0] - sum[0]
                        m_offset = T.floordiv(num_tiles, tiles_per_row) * BLK_M + row[0]
                        n_offset = T.floormod(num_tiles, tiles_per_row) * BLK_N
                        with T.sblock("gemm"):
                            T.reads(
                                row[1],
                                X[m_offset : m_offset + BLK_M, :],
                                w[e, n_offset : n_offset + BLK_N, :],
                                scale[e, n_offset : n_offset + BLK_N, :],
                            )
                            T.writes(
                                out[
                                    m_offset : m_offset + BLK_M,
                                    n_offset : n_offset + BLK_N,
                                ]
                            )
                            X_tile = T.sblock_alloc_buffer((BLK_M, K), model_dtype, scope="shared")
                            W_tile = T.sblock_alloc_buffer((BLK_N, K), model_dtype, scope="shared")
                            O_tile = T.sblock_alloc_buffer((BLK_M, BLK_N), "float32", scope="local")
                            for a0, a1 in T.grid(BLK_M, K):
                                with T.sblock("X_shared"):
                                    i, j = T.axis.remap("SS", [a0, a1])
                                    X_tile[i, j] = T.if_then_else(
                                        m_offset + i < row[1],
                                        X[m_offset + i, j],
                                        zero,
                                    )
                            for a0, a1 in T.grid(BLK_N, K):
                                with T.sblock("W_shared"):
                                    i, j = T.axis.remap("SS", [a0, a1])
                                    W_tile[i, j] = T.if_then_else(
                                        n_offset + i < N,
                                        _dequantize(w, scale, e, n_offset + i, j),
                                        zero,
                                    )
                            for a0, a1, a2 in T.grid(BLK_M, BLK_N, K):
                                with T.sblock("compute"):
                                    i, j, k = T.axis.remap("SSR", [a0, a1, a2])
                                    with T.init():
                                        O_tile[i, j] = zero
                                    O_tile[i, j] += X_tile[i, k] * W_tile[j, k]
                            for a0, a1 in T.grid(BLK_M, BLK_N):
                                with T.sblock("store"):
                                    i, j = T.axis.remap("SS", [a0, a1])
                                    if m_offset + i < row[1] and n_offset + j < N:
                                        out[m_offset + i, n_offset + j] = O_tile[i, j]
                    # move to next tile
                    tile_id[0] += CTA_COUNT

    def _schedule():
        sch = s_tir.Schedule(_func)

        def _cooperative_fetch(block, vec_len):
            num_loops = len(sch.get_loops(block))
            sch.compute_at(block, ko, preserve_unit_loops=True)
            loops = sch.get_loops(block)[-num_loops:]
            ty, tx, _, vec = sch.split(
                sch.fuse(*loops),
                factors=[TY, TX, None, vec_len],
            )
            sch.vectorize(vec)
            sch.bind(ty, "threadIdx.y")
            sch.bind(tx, "threadIdx.x")
            if STORAGE_ALIGN:
                sch.storage_align(block, 0, axis=1, factor=8, offset=vec_len)
            return block

        main_block = sch.get_sblock("compute")
        x, y, k = sch.get_loops(main_block)
        ty, yi = sch.split(y, [TY, None])
        tx, xi, vec_c = sch.split(x, [TX, None, VEC_DOT])
        ko, ki = sch.split(k, factors=[None, BLK_K])
        sch.reorder(ty, tx, ko, ki, yi, xi, vec_c)
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec_c)
        if UNROLL > 0:
            sch.annotate(tx, ann_key="pragma_auto_unroll_max_step", ann_val=UNROLL)
            sch.annotate(tx, ann_key="pragma_unroll_explicit", ann_val=1)
        l2g = sch.get_sblock("store")
        sch.reverse_compute_at(l2g, tx, preserve_unit_loops=True)
        _, v = sch.split(sch.get_loops(l2g)[-1], [None, VEC_O])
        sch.vectorize(v)
        _cooperative_fetch(sch.get_sblock("X_shared"), vec_len=VEC_X)
        _cooperative_fetch(sch.get_sblock("W_shared"), vec_len=VEC_W)
        sch.decompose_reduction(main_block, ko)
        return sch.mod["main"]

    return op.tensor_ir_op(
        _schedule(),
        "dequantize_group_gemm",
        args=[x, w, scale, indptr],
        out=Tensor.placeholder([x.shape[0], out_features], model_dtype),
    )
