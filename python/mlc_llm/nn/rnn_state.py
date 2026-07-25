"""RNN State modeling."""

from collections.abc import Sequence
from typing import Tuple, Union

import tvm
from tvm import relax as rx
from tvm import s_tir, tirx
from tvm.relax.frontend.nn import Object, Tensor
from tvm.script import tirx as T


def _schedule_state_copy(prim_func: tirx.PrimFunc, dtype: str) -> tirx.PrimFunc:
    """Bind state-copy PrimFunc loops to (block, thread) with a vectorized inner.

    The default dlight Fallback splits with 1024 threads/block doing 1 elt each,
    capping memory throughput around 19% of LPDDR5 peak on Orin AGX (ncu confirmed).
    This schedule pushes per-thread work to 16 bytes (LDG.128/STG.128) and shrinks
    block size to 256, which gets multiple in-flight blocks per SM and saturates
    the wide-load path.

    Falls back to a non-vectorized binding if the innermost extent isn't a
    multiple of the vector width — guards safely against odd state shapes.
    """
    bits = tvm.DataType(dtype).bits
    if bits not in (16, 32):
        # exotic dtype — leave to dlight Fallback
        return prim_func
    vec_width = 16 // (bits // 8)  # 16 bytes / per-elem bytes  → 4 (fp32), 8 (fp16)

    sch = s_tir.Schedule(prim_func)
    block = sch.get_sblock("copy")
    loops = sch.get_loops(block)
    if len(loops) < 2:
        return prim_func  # not enough loops to vectorize against
    inner = loops[-1]
    inner_extent = sch.get(inner).extent
    # Need a static, divisible inner extent — fall back if not.
    if not isinstance(inner_extent, tvm.tirx.IntImm) or int(inner_extent) % vec_width != 0:
        return prim_func

    inner_outer, inner_vec = sch.split(inner, factors=[None, vec_width])
    fused = sch.fuse(*loops[:-1], inner_outer)
    bx, tx = sch.split(fused, factors=[None, 256])
    sch.bind(bx, "blockIdx.x")
    sch.bind(tx, "threadIdx.x")
    sch.vectorize(inner_vec)
    new_func = sch.mod["main"]
    return new_func.with_attr("tirx.is_scheduled", 1)


class RNNState(Object):
    """The RNN State used in Space State Models"""

    @staticmethod
    def create(
        max_batch_size: tirx.Var,
        num_hidden_layers: int,
        max_history: int,
        init_values: Sequence[rx.Constant],
        name: str = "rnn_state",
    ) -> "RNNState":
        """Create a RNN state object.

        Parameters
        ----------
        max_batch_size : tirx.Var
            The maximum batch size.
        num_hidden_layers : int
            The number of hidden layers.
        max_history : int
            The maximum history length.
        init_values : Sequence[rx.Constant]
            The initial values of the RNN state. Must be compile-time Relax constants
            (e.g. R.const(np.zeros(...))).
        """

        bb = rx.BlockBuilder.current()
        state_infos = [
            (tuple(int(x) for x in v.data.shape), str(v.data.dtype)) for v in init_values
        ]

        f_gets = [
            bb.add_func(
                RNNState.create_get_func(shape, dtype, max_batch_size, max_history, id),
                f"rnn_state_get_{id}",
            )
            for id, (shape, dtype) in enumerate(state_infos)
        ]
        f_sets = [
            bb.add_func(
                RNNState.create_set_func(shape, dtype, max_batch_size, max_history, id),
                f"rnn_state_set_{id}",
            )
            for id, (shape, dtype) in enumerate(state_infos)
        ]
        f_sets_with_history = [
            bb.add_func(
                RNNState.create_set_with_history_func(
                    shape, dtype, max_batch_size, max_history, id
                ),
                f"rnn_state_set_with_history_{id}",
            )
            for id, (shape, dtype) in enumerate(state_infos)
        ]

        ret = RNNState(
            _expr=rx.call_pure_packed(
                "vm.builtin.rnn_state_create",
                rx.PrimValue(num_hidden_layers),
                max_batch_size,
                max_history,
                f_gets,
                f_sets,
                f_sets_with_history,
                list(init_values),
                sinfo_args=[rx.ObjectStructInfo()],
            ),
            _name=name,
        )
        return ret

    def get(
        self,
        layer_id: int,
        state_id: int,
        shape: Sequence[tirx.PrimExpr],
        dtype: str,
    ) -> Tensor:
        """Get the state of the RNN layer.

        - If there is only one sequence, we can directly use the storage memory,
        without copying the data.
        - If there are multiple sequences, we need to copy the data to get a contiguous
        memory.

        Parameters
        ----------
        layer_id : int
            The layer id.
        state_id : int
            The state id.
        shape : Sequence[tirx.PrimExpr]
            The shape of the state tensor.
        dtype: str
            The data type of the state tensor.

        Returns
        -------
        Tensor
            The state tensor, with shape `(batch_size, *state_size)`.
        """
        bb = rx.BlockBuilder.current()

        return Tensor(
            _expr=bb.emit(
                rx.call_dps_packed(
                    "vm.builtin.rnn_state_get",
                    [self._expr, layer_id, state_id],
                    out_sinfo=rx.TensorStructInfo(shape, dtype),
                )
            )
        )

    def storage(
        self,
        layer_id: int,
        state_id: int,
        shape: Sequence[tirx.PrimExpr],
        dtype: str,
    ) -> Tensor:
        """Raw handle on the whole state storage, for kernels that update the slot in place.

        Returns the `(max_batch_size, max_history, *state_size)` buffer itself — no copy, and
        no slot indexing. Pair it with `slot_ids()` and index inside the consumer kernel.

        `max_batch_size` and `max_history` are runtime arguments to `create_rnn_state`, so no
        Relax symbolic var naming them is in scope at the use site. The caller therefore
        passes fresh vars for those two leading dims and this method binds them with a
        `match_cast` — the standard Relax way to introduce shape vars from an otherwise
        opaque value. Reusing one pair of vars across every layer is intended: the second
        and later `match_cast`s degrade to cheap host-side assertions that all layers really
        do share a storage geometry.

        This exists so a fused recurrence kernel can replace the `get` -> compute -> `set`
        pair, which costs two full state copies per layer per token (245 MiB/token on the
        35B). Note the docstring on `get` has long claimed the single-sequence case "can
        directly use the storage memory, without copying" — it never did, because `get` is
        emitted as `call_dps_packed`, which by construction allocates a destination for the
        builtin to fill.

        Why not hand back a view of just the active slot instead: the slot offset is
        `(seq_slot_id * max_history + history_slot_id) * state_size`, and `history_slot_id`
        advances every step whenever `max_history > 1` (Phase 8 prefix caching sets it to 64).
        A pointer baked into a captured CUDA graph would then address the wrong slot. Keeping
        the addressing dynamic — whole buffer plus device-side index arrays — is cudagraph-safe
        under any `max_history`.
        """
        bb = rx.BlockBuilder.current()
        raw = bb.emit(
            rx.call_pure_packed(
                "vm.builtin.rnn_state_storage",
                self._expr,
                rx.PrimValue(layer_id),
                rx.PrimValue(state_id),
                sinfo_args=[rx.TensorStructInfo(ndim=len(shape), dtype=dtype)],
            )
        )
        return Tensor(_expr=bb.match_cast(raw, rx.TensorStructInfo(shape, dtype)))

    def slot_ids(self, batch_size: tirx.PrimExpr, dtype: str = "int32") -> Tuple[Tensor, Tensor]:
        """The device-side `(seq_slot_ids, history_slot_ids)` arrays, shape `(batch_size,)`.

        Only valid between `BeginForward` and `EndForward` — the runtime enforces this with
        the same synchronization check `get`/`set` make. Use with `storage()`.
        """
        bb = rx.BlockBuilder.current()
        seq = Tensor(
            _expr=bb.emit(
                rx.call_pure_packed(
                    "vm.builtin.rnn_state_seq_slot_ids",
                    self._expr,
                    sinfo_args=[rx.TensorStructInfo((batch_size,), dtype)],
                )
            )
        )
        hist = Tensor(
            _expr=bb.emit(
                rx.call_pure_packed(
                    "vm.builtin.rnn_state_history_slot_ids",
                    self._expr,
                    sinfo_args=[rx.TensorStructInfo((batch_size,), dtype)],
                )
            )
        )
        return seq, hist

    def set(self, layer_id: int, state_id: int, value: Tensor) -> "RNNState":
        """Set the state of the RNN layer.

        Parameters
        ----------
        layer_id : int
            The layer id.
        state_id : int
            The state id.
        value : Tensor
            The state tensor, with shape `(batch_size, *state_size)`.
        """
        bb = rx.BlockBuilder.current()
        return RNNState(
            _expr=bb.emit(
                rx.call_pure_packed(
                    "vm.builtin.rnn_state_set",
                    self._expr,
                    rx.PrimValue(layer_id),
                    rx.PrimValue(state_id),
                    value._expr,
                    sinfo_args=[rx.ObjectStructInfo()],
                )
            ),
            _name="rnn_state_set",
        )

    def set_with_history(self, layer_id: int, state_id: int, value: Tensor) -> "RNNState":
        """Scatter per-position state into history slots [H+1..H+seq_len].

        Used by speculative-decoding verify on hybrid (attention + recurrent) models so
        a partial accept can roll back the recurrent state to any intermediate position
        via the existing PopN API. Caller must arm the `set_use_history_mode(True)` flag
        before BeginForward so that EndForward advances `history_slot_id` by `seq_len`
        rather than by 1.

        Parameters
        ----------
        layer_id : int
            The layer id.
        state_id : int
            The state id.
        value : Tensor
            The per-position state tensor, with shape `(batch_size, seq_len, *state_size)`.
        """
        bb = rx.BlockBuilder.current()
        return RNNState(
            _expr=bb.emit(
                rx.call_pure_packed(
                    "vm.builtin.rnn_state_set_with_history",
                    self._expr,
                    rx.PrimValue(layer_id),
                    rx.PrimValue(state_id),
                    value._expr,
                    sinfo_args=[rx.ObjectStructInfo()],
                )
            ),
            _name="rnn_state_set_with_history",
        )

    @staticmethod
    def create_get_func(
        shape: Sequence[Union[int, tirx.Var]],
        dtype: str,
        max_batch_size: Union[int, tirx.Var],
        max_history: Union[int, tirx.Var],
        state_id: int,
    ) -> tirx.PrimFunc:
        """Create the get function with given state shape.

        Parameters
        ----------
        shape : Sequence[Union[int, tirx.Var]]
            The shape of the state tensor.

        dtype: str
            The data type of the state tensor.

        max_batch_size : Union[int, tirx.Var]
            The maximum batch size.

        max_history : Union[int, tirx.Var]
            The maximum history length.

        state_id : int
            The id of the state, used for naming the function.

        Returns
        -------
        tirx.PrimFunc
            The get function.
        """

        def _func_one_dim():
            @T.prim_func
            def f(
                var_storage: T.handle,
                var_seq_slot_ids: T.handle,
                var_history_slot_ids: T.handle,
                var_output: T.handle,
            ):
                batch_size = T.int32(is_size_var=True)
                T.func_attr({"global_symbol": f"rnn_state_get_{state_id}"})

                storage = T.match_buffer(
                    var_storage, (max_batch_size, max_history, shape[0]), dtype
                )
                seq_slot_ids = T.match_buffer(var_seq_slot_ids, (batch_size,), "int32")
                history_slot_ids = T.match_buffer(var_history_slot_ids, (batch_size,), "int32")
                output = T.match_buffer(var_output, (batch_size, shape[0]), dtype)

                for i in range(batch_size):
                    for s in range(shape[0]):
                        with T.sblock("copy"):
                            vi, vs = T.axis.remap("SS", [i, s])
                            seq_id: T.int32 = seq_slot_ids[vi]
                            history_id: T.int32 = history_slot_ids[vi]
                            output[vi, vs] = storage[seq_id, history_id, vs]

            return f

        def _func_high_dim():
            # Add a wrapper function to avoid parse the following code when len(shape) = 1
            @T.prim_func
            def f(
                var_storage: T.handle,
                var_seq_slot_ids: T.handle,
                var_history_slot_ids: T.handle,
                var_output: T.handle,
            ):
                batch_size = T.int32(is_size_var=True)
                T.func_attr({"global_symbol": f"rnn_state_get_{state_id}"})

                storage = T.match_buffer(var_storage, (max_batch_size, max_history, *shape), dtype)
                seq_slot_ids = T.match_buffer(var_seq_slot_ids, (batch_size,), "int32")
                history_slot_ids = T.match_buffer(var_history_slot_ids, (batch_size,), "int32")
                output = T.match_buffer(var_output, (batch_size, *shape), dtype)

                for i in range(batch_size):
                    for s in T.grid(*shape):
                        with T.sblock("copy"):
                            vi, *vs = T.axis.remap("S" * (len(shape) + 1), [i, *s])
                            seq_id: T.int32 = seq_slot_ids[vi]
                            history_id: T.int32 = history_slot_ids[vi]
                            # The following line is equivalent to:
                            # `output[vi, *vs] = storage[seq_id, history_id, *vs]`
                            # However, unpacking operator in subscript requires Python 3.11 or newer
                            T.buffer_store(
                                output,
                                T.BufferLoad(storage, [seq_id, history_id, *vs]),
                                [vi, *vs],
                            )

            return f

        f = _func_one_dim() if len(shape) == 1 else _func_high_dim()
        return _schedule_state_copy(f, dtype)

    @staticmethod
    def create_set_with_history_func(
        shape: Sequence[Union[int, tirx.Var]],
        dtype: str,
        max_batch_size: Union[int, tirx.Var],
        max_history: Union[int, tirx.Var],
        state_id: int,
    ) -> tirx.PrimFunc:
        """Per-position scatter-set kernel.

        Writes `data[i, t, *vs]` to `storage[seq_slot_ids[i],
        (history_slot_ids[i] + 1 + t) mod max_history, *vs]` for each batch element `i`
        and inner-seq position `t`. Caller must guarantee `max_history >= seq_len + 1`
        so writes do not collide.
        """

        def _func_one_dim():
            @T.prim_func
            def f(
                var_storage: T.handle,
                var_seq_slot_ids: T.handle,
                var_history_slot_ids: T.handle,
                var_data: T.handle,
            ):
                batch_size = T.int32(is_size_var=True)
                seq_len = T.int32(is_size_var=True)
                T.func_attr({"global_symbol": f"rnn_state_set_with_history_{state_id}"})

                storage = T.match_buffer(
                    var_storage, (max_batch_size, max_history, shape[0]), dtype
                )
                seq_slot_ids = T.match_buffer(var_seq_slot_ids, (batch_size,), "int32")
                history_slot_ids = T.match_buffer(var_history_slot_ids, (batch_size,), "int32")
                data = T.match_buffer(var_data, (batch_size, seq_len, shape[0]), dtype)

                for i, t in T.grid(batch_size, seq_len):
                    for s in range(shape[0]):
                        with T.sblock("copy"):
                            vi, vt, vs = T.axis.remap("SSS", [i, t, s])
                            seq_id: T.int32 = seq_slot_ids[vi]
                            history_id: T.int32 = (
                                history_slot_ids[vi] + 1 + vt
                            ) % T.cast(max_history, "int32")
                            storage[seq_id, history_id, vs] = data[vi, vt, vs]

            return f

        def _func_high_dim():
            @T.prim_func
            def f(
                var_storage: T.handle,
                var_seq_slot_ids: T.handle,
                var_history_slot_ids: T.handle,
                var_data: T.handle,
            ):
                batch_size = T.int32(is_size_var=True)
                seq_len = T.int32(is_size_var=True)
                T.func_attr({"global_symbol": f"rnn_state_set_with_history_{state_id}"})

                storage = T.match_buffer(var_storage, (max_batch_size, max_history, *shape), dtype)
                seq_slot_ids = T.match_buffer(var_seq_slot_ids, (batch_size,), "int32")
                history_slot_ids = T.match_buffer(var_history_slot_ids, (batch_size,), "int32")
                data = T.match_buffer(var_data, (batch_size, seq_len, *shape), dtype)

                for i, t in T.grid(batch_size, seq_len):
                    for s in T.grid(*shape):
                        with T.sblock("copy"):
                            vi, vt, *vs = T.axis.remap("S" * (len(shape) + 2), [i, t, *s])
                            seq_id: T.int32 = seq_slot_ids[vi]
                            history_id: T.int32 = (
                                history_slot_ids[vi] + 1 + vt
                            ) % T.cast(max_history, "int32")
                            T.buffer_store(
                                storage,
                                T.BufferLoad(data, [vi, vt, *vs]),
                                [seq_id, history_id, *vs],
                            )

            return f

        f = _func_one_dim() if len(shape) == 1 else _func_high_dim()
        return _schedule_state_copy(f, dtype)

    @staticmethod
    def create_set_func(
        shape: Sequence[Union[int, tirx.Var]],
        dtype: str,
        max_batch_size: Union[int, tirx.Var],
        max_history: Union[int, tirx.Var],
        state_id: int,
    ) -> tirx.PrimFunc:
        """Create the set function with given state shape.

        Parameters
        ----------
        shape : Sequence[Union[int, tirx.Var]]
            The shape of the state tensor.

        dtype: str
            The data type of the state tensor.

        max_batch_size : Union[int, tirx.Var]
            The maximum batch size.

        max_history : Union[int, tirx.Var]
            The maximum history length.

        state_id : int
            The id of the state, used for naming the function.

        Returns
        -------
        tirx.PrimFunc
            The set function.
        """

        def _func_one_dim():
            @T.prim_func
            def f(
                var_storage: T.handle,
                var_seq_slot_ids: T.handle,
                var_history_slot_ids: T.handle,
                var_data: T.handle,
            ):
                batch_size = T.int32(is_size_var=True)
                T.func_attr({"global_symbol": f"rnn_state_set_{state_id}"})

                storage = T.match_buffer(
                    var_storage, (max_batch_size, max_history, shape[0]), dtype
                )
                seq_slot_ids = T.match_buffer(var_seq_slot_ids, (batch_size,), "int32")
                history_slot_ids = T.match_buffer(var_history_slot_ids, (batch_size,), "int32")
                data = T.match_buffer(var_data, (batch_size, shape[0]), dtype)

                for i in range(batch_size):
                    for s in range(shape[0]):
                        with T.sblock("copy"):
                            vi, vs = T.axis.remap("SS", [i, s])
                            seq_id: T.int32 = seq_slot_ids[vi]
                            history_id: T.int32 = (history_slot_ids[vi] + 1) % T.cast(
                                max_history, "int32"
                            )
                            storage[seq_id, history_id, vs] = data[vi, vs]

            return f

        def _func_high_dim():
            @T.prim_func
            def f(
                var_storage: T.handle,
                var_seq_slot_ids: T.handle,
                var_history_slot_ids: T.handle,
                var_data: T.handle,
            ):
                batch_size = T.int32(is_size_var=True)
                T.func_attr({"global_symbol": f"rnn_state_set_{state_id}"})

                storage = T.match_buffer(var_storage, (max_batch_size, max_history, *shape), dtype)
                seq_slot_ids = T.match_buffer(var_seq_slot_ids, (batch_size,), "int32")
                history_slot_ids = T.match_buffer(var_history_slot_ids, (batch_size,), "int32")
                data = T.match_buffer(var_data, (batch_size, *shape), dtype)

                for i in range(batch_size):
                    for s in T.grid(*shape):
                        with T.sblock("copy"):
                            vi, *vs = T.axis.remap("S" * (len(shape) + 1), [i, *s])
                            seq_id: T.int32 = seq_slot_ids[vi]
                            history_id: T.int32 = (history_slot_ids[vi] + 1) % T.cast(
                                max_history, "int32"
                            )
                            # The following line is equivalent to:
                            # `storage[seq_id, history_id, *vs] = data[vi, *vs]`
                            # However, unpacking operator in subscript requires Python 3.11 or newer
                            T.buffer_store(
                                storage,
                                T.BufferLoad(data, [vi, *vs]),
                                [seq_id, history_id, *vs],
                            )

            return f

        f = _func_one_dim() if len(shape) == 1 else _func_high_dim()
        return _schedule_state_copy(f, dtype)
