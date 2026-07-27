"""A compiler pass that dispatches patterns to CUBLAS."""

import os

import tvm
from tvm import IRModule, relax
from tvm.relax.backend import get_patterns_with_prefix

try:
    import tvm.relax.backend.cuda.cublas as _cublas  # noqa: F401
    import tvm.relax.backend.rocm.hipblas as _hipblas  # noqa: F401
except ImportError:
    # Note: legacy path of cublas/hipblas for backward compatibility
    pass


def _region_needs_tir_vars(context: relax.transform.PatternCheckContext) -> bool:
    """Would fusing this match produce a composite function with a `R.Shape` parameter?

    When `FuseOpsByPattern` lifts a matched region into a function, any symbolic variable
    the region *uses* but that none of its parameters *define* is passed in through an
    appended `tir_vars: R.Shape([...])` parameter
    ([fuse_ops.cc:567](3rdparty/tvm/src/relax/transform/fuse_ops.cc#L567)). The BYOC JSON
    serializer then walks every parameter and requires a `TensorStructInfo`
    ([codegen_json.h:289](3rdparty/tvm/src/relax/backend/contrib/codegen_json/codegen_json.h#L289)),
    so that extra parameter aborts the whole compile with

        Check failed: (tensor_sinfo) is false:
            Expect TensorStructInfo, but received: relax.ShapeStructInfo

    A variable counts as *defined* by a parameter only when it appears as a bare `tir.Var`
    in that parameter's shape. `R.Tensor((seq_len, 2048))` defines `seq_len`;
    `R.Tensor((num_patches // 4, 3072))` defines nothing, and `num_patches` is then free.
    The second form is what the VL patch merger produces —
    [qwen3_vl_vit.py](../model/vision/qwen3_vl_vit.py) reshapes `(n, hidden)` to
    `(n // merge_sq, merge_sq * hidden)` before `linear_fc1`/`linear_fc2` — which is why
    this fires on the vision tower and on nothing in the text model.

    Rather than let the pass die, decline the match: those two matmuls fall back to the
    normal generated kernel and every other cuBLAS offload in the model is kept. Declining
    is always safe — a pattern check returning False is the supported way to say "not this
    one" — and it is far narrower than the `cublas_gemm=0` workaround it replaces.
    """
    matched = set(context.matched_bindings.keys())
    used, definable = set(), set()

    def _account(sinfo, is_input: bool) -> None:
        used.update(relax.analysis.tir_vars_in_struct_info(sinfo))
        if is_input:
            definable.update(relax.analysis.definable_tir_vars_in_struct_info(sinfo))

    for value in context.matched_bindings.values():
        _account(value.struct_info, is_input=False)
        for arg in getattr(value, "args", []):
            # Only a *parameter* of the lifted function can define a variable. An argument
            # bound by another matched binding is internal to the region; a ShapeExpr or
            # PrimValue argument is inlined into the body rather than parameterised
            # ([fuse_ops.cc:644](3rdparty/tvm/src/relax/transform/fuse_ops.cc#L644)), so a
            # bare `tir.Var` sitting in one of those does not define it either. Counting
            # them would be the unsafe direction — it would let a match through that then
            # kills the compile.
            is_param = arg not in matched and not isinstance(
                arg, relax.ShapeExpr | relax.PrimValue
            )
            _account(arg.struct_info, is_input=is_param)

    return bool(used - definable)


def _region_is_fp32(context: relax.transform.PatternCheckContext) -> bool:
    """Does this match compute in fp32?

    Used to decline the offload — see `_decline_free_symbolic_vars` for why that
    is the safe way to say "not this one". The rule is measured, not assumed
    (workplan §20):

    On the Qwen3.5-VL vision tower cuBLAS picks `ampere_sgemm_128x128_tn` for the
    fp32 `matmul(q32, k_t)` and `ampere_fp16_s16816gemm_*` for the fp16 FFN GEMMs.
    The fp16 kernels are tensor-core; the fp32 one is plain SIMT. So the fp32
    offload buys a *scheduling* win only — 2.69 vs 1.03 TFLOP/s, real but small in
    absolute terms — while costing the `matmul -> multiply` fusion it displaces.

    That trade is decided by the output tensor, not the GEMM. The tower's score
    tensor is `(12, 2520, 2520)` fp32 = 305 MB, so breaking the fusion adds a
    610 MB/layer DRAM round trip: 3.91 ms/layer predicted at the 156 GB/s wall,
    3.94 ms measured. Twelve layers turns cuBLAS's 3.6 ms GEMM win into a
    **47 ms loss**, and `image_embed` goes 297 -> 337 ms.

    Restricted to fp32 because that is the case where cuBLAS brings no tensor
    cores to the trade and therefore cannot win back a fusion. The fp16 offloads
    measured +3.0 ms/iteration in the tower's favour and are kept.

    ⚠️ **Scope, stated honestly.** The mechanism is really "the displaced fusion
    writes a tensor bigger than the GEMM reads", and fp32 is a *proxy* for it, not
    the thing itself. The precise test — is the matmul expanding? — is not
    computable in a pattern check here: the tower's shapes are symbolic in
    `num_patches`, and `12*s*s > 4*(12*s*64 + 12*64*s)` is unprovable without a
    bound on `s`, so an analyzer-based version would decline nothing and fix
    nothing. The proxy is exact on every configuration in this project, because
    `_cublas_gemm` only enables the pass for `q0f16`/`q0bf16`/`q0f32`/fp8
    ([compiler_flags.py:103](../interface/compiler_flags.py#L103)) and the VL
    `q0f16` build is the only one of those here — where the sole fp32 GEMMs are
    the vision tower's attention. The 35B and 0.8B text models are `q4f16_1`, so
    the whole pass is off for them and this guard is unreachable.

    On a **`q0f32`** model, though, this would decline cuBLAS wholesale, and that
    case is unmeasured. `MLC_BLAS_SKIP_FP32=0` restores the previous behaviour.
    """
    for value in context.matched_bindings.values():
        sinfo = value.struct_info
        if isinstance(sinfo, relax.TensorStructInfo) and sinfo.dtype == "float32":
            return True
    return False


def _decline_free_symbolic_vars(pattern: relax.transform.FusionPattern, skip_fp32: bool):
    """Wrap a fusion pattern's check with the guards above."""
    inner = pattern.check

    def _check(context: relax.transform.PatternCheckContext) -> bool:
        if inner is not None and not inner(context):
            return False
        if skip_fp32 and _region_is_fp32(context):
            return False
        return not _region_needs_tir_vars(context)

    return relax.transform.FusionPattern(
        pattern.name, pattern.pattern, pattern.annotation_patterns,
        _check, pattern.attrs_getter,
    )


@tvm.transform.module_pass(opt_level=0, name="BLASDispatch")
class BLASDispatch:
    """A compiler pass that dispatches patterns to cuBLAS/hipBLAS."""

    def __init__(self, target: tvm.target.Target) -> None:
        # A/B knob for the fp32 guard (§20.3). **Default `0` — superseded (§20.5).**
        #
        # It shipped at `1` for one session and was right about the mechanism and
        # wrong about the fix. Declining the offload avoided the broken fusion; it
        # also gave up cuBLAS's 2.46x-faster GEMM, leaving 5.3 ms/layer unclaimed.
        # Moving the score scale onto `q` (`MLC_QWEN35_VL_PRESCALE_Q`, default `1`)
        # removes the *reason* for the fusion instead, so the QK matmul is a bare
        # GEMM cuBLAS can take for free: 9.46 -> 3.63 ms/layer, `image_embed`
        # 293 -> 223 ms. With that change the guard only costs, so it defaults off.
        #
        # ⚠️ The two knobs are coupled. `MLC_QWEN35_VL_PRESCALE_Q=0` restores the
        # `matmul -> multiply` fusion, and *then* this guard is worth `1` again —
        # without it that configuration pays §20.2's 47 ms. Do not set one to its
        # non-default without considering the other.
        skip_fp32 = os.environ.get("MLC_BLAS_SKIP_FP32", "0") == "1"
        if target.kind.name == "cuda":
            self.has_blas = tvm.get_global_func("relax.ext.cublas", True)
            if not self.has_blas:
                raise Exception("cuBLAS is not enabled.")
            self.patterns = [
                _decline_free_symbolic_vars(p, skip_fp32)
                for p in get_patterns_with_prefix("cublas")
            ]
        elif target.kind.name == "rocm":
            self.has_blas = tvm.get_global_func("relax.ext.hipblas", True)
            if not self.has_blas:
                raise Exception("hipBLAS is not enabled.")
            # Same serializer, same failure mode — hipBLAS goes through the identical
            # BYOC JSON path, so it gets the identical guard. The fp32 guard carries
            # over on the same argument: it is the broken fusion on a 305 MB tensor
            # that decides it, not anything Ampere-specific.
            self.patterns = [
                _decline_free_symbolic_vars(p, skip_fp32)
                for p in get_patterns_with_prefix("hipblas")
            ]
        else:
            raise Exception(f"Unsupported target {target.kind.name} for BLAS dispatch.")

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        model_names = [
            gv.name_hint for gv, func in mod.functions.items() if isinstance(func, relax.Function)
        ]
        # exclude single batch decode
        model_names = [name for name in model_names if "batch" in name or "decode" not in name]
        mod = tvm.transform.Sequential(
            [
                relax.transform.FuseOpsByPattern(
                    self.patterns,
                    bind_constants=False,
                    annotate_codegen=True,
                    entry_functions=model_names,
                ),
                relax.transform.RunCodegen({}, entry_functions=model_names),
            ]
        )(mod)
        return mod
