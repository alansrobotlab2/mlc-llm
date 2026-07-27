"""A compiler pass that dispatches patterns to CUBLAS."""

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


def _decline_free_symbolic_vars(pattern: relax.transform.FusionPattern):
    """Wrap a fusion pattern's check with `_region_needs_tir_vars`."""
    inner = pattern.check

    def _check(context: relax.transform.PatternCheckContext) -> bool:
        if inner is not None and not inner(context):
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
        if target.kind.name == "cuda":
            self.has_blas = tvm.get_global_func("relax.ext.cublas", True)
            if not self.has_blas:
                raise Exception("cuBLAS is not enabled.")
            self.patterns = [
                _decline_free_symbolic_vars(p) for p in get_patterns_with_prefix("cublas")
            ]
        elif target.kind.name == "rocm":
            self.has_blas = tvm.get_global_func("relax.ext.hipblas", True)
            if not self.has_blas:
                raise Exception("hipBLAS is not enabled.")
            # Same serializer, same failure mode — hipBLAS goes through the identical
            # BYOC JSON path, so it gets the identical guard.
            self.patterns = [
                _decline_free_symbolic_vars(p) for p in get_patterns_with_prefix("hipblas")
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
