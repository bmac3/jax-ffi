from functools import partial

from . import gpu_ops

import jax
import jax.numpy as jnp
from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import xla
from jax.lib import xla_client

from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call

_foo_fwd = core.Primitive("foo_fwd")
_foo_fwd.multiple_results = True
_foo_fwd.def_impl(partial(xla.apply_primitive, _foo_fwd))

_foo_bwd = core.Primitive("foo_bwd")
_foo_bwd.multiple_results = True
_foo_bwd.def_impl(partial(xla.apply_primitive, _foo_bwd))


def foo_fwd(a, b):
    output, b_plus_1 = _foo_fwd.bind(
        a,
        b,
    )
    return output, (output, b_plus_1)


def foo_bwd(grad, output, b_plus_1):
    a_grad, b_grad = _foo_bwd.bind(
        grad,
        output,
        b_plus_1,
    )
    return a_grad, b_grad


@partial(jax.custom_vjp)
def foo(a, b):
    output, b_plus_1 = foo_fwd(a, b)
    return output


foo.defvjp(foo_fwd, foo_bwd)


def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


def _foo_fwd_cuda_lowering(ctx, a, b):
    a_type = ir.RankedTensorType(a.type)
    b_type = ir.RankedTensorType(b.type)

    size = np.prod(a_type.shape).astype(np.int64)

    opaque = gpu_ops.build_foo_fwd_descriptor(
        size
    )

    out = custom_call(
        b"gpu_foo_fwd",
        result_types=[
            ir.RankedTensorType.get(a_type.shape, a_type.element_type),
            ir.RankedTensorType.get(a_type.shape, a_type.element_type),
        ],
        operands=[a, b],
        backend_config=opaque,
        operand_layouts=default_layouts(
            a_type.shape, b_type.shape
        ),
        result_layouts=default_layouts(a_type.shape, a_type.shape),
    )
    return out.results


def _foo_bwd_cuda_lowering(
    ctx,
    grad,
    output,
    b_plus_1,
    a,
    b
):
    a_type = ir.RankedTensorType(a.type)
    b_type = ir.RankedTensorType(b.type)

    size = np.prod(a_type.shape).astype(np.int64)

    opaque = gpu_ops.build_foo_bwd_descriptor(
        size
    )

    out = custom_call(
        b"gpu_foo_bwd",
        result_types=[
            ir.RankedTensorType.get(a_type.shape, a_type.element_type),
            ir.RankedTensorType.get(b_type.shape, b_type.element_type),
        ],
        operands=[grad, output, b_plus_1],
        backend_config=opaque,
        operand_layouts=default_layouts(
            a_type.shape,
            a_type.shape,
        ),
        result_layouts=default_layouts(
            a_type.shape,
            b_type.shape,
        ),
    )
    return out.results


def _foo_fwd_abstract(a, b):
    a_dtype = dtypes.canonicalize_dtype(a.dtype)
    b_dtype = dtypes.canonicalize_dtype(b.dtype)
    assert a_dtype == b_dtype
    assert a.shape == b.shape
    return (
        ShapedArray(a.shape, a_dtype),
        ShapedArray(a.shape, a_dtype),
    )


def _foo_bwd_abstract(
    grad,
    output,
    b_plus_1,
    a,
    b,
):
    a_dtype = dtypes.canonicalize_dtype(a.dtype)
    b_dtype = dtypes.canonicalize_dtype(b.dtype)
    assert a_dtype == b_dtype
    assert a.shape == b.shape
    return (
        ShapedArray(a.shape, a_dtype),
        ShapedArray(a.shape, a_dtype),
    )


def _register():
    xla_client.register_custom_call_target(b"gpu_foo_fwd", _foo_fwd_cuda_lowering, platform="gpu")
    xla_client.register_custom_call_target(b"gpu_foo_bwd", _foo_bwd_cuda_lowering, platform="gpu")

    mlir.register_lowering(
        _foo_fwd,
        _foo_fwd_cuda_lowering,
        platform="gpu",
    )

    mlir.register_lowering(
        _foo_bwd,
        _foo_bwd_cuda_lowering,
        platform="gpu",
    )

    _foo_fwd.def_abstract_eval(_foo_fwd_abstract)

    _foo_bwd.def_abstract_eval(_foo_bwd_abstract)


_register()