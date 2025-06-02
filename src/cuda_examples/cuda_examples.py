# -*- coding: utf-8 -*-

__all__ = ["foo"]

from functools import partial

import numpy as np
from jax import core, dtypes, lax
from jax import numpy as jnp
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir, xla
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call


# If the GPU version exists, also register those
try:
    from . import gpu_ops
except ImportError:
    gpu_ops = None
else:
    for _name, _value in gpu_ops.registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")

# This function exposes the primitive to user code and this is the only
# public-facing function in this module


def foo(a, b):
    return _foo_prim.bind(a, b)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _foo_abstract(a, b):
    shape = a.shape
    dtype = dtypes.canonicalize_dtype(a.dtype)
    assert dtypes.canonicalize_dtype(b.dtype) == dtype
    assert b.shape == shape
    return (ShapedArray(shape, dtype), ShapedArray(shape, dtype))


# We also need a lowering rule to provide an MLIR "lowering" of out primitive.
# This provides a mechanism for exposing our custom C++ and/or CUDA interfaces
# to the JAX XLA backend. We're wrapping two translation rules into one here:
# one for the CPU and one for the GPU
def _foo_lowering(ctx, a, b, *, platform="gpu"):

    # Checking that input types and shape agree
    assert a.type == b.type

    # Extract the numpy type of the inputs
    a_aval, _ = ctx.avals_in
    np_dtype = np.dtype(a_aval.dtype)

    # The inputs and outputs all have the same shape and memory layout
    # so let's predefine this specification
    dtype = mlir.ir.RankedTensorType(a.type)
    dims = dtype.shape
    layout = tuple(range(len(dims) - 1, -1, -1))

    # The total size of the input is the product across dimensions
    size = np.prod(dims).astype(np.int64)

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        # On the CPU, we pass the size of the data as a the first input
        # argument
        raise NotImplementedError("CPU support not implemented")

    elif platform == "gpu":
        if gpu_ops is None:
            raise ValueError(
                "The 'cuda_examples' module was not compiled with CUDA support"
            )
        # On the GPU, we do things a little differently and encapsulate the
        # dimension using the 'opaque' parameter
        opaque = gpu_ops.build_kepler_descriptor(size)

        return custom_call(
            op_name,
            # Output types
            result_types=[dtype, dtype],
            # The inputs:
            operands=[a, b],
            # Layout specification:
            operand_layouts=[layout, layout],
            result_layouts=[layout, layout],
            # GPU specific additional data
            backend_config=opaque
        ).results

    raise ValueError(
        "Unsupported platform; this must be either 'cpu' or 'gpu'"
    )


# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************

# Here we define the differentiation rules using a JVP derived using implicit
# differentiation of Kepler's equation:
#
#  M = E - e * sin(E)
#  -> dM = dE * (1 - e * cos(E)) - de * sin(E)
#  -> dE/dM = 1 / (1 - e * cos(E))  and  de/dM = sin(E) / (1 - e * cos(E))
#
# In this case we don't need to define a transpose rule in order to support
# reverse and higher order differentiation. This might not be true in other
# applications, so check out the "How JAX primitives work" tutorial in the JAX
# documentation for more info as necessary.
def _kepler_jvp(args, tangents):
    mean_anom, ecc = args
    d_mean_anom, d_ecc = tangents

    # We use "bind" here because we don't want to mod the mean anomaly again
    sin_ecc_anom, cos_ecc_anom = _kepler_prim.bind(mean_anom, ecc)

    def zero_tangent(tan, val):
        return lax.zeros_like_array(val) if type(tan) is ad.Zero else tan

    # Propagate the derivatives
    d_ecc_anom = (
        zero_tangent(d_mean_anom, mean_anom)
        + zero_tangent(d_ecc, ecc) * sin_ecc_anom
    ) / (1 - ecc * cos_ecc_anom)

    return (sin_ecc_anom, cos_ecc_anom), (
        cos_ecc_anom * d_ecc_anom,
        -sin_ecc_anom * d_ecc_anom,
    )




# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_kepler_prim = core.Primitive("kepler")
_kepler_prim.multiple_results = True
_kepler_prim.def_impl(partial(xla.apply_primitive, _kepler_prim))
_kepler_prim.def_abstract_eval(_kepler_abstract)

# Connect the XLA translation rules for JIT compilation
for platform in ["gpu"]:
    mlir.register_lowering(
        _kepler_prim,
        partial(_kepler_lowering, platform=platform),
        platform=platform)

# Connect the JVP and batching rules
ad.primitive_jvps[_kepler_prim] = _kepler_jvp
