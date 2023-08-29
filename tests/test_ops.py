import jax.random

from jag.ops import (
    get_op_registration,
    squeeze,
    unsqueeze,
    repeat,
    reshape,
    transpose,
    sum,
    add,
    subtract,
    negative,
    multiply,
    divide,
    matmul,
    where,
)
import functools
import pytest
import jax.numpy as jnp
import numpy as np


@pytest.mark.parametrize("op, kwargs, jax_op",
                         [(squeeze, {}, None),
                          (unsqueeze, {"axis": 4}, jnp.expand_dims),
                          (repeat, {"repeats": 2, "axis": 1}, None),
                          (reshape, {"newshape": (-1, 6)}, None),
                          (transpose, {"axes": (0, 2, 3, 1)}, None),
                          (sum, {"axis": 2}, None),
                          (negative, {}, None)]
                         )
def test_jvp_unary_ops(op, kwargs, jax_op):
    primal = np.random.random((1, 2, 3, 4))
    tangent = np.random.random((1, 2, 3, 4))
    jvp_fn = get_op_registration(op.name)['jvp']
    tangent_out = jvp_fn(tangent, primal, **kwargs)
    jax_op = jax_op or op.op
    jax_tangent_out = jax.jvp(functools.partial(jax_op, **kwargs), (primal,), (tangent,))[1]
    assert jnp.allclose(tangent_out, jax_tangent_out)


@pytest.mark.parametrize("op, kwargs, jax_op",
                         [(add, {}, jnp.add),
                          (subtract, {}, jnp.subtract),
                          (multiply, {}, jnp.multiply),
                          (divide, {}, jnp.divide),
                          (matmul, {}, jnp.matmul)])
def test_jvp_binary_ops(op, kwargs, jax_op):
    if op.name == 'matmul':
        primal = np.random.random((1, 2, 3, 4)), np.random.random((1, 2, 4, 3))
        tangent = np.random.random((1, 2, 3, 4)), np.random.random((1, 2, 4, 3))
    else:
        primal = np.random.random((1, 2, 3, 4)), np.random.random((1, 2, 3, 4))
        tangent = np.random.random((1, 2, 3, 4)), np.random.random((1, 2, 3, 4))
    jvp_fn = get_op_registration(op.name)['jvp']
    tangent_out = jvp_fn(*tangent, *primal, **kwargs)
    jax_op = jax_op or op.op
    jax_tangent_out = jax.jvp(functools.partial(jax_op, **kwargs), primal, tangent)[1]
    assert jnp.allclose(tangent_out, jax_tangent_out)
