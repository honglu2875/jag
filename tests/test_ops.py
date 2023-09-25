import functools

import jax.numpy as jnp
import jax.random
import numpy as np
import pytest

from jag.ops import (
    add,
    at,
    divide,
    exp,
    get_op_registration,
    log,
    matmul,
    multiply,
    negative,
    ones_like,
    power,
    repeat,
    replace,
    reshape,
    squeeze,
    subtract,
    sum,
    transpose,
    unsqueeze,
    where,
    zeros_like,
)


@pytest.mark.parametrize(
    "op, kwargs, jax_op",
    [
        (squeeze, {}, None),
        (unsqueeze, {"axis": 4}, jnp.expand_dims),
        (repeat, {"repeats": 2, "axis": 1}, None),
        (reshape, {"newshape": (-1, 6)}, None),
        (transpose, {"axes": (0, 2, 3, 1)}, None),
        (sum, {"axis": 2}, None),
        (negative, {}, None),
        (zeros_like, {}, jnp.zeros_like),
        (ones_like, {}, jnp.ones_like),
        (log, {}, jnp.log),
        (exp, {}, jnp.exp),
        (at, {"idx": slice(1, 2)}, at.op),
        (at, {"idx": (slice(1, 2), slice(1, 2))}, at.op),
    ],
)
def test_jvp_and_vjp_unary_ops(op, kwargs, jax_op):
    primal = np.random.random((1, 2, 3, 4)).astype(np.float64)
    tangent = np.random.random((1, 2, 3, 4)).astype(np.float64)
    jvp_fn = get_op_registration(op.name)["jvp"]
    tangent_out = jvp_fn(tangent, primal, **kwargs)
    jax_op = jax_op or op.op
    jax_tangent_out = jax.jvp(
        functools.partial(jax_op, **kwargs), (primal,), (tangent,)
    )[1]
    assert jnp.allclose(tangent_out, jax_tangent_out)

    # could have used tangent_out, but for rigorousness we re-initialize cotangent vector
    cotangent = np.random.random(tangent_out.shape)
    vjp_fn = get_op_registration(op.name)["vjp"]
    cotangent_out = vjp_fn(cotangent, primal, **kwargs)
    jax_cotangent_out = jax.vjp(functools.partial(jax_op, **kwargs), primal)[1](
        cotangent
    )
    assert jnp.allclose(cotangent_out[0], jax_cotangent_out[0])


@pytest.mark.parametrize(
    "op, kwargs, jax_op",
    [
        (add, {}, jnp.add),
        (subtract, {}, jnp.subtract),
        (multiply, {}, jnp.multiply),
        (divide, {}, jnp.divide),
        (matmul, {}, jnp.matmul),
        (
            where,
            {"condition": np.random.random((1, 2, 3, 4)) > 0.5},
            lambda x, y, condition=None: jnp.where(condition, x, y),
        ),
        (power, {}, jnp.power),
        (
            replace,
            {"idx": (slice(0, 1), slice(1, 2), slice(1, 2), slice(0, 2))},
            lambda x, y, idx=None: x.at[idx].set(y),
        ),
    ],
)
def test_jvp_and_vjp_binary_ops(op, kwargs, jax_op):
    if op.name == "matmul":
        primal = np.random.random((1, 2, 3, 4)).astype(np.float64), np.random.random(
            (1, 2, 4, 3)
        ).astype(np.float64)
        tangent = np.random.random((1, 2, 3, 4)).astype(np.float64), np.random.random(
            (1, 2, 4, 3)
        ).astype(np.float64)
    elif op.name == "replace":
        primal = np.random.random((1, 2, 3, 4)).astype(np.float64), np.random.random(
            (1, 1, 1, 2)
        ).astype(np.float64)
        tangent = np.random.random((1, 2, 3, 4)).astype(np.float64), np.random.random(
            (1, 1, 1, 2)
        ).astype(np.float64)
    else:
        primal = np.random.random((1, 2, 3, 4)).astype(np.float64), np.random.random(
            (1, 2, 3, 4)
        ).astype(np.float64)
        tangent = np.random.random((1, 2, 3, 4)).astype(np.float64), np.random.random(
            (1, 2, 3, 4)
        ).astype(np.float64)
    jvp_fn = get_op_registration(op.name)["jvp"]
    tangent_out = jvp_fn(*tangent, *primal, **kwargs)
    jax_op = jax_op or op.op
    jax_tangent_out = jax.jvp(functools.partial(jax_op, **kwargs), primal, tangent)[1]
    assert jnp.allclose(tangent_out, jax_tangent_out)

    cotangent = np.random.random(tangent_out.shape)
    vjp_fn = get_op_registration(op.name)["vjp"]
    try:
        cotangent_out = vjp_fn(cotangent, *primal, **kwargs)
    except Exception as e:
        print(cotangent.shape)
        raise e
    jax_cotangent_out = jax.vjp(functools.partial(jax_op, **kwargs), *primal)[1](
        cotangent
    )
    assert jnp.allclose(cotangent_out[0], jax_cotangent_out[0])
    assert jnp.allclose(cotangent_out[1], jax_cotangent_out[1])
