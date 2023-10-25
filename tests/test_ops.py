# Copyright (c) 2023 Honglu Fan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import functools

import jax.numpy as jnp
import jax.random
import numpy as np
import pytest

from jag.ops import (TraceableOp, absolute, add, arccos, arcsin, arctan, at,
                     clip, cos, cosh, divide, exp, get_op_registration,
                     greater, greater_equal, less, less_equal, log, matmul,
                     mean, multiply, negative, ones_like, power, repeat,
                     replace, reshape, sign, sin, sinh, squeeze, subtract, sum,
                     tan, tanh, transpose, unsqueeze, where, zeros_like)


@pytest.mark.parametrize(
    "op, kwargs, jax_op",
    [
        (squeeze, {}, None),
        (unsqueeze, {"axis": 4}, jnp.expand_dims),
        (repeat, {"repeats": 2, "axis": 1}, None),
        (reshape, {"newshape": (-1, 6)}, None),
        (transpose, {"axes": (0, 2, 3, 1)}, None),
        (sum, {"axis": 2}, None),
        (mean, {"axis": 2}, None),
        (sign, {}, jnp.sign),
        (absolute, {}, jnp.absolute),
        (negative, {}, None),
        (zeros_like, {}, jnp.zeros_like),
        (ones_like, {}, jnp.ones_like),
        (log, {}, jnp.log),
        (exp, {}, jnp.exp),
        (at, {"idx": slice(1, 2)}, at.op),
        (at, {"idx": (slice(1, 2), slice(1, 2))}, at.op),
        (clip, {"a_min": 0.5, "a_max": 1.0}, jnp.clip),
        (sin, {}, jnp.sin),
        (cos, {}, jnp.cos),
        (tan, {}, jnp.tan),
        (sinh, {}, jnp.sinh),
        (cosh, {}, jnp.cosh),
        (tanh, {}, jnp.tanh),
        (arcsin, {}, jnp.arcsin),
        (arccos, {}, jnp.arccos),
        (arctan, {}, jnp.arctan),
    ],
)
def test_jvp_and_vjp_unary_ops(op: TraceableOp, kwargs, jax_op):
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
        # Different behavior in JAX: below derivatives would result in b'' instead of 0 in JAX
        # (greater, {}, jnp.greater),
        # (less, {}, jnp.less),
        # (greater_equal, {}, jnp.greater_equal),
        # (less_equal, {}, jnp.less_equal),
    ],
)
def test_jvp_and_vjp_binary_ops(op: TraceableOp, kwargs, jax_op):
    _type = np.float32
    j_type = jnp.float32

    if op.name == "matmul":
        primal = np.random.random((1, 2, 3, 4)).astype(_type), np.random.random(
            (1, 2, 4, 3)
        ).astype(_type)
        tangent = np.random.random((1, 2, 3, 4)).astype(_type), np.random.random(
            (1, 2, 4, 3)
        ).astype(_type)
    elif op.name == "replace":
        primal = np.random.random((1, 2, 3, 4)).astype(_type), np.random.random(
            (1, 1, 1, 2)
        ).astype(_type)
        tangent = np.random.random((1, 2, 3, 4)).astype(_type), np.random.random(
            (1, 1, 1, 2)
        ).astype(_type)
    else:
        primal = (
            np.random.random((1, 2, 3, 4)).astype(_type),
            np.random.random((1, 2, 3, 4)).astype(_type) + 1.0,
        )  # to avoid division by zero
        tangent = np.random.random((1, 2, 3, 4)).astype(_type), np.random.random(
            (1, 2, 3, 4)
        ).astype(_type)
    jvp_fn = get_op_registration(op.name)["jvp"]
    tangent_out = jvp_fn(*tangent, *primal, **kwargs)
    jax_op = jax_op or op.op
    jax_tangent_out = jax.jvp(functools.partial(jax_op, **kwargs), primal, tangent)[1]
    assert jnp.allclose(
        jnp.array(tangent_out, dtype=j_type), jax_tangent_out.astype(j_type)
    )

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
    assert jnp.allclose(
        jnp.array(cotangent_out[0], dtype=j_type), jax_cotangent_out[0].astype(j_type)
    )
    assert jnp.allclose(
        jnp.array(cotangent_out[1], dtype=j_type), jax_cotangent_out[1].astype(j_type)
    )
