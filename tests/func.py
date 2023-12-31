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
import jax.numpy as jnp
import numpy as np


def f1(x, y, z):
    return np.sum((x @ y - z) ** 2)


def d1(x, y, z):
    return 2 * (x @ y - z) @ y.T, (2 * (x @ y - z).T @ x).T, -2 * (x @ y - z)


def f2(x, y, z):
    return 1 / (np.reshape(x, -1) @ np.reshape(y, -1))


def d2(x, y, z):
    return (
        np.reshape(
            -np.reshape(y, -1) / (np.reshape(x, -1) @ np.reshape(y, -1)) ** 2,
            x.shape,
        ),
        np.reshape(
            -np.reshape(x, -1) / (np.reshape(x, -1) @ np.reshape(y, -1)) ** 2,
            y.shape,
        ),
        np.zeros_like(z),
    )


def f3(x, y, z):
    return np.exp(x)[:2, :2] + np.log(y)[:2, :2] - np.power(z, 2)[:2, :2]


def jf3(x, y, z):
    return jnp.exp(x)[:2, :2] + jnp.log(y)[:2, :2] - jnp.power(z, 2)[:2, :2]


def d3(x, y, z):
    xx = np.zeros_like(x)
    yy = np.zeros_like(y)
    zz = np.zeros_like(z)
    xx[:2, :2] = np.exp(x)[:2, :2]
    yy[:2, :2] = (1 / y)[:2, :2]
    zz[:2, :2] = (-2 * z)[:2, :2]
    return xx, yy, zz


def f1_kw(x, y, z):
    return np.sum(x @ y, axis=-1)


def f4(x, *args):
    return (np.exp(x) + np.exp(-x)) / x


def jf4(x, *args):
    return (jnp.exp(x) + jnp.exp(-x)) / x


def d4(x, *args):
    return (((np.exp(x) - np.exp(-x)) * x - (np.exp(x) + np.exp(-x))) / (x**2),)


def dd4(x):
    return (
        f4(x)
        - 2 * (np.exp(x) - np.exp(-x)) / (x**2)
        + 2 * (np.exp(x) + np.exp(-x)) / (x**3),
    )


def f5(x, *args):
    y = x[:2, :2]
    return np.sum(np.log(y @ y) ** 2) ** 2


def jf5(x, *args):
    y = x[:2, :2]
    return jnp.sum(jnp.log(y @ y) ** 2) ** 2


def d5(x, *args):
    y = x[:2, :2]
    r = 2 * np.sum(np.log(y @ y) ** 2) * 2 * np.log(y @ y) / (y @ y)
    r = y.T @ r + r @ y.T
    # Test assignment
    xx = np.zeros_like(x)
    xx[:2, :2] = r
    return (xx,)


def f6(x, y, z, *args):
    a = x @ y
    b = a - z
    c = (y @ x)[:2, :2]
    # implement relu
    relu = np.sum(np.maximum(a, 0))
    # implement gelu
    gelu = np.sum(b * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (b + 0.044715 * b**3))))
    # implement elu
    elu = np.sum(np.where(c > 0, c, np.exp(c) - 1))
    return relu + gelu + elu
