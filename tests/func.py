import numpy as np
import jax.numpy as jnp


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
    return xx,
