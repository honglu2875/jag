import jag.ops as jnp


def f1(x, y, z):
    return jnp.sum((x @ y - z) ** 2)


def d1(x, y, z):
    return 2 * (x @ y - z) @ y.T, (2 * (x @ y - z).T @ x).T, -2 * (x @ y - z)


def f2(x, y, z):
    return 1 / (jnp.reshape(x, -1) @ jnp.reshape(y, -1))


def d2(x, y, z):
    return (
        jnp.reshape(
            -jnp.reshape(y, -1) / (jnp.reshape(x, -1) @ jnp.reshape(y, -1)) ** 2,
            x.shape,
        ),
        jnp.reshape(
            -jnp.reshape(x, -1) / (jnp.reshape(x, -1) @ jnp.reshape(y, -1)) ** 2,
            y.shape,
        ),
        jnp.zeros_like(z),
    )


def f3(x, y, z):
    return jnp.exp(x)[:2, :2] + jnp.log(y)[:2, :2] - jnp.power(z, 2)[:2, :2]


def d3(x, y, z):
    return (
        jnp.replace(jnp.zeros_like(x), jnp.exp(x)[:2, :2], idx=(slice(2), slice(2))),
        jnp.replace(jnp.zeros_like(y), (1 / y)[:2, :2], idx=(slice(2), slice(2))),
        jnp.replace(jnp.zeros_like(z), (-2 * z)[:2, :2], idx=(slice(2), slice(2))),
    )


def f1_kw(x, y, z):
    return jnp.sum(x @ y, axis=-1)
