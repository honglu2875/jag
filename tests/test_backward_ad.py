from jag.diff import vjp
import jag.ops as jnp
from jag.trace import trace
import pytest
import numpy as np


def f1(x, y, z):
    return jnp.sum((x @ y - z) ** 2)


def d1(x, y, z):
    return 2 * (x @ y - z) @ y.T, (2 * (x @ y - z).T @ x).T, -2 * (x @ y - z)


@pytest.mark.parametrize("func, der", [(f1, d1)])
def test_backward(func, der):
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[1, 2], [3, 4], [5, 6]])
    z = np.array([[1, 2], [3, 4]])
    graph = trace(func, x, y, z)
    g = np.array(1)
    results = vjp(graph)(g, x, y, z)
    truth = der(x, y, z)
    print(vjp(graph)(g, x, y, z))
    for i in range(len(results)):
        assert np.allclose(results[i], truth[i])
