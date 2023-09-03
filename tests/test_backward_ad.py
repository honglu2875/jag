import numpy as np
import pytest
from func import d1, d2, d3, f1, f2, f3

from jag.diff import vjp
from jag.trace import trace


@pytest.mark.parametrize("func, der", [(f1, d1), (f2, d2), (f3, d3)])
def test_backward_grad(func, der):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    y = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    z = np.array([[1, 2], [3, 4]], dtype=np.float64)
    graph = trace(func, x, y, z)
    g = np.array(1)
    results = vjp(graph)(g, x, y, z)
    truth = der(x, y, z)
    print(vjp(graph)(g, x, y, z))
    for i in range(len(results)):
        assert np.allclose(results[i], truth[i])
