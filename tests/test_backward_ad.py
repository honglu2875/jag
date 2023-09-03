from jag.diff import vjp
from jag.trace import trace
import pytest
import numpy as np
from func import f1, f2, d1, d2


@pytest.mark.parametrize("func, der", [(f1, d1), (f2, d2)])
def test_backward_grad(func, der):
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
