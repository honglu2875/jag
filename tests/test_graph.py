from jag.diff import vjp
import jag.ops as jnp
import numpy as np
from jag.trace import trace
import pytest
from func import f1, f1_kw


@pytest.mark.parametrize("func", [f1, f1_kw])
def test_graph_execute(func):
    x = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([[1, 2], [3, 4], [5, 6]])
    z = np.array([[1, 2], [3, 4]])
    graph = trace(func, x, y, z)
    print(graph.to_str())
    assert np.allclose(graph.execute(x, y, z), func(x, y, z))
