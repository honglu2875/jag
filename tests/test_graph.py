import numpy as np
import pytest
from func import f1, f1_kw

import jag.ops as jnp
from jag.diff import vjp
from jag.trace import trace


@pytest.mark.parametrize("func", [f1, f1_kw])
def test_graph_execute(func):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    y = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    z = np.array([[1, 2], [3, 4]], dtype=np.float64)
    graph = trace(func, x, y, z)
    print(graph.to_str())
    assert np.allclose(graph.execute(x, y, z), func(x, y, z))
