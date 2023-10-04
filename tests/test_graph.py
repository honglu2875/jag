import numpy as np
import pytest
from func import f1, f1_kw, f2, f6

import jag
from jag.graph import trace
from jag.utils import map_nodes


@pytest.mark.parametrize("func", [f1, f1_kw])
def test_graph_execute(func):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    y = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    z = np.array([[1, 2], [3, 4]], dtype=np.float64)
    graph = trace(func, x, y, z)
    print(graph.to_str())
    assert np.allclose(graph.execute(x, y, z), func(x, y, z))


def test_name_mapping():
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    y = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    z = np.array([[1, 2], [3, 4]], dtype=np.float64)
    graph = trace(f6, x, y, z)
    name_map = map_nodes(graph, include_kwargs=True)
    leaves = list(graph.leaves(include_constant=True, unique=True))
    assert len(leaves) == len(set([id(l) for l in leaves]))
    assert len(name_map.values()) == len(set(name_map.values()))
