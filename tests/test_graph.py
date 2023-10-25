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
