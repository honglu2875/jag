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
from func import d1, d2, d3, d4, d5, dd4, f1, f2, f3, f4, f5

from jag.diff import grad, vjp
from jag.graph import trace


@pytest.mark.parametrize(
    "func, der", [(f1, d1), (f2, d2), (f3, d3), (f4, d4), (f5, d5)]
)
def test_backward_grad(func, der):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    y = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    z = np.array([[1, 2], [3, 4]], dtype=np.float64)
    graph = trace(func, x, y, z)
    g = np.array(1)
    results = vjp(graph)(g, x, y, z)
    truth = der(x, y, z)
    print(results)
    print(truth)
    for i in range(len(results)):
        assert np.allclose(results[i], truth[i])


@pytest.mark.parametrize("func, der, dder", [(f4, d4, dd4)])
def test_double_diff(func, der, dder):
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    graph = trace(func, x)
    vjp_fn = vjp(graph)  # g, x |----> gJf(x), same shape as x
    vjp_graphs = trace(vjp_fn, np.ones_like(x), x)
    assert len(vjp_graphs) == 1
    dd_fn = vjp(vjp_graphs[0])  # h, g, x |----> hJf(x)^T, g^ThJJf
    res = dd_fn(np.ones_like(x), np.ones_like(x), x)
    assert np.allclose(res[0], der(x)[0])
    assert np.allclose(res[1], dder(x)[0])
