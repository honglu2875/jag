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
import jax
import numpy as np
import pytest
from func import f1, f2, f3, f4, f5, jf3, jf4, jf5

from jag.diff import jvp
from jag.graph import trace


@pytest.mark.parametrize(
    "func, jax_func", [(f1, None), (f2, None), (f3, jf3), (f4, jf4), (f5, jf5)]
)
def test_backward_grad(func, jax_func):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    y = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    z = np.array([[1, 2], [3, 4]], dtype=np.float64)
    graph = trace(func, x, y, z)
    xx = np.random.random(x.shape)
    yy = np.random.random(y.shape)
    zz = np.random.random(z.shape)
    results = jvp(graph)(x, y, z, xx, yy, zz)
    truth = jax.jvp(jax_func or func, (x, y, z), (xx, yy, zz))

    for i in range(len(results)):
        assert np.allclose(results[i], truth[i])
