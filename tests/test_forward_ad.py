import jax
import numpy as np
import pytest
from func import f1, f2, f3, f4, f5, jf3, jf4, jf5

from jag.diff import jvp
from jag.trace import trace


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
