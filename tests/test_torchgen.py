from func import f1, f2, f3, f4, f5, f6
import pytest
from jag.codegen import torchgen
from jag import trace
import numpy as np
import torch
import inspect


@pytest.mark.parametrize("func", [f1, f2, f3, f4, f5, f6])
def test_torchgen(func):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    y = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    z = np.array([[1, 2], [3, 4]], dtype=np.float64)

    graph = trace(func, x, y, z)
    code = torchgen(graph, func_name="func_torch")
    import_line = "import torch\n"

    print(import_line + code)

    exec(import_line + code, None, locals())
    params = inspect.signature(locals()["func_torch"]).parameters
    args = [torch.tensor(x), torch.tensor(y), torch.tensor(z)][: len(params)]
    assert np.allclose(func(x, y, z), locals()["func_torch"](*args).numpy())