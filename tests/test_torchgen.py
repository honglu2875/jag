import inspect
from typing import Callable

import jax
import numpy as np
import pytest
import torch
from func import f1, f2, f3, f4, f5, f6, jf3, jf4, jf5

from jag.codegen import torchgen
from jag.graph import trace


def _test_torch_fun(code: str, fun: Callable, args: tuple) -> bool:
    import_line = "import torch\n"
    exec(import_line + code, None, locals())
    params = inspect.signature(locals()["func_torch"]).parameters
    torch_args = [torch.tensor(x) for x in args][: len(params)]
    return np.allclose(fun(*args), locals()["func_torch"](*torch_args).numpy())


@pytest.mark.parametrize("func", [f1, f2, f3, f4, f5, f6])
def test_torchgen(func):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    y = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    z = np.array([[1, 2], [3, 4]], dtype=np.float64)

    graph = trace(func, x, y, z)
    code = torchgen(graph, func_name="func_torch")

    print(code)

    args = (x, y, z)
    assert _test_torch_fun(code, func, args)


@pytest.mark.parametrize(
    "func, jax_func, do_grad",
    [
        (f1, f1, True),
        (f2, f2, True),
        (f3, jf3, False),
        (f4, jf4, False),
        (f5, jf5, False),
    ],
)
def test_diff_torchgen(func, jax_func, do_grad):
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    y = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    z = np.array([[1, 2], [3, 4]], dtype=np.float64)
    out = func(x, y, z)

    tx = np.random.random(x.shape)
    ty = np.random.random(y.shape)
    tz = np.random.random(z.shape)

    tout = np.random.random(out.shape)

    graph = trace(func, x, y, z)
    for g, fun, truth, args, argnames in [
        (
            graph.grad(),
            graph.call_grad,
            jax.grad(jax_func, argnums=(0, 1, 2))(x, y, z) if do_grad else None,
            (x, y, z),
            ("x", "y", "z"),
        ),
        (
            graph.vjp(),
            graph.call_vjp,
            jax.vjp(jax_func, x, y, z)[1](tout),
            (tout, x, y, z),
            ("g", "x", "y", "z"),
        ),
        (
            graph.jvp(),
            graph.call_jvp,
            jax.jvp(jax_func, (x, y, z), (tx, ty, tz)),
            (x, y, z, tx, ty, tz),
            ("x", "y", "z", "dx", "dy", "dz"),
        ),
    ]:
        res = fun(*args)
        for i in range(len(res)):
            code = torchgen(g[i], func_name="func_torch", arg_list=argnames)
            code = f"def func_torch({', '.join(argnames)}):\n" + "\n".join(
                code.split("\n")[1:]
            )
            print(code)
            if truth is not None:
                assert np.allclose(res[i], truth[i])
            assert _test_torch_fun(code, lambda *_: res[i], args)
