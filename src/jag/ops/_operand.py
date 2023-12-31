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
from numbers import Number
from typing import Callable

import numpy as np

from ._ops import (add, at, clip, divide, greater, greater_equal, less,
                   less_equal, matmul, multiply, negative, power, subtract,
                   sum, transpose, unsqueeze, where)
from ._traceable_op import TraceableOp

_implemented_ufunc_call = {
    name: v["op"] for name, v in TraceableOp._traceable_op_registry.items()
}
# Special names
_implemented_ufunc_call.update(
    {
        "expand_dims": unsqueeze,
    }
)
_ufunc_reduce = {
    "add": sum,
}


class Operand(np.lib.mixins.NDArrayOperatorsMixin):
    def __neg__(self):
        return negative(self)

    def __gt__(self, other):
        return greater(self, other)

    def __lt__(self, other):
        return less(self, other)

    def __ge__(self, other):
        return greater_equal(self, other)

    def __le__(self, other):
        return less_equal(self, other)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other):
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(other, self)

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(other, self)

    def __div__(self, other):
        return divide(self, other)

    def __rdiv__(self, other):
        return divide(other, self)

    def __truediv__(self, other):
        return divide(self, other)

    def __rtruediv__(self, other):
        return divide(other, self)

    def __pow__(self, p, modulo=None):
        return power(self, p)

    def __getitem__(self, item):
        return at(self, item)

    def __setitem__(self, key, value):
        # self = replace(self, value, idx=key)
        raise NotImplementedError("Not implemented.")

    @property
    def T(self):
        return transpose(self)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Special treatment for "where"
        if ufunc.__name__ == "where":
            return where(inputs[1], inputs[2], inputs[0], *inputs[3:], **kwargs)
        # Call corresponding wrappers
        if method == "__call__" and ufunc.__name__ in _implemented_ufunc_call:
            return _implemented_ufunc_call[ufunc.__name__](*inputs, **kwargs)
        elif method == "reduce" and ufunc.__name__ in _ufunc_reduce:
            return _ufunc_reduce[ufunc.__name__](*inputs, **kwargs)
        else:
            raise NotImplementedError(
                f"Not implemented.\nufunc: {ufunc}, method: {method}"
            )

    def __array_function__(self, func, types, args, kwargs):
        if func.__name__ == "where":
            return where(args[1], args[2], args[0], *args[3:], **kwargs)
        if func.__name__ in _implemented_ufunc_call:
            return _implemented_ufunc_call[func.__name__](*args, **kwargs)
        else:
            raise NotImplementedError(f"Not implemented.\nfunc: {func}, types: {types}")

    @staticmethod
    def to_dict(o):
        if isinstance(o, (Number, str, bool)):
            return o
        elif isinstance(o, (Callable, np.dtype)) or o is None:
            return str(o)
        elif isinstance(o, (list, tuple)):
            return [Operand.to_dict(v) for v in o]
        elif isinstance(o, dict):
            return {k: Operand.to_dict(v) for k, v in o.items()}
        elif isinstance(o, Operand):
            return {k: Operand.to_dict(v) for k, v in o.__dict__.items()}
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            raise RuntimeError(f"Unknown type {type(o)} for to_dict.")
