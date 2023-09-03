from numbers import Number
from typing import Callable

import numpy as np

from ._ops import add, divide, matmul, multiply, negative, power, subtract, transpose


class Operand:
    def __neg__(self):
        return negative(self)

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

    @property
    def T(self):
        return transpose(self)

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
