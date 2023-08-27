from dataclasses import dataclass
import numpy as np
from typing import Callable, Optional, Any
from numbers import Number
from functools import singledispatchmethod
from jag.type import ArrayLike


class Operand:
    def __neg__(self): return negative(self)
    def __add__(self, other): return add(self, other)
    def __radd__(self, other): return add(other, self)
    def __sub__(self, other): return subtract(self, other)
    def __rsub__(self, other): return subtract(other, self)
    def __mul__(self, other): return multiply(self, other)
    def __rmul__(self, other): return multiply(other, self)
    def __matmul__(self, other): return matmul(self, other)
    def __rmatmul__(self, other): return matmul(other, self)
    def __div__(self, other): return divide(self, other)
    def __rdiv__(self, other): return divide(other, self)
    def __truediv__(self, other): return divide(self, other)
    def __rtruediv__(self, other): return divide(other, self)
    def __pow__(self, p, modulo=None): return power(self, p)

    @property
    def T(self): return transpose(self)

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


@dataclass
class Node(Operand):
    op: Any
    operands: list
    shape: tuple
    kwargs: Optional[dict] = None

    def __post_init__(self):
        self.kwargs = self.kwargs or {}


@dataclass
class TracedArray(Operand):
    shape: tuple
    dtype: np.dtype
    value: Optional[ArrayLike] = None

    @property
    def require_grad(self):
        return True

    @singledispatchmethod
    def to_const(self):
        if self.value is None:
            raise ValueError(f"To convert to a constant leaf, a 'value' must be given.")
        return ConstantArray(self.value)

    @to_const.register
    def _(self, value: ArrayLike):
        return ConstantArray(value)


class ConstantArray(TracedArray):
    def __init__(self, value: ArrayLike):
        super().__init__(
            shape=np.array(value).shape,
            dtype=np.array(value).dtype,
        )
        self.value = np.array(value)

    @property
    def require_grad(self):
        return False

    def to_abstract(self):
        return TracedArray(shape=self.shape, dtype=self.dtype, value=self.value)
