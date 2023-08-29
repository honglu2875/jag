from dataclasses import dataclass
import numpy as np
from typing import Optional, Any
from functools import singledispatchmethod
from numbers import Number

from jag.ops import Operand, TraceableOp
from jag.type import ArrayLike


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


def to_traceable(arg: Any):
    """
    Convert the argument into a part of the computation graph for tracing:
    1. if arg is a subclass of Operand, it is a part of the graph and is already traceable.
    2. if arg is a concrete value, it is turned into a 'ConstantArray' object as leaves of a graph.
    """
    if isinstance(arg, (Node, TracedArray)):
        return arg
    elif isinstance(arg, (Number, np.ndarray)):
        return ConstantArray(np.array(arg))
    else:
        raise RuntimeError(f"Unsupported argument type: {type(arg)}.")
