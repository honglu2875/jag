import math
from dataclasses import dataclass
from functools import singledispatchmethod
from numbers import Number
from typing import Any, Callable, Optional, Sequence

import numpy as np

from jag.diff import jvp, vjp
from jag.ops import Operand, TraceableOp
from jag.type import ArrayLike, GraphNode


def get_value(node: "TracedArray") -> str:
    return (
        str(node.value.tolist())
        if node.value is not None
        else f"Array({node.shape}, dtype={node.dtype})"
    )


@dataclass(kw_only=True)
class Node(Operand, GraphNode):
    op: Any
    shape: tuple
    dtype: np.dtype
    kwargs: Optional[dict] = None
    # _vjp and _jvp would be created by calling vjp and jvp methods if not assigned
    _vjp: Optional[Any] = None
    _jvp: Optional[Any] = None

    def __post_init__(self):
        self.kwargs = self.kwargs or {}

    def _execute(self, value_dict: dict):
        """
        Recursively execute the graph from the node, with the given values of the leaves.
        Args:
            value_dict: the dictionary of values of the leaves.
        Returns:
            the value of the node.
        """
        return self.op(
            *[child.execute(value_dict) for child in self.operands],
            **self.kwargs,
        )

    def print_summary(self):
        """
        Print a summary of the graph.
        """
        print("Leaves:")
        for leaf in self.leaves(include_constant=False):
            leaf.print_summary(indent=2)
        print("Number of vertices:", self.num_nodes)

    def random_leaf_values(self):
        """
        Get random concrete values of the leaves.
        Returns:
            a list of arrays according to the order of the results of self.leaves()
        """
        return [
            np.random.random(leaf.shape).astype(leaf.dtype)
            for leaf in self.leaves(include_constant=False)
        ]

    def call_vjp(self, g: ArrayLike, *args, **kwargs):
        """
        Call the vjp function of the graph.
        """
        if self._vjp is None:
            self._vjp = vjp(self)
        return self._vjp(g, *args, **kwargs)

    def call_jvp(self, *args, **kwargs):
        """
        Call the jvp function of the graph.
        """
        if self._jvp is None:
            self._jvp = jvp(self)
        return self._jvp(*args, **kwargs)

    def grad(self):
        """
        Get the graph of its gradient function with respect to non-constant leaves.
        """
        return trace(
            self.call_vjp(np.array(1, dtype=self.dtype)),
            *self.random_leaf_values(),
            **self.kwargs,
        )

    def vjp(self):
        """
        Get the graph of its vjp function.
        """
        return trace(
            self.call_vjp(
                np.random.random(self.shape).astype(self.dtype),
                *self.random_leaf_values(),
                **self.kwargs,
            )
        )

    def jvp(self):
        """
        Get the graph of its jvp function.
        """
        return trace(self.call_jvp(*self.random_leaf_values(), **self.kwargs))

    def to_str(self):
        """
        I do not override __str__ because I have not decided whether to use this way to serialize into pseudo-codes.
        """

        def _to_str(node: Operand, depth: int):
            spaces = " " * (depth * 2)
            if isinstance(node, TracedArray):
                return f"{spaces}{node.name or '<unnamed>'} = {get_value(node)}"
            elif isinstance(node, Node):
                if isinstance(node.op, TraceableOp):
                    op_name = node.op.name
                else:
                    op_name = node.op.__name__
                delimiter = ",\n"
                kwarg_str = (
                    [", ".join([f"{spaces}  {k}={v}" for k, v in node.kwargs.items()])]
                    if node.kwargs
                    else []
                )
                return (
                    f"{spaces}{op_name}(\n"
                    f"{delimiter.join([_to_str(child, depth + 1) for child in node.operands] + kwarg_str)}\n"
                    f"{spaces})"
                )
            else:
                raise ValueError(f"Unsupported node type: {type(node)}.")

        return _to_str(self, 0)

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def num_nodes(self) -> int:
        count = 0
        for node in self.operands + list(self.kwargs.values()):
            if isinstance(node, Node):
                count += node.num_nodes
            elif isinstance(node, TracedArray):
                count += 1
        return count


@dataclass(kw_only=True)
class TracedArray(Operand, GraphNode):
    shape: tuple
    dtype: np.dtype
    value: Optional[ArrayLike] = None
    _requires_grad: bool = True

    def print_summary(self, indent: int = 0):
        print(f"{' ' * indent}{self.name or '<unspecified>'}:")
        print(f"{' ' * indent}  shape: {self.shape}")
        print(f"{' ' * indent}  dtype: {self.dtype}")
        if self.value is not None:
            print(f"{' ' * indent}  value: {self.value}")

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self._requires_grad = value

    def _execute(self, value_dict: dict):
        return value_dict.get(id(self), self.value)

    @singledispatchmethod
    def to_const(self):
        if self.value is None:
            raise ValueError(f"To convert to a constant leaf, a 'value' must be given.")
        return ConstantArray(self.value)

    @to_const.register
    def _(self, value: ArrayLike):
        return ConstantArray(value)


@dataclass(kw_only=True)
class ConstantArray(TracedArray):
    def __init__(self, value: ArrayLike):
        super().__init__(
            shape=np.array(value).shape,
            dtype=np.array(value).dtype,
        )
        self.value = np.array(value)

    @property
    def requires_grad(self):
        return False

    def to_abstract(self):
        return TracedArray(
            name=self.name, shape=self.shape, dtype=self.dtype, value=self.value
        )

    def _execute(self, value_dict: dict):
        return self.value

    @requires_grad.setter
    def requires_grad(self, value):
        raise ValueError("Cannot set requires_grad for a constant node.")


def to_traceable(arg: Any) -> Node | TracedArray:
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


def trace(func: Callable, *args, constants: Optional[Sequence] = None, **kwargs):
    """
    Utility function to trace the computation graph of a function.
    """
    constants = constants or []
    args = [
        TracedArray(shape=np.array(arg).shape, dtype=np.array(arg).dtype)
        if i not in constants
        else ConstantArray(np.array(arg))
        for i, arg in enumerate(args)
    ]
    return func(*args, **kwargs)
