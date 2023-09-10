from dataclasses import dataclass
from functools import singledispatchmethod
from numbers import Number
from typing import Any, Optional

import numpy as np

from jag.ops import Operand, TraceableOp
from jag.type import ArrayLike


def get_value(node: "TracedArray") -> str:
    return (
        str(node.value.tolist())
        if node.value is not None
        else f"Array({node.shape}, dtype={node.dtype})"
    )


@dataclass
class Node(Operand):
    op: Any
    operands: list
    shape: tuple
    dtype: np.dtype
    name: Optional[str] = None
    kwargs: Optional[dict] = None

    def __post_init__(self):
        self.kwargs = self.kwargs or {}

    @singledispatchmethod
    def execute(self, *args):
        return Node._execute(self, dict(zip([id(l) for l in get_leaves(self)], args)))

    @execute.register
    def _(self, node_values: dict):
        return Node._execute(self, node_values)

    @staticmethod
    def _execute(node: Operand, value_dict: dict):
        """
        Recursively execute the graph from the node, with the given values of the leaves.
        Args:
            node: the node to execute.
            value_dict: the dictionary of values of the leaves.
        Returns:
            the value of the node.
        """
        if isinstance(node, ConstantArray):
            return node.value
        elif isinstance(node, TracedArray):
            return value_dict[id(node)]
        elif isinstance(node, Node):
            return node.op(
                *[Node._execute(child, value_dict) for child in node.operands],
                **node.kwargs,
            )
        else:
            raise ValueError(f"Unsupported node type: {type(node)}.")

    def to_str(self):
        """
        I do not override __str__ because I have not decided whether to use this way to serialize into pseudo-codes.
        """

        def _to_str(node: Operand, depth: int):
            spaces = " " * (depth * 2)
            if isinstance(node, TracedArray):
                return f"{spaces}{node.name} = {get_value(node)}"
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


@dataclass
class TracedArray(Operand):
    shape: tuple
    dtype: np.dtype
    value: Optional[ArrayLike] = None
    name: Optional[str] = None

    @property
    def requires_grad(self):
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
    def requires_grad(self):
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


def get_leaves(node: Node, include_constant=False, unique=True) -> list[TracedArray]:
    """
    Get the leaves of the graph.
    """
    leaves = []
    used_leaf = set()

    def _get_leaves(node: Node):
        if not include_constant and isinstance(node, ConstantArray):
            return
        if isinstance(node, TracedArray):
            if not unique or id(node) not in used_leaf:
                used_leaf.add(id(node))
                leaves.append(node)
        else:
            for operand in node.operands:
                _get_leaves(operand)

    _get_leaves(node)

    return leaves
