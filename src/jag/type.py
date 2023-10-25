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
import abc
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Any, Optional, Union

import numpy as np

ArrayLike = Union[np.ndarray]


@dataclass(kw_only=True)
class GraphNode:
    """
    Abstract class of a node in the computation graph.
    Only define the bare minimum for autodiff to work.
    """

    name: Optional[str] = None
    operands: Optional[list] = None
    shape: Optional[tuple] = None
    dtype: Optional[Any] = None

    def __post_init__(self):
        self.operands = self.operands or []

    @singledispatchmethod
    def execute(self, *args):
        return self._execute(dict(zip([id(l) for l in self.leaves()], args)))

    @execute.register
    def _(self, node_values: dict):
        return self._execute(node_values)

    @abc.abstractmethod
    def _execute(self, value_dict: dict):
        """
        Recursively execute the graph from the node, with the given values of the leaves.
        Args:
            value_dict: the dictionary of values of the leaves.
        Returns:
            the value of the node.
        """
        raise NotImplementedError("Not implemented.")

    def leaves(
        self, include_constant=False, include_kwarg=False, unique=True, used_leaves=None
    ):
        """
        Get the leaves of the graph.
        Args:
            include_constant: whether to include constant nodes as leaves.
            include_kwarg: whether to include nodes in kwargs as leaves.
            unique: whether to return a unique list of leaves.
            used_leaves: a set of ids of leaves that have been used and should not return. Used for recursion.
        Returns:
            the leaves of the graph.
        """
        used_leaves = used_leaves if used_leaves is not None else set()
        if self.is_leaf():
            if not unique or id(self) not in used_leaves:
                yield self
                used_leaves.add(id(self))
        else:
            for leaf in (
                self.operands + list(self.kwargs.values())
                if include_kwarg
                else self.operands
            ):
                if not include_constant and leaf.is_constant():
                    continue
                yield from leaf.leaves(
                    include_constant, include_kwarg, unique, used_leaves
                )

    def is_leaf(self):
        return not self.operands

    @property
    def requires_grad(self):
        """
        Default is False for leaves. TracedArray will override it.
        """
        if not self.is_leaf():
            return any([child.requires_grad for child in self.operands])
        else:
            return False

    @requires_grad.setter
    def requires_grad(self, value):
        """
        TracedArray: can have mutable requires_grad
        ConstantArray: cannot change the value of requires_grad
        leaving the ValueError as default and let TracedArray override it
        """
        raise ValueError("Cannot set requires_grad for a leaf node.")

    def is_constant(self):
        return self.is_leaf() and not self.requires_grad


def _assure_node_with_op(node: GraphNode):
    assert hasattr(
        node, "op"
    ), f"node must be Node class, but get {type(node)} instead."


def _assure_kwargs(node: GraphNode):
    if node.is_leaf():
        return
    if not hasattr(node, "kwargs"):
        raise ValueError(
            f"include_kwargs=True requires the node {node} to have kwargs."
        )
    if not isinstance(node.kwargs, dict):
        raise ValueError(
            f"include_kwargs=True requires the node {node} to have kwargs of type dict."
        )
