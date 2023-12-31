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
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Type

import numpy as np

from jag.type import GraphNode

"""
Traceable ops along with its vector-Jacobian function and Jacobian-vector function 
are registered in `_traceable_op_registry`. We expose the following functions
for its manipulation.
- `register_op`
- `get_registered_ops`
"""


def _assert_non_traceable(kwargs, cls):
    assert all(not isinstance(arg, cls) for arg in kwargs), (
        "All values of keyword arguments must be non-traceable. "
        "Or it might invoke an infinite recursion during tracing."
    )


def _is_all_zero(x) -> bool:
    """
    Decide whether x is
        1. a constant zero integer
        2. or a numpy array with all zeros
        2. or a ConstantArray with an all-zero numpy array as value
    """
    if isinstance(x, int):
        return x == 0
    if isinstance(x, np.ndarray):
        return np.all(x == 0)
    elif isinstance(x, GraphNode):
        return (
            x.is_constant() and isinstance(x.value, np.ndarray) and np.all(x.value == 0)
        )
    else:
        return False


@dataclass
class TraceableOp:
    """
    The base class of traceable ops. A traceable function must form a computational graph using traceable ops.

    Important notes:
    1. Best way to use the op: all non-differentiable arguments are put in the keyword arguments.
       But to make it more robust, I include 'non_diff_args' in the registration to optionally move
         specified positional arguments into keyword arguments.
    2. The shape of the output must be inferred from the shapes of the operands.

    Attributes:
        op: the base implementation of the op.
        name: the name of the op. Must be unique.
        shape_fn: a function that takes the shapes of the operands and returns the shape of the output.
                  The default assumes the base implementation is a numpy function and uses all-one dummies
                  to infer the output shape.
        non_diff_args: a sequence of (argnum, argname) for non-differentiable arguments.
    """

    op: Callable
    name: str
    shape_fn: Optional[Callable] = None
    out_dtype: Optional[np.dtype] = None
    non_diff_args: Optional[Sequence[tuple[int, str]]] = None
    # Mutable members are shared across class objects. It is a feature, not a bug.
    # We use this feature below to avoid circular import by pushing the Node object (dependent on TraceableOp itself)
    #   at import-time.
    _global_node_cls = []
    _global_leaf_cls = []
    _to_traceable_fn = []
    # Ops registration is shared across class objects.
    _traceable_op_registry = {}

    def __post_init__(self):
        if self.shape_fn is None:

            def shape_fn(*shapes, **kwargs):
                _assert_non_traceable(kwargs, (self.node_cls, self.leaf_cls))
                dummies = [np.ones(shape) for shape in shapes]
                return self.op(*dummies, **kwargs).shape

            self.shape_fn = shape_fn

    def _arg_preprocess(self, args, kwargs):
        """
        Preprocess the arguments to make sure all non-differentiable arguments are put in the keyword arguments.
        Rules:
            - if the non-differentiable-argument (element in non_diff_args) has argnum > len(args), ignore
            - if otherwise, move the argument from args to kwargs
        Args:
            args: the positional arguments.
            kwargs: the keyword arguments.
        """
        removed_args = set()
        for argnum, argname in self.non_diff_args or []:
            if argnum >= len(args):
                continue
            if argname in kwargs:
                raise ValueError(
                    f"Argument '{argname}' is specified both as a positional argument "
                    f"and a keyword argument."
                )
            kwargs[argname] = args[argnum]
            removed_args.add(argnum)
        args = tuple(arg for idx, arg in enumerate(args) if idx not in removed_args)
        return args, kwargs

    def __call__(self, *args, **kwargs):
        args, kwargs = self._arg_preprocess(args, kwargs)

        trace = kwargs.pop("trace", False)
        if any(isinstance(arg, GraphNode) for arg in args + tuple(kwargs.values())):
            trace = True

        if trace:
            # Pruning: if any of the operands is all-zero during mul or matmul, the result is all-zero.
            if self.name == "multiply" or self.name == "matmul":
                if any(_is_all_zero(a) for a in args):
                    return self.leaf_cls(
                        value=np.zeros(
                            shape=self.shape_fn(*[arg.shape for arg in args], **kwargs),
                            dtype=self.out_dtype
                            or self._get_implied_dtype(args, **kwargs),
                        )
                    ).to_const()
            # Pruning: if any of the operands is all-zero during add or subtract, the result is the other operand.
            elif self.name == "add":
                if any(_is_all_zero(a) for a in args):
                    return args[0] if _is_all_zero(args[1]) else args[1]
            elif self.name == "subtract":
                if _is_all_zero(args[1]):
                    return args[0]
                elif _is_all_zero(args[0]):
                    return -args[1]

            traceable_args = [self.to_traceable(arg) for arg in args]
            dummy_kwargs = self._convert_kwargs_to_nontraceable(kwargs)
            return self.node_cls(
                op=self,
                operands=traceable_args,
                shape=self.shape_fn(
                    *[arg.shape for arg in traceable_args], **dummy_kwargs
                ),
                dtype=self.out_dtype
                or self._get_implied_dtype(traceable_args, **dummy_kwargs),
                kwargs=kwargs,
            )
        else:
            return self.op(*args, **kwargs)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self.__dict__)

    def _get_implied_dtype(self, args, **kwargs) -> np.dtype:
        _assert_non_traceable(kwargs, GraphNode)
        dummies = [np.ones(arg.shape, dtype=arg.dtype) for arg in args]
        return np.array(self.op(*dummies, **kwargs)).dtype

    @staticmethod
    def _convert_kwargs_to_nontraceable(kwargs):
        """A utility function to convert all traceable arguments in kwargs to dummy zeros."""
        return {
            k: np.zeros(shape=v.shape, dtype=v.dtype) if isinstance(v, GraphNode) else v
            for k, v in kwargs.items()
        }

    @property
    def node_cls(self) -> Type:
        assert len(self._global_node_cls) == 1, (
            "The 'Node' class is either not defined or ambiguous. "
            "Most likely the library is not imported correctly."
        )
        return self._global_node_cls[0]

    @property
    def leaf_cls(self) -> Type:
        assert len(self._global_leaf_cls) == 1, (
            "The 'Leaf' class is either not defined or ambiguous. "
            "Most likely the library is not imported correctly."
        )
        return self._global_leaf_cls[0]

    @property
    def to_traceable(self) -> Callable:
        assert len(self._to_traceable_fn) == 1, (
            "The 'to_traceable' function is either not defined or ambiguous. "
            "Most likely the library is not imported correctly."
        )
        return self._to_traceable_fn[0]

    def register_op(
        self,
        name: Optional[str] = None,
        vjp: Optional[Callable] = None,
        jvp: Optional[Callable] = None,
        overwrite: bool = False,
    ):
        name = name or self.name
        if not overwrite and name in TraceableOp._traceable_op_registry:
            raise ValueError(
                f"'{name}' already exists in the op registry. "
                f"If you wish to overwrite, please set 'overwrite=True'."
            )
        self._traceable_op_registry[name] = {"op": self, "vjp": vjp, "jvp": jvp}
        return self

    @classmethod
    def get_op_registration(cls, name: str) -> dict:
        return cls._traceable_op_registry.get(name, None)


def register_op(
    name: str,
    traceable_op: TraceableOp,
    vjp: Optional[Callable],
    jvp: Optional[Callable],
    overwrite: bool = False,
):
    """
    Convenient global function to register an op.
    """
    traceable_op.register_op(name, vjp, jvp, overwrite)


def get_op_registration(name: str) -> dict:
    """
    Convenient global function to get an op registration.
    """
    return TraceableOp.get_op_registration(name)
