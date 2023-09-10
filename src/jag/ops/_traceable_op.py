from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Type

import numpy as np


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

    def __post_init__(self):
        if self.shape_fn is None:

            def shape_fn(*shapes, **kwargs):
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
        if any(isinstance(arg, (self.node_cls, self.leaf_cls)) for arg in args):
            trace = True

        if trace:
            traceable_args = [self.to_traceable(arg) for arg in args]
            return self.node_cls(
                op=self,
                operands=traceable_args,
                shape=self.shape_fn(*[arg.shape for arg in traceable_args], **kwargs),
                dtype=self.out_dtype
                or self._get_implied_dtype(traceable_args, **kwargs),
                kwargs=kwargs,
            )
        else:
            return self.op(*args, **kwargs)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self.__dict__)

    def _get_implied_dtype(self, args, **kwargs) -> np.dtype:
        dummies = [np.ones(arg.shape, dtype=arg.dtype) for arg in args]
        return self.op(*dummies, **kwargs).dtype

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
