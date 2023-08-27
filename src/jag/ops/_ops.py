import warnings
from typing import Callable, Any, Optional, Concatenate, Sequence
from numbers import Number
from dataclasses import dataclass
import numpy as np
from jag.graph import Node, Operand, ConstantArray, TracedArray


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
                  The default assumes the base implementation is a numpy function and uses all-zero dummies
                  to infer the output shape.
        non_diff_args: a sequence of (argnum, argname) for non-differentiable arguments.
    """
    op: Callable
    name: str
    shape_fn: Optional[Callable] = None
    non_diff_args: Optional[Sequence[tuple[int, str]]] = None

    def __post_init__(self):
        if self.shape_fn is None:
            def shape_fn(*shapes, **kwargs):
                dummies = [np.zeros(shape) for shape in shapes]
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
                raise ValueError(f"Argument '{argname}' is specified both as a positional argument "
                                 f"and a keyword argument.")
            kwargs[argname] = args[argnum]
            removed_args.add(argnum)
        args = tuple(arg for idx, arg in enumerate(args) if idx not in removed_args)
        return args, kwargs

    def __call__(self, *args, **kwargs):
        args, kwargs = self._arg_preprocess(args, kwargs)

        trace = kwargs.pop('trace', False)
        if any(isinstance(arg, Node) for arg in args):
            warnings.warn(f"Arguments contain abstract objects (Operand, Node, TracedArray, etc.). "
                          f"Forcing 'trace=True' to trace the computation graph.")
            trace = True

        if trace:
            return Node(op=self,
                        operands=[to_traceable(arg) for arg in args],
                        shape=self.shape_fn(*[arg.shape for arg in args], **kwargs),
                        kwargs=kwargs)
        else:
            return self.op(*args, **kwargs)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self.__dict__)


"""
Traceable ops along with its vector-Jacobian function and Jacobian-vector function 
are registered in `_traceable_op_registry`. We expose the following functions
for its manipulation.
- `register_op`
- `get_registered_ops`
"""

_traceable_op_registry = {}


def register_op(name: str,
                traceable_op: TraceableOp,
                vjp: Optional[Callable],
                jvp: Optional[Callable],
                overwrite: bool = False):
    if not isinstance(traceable_op, TraceableOp):
        raise ValueError(f"'traceable_op' must be a TraceableOp object. Got {type(traceable_op)} instead.")
    if not overwrite and name in _traceable_op_registry:
        raise ValueError(f"'{name}' already exists in the op registry. "
                         f"If you wish to overwrite, please set 'overwrite=True'.")
    _traceable_op_registry[name] = {'op': traceable_op,
                                    'vjp': vjp,
                                    'jvp': jvp}


def get_registered_op() -> list:
    return list(_traceable_op_registry.values())


# ----------------------------------- Primitive ops implementations ----------------------------------- #


# Shape manipulations
squeeze = TraceableOp(np.squeeze, 'squeeze')
unsqueeze = TraceableOp(np.expand_dims, 'unsqueeze')
repeat = TraceableOp(np.repeat, 'repeat', non_diff_args=[(1, 'repeats')])
reshape = TraceableOp(np.reshape, 'reshape', non_diff_args=[(1, 'newshape')])
transpose = TraceableOp(np.transpose, 'transpose')
expand_dims = TraceableOp(np.expand_dims, 'expand_dims')

# Basic arithmetics
sum = TraceableOp(np.sum, 'sum')
add = TraceableOp(np.add, 'add')
subtract = TraceableOp(np.subtract, 'subtract')
negative = TraceableOp(lambda x: -x, 'negative')
multiply = TraceableOp(np.multiply, 'multiply')
divide = TraceableOp(np.divide, 'divide')
matmul = TraceableOp(np.matmul, 'multiply')
where = TraceableOp(lambda x, y, condition=None: np.where(condition, x, y), 'where')


# ----------------------------------- vjp and jvp implementations ----------------------------------- #


def unbroadcast(target, g, broadcast_idx=0):
    """
    'target' is the operand that is broadcasted to 'g'. We need to sum 'g' along the broadcasted axes.
    In the ordinary broadcasting convention, 'target' is broadcasted to 'g' by
    1. first adding leading singleton dimensions to 'target' until it has the same number of dimensions as 'g'
    2. then repeating 'target' along the singleton dimensions until it has the same shape as 'g'
    """
    if broadcast_idx > 0:
        summed_axis = tuple(range(broadcast_idx, broadcast_idx + len(g.shape) - len(target.shape)))
    else:
        summed_axis = tuple(range(broadcast_idx, broadcast_idx - len(g.shape) + len(target.shape), -1))
    if summed_axis:
        g = sum(g, axis=summed_axis)

    summed_axis = tuple(idx for idx, size in enumerate(target.shape) if size == 1 and g.shape[idx] != 1)
    if summed_axis:
        g = sum(g, axis=summed_axis, keepdims=True)
    return g


def squeeze_vjp(g, x, **kwargs):
    """
    The only problem with the vjp of 'np.squeeze' is when axis=None. We take special care of this case.
    """
    if kwargs.get('axis', None) is None:
        return unsqueeze(g, axis=tuple(idx for idx, size in enumerate(x.shape) if size == 1))
    else:
        return unsqueeze(g, axis=kwargs['axis'])


def repeat_vjp(g, x, **kwargs):
    """
    The vjp of 'np.repeat' is to sum the vector along the repeated axis.
    We put 'repeats' into keyword arguments because it is non-differentiable.
    Now there are two cases:
    1. if repeats is an 'int', the case is easy and we just sum across the corresponding axis.
    2. if the repeats is an array, we need to sum over the axis with extra care.
    In addition, if axis=None, 'np.repeat' flattens the array and vjp reconstructs the original shape.
    """
    axis = kwargs.get('axis', None)
    assert axis is None or isinstance(axis, int), f"Unsupported type for 'axis': {type(axis)}."
    repeats = kwargs['repeats']
    if isinstance(repeats, int):
        if axis is None:
            return sum(reshape(g, (*x.shape, repeats)), axis=-1)
        else:
            return sum(reshape(g, x.shape[:axis] + (repeats,) + x.shape[axis:]), axis=axis)
    elif isinstance(repeats, np.ndarray):
        # TODO: implement this
        raise NotImplementedError
    else:
        raise RuntimeError(f"Unsupported type for 'repeats': {type(repeats)}.")


def sum_vjp(g, x, **kwargs):
    axis = kwargs.get('axis', None)
    keepdims = kwargs.get('keepdims', False)
    # TODO: implement this


"""
def matmul_vjp(g, x, y):
    if isinstance(x, Node) or isinstance(y, Node) or isinstance(g, Node):
        return matmul(g, y.T), matmul(x.T, g)

    def dot_lhs(g, lhs, rhs):
        if len(rhs.shape) == 0:
            return sum(multiply(rhs, g))
        if len(lhs.shape) == 1 and len(rhs.shape) == 1:
            return multiply(g, rhs)
        if len(lhs.shape) == 2 and len(rhs.shape) == 1:
            return multiply(g[:, None], rhs)
        if len(lhs.shape) == 1 and len(rhs.shape) == 2:
            return matmul(rhs, g)
        return matmul(g, rhs.T)

    def dot_rhs(g, lhs, rhs):
        if len(rhs.shape) == 0:
            return sum(multiply(lhs, g))
        if len(lhs.shape) == 1 and len(rhs.shape) == 1:
            return multiply(g, lhs)
        if len(lhs.shape) == 2 and len(rhs.shape) == 1:
            return matmul(g, lhs)
        if len(lhs.shape) == 1 and len(rhs.shape) == 2:
            return multiply(lhs[:, None], g)
        return matmul(lhs.T, g)

    return dot_lhs(g, x, y), dot_rhs(g, x, y)
"""


# ----------------------------------- Register ops+vjp+jvp as a whole ----------------------------------- #


register_op('squeeze',
            squeeze,
            vjp=squeeze_vjp,  # Call unsqueeze and take special care of axis=None
            jvp=lambda g, x, **kwargs: squeeze(g, axis=kwargs.get('axis', None)))
register_op('unsqueeze',
            unsqueeze,
            vjp=lambda g, x, **kwargs: squeeze(g, axis=kwargs['axis']),  # unsqueeze must have a broadcasted axis
            jvp=lambda g, x, **kwargs: unsqueeze(g, axis=kwargs['axis']))
register_op('repeat',
            repeat,
            vjp=repeat_vjp,  # sum the vector along the repeated axis with extra care on special cases
            jvp=lambda g, x, **kwargs: repeat(g, kwargs['repeats'], axis=kwargs.get('axis', None)))
register_op('reshape',
            reshape,
            vjp=lambda g, x, **kwargs: reshape(g, newshape=x.shape),  # restore the vector to the original shape
            jvp=lambda g, x, **kwargs: reshape(g, **kwargs))  # reshape the vector directly to the new shape
register_op('transpose',
            transpose,
            vjp=lambda g, x, **kwargs: transpose(g, axes=np.argsort(kwargs['axes'])),  # restore the transposition
            jvp=lambda g, x, **kwargs: transpose(g, axes=kwargs['axes']))

register_op('sum',
            sum,
            vjp=sum_vjp,  # expand the vector according to the summed axis
            jvp=lambda g, x, **kwargs: sum(g, **kwargs))  # sum the vector along the summed axis
register_op('add',
            add,
            vjp=lambda g, x, y, **kwargs: (unbroadcast(x, g), unbroadcast(y, g)),
            jvp=lambda g, x, y, **kwargs: (unbroadcast(x, g), unbroadcast(y, g)))
register_op('multiply',
            multiply,
            vjp=lambda g, x, y, **kwargs: (unbroadcast(x, g * y), unbroadcast(y, g * x)),
            jvp=lambda g, x, y, **kwargs: (unbroadcast(x, y * g), unbroadcast(y, x * g)))
register_op('subtract',
            subtract,
            vjp=lambda g, x, y, **kwargs: (unbroadcast(x, g), unbroadcast(y, -g)),
            jvp=lambda g, x, y, **kwargs: (unbroadcast(x, g), unbroadcast(y, -g)))
register_op('negative',
            negative,
            vjp=lambda g, x, **kwargs: negative(g),
            jvp=lambda g, x, **kwargs: negative(g))
register_op('divide',
            divide,
            vjp=lambda g, x, y, **kwargs: (unbroadcast(x, g / y), unbroadcast(y, -g * x / y ** 2)),
            jvp=lambda g, x, y, **kwargs: (unbroadcast(x, g / y), unbroadcast(y, -g * x / y ** 2)))
#register_op('matmul',
#            matmul,
#            vjp=matmul_vjp,
#            jvp=matmul_jvp)
