from typing import Callable, Optional
import numpy as np
from ._traceable_op import TraceableOp

"""
Traceable ops along with its vector-Jacobian function and Jacobian-vector function 
are registered in `_traceable_op_registry`. We expose the following functions
for its manipulation.
- `register_op`
- `get_registered_ops`
"""

_traceable_op_registry = {}


def register_op(
    name: str,
    traceable_op: TraceableOp,
    vjp: Optional[Callable],
    jvp: Optional[Callable],
    overwrite: bool = False,
):
    if not isinstance(traceable_op, TraceableOp):
        raise ValueError(
            f"'traceable_op' must be a TraceableOp object. Got {type(traceable_op)} instead."
        )
    if not overwrite and name in _traceable_op_registry:
        raise ValueError(
            f"'{name}' already exists in the op registry. "
            f"If you wish to overwrite, please set 'overwrite=True'."
        )
    _traceable_op_registry[name] = {"op": traceable_op, "vjp": vjp, "jvp": jvp}


def get_op_registration(name: str) -> dict:
    return _traceable_op_registry.get(name, None)


# ----------------------------------- Primitive op implementations ----------------------------------- #


# Shape manipulations
squeeze = TraceableOp(np.squeeze, "squeeze")
unsqueeze = TraceableOp(np.expand_dims, "unsqueeze")
repeat = TraceableOp(np.repeat, "repeat", non_diff_args=[(1, "repeats")])
reshape = TraceableOp(np.reshape, "reshape", non_diff_args=[(1, "newshape")])
transpose = TraceableOp(np.transpose, "transpose")

# Basic arithmetics
sum = TraceableOp(np.sum, "sum")
add = TraceableOp(np.add, "add")
subtract = TraceableOp(np.subtract, "subtract")
negative = TraceableOp(lambda x: -x, "negative")
multiply = TraceableOp(np.multiply, "multiply")
divide = TraceableOp(np.divide, "divide")
matmul = TraceableOp(np.matmul, "matmul")
where = TraceableOp(lambda x, y, condition=None: np.where(condition, x, y), "where")
power = TraceableOp(np.power, "power")


# ----------------------------------- vjp and jvp implementations ----------------------------------- #


def unbroadcast(target, g, broadcast_idx=0):
    """
    'target' is the operand that is broadcasted to 'g'. We need to sum 'g' along the broadcasted axes.
    In the ordinary broadcasting convention, 'target' is broadcasted to 'g' by
    1. first adding leading singleton dimensions to 'target' until it has the same number of dimensions as 'g'
    2. then repeating 'target' along the singleton dimensions until it has the same shape as 'g'
    """
    if broadcast_idx > 0:
        summed_axis = tuple(
            range(broadcast_idx, broadcast_idx + len(g.shape) - len(target.shape))
        )
    else:
        summed_axis = tuple(
            range(broadcast_idx, broadcast_idx - len(g.shape) + len(target.shape), -1)
        )
    if summed_axis:
        g = sum(g, axis=summed_axis)

    summed_axis = tuple(
        idx for idx, size in enumerate(target.shape) if size == 1 and g.shape[idx] != 1
    )
    if summed_axis:
        g = sum(g, axis=summed_axis, keepdims=True)
    return g


def squeeze_vjp(g, x, **kwargs):
    """
    The only problem with the vjp of 'np.squeeze' is when axis=None. We take special care of this case.
    """
    if kwargs.get("axis", None) is None:
        return unsqueeze(
            g, axis=tuple(idx for idx, size in enumerate(x.shape) if size == 1)
        )
    else:
        return unsqueeze(g, axis=kwargs["axis"])


def repeat_vjp(g, x, **kwargs):
    """
    The vjp of 'np.repeat' is to sum the vector along the repeated axis.
    We put 'repeats' into keyword arguments because it is non-differentiable.
    Now there are two cases:
    1. if repeats is an 'int', the case is easy and we just sum across the corresponding axis.
    2. if the repeats is an array, we need to sum over the axis with extra care.
    In addition, if axis=None, 'np.repeat' flattens the array and vjp reconstructs the original shape.
    """
    axis = kwargs.get("axis", None)
    assert axis is None or isinstance(
        axis, int
    ), f"Unsupported type for 'axis': {type(axis)}."
    repeats = kwargs["repeats"]
    if isinstance(repeats, int):
        if axis is None:
            return sum(reshape(g, (*x.shape, repeats)), axis=-1)
        else:
            return sum(
                reshape(g, x.shape[:axis] + (repeats,) + x.shape[axis:]), axis=axis
            )
    elif isinstance(repeats, np.ndarray):
        # TODO: implement this
        raise NotImplementedError
    else:
        raise RuntimeError(f"Unsupported type for 'repeats': {type(repeats)}.")


def sum_vjp(g, x, **kwargs):
    axis = kwargs.get("axis", None)
    keepdims = kwargs.get("keepdims", False)
    if axis is None:
        assert not keepdims, f"Cannot keep dimension because axis=None."
        assert not g.shape or g.shape == (
            1,
        ), f"Invalid shape for the out-gradient tensor. Got {g.shape}."
        axis = tuple(range(len(x.shape)))
    elif isinstance(axis, int):
        axis = (axis,)
    if not keepdims:
        g = unsqueeze(g, axis=axis)
    for a in axis:
        assert (
            g.shape[a] == 1
        ), f"Invalid dimension {a} on the out-gradient tensor. Got {g.shape[a]}."
        g = repeat(g, repeats=x.shape[axis], axis=a)
    return g


def matmul_vjp(g, x, y, **kwargs):
    def dot_lhs(g, lhs, rhs):
        if len(rhs.shape) == 0:
            return sum(multiply(rhs, g))
        if len(lhs.shape) == 1 and len(rhs.shape) == 1:
            return multiply(g, rhs)
        if len(lhs.shape) == 2 and len(rhs.shape) == 1:
            return multiply(unsqueeze(g, axis=-1), rhs)
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
            return multiply(unsqueeze(lhs, axis=-1), g)
        return matmul(lhs.T, g)

    return dot_lhs(g, x, y), dot_rhs(g, x, y)


# ----------------------------------- Register ops+vjp+jvp as a whole ----------------------------------- #


register_op(
    "squeeze",
    squeeze,
    vjp=squeeze_vjp,  # Call unsqueeze and take special care of axis=None
    jvp=lambda g, x, **kwargs: squeeze(g, axis=kwargs.get("axis", None)),
)
register_op(
    "unsqueeze",
    unsqueeze,
    vjp=lambda g, x, **kwargs: squeeze(
        g, axis=kwargs["axis"]
    ),  # unsqueeze must have a broadcasted axis
    jvp=lambda g, x, **kwargs: unsqueeze(g, axis=kwargs["axis"]),
)
register_op(
    "repeat",
    repeat,
    vjp=repeat_vjp,  # sum the vector along the repeated axis with extra care on special cases
    jvp=lambda g, x, **kwargs: repeat(
        g, kwargs["repeats"], axis=kwargs.get("axis", None)
    ),
)
register_op(
    "reshape",
    reshape,
    vjp=lambda g, x, **kwargs: reshape(
        g, newshape=x.shape
    ),  # restore the vector to the original shape
    jvp=lambda g, x, **kwargs: reshape(g, **kwargs),
)  # reshape the vector directly to the new shape
register_op(
    "transpose",
    transpose,
    vjp=lambda g, x, **kwargs: transpose(
        g, axes=np.argsort(kwargs["axes"])
    ),  # restore the transposition
    jvp=lambda g, x, **kwargs: transpose(g, axes=kwargs["axes"]),
)

register_op(
    "sum",
    sum,
    vjp=sum_vjp,  # expand the vector according to the summed axis
    jvp=lambda g, x, **kwargs: sum(g, **kwargs),
)  # sum the vector along the summed axis
register_op(
    "add",
    add,
    vjp=lambda g, x, y, **kwargs: (unbroadcast(x, g), unbroadcast(y, g)),
    jvp=lambda g, h, x, y, **kwargs: g + h,
)
register_op(
    "multiply",
    multiply,
    vjp=lambda g, x, y, **kwargs: (unbroadcast(x, g * y), unbroadcast(y, g * x)),
    jvp=lambda g, h, x, y, **kwargs: g * y + h * x,
)
register_op(
    "subtract",
    subtract,
    vjp=lambda g, x, y, **kwargs: (unbroadcast(x, g), unbroadcast(y, -g)),
    jvp=lambda g, h, x, y, **kwargs: g - h,
)
register_op(
    "negative",
    negative,
    vjp=lambda g, x, **kwargs: negative(g),
    jvp=lambda g, x, **kwargs: negative(g),
)
register_op(
    "divide",
    divide,
    vjp=lambda g, x, y, **kwargs: (
        unbroadcast(x, g / y),
        unbroadcast(y, -g * x / y**2),
    ),
    jvp=lambda g, h, x, y, **kwargs: (g * y - x * h) / y**2,
)
register_op(
    "matmul",
    matmul,
    vjp=matmul_vjp,
    jvp=lambda g, h, x, y, **kwargs: matmul(g, y) + matmul(x, h),
)
# TODO: where and power
