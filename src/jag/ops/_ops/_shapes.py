from ._funcs import squeeze_vjp, repeat_vjp, _replace
from .._traceable_op import TraceableOp
import numpy as np


squeeze = TraceableOp(np.squeeze, "squeeze").register_op(
    vjp=squeeze_vjp,  # Call unsqueeze and take special care of axis=None
    jvp=lambda g, x, **kwargs: squeeze(g, axis=kwargs.get("axis", None)),
)
unsqueeze = TraceableOp(np.expand_dims, "unsqueeze").register_op(
    vjp=lambda g, x, **kwargs: (
        squeeze(g, axis=kwargs["axis"]),
    ),  # unsqueeze must have a broadcast axis
    jvp=lambda g, x, **kwargs: unsqueeze(g, axis=kwargs["axis"]),
)
repeat = TraceableOp(np.repeat, "repeat", non_diff_args=[(1, "repeats")]).register_op(
    vjp=repeat_vjp,  # sum the vector along the repeated axis with extra care on special cases
    jvp=lambda g, x, **kwargs: repeat(
        g, kwargs["repeats"], axis=kwargs.get("axis", None)
    ),
)
reshape = TraceableOp(
    np.reshape, "reshape", non_diff_args=[(1, "newshape")]
).register_op(
    vjp=lambda g, x, **kwargs: (
        reshape(g, newshape=x.shape),
    ),  # restore the vector to the original shape
    jvp=lambda g, x, **kwargs: reshape(g, **kwargs),
)
transpose = TraceableOp(np.transpose, "transpose").register_op(
    vjp=lambda g, x, **kwargs: (
        transpose(g, axes=np.argsort(kwargs["axes"])),
    ),  # restore the transposition
    jvp=lambda g, x, **kwargs: transpose(g, axes=kwargs["axes"]),
)
zeros_like = TraceableOp(np.zeros_like, "zeros_like").register_op(
    vjp=lambda g, x, **kwargs: (zeros_like(x),),
    jvp=lambda g, x, **kwargs: zeros_like(x),
)
ones_like = TraceableOp(np.ones_like, "ones_like").register_op(
    vjp=lambda g, x, **kwargs: (zeros_like(x),),
    jvp=lambda g, x, **kwargs: zeros_like(x),
)
at = TraceableOp(lambda x, idx: x[idx], "at", non_diff_args=[(1, "idx")]).register_op(
    vjp=lambda g, x, **kwargs: (replace(zeros_like(x), g, idx=kwargs["idx"]),),
    jvp=lambda g, x, **kwargs: at(g, kwargs["idx"]),
)
replace = TraceableOp(_replace, "replace", non_diff_args=[(2, "idx")]).register_op(
    vjp=lambda g, x, y, **kwargs: (
        replace(g, zeros_like(y), idx=kwargs["idx"]),
        at(g, idx=kwargs["idx"]),
    ),
    jvp=lambda g, h, x, y, **kwargs: replace(g, h, idx=kwargs["idx"]),
)
