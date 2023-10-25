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
import numpy as np

from .._traceable_op import TraceableOp
from ._funcs import _replace, repeat_vjp, squeeze_vjp

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
