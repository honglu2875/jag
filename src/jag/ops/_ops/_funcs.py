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


def _replace(x, y, idx):
    """
    Replace the elements in 'x' with the elements in 'y' at the given indices.
    """
    x = x.copy()
    x[idx] = y
    return x


def unbroadcast(target, g, broadcast_idx=0):
    """
    'target' is the operand that is broadcast to 'g'. We need to sum 'g' along the broadcast axes.
    In the ordinary broadcasting convention, 'target' is broadcast to 'g' by
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
        g = np.sum(g, axis=summed_axis)

    summed_axis = tuple(
        idx for idx, size in enumerate(target.shape) if size == 1 and g.shape[idx] != 1
    )
    if summed_axis:
        g = np.sum(g, axis=summed_axis, keepdims=True)
    return g


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
        g = np.expand_dims(g, axis=axis)
    for a in axis:
        assert (
            g.shape[a] == 1
        ), f"Invalid dimension {a} on the out-gradient tensor. Got {g.shape[a]}."
        g = np.repeat(g, repeats=x.shape[a], axis=a)
    return (g,)


def matmul_vjp(g, x, y, **kwargs):
    def dot_lhs(g, lhs, rhs):
        if len(rhs.shape) == 0:
            return np.sum(np.multiply(rhs, g))
        if len(lhs.shape) == 1 and len(rhs.shape) == 1:
            return np.multiply(g, rhs)
        if len(lhs.shape) == 2 and len(rhs.shape) == 1:
            return np.multiply(np.expand_dims(g, axis=-1), rhs)
        if len(lhs.shape) == 1 and len(rhs.shape) == 2:
            return np.matmul(rhs, g)
        ndim = len(rhs.shape)
        return np.matmul(
            g, np.transpose(rhs, axes=tuple(range(ndim - 2)) + (ndim - 1, ndim - 2))
        )

    def dot_rhs(g, lhs, rhs):
        if len(rhs.shape) == 0:
            return np.sum(np.multiply(lhs, g))
        if len(lhs.shape) == 1 and len(rhs.shape) == 1:
            return np.multiply(g, lhs)
        if len(lhs.shape) == 2 and len(rhs.shape) == 1:
            return np.matmul(g, lhs)
        if len(lhs.shape) == 1 and len(rhs.shape) == 2:
            return np.multiply(np.expand_dims(lhs, axis=-1), g)
        ndim = len(lhs.shape)
        return np.matmul(
            np.transpose(lhs, axes=tuple(range(ndim - 2)) + (ndim - 1, ndim - 2)), g
        )

    return dot_lhs(g, x, y), dot_rhs(g, x, y)


def squeeze_vjp(g, x, **kwargs):
    """
    The only problem with the vjp of 'np.squeeze' is when axis=None. We take special care of this case.
    """
    if kwargs.get("axis", None) is None:
        return np.expand_dims(
            g, axis=tuple(idx for idx, size in enumerate(x.shape) if size == 1)
        )
    else:
        return (np.expand_dims(g, axis=kwargs["axis"]),)


def repeat_vjp(g, x, **kwargs):
    """
    The vjp of 'np.repeat' is to sum the vector along the repeated axis.
    We put 'repeats' into keyword arguments because it is non-differentiable.
    Now there are two cases:
    1. if repeats is an 'int', the case is easy, and we just sum across the corresponding axis.
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
            return (np.sum(np.reshape(g, (*x.shape, repeats)), axis=-1),)
        else:
            # [aabbccddee] gets reshaped into [aa,bb,cc,dd,ee] and the repeated dim is summed over.
            return (
                np.sum(
                    np.reshape(
                        g,
                        x.shape[:axis] + (x.shape[axis], repeats) + x.shape[axis + 1 :],
                    ),
                    axis=axis + 1,
                ),
            )
    elif isinstance(repeats, np.ndarray):
        # TODO: implement this
        raise NotImplementedError
    else:
        raise RuntimeError(f"Unsupported type for 'repeats': {type(repeats)}.")
