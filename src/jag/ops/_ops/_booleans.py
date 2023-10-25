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

where = TraceableOp(
    lambda x, y, condition=None: np.where(condition, x, y),
    "where",
    non_diff_args=[(2, "condition")],
).register_op(
    vjp=lambda g, x, y, **kwargs: (
        where(g, np.zeros_like(g), condition=kwargs["condition"]),
        where(np.zeros_like(g), g, condition=kwargs["condition"]),
    ),
    jvp=lambda g, h, x, y, **kwargs: where(g, h, condition=kwargs["condition"]),
)
maximum = TraceableOp(np.maximum, "maximum").register_op(
    vjp=lambda g, x, y, **kwargs: (
        where(g, np.zeros_like(g), condition=x > y),
        where(g, np.zeros_like(g), condition=x <= y),
    ),
    jvp=lambda g, h, x, y, **kwargs: where(g, h, condition=x > y),
)
greater = TraceableOp(np.greater, "greater").register_op(
    vjp=lambda g, x, y, **kwargs: (np.zeros_like(x), np.zeros_like(y)),
    jvp=lambda g, h, x, y, **kwargs: np.zeros_like(x),
)
less = TraceableOp(np.less, "less").register_op(
    vjp=lambda g, x, y, **kwargs: (np.zeros_like(x), np.zeros_like(y)),
    jvp=lambda g, h, x, y, **kwargs: np.zeros_like(x),
)
greater_equal = TraceableOp(np.greater_equal, "greater_equal").register_op(
    vjp=lambda g, x, y, **kwargs: (np.zeros_like(x), np.zeros_like(y)),
    jvp=lambda g, h, x, y, **kwargs: np.zeros_like(x),
)
less_equal = TraceableOp(np.less_equal, "less_equal").register_op(
    vjp=lambda g, x, y, **kwargs: (np.zeros_like(x), np.zeros_like(y)),
    jvp=lambda g, h, x, y, **kwargs: np.zeros_like(x),
)
clip = TraceableOp(
    np.clip, "clip", non_diff_args=[(1, "a_min"), (2, "a_max")]
).register_op(
    vjp=lambda g, x, **kwargs: (
        where(
            g, np.zeros_like(g), condition=(x > kwargs["a_min"]) & (x < kwargs["a_max"])
        ),
    ),
    jvp=lambda g, x, **kwargs: where(
        g, np.zeros_like(g), condition=(x > kwargs["a_min"]) & (x < kwargs["a_max"])
    ),
)
