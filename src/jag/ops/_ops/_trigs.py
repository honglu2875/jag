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

sin = TraceableOp(np.sin, "sin").register_op(
    vjp=lambda g, x, **kwargs: (g * np.cos(x),),
    jvp=lambda g, x, **kwargs: g * np.cos(x),
)
cos = TraceableOp(np.cos, "cos").register_op(
    vjp=lambda g, x, **kwargs: (-g * np.sin(x),),
    jvp=lambda g, x, **kwargs: -g * np.sin(x),
)
tan = TraceableOp(np.tan, "tan").register_op(
    vjp=lambda g, x, **kwargs: (g / np.cos(x) ** 2,),
    jvp=lambda g, x, **kwargs: g / np.cos(x) ** 2,
)
arcsin = TraceableOp(np.arcsin, "arcsin").register_op(
    vjp=lambda g, x, **kwargs: (g / np.power(1 - x**2, 0.5),),
    jvp=lambda g, x, **kwargs: g / np.power(1 - x**2, 0.5),
)
arccos = TraceableOp(np.arccos, "arccos").register_op(
    vjp=lambda g, x, **kwargs: (-g / np.power(1 - x**2, 0.5),),
    jvp=lambda g, x, **kwargs: -g / np.power(1 - x**2, 0.5),
)
arctan = TraceableOp(np.arctan, "arctan").register_op(
    vjp=lambda g, x, **kwargs: (g / (1 + x**2),),
    jvp=lambda g, x, **kwargs: g / (1 + x**2),
)
sinh = TraceableOp(np.sinh, "sinh").register_op(
    vjp=lambda g, x, **kwargs: (g * np.cosh(x),),
    jvp=lambda g, x, **kwargs: g * np.cosh(x),
)
cosh = TraceableOp(np.cosh, "cosh").register_op(
    vjp=lambda g, x, **kwargs: (g * np.sinh(x),),
    jvp=lambda g, x, **kwargs: g * np.sinh(x),
)
tanh = TraceableOp(np.tanh, "tanh").register_op(
    vjp=lambda g, x, **kwargs: (g / np.cosh(x) ** 2,),
    jvp=lambda g, x, **kwargs: g / np.cosh(x) ** 2,
)
