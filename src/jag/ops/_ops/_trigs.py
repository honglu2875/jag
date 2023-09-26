from .._traceable_op import TraceableOp
import numpy as np


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
