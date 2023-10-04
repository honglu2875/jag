import numpy as np

from .._traceable_op import TraceableOp
from ._funcs import matmul_vjp, sum_vjp, unbroadcast

sum = TraceableOp(np.sum, "sum").register_op(
    vjp=sum_vjp, jvp=lambda g, x, **kwargs: np.sum(g, **kwargs)
)
mean = TraceableOp(np.mean, "mean").register_op(
    vjp=lambda g, x, **kwargs: (
        sum_vjp(g, x, **kwargs)[0] / (x.size // g.size),
    ),  # todo: make it better?
    jvp=lambda g, x, **kwargs: np.mean(g, **kwargs),
)
add = TraceableOp(np.add, "add").register_op(
    vjp=lambda g, x, y, **kwargs: (unbroadcast(x, g), unbroadcast(y, g)),
    jvp=lambda g, h, x, y, **kwargs: g + h,
)
sign = TraceableOp(np.sign, "sign").register_op(
    vjp=lambda g, x, **kwargs: (np.zeros_like(x),),
    jvp=lambda g, x, **kwargs: np.zeros_like(x),
)
absolute = TraceableOp(np.abs, "absolute").register_op(
    vjp=lambda g, x, **kwargs: (g * np.sign(x),),
    jvp=lambda g, x, **kwargs: g * np.sign(x),
)
subtract = TraceableOp(np.subtract, "subtract").register_op(
    vjp=lambda g, x, y, **kwargs: (unbroadcast(x, g), unbroadcast(y, -g)),
    jvp=lambda g, h, x, y, **kwargs: g - h,
)
negative = TraceableOp(lambda x: -x, "negative").register_op(
    vjp=lambda g, x, **kwargs: (np.negative(g),),
    jvp=lambda g, x, **kwargs: np.negative(g),
)
multiply = TraceableOp(np.multiply, "multiply").register_op(
    vjp=lambda g, x, y, **kwargs: (unbroadcast(x, g * y), unbroadcast(y, g * x)),
    jvp=lambda g, h, x, y, **kwargs: g * y + h * x,
)
divide = TraceableOp(np.divide, "divide").register_op(
    vjp=lambda g, x, y, **kwargs: (
        unbroadcast(x, g / y),
        unbroadcast(y, -g * x / y**2),
    ),
    jvp=lambda g, h, x, y, **kwargs: (g * y - x * h) / y**2,
)
matmul = TraceableOp(np.matmul, "matmul").register_op(
    vjp=matmul_vjp,
    jvp=lambda g, h, x, y, **kwargs: np.matmul(g, y) + np.matmul(x, h),
)
log = TraceableOp(np.log, "log").register_op(
    vjp=lambda g, x, **kwargs: (g / x,),
    jvp=lambda g, x, **kwargs: g / x,
)
exp = TraceableOp(np.exp, "exp").register_op(
    vjp=lambda g, x, **kwargs: (g * np.exp(x),),
    jvp=lambda g, x, **kwargs: g * np.exp(x),
)
power = TraceableOp(np.power, "power").register_op(
    vjp=lambda g, x, y, **kwargs: (
        unbroadcast(x, g * y * np.power(x, np.where(y, y - 1, 1.0))),
        unbroadcast(y, g * np.log(np.where(x, x, 1.0)) * np.power(x, y)),
    ),
    jvp=lambda g, h, x, y, **kwargs: (h * np.log(x) + y * g / x) * np.power(x, y),
)
sqrt = TraceableOp(np.sqrt, "sqrt").register_op(
    vjp=lambda g, x, **kwargs: (g / (2 * np.sqrt(x)),),
    jvp=lambda g, x, **kwargs: g / (2 * np.sqrt(x)),
)
