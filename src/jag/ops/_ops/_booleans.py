from .._traceable_op import TraceableOp
import numpy as np


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
