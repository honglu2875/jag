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
