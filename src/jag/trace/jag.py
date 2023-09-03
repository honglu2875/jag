from typing import Callable, Optional, Sequence

import numpy as np

from jag.graph import ConstantArray, TracedArray, to_traceable


def trace(func: Callable, *args, constants: Optional[Sequence] = None, **kwargs):
    """
    Utility function to trace the computation graph of a function.
    """
    constants = constants or []
    args = [
        TracedArray(shape=np.array(arg).shape, dtype=np.array(arg).dtype)
        if i not in constants
        else ConstantArray(np.array(arg))
        for i, arg in enumerate(args)
    ]
    return func(*args, **kwargs)
