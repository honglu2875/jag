# TODO: implement a tracer for torch function

try:
    import torch
except ImportError:
    raise RuntimeError(
        f"The {__file__.replace('.py', '')} package is applied to PyTorch but "
        f"PyTorch is not installed."
    )
