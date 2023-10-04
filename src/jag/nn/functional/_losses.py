import numpy as np


def binary_cross_entropy(logits, labels):
    return np.mean(
        np.maximum(logits, 0) - logits * labels + np.log(1 + np.exp(-np.abs(logits))),
        axis=-1,
    )
