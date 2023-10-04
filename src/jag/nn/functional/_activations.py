import numpy as np

"""
Activation functions implemented by numpy functions. Guaranteed to be traceable.
"""


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1, keepdims=False):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=keepdims)


def log_softmax(x, axis=-1, keepdims=False):
    return np.log(softmax(x, axis=axis, keepdims=keepdims))


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)
