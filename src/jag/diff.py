import functools
from typing import Tuple

import numpy as np

from jag.graph import ConstantArray, Node, TracedArray, get_leaves
from jag.ops import get_op_registration
from jag.type import ArrayLike
from jag.utils import topsort


def vjp(root: Node):
    """
    Compute the vjp function of the computational graph.
    Args:
        root: the root node
    Returns:
        a traceable vjp function corresponding to the computational graph.
        The vjp function specification is the following.
        Positional args:
            1. g: Array, primal: tuple, where the tuple order is the same as the flattened order of the
                leaves (when calling jag.graph.get_leaves).
            2. g: Array, primal: dict, where the dicts map the names of the leaves to the corresponding
                values.
            3. g: Array, *primal where the tuple order is the same as the flattened order of the leaves.
        Keyword args:
            as_tuple: (Default: True) whether the vjp function returns the pull-back vectors as a tuple
        Returns:
            1. if as_tuple=True, the return is a tuple of pull-back vectors with the order the same as
                flattened leaves
            2. if as_tuple=False, the return is a dict mapping the names of the leaves to the pull-back
                vector
    """
    sorted_nodes = topsort(
        root
    )  # top sorted from the output-node to leaves (variables to be diff'ed)
    leaves = get_leaves(root)  # list of leaves (variables to be diff'ed)
    id_to_obj = {id(node): node for node in sorted_nodes}

    def propagate_values(node_values: dict):
        """
        Propagate leaf values in-place
        """
        for node in sorted_nodes[::-1]:
            if id(node) not in node_values:
                if isinstance(node, ConstantArray):
                    node_values[id(node)] = node.value
                else:
                    assert isinstance(node, Node), (
                        f"Cannot propagate value for node {node}.\n"
                        f"Likely a TracedArray is not given a concrete value."
                    )
                    node_values[id(node)] = node.op(
                        *[node_values[id(child)] for child in node.operands],
                        **node.kwargs,
                    )

    def backprop(g: ArrayLike, node_values: dict) -> dict:
        backprop_values = {id(l): np.zeros(l.shape, dtype=l.dtype) for l in leaves}
        backprop_values[id(sorted_nodes[0])] = g  # Initial vector to be pulled back

        for node in sorted_nodes:
            assert (
                id(node) in backprop_values
            ), f"Backpropagation error. Most likely the graph is broken.\nGraph:{root}"
            if isinstance(node, TracedArray):
                continue
            node_output: Tuple = get_op_registration(node.op.name)["vjp"](
                backprop_values[id(node)],
                *[node_values[id(child)] for child in node.operands],
                **node.kwargs,
            )

            for child, value in zip(node.operands, node_output):
                # In case of custom ring structure, one can change "+" into the corresponding op.
                backprop_values[id(child)] = value + backprop_values.setdefault(
                    id(child), np.zeros(child.shape, dtype=child.dtype)
                )

        return backprop_values

    def compute_leaf_gradients(
        g: ArrayLike, node_values: dict, **kwargs
    ) -> tuple | dict:
        # Get keyword arguments
        as_tuple = kwargs.get("as_tuple", True)

        propagate_values(node_values)
        result = backprop(g, node_values)
        if as_tuple:
            return tuple(result.get(id(l), ...) for l in leaves)
        else:
            assert all(id_to_obj[id(l)].name for l in leaves)
            return {id_to_obj[id(l)].name: result[l] for l in leaves}

    def vjp_func(g: ArrayLike, *args, **kwargs):
        if not args:
            raise ValueError("No leaf values are provided.")
        if len(args) == 1 and isinstance(args[0], dict):
            node_values = {
                id(l): args[0][l.name] if l.name else l.value for l in leaves
            }
        elif len(args) == 1 and isinstance(args[0], tuple):
            node_values = {id(l): val for l, val in zip(leaves, args[0])}
        else:
            node_values = {id(l): val for l, val in zip(leaves, args)}
        return compute_leaf_gradients(g, node_values, **kwargs)

    return vjp_func


def grad(root: Node):
    assert (
        np.prod(root.shape) == 1
    ), "The output must be a singleton in order for gradient to make sense."
    out_grad = np.ones(root.shape, dtype=root.dtype)
    vjp_func = vjp(root)

    def grad_func(*args, **kwargs):
        return vjp_func(out_grad, *args, **kwargs)

    return grad_func


def jvp(root: Node, include_value=True):
    """
    Compute the jvp function of the computational graph.
    Args:
        root: the root node
        include_value: whether the return of the jvp function includes the evaluation result
            if False, the result of the jvp function will just be the push-forward vector
            if True, the result of the jvp function will be the tuple (evaluation value, push-forward vector)
    Returns:
        a traceable jvp function corresponding to the computational graph.
        The jvp function takes:
            1. primal: tuple, tangent: tuple, where the tuple order is the same as the flattened order of the
                leaves (when calling jag.graph.get_leaves).
            2. primal: dict, tangent: dict, where the dicts map the names of the leaves to the corresponding
                values.
            3. *primal, *tangent, where the tuple order is the same as the flattened order of the leaves.
    """
    sorted_nodes = topsort(
        root
    )  # top sorted from the output-node to leaves (variables to be diff'ed)
    leaves = get_leaves(root)  # list of leaves (variables to be diff'ed)

    def propagate_vector_values(node_values: dict, tangent_values: dict):
        for node in sorted_nodes[::-1]:
            if isinstance(node, ConstantArray):
                node_values[id(node)] = node.value
                tangent_values[id(node)] = np.zeros(shape=node.shape, dtype=node.dtype)
            elif isinstance(node, TracedArray):
                assert id(node) in node_values, f"The leaf {id(node)} is not provided a primal value."
                assert id(node) in tangent_values, f"The leaf {id(node)} is not provided a tangent vector."
            else:
                node_values[id(node)] = node.op(*[node_values[id(child)] for child in node.operands], **node.kwargs)
                tangent_values[id(node)] = get_op_registration(node.op.name)["jvp"](*[tangent_values[id(child)] for child in node.operands],
                                                                                    *[node_values[id(child)] for child in node.operands],
                                                                                    **node.kwargs)

    def jvp_func(*args, **kwargs):
        if not args:
            raise ValueError("No leaf values are provided.")
        elif len(args) == 2:
            if isinstance(args[0], tuple) and isinstance(args[1], tuple):
                node_values = {id(l): val for l, val in zip(leaves, args[0])}
                tangent_values = {id(l): val for l, val in zip(leaves, args[1])}
            elif isinstance(args[0], dict) and isinstance(args[1], dict):
                node_values = {
                    id(l): args[0][l.name] if l.name else l.value for l in leaves
                }
                tangent_values = {
                    id(l): args[1][l.name] if l.name else l.value for l in leaves
                }
            else:
                raise ValueError("When there are two non-array positional arguments, the function accepts\n"
                                 "1. tangent: tuple, primal: tuple, where the order of tuple corresponds to "
                                 "the flattened order of leaves (when calling jag.graph.get_leaves).\n"
                                 "2. tangent: dict, primal: dict, where the dicts map the names of the leaves "
                                 "to the corresponding values.")
        else:
            node_values = {id(l): val for l, val in zip(leaves, args[: len(args) // 2])}
            tangent_values = {id(l): val for l, val in zip(leaves, args[len(args) // 2 :])}

        propagate_vector_values(node_values, tangent_values)
        if include_value:
            return node_values[id(root)], tangent_values[id(root)]
        else:
            return tangent_values[id(root)]

    return jvp_func

