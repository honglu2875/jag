from functools import singledispatch
from typing import Tuple

from jag.graph import Node, get_leaves, ConstantArray, TracedArray
from jag.ops import get_op_registration
from jag.type import ArrayLike
from jag.utils import topsort


def vjp(root: Node):
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
                    assert isinstance(node, Node), f"Cannot propagate value for node {node}.\n" \
                                                   f"Likely a TracedArray is not given a concrete value."
                    node_values[id(node)] = node.op(
                        *[node_values[id(child)] for child in node.operands], **node.kwargs
                    )

    def backprop(g: ArrayLike, node_values: dict) -> dict:
        backprop_values = {}
        for node in sorted_nodes:
            if not backprop_values:
                backprop_values[id(node)] = g

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
            backprop_values.update(
                {id(child): value for child, value in zip(node.operands, node_output)}
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
            return tuple(result[id(l)] for l in leaves)
        else:
            assert all(id_to_obj[id(l)].name for l in leaves)
            return {id_to_obj[id(l)].name: result[l] for l in leaves}

    def vjp_func(g: ArrayLike, *args, **kwargs):
        if not args:
            return ValueError("No leaf values are provided.")
        if len(args) == 1 and isinstance(args[0], dict):
            node_values = {id(l): args[0][l.name] if l.name else l.value for l in leaves}
        elif len(args) == 1 and isinstance(args[0], tuple):
            node_values = {id(l): val for l, val in zip(leaves, args[0])}
        else:
            node_values = {id(l): val for l, val in zip(leaves, args)}
        return compute_leaf_gradients(g, node_values, **kwargs)

    return vjp_func
