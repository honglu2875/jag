from typing import Any

from jag.codegen._torch_special_ops import (_args_kwargs_post_process,
                                            _in_place_ops,
                                            _kwargs_value_post_process,
                                            _special_ops)
from jag.type import GraphNode, _assure_kwargs, _assure_node_with_op
from jag.utils import map_nodes, topsort

_torch_op_maps = {
    "repeat": "torch.repeat_interleave",
    "transpose": "torch.permute",
}
_simple_ops = {
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "divide": "/",
    "power": "**",
    "negative": "-",
    "matmul": "@",
    "greater": ">",
    "less": "<",
}
_kwargs_mapping = {"reshape": {"newshape": "shape"}, "transpose": {"axes": "dims"}}


def _get_repr(value: Any, _unexpanded_nodes, _full_name_map) -> Any:
    """
    Get the string representation of a node or a traced array.
    """
    if isinstance(value, GraphNode):
        return _unexpanded_nodes.get(id(value), _full_name_map[id(value)])
    else:
        return value


def _get_key_of_kwarg(op_name: str, key: str) -> str:
    if key in _kwargs_mapping.get(op_name, {}):
        return _kwargs_mapping[op_name][key]
    return key


def _get_value_of_kwarg(
    op_name: str, key: str, value: Any, _unexpanded_nodes, _full_name_map
) -> Any:
    # Expand traceable nodes
    value = _get_repr(value, _unexpanded_nodes, _full_name_map)
    # Apply post-processing
    if (
        op_name in _kwargs_value_post_process
        and key in _kwargs_value_post_process[op_name]
    ):
        return _kwargs_value_post_process[op_name][key](value)
    return value


def _get_op_name(node: GraphNode):
    _assure_node_with_op(node)
    if node.op.name in _torch_op_maps:
        return _torch_op_maps[node.op.name]
    else:
        return f"torch.{node.op.name}"


def _get_args_kwargs(
    node: GraphNode, operands_repr: list, _unexpanded_nodes: dict, _full_name_map: dict
) -> tuple:
    _assure_node_with_op(node)
    _assure_kwargs(node)

    if node.op.name in _args_kwargs_post_process:
        _args, _kwargs = _args_kwargs_post_process[node.op.name](
            operands_repr, node.kwargs
        )
    else:
        _args, _kwargs = operands_repr, node.kwargs
    _args = tuple(_get_repr(arg, _unexpanded_nodes, _full_name_map) for arg in _args)
    _kwargs = {
        _get_key_of_kwarg(node.op.name, k): _get_value_of_kwarg(
            node.op.name, k, v, _unexpanded_nodes, _full_name_map
        )
        for k, v in _kwargs.items()
    }
    return _args, _kwargs


def _get_signature_arg_list(arg_list: list, leaf_names_list: list) -> list:
    leaf_names = set(leaf_names_list)
    arg_list = [arg for arg in arg_list if arg in leaf_names]
    arg_set = set(arg_list)
    return arg_list + [leaf for leaf in leaf_names_list if leaf not in arg_set]


def torchgen(
    root: GraphNode,
    name_map: dict = None,
    func_name: str = "func",
    max_unexpanded_len: int = 25,
    arg_list: list | None = None,
) -> str:
    """
    Generate the PyTorch code from the computational graph with `node` as the root.
    Args:
        root: the root node
        name_map: a dictionary mapping the id of a node to its variable name
        func_name: the name of the function
        max_unexpanded_len: the maximum length of a node that is not expanded
        arg_list: list of names of positional arguments. Unmatched arg names will be ignored,
            and unmatched leaves will be appended at the end of arg list.
    Returns:
        a string of PyTorch code.
    """
    sorted_nodes = topsort(root, include_kwargs=True)[
        ::-1
    ]  # top sorted from the leaves to the root
    _full_name_map = map_nodes(
        root, name_map, include_kwargs=True
    )  # full mapping from id to name
    _unexpanded_nodes = (
        {}
    )  # some simple ops are not expanded immediately in new lines, such as (x + y * z), etc.
    arg_list = arg_list or []

    tabs = " " * 4

    # The order of the arguments respect the following rules:
    # 1. The `arg_list` is given the first priority with unmatched names ignored.
    # 2. The rest respects the order out of `root.leaves` method which uses the topological order.

    leaf_names_list = list(
        _full_name_map[id(leaf)] for leaf in root.leaves(include_constant=False)
    )
    arg_list = _get_signature_arg_list(arg_list, leaf_names_list)

    signature = f"def {func_name}({', '.join(arg_list)}):\n"
    output_codes = []

    for node in sorted_nodes:
        assert isinstance(node, GraphNode)
        if node.is_constant():
            # When encountering a constant, we either
            # 1. expand the assignment in a new line, or
            # 2. keep the expression if it is simple enough.
            if node.shape == () or node.shape == (1,):
                _unexpanded_nodes[
                    id(node)
                ] = f"torch.tensor({str(node.value.tolist())}, dtype=torch.{node.dtype})"
            else:
                output_codes.append(
                    f"{_full_name_map[id(node)]} = torch.tensor({node.value.tolist()})"
                )
        elif node.is_leaf():
            pass
        else:
            # When encountering a non-leaf node, we first check each operand to see
            # whether we refer to them as an unexpanded expression or its var name.
            operands_repr = [
                _get_repr(child, _unexpanded_nodes, _full_name_map)
                for child in node.operands
            ]
            if node.op.name in _special_ops:
                joined_repr = _special_ops[node.op.name](node, operands_repr)
                if node.op.name in _in_place_ops:
                    # If in-place,
                    # 1. expand the first operand immediately in a new line,
                    # 2. the line of code is immediately appended
                    # 3. the node is marked with its first operand's name.
                    target_var_name = _full_name_map[id(node.operands[0])]
                    if id(node.operands[0]) in _unexpanded_nodes:
                        output_codes.append(f"{target_var_name} = {operands_repr[0]}")
                    output_codes.append(
                        _special_ops[node.op.name](
                            node, [target_var_name] + operands_repr[1:]
                        )
                    )
                    _unexpanded_nodes[id(node)] = target_var_name
                    continue
            elif node.op.name in _simple_ops and not node.kwargs:  # If the op is simple
                if len(node.operands) == 1:
                    joined_repr = f"{_simple_ops[node.op.name]}{operands_repr[0]}"
                else:
                    joined_repr = f" {_simple_ops[node.op.name]} ".join(operands_repr)
            else:  # The general case
                op_name = _get_op_name(node)
                _args, _kwargs = _get_args_kwargs(
                    node, operands_repr, _unexpanded_nodes, _full_name_map
                )
                _kwargs_repr = [f"{k}={v}" for k, v in _kwargs.items()]

                joined_repr = (
                    f"{op_name}(" f"{', '.join(_args)}, " f"{', '.join(_kwargs_repr)})"
                )
            # We set a length threshold to determine whether an expression should be
            # assigned to a new variable in a new line.
            if len(joined_repr) < max_unexpanded_len and id(node) != id(root):
                _unexpanded_nodes[id(node)] = (
                    f"({joined_repr})" if node.op.name in _simple_ops else joined_repr
                )
            else:
                output_codes.append(f"{_full_name_map[id(node)]} = {joined_repr}")

    return_line = f"{tabs}return {_get_repr(root, _unexpanded_nodes, _full_name_map)}\n"

    return signature + tabs + f"\n{tabs}".join(output_codes) + "\n" + return_line
