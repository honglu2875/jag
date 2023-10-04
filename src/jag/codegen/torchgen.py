from typing import Any

from jag.type import GraphNode, _assure_kwargs, _assure_node_with_op
from jag.utils import map_nodes, topsort

_torch_op_maps = {}
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
_kwargs_mapping = {
    "reshape": {"newshape": "shape"},
}
_kwargs_value_post_process = {}
_args_kwargs_post_process = {}
_special_ops = {}


def register_value_process(name: str, key: str):
    def decorator(func):
        _kwargs_value_post_process.setdefault(name, {}).update({key: func})
        return func

    return decorator


def register_args_kwargs_post_process(name: str):
    def decorator(func):
        _args_kwargs_post_process[name] = func
        return func

    return decorator


def register_special_op(name: str):
    def decorator(func):
        _special_ops[name] = func
        return func

    return decorator


def _get_repr(value: Any, _unexpanded_nodes, _full_name_map) -> Any:
    """
    Get the string representation of a node or a traced array.
    """
    if isinstance(value, GraphNode):
        return (
            _unexpanded_nodes[id(value)]
            if id(value) in _unexpanded_nodes
            else _full_name_map[id(value)]
        )
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


@register_special_op("at")
def _process_at(node: GraphNode, operands_repr) -> str:
    assert len(node.operands) == 1, f"Expected 1 operand, got {len(node.operands)}."
    idx = node.kwargs["idx"]
    idx_repr = []
    for i in idx:
        if isinstance(i, int):
            idx_repr.append(str(i))
        elif isinstance(i, slice):
            if i.step is None or i.step == 1:
                idx_repr.append(f"{i.start or ''}:{i.stop or ''}")
            else:
                idx_repr.append(f"{i.start or ''}:{i.stop or ''}:{i.step}")
        else:
            raise ValueError(f"Unsupported index type: {type(i)}.")

    return f"{operands_repr[0]}[{', '.join(idx_repr)}]"


@register_value_process("reshape", "newshape")
def _reshape_newshape(value):
    if isinstance(value, int):
        return (value,)
    return value


@register_args_kwargs_post_process("where")
def _where_args_kwargs(args, kwargs):
    assert len(args) in (2, 3), f"Expected 2 or 3 arguments, got {len(args)}."
    if len(args) == 2:
        print(args, kwargs)
        return (kwargs["condition"], *args), {
            k: v for k, v in kwargs.items() if k != "condition"
        }
    else:
        return (args[2], args[0], args[1]), kwargs


def torchgen(
    root: GraphNode,
    name_map: dict = None,
    func_name: str = "func",
    max_unexpanded_len: int = 25,
) -> str:
    """
    Generate the PyTorch code from the computational graph with `node` as the root.
    Args:
        root: the root node
        name_map: a dictionary mapping the id of a node to its variable name
        func_name: the name of the function
        max_unexpanded_len: the maximum length of a node that is not expanded
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
    )  # some simple nodes are not expanded, such as ConstantArray(2.0), (x + y), etc.

    tabs = " " * 4
    leaves = root.leaves(include_constant=False)
    signature = f"def {func_name}({', '.join([_full_name_map[id(leaf)] for leaf in leaves])}):\n"
    return_line = f"{tabs}return {_full_name_map[id(root)]}\n"
    output_codes = []

    for node in sorted_nodes:
        assert isinstance(node, GraphNode)
        if node.is_constant():
            if node.shape == () or node.shape == (1,):
                _unexpanded_nodes[
                    id(node)
                ] = f"torch.tensor({str(node.value)}, dtype=torch.{node.dtype})"
            else:
                output_codes.append(
                    f"{_full_name_map[id(node)]} = torch.tensor({node.value})"
                )
        elif node.is_leaf():
            pass
        else:
            operands_repr = [
                _get_repr(child, _unexpanded_nodes, _full_name_map)
                for child in node.operands
            ]
            if node.op.name in _special_ops:
                joined_repr = _special_ops[node.op.name](node, operands_repr)
            elif node.op.name in _simple_ops and not node.kwargs:
                if len(node.operands) == 1:
                    joined_repr = f"{_simple_ops[node.op.name]}{operands_repr[0]}"
                else:
                    joined_repr = f" {_simple_ops[node.op.name]} ".join(operands_repr)
            else:  # The generic case
                op_name = _get_op_name(node)
                _args, _kwargs = _get_args_kwargs(
                    node, operands_repr, _unexpanded_nodes, _full_name_map
                )
                _kwargs_repr = [f"{k}={v}" for k, v in _kwargs.items()]

                joined_repr = (
                    f"{op_name}(" f"{', '.join(_args)}, " f"{', '.join(_kwargs_repr)})"
                )

            if len(joined_repr) < max_unexpanded_len and id(node) != id(root):
                _unexpanded_nodes[id(node)] = (
                    f"({joined_repr})" if node.op.name in _simple_ops else joined_repr
                )
            else:
                output_codes.append(f"{_full_name_map[id(node)]} = {joined_repr}")

    return signature + tabs + f"\n{tabs}".join(output_codes) + "\n" + return_line
