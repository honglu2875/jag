from jag.graph import Node, TracedArray, ConstantArray
from jag.utils import topsort, map_nodes
from typing import Any


_torch_op_maps = {}
_simple_ops = {
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "divide": "/",
    "power": "**",
    "negative": "-",
    "matmul": "@",
}
_kwargs_mapping = {
    "reshape": {"newshape": "shape"},
}
_kwargs_post_process = {}
_special_ops = {}
_max_unexpanded_len = 25


def register_post_process(name: str, key: str):
    def decorator(func):
        _kwargs_post_process.setdefault(name, {}).update({key: func})
        return func
    return decorator


def register_special_op(name: str):
    def decorator(func):
        _special_ops[name] = func
        return func
    return decorator


def _map_kwarg(op_name: str, key: str, value: Any) -> Any:
    if op_name in _kwargs_post_process and key in _kwargs_post_process[op_name]:
        return _kwargs_post_process[op_name][key](value)
    return value


@register_special_op("at")
def _process_at(node: Node, operands_repr) -> str:
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


@register_post_process("reshape", "shape")
def _post_reshape(value):
    if isinstance(value, int):
        return value,
    return value


def torchgen(root: Node, name_map: dict = None, func_name: str = "func") -> str:
    """
    Generate the PyTorch code from the computational graph with `node` as the root.
    Args:
        root: the root node
        name_map: a dictionary mapping the id of a node to its variable name
        func_name: the name of the function
    Returns:
        a string of PyTorch code.
    """
    sorted_nodes = topsort(root)[::-1]  # top sorted from the leaves to the root
    _full_name_map = map_nodes(root, name_map)  # full mapping from id to name
    _unexpanded_nodes = (
        {}
    )  # some simple nodes are not expanded, such as ConstantArray(2.0), (x + y), etc.

    tabs = " " * 4
    leaves = [
        node
        for node in sorted_nodes
        if isinstance(node, TracedArray) and not isinstance(node, ConstantArray)
    ]
    signature = f"def {func_name}({', '.join([_full_name_map[id(leaf)] for leaf in leaves])}):\n"
    return_line = f"{tabs}return {_full_name_map[id(root)]}\n"
    output_codes = []

    for node in sorted_nodes:
        if isinstance(node, ConstantArray):
            if node.shape == () or node.shape == (1,):
                _unexpanded_nodes[
                    id(node)
                ] = f"torch.tensor({str(node.value)}, dtype=torch.{node.dtype})"
            else:
                output_codes.append(
                    f"{_full_name_map[id(node)]} = torch.tensor({node.value})"
                )
        elif isinstance(node, Node):
            operands_repr = [
                _unexpanded_nodes[id(child)]
                if id(child) in _unexpanded_nodes
                else _full_name_map[id(child)]
                for child in node.operands
            ]
            if node.op.name in _special_ops:
                joined_repr = _special_ops[node.op.name](node, operands_repr)
            elif node.op.name in _simple_ops and not node.kwargs:
                if len(node.operands) == 1:
                    joined_repr = f"{_simple_ops[node.op.name]}{operands_repr[0]}"
                else:
                    joined_repr = f" {_simple_ops[node.op.name]} ".join(operands_repr)
            else:
                if node.op.name in _torch_op_maps:
                    op_name = _torch_op_maps[node.op.name]
                else:
                    op_name = f"torch.{node.op.name}"

                k_map = _kwargs_mapping.get(node.op.name, {})
                kwargs = [
                    f"{k_map[k]}={_map_kwarg(node.op.name, k_map[k], v)}"
                    if k in k_map
                    else f"{k}={_map_kwarg(node.op.name, k, v)}"
                    for k, v in node.kwargs.items()
                ]

                joined_repr = (
                    f"{op_name}("
                    f"{', '.join(operands_repr)}, "
                    f"{', '.join(kwargs)})"
                )

            if len(joined_repr) < _max_unexpanded_len and id(node) != id(root):
                _unexpanded_nodes[id(node)] = (
                    f"({joined_repr})" if node.op.name in _simple_ops else joined_repr
                )
            else:
                output_codes.append(f"{_full_name_map[id(node)]} = {joined_repr}")

    return signature + tabs + f"\n{tabs}".join(output_codes) + "\n" + return_line
