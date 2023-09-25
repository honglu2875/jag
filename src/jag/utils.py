from typing import List, Set, Optional

from jag.graph import Node, Operand, TracedArray


def topsort(node: Node):
    """
    Topological sort the computation graph.
    """
    visited = set()
    stack = []

    def _topsort(node: Operand, stack: List[Node], visited: Set):
        if not isinstance(
            node, TracedArray
        ):  # Not a leaf -> carry out recursion on the node
            assert isinstance(node, Node)
            for operand in node.operands:
                if id(operand) not in visited:
                    visited.add(id(operand))
                    _topsort(operand, stack, visited)
        stack.append(node)

    _topsort(node, stack, visited)

    return stack[::-1]


def name_generator():
    """
    Generate a unique name for each node.
    """
    base_char = ["a", "b", "c", "d", "x", "y", "z", "u", "v", "w"]
    cnt = 0
    while True:
        var_name = base_char[cnt % len(base_char)]
        yield var_name if cnt < len(base_char) else var_name + "_" + str(
            cnt // len(base_char)
        )
        cnt += 1


def map_nodes(root: Node, incomplete_map: Optional[dict] = None) -> dict:
    """
    Create a mapping from the id of each node to their unique variable names.
    Args:
        root: the root node of the graph.
        incomplete_map: a possibly incomplete dictionary mapping the id of a node to its variable name.
    Returns:
        a complete mapping from the id of each node to their unique variable names.
    """
    name_gen = name_generator()
    incomplete_map = incomplete_map or {}
    _name_map = {}
    _used_names = set()

    def _map_nodes(node: Node):
        """
        Priority:
        0. if the node is already in _name_map, skip.
        1. if the node has a name, use it.
        2. if the node is in the incomplete_map, use the name in the incomplete_map.
        3. otherwise, generate a new name.

        Jump to the next whenever there is a name conflict.
        """
        if id(node) not in _name_map:
            if node.name is not None and node.name not in _used_names:
                _name_map[id(node)] = node.name
            elif (
                id(node) in incomplete_map
                and incomplete_map[id(node)] not in _used_names
            ):
                _name_map[id(node)] = incomplete_map[id(node)]
            else:
                name = next(name_gen)
                while name in _name_map:
                    name = next(name_gen)
                _name_map[id(node)] = name
            _used_names.add(_name_map[id(node)])

        if isinstance(node, Node):
            for operand in node.operands:
                _map_nodes(operand)

    _map_nodes(root)

    return _name_map
