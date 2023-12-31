# Copyright (c) 2023 Honglu Fan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from typing import List, Optional, Set

from jag.type import GraphNode, _assure_kwargs


def topsort(node: GraphNode, include_kwargs: bool = False):
    """
    Topological sort the computation graph.
    Args:
        node: the root node of the graph.
        include_kwargs: whether to include the nodes in kwargs at each node as its children.
    Returns:
        a list of nodes sorted from the leaves to the root.
    """
    assert isinstance(node, GraphNode), f"{node} is not a GraphNode."

    visited = set()
    stack = []

    def _topsort(node: GraphNode, stack: List[GraphNode], visited: Set):
        if not node.is_leaf():
            for operand in node.operands:
                if id(operand) not in visited:
                    visited.add(id(operand))
                    _topsort(operand, stack, visited)
            if include_kwargs and not node.is_leaf():
                _assure_kwargs(node)
                for operand in node.kwargs.values():
                    if isinstance(operand, GraphNode) and id(operand) not in visited:
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


def map_nodes(
    root: GraphNode, incomplete_map: Optional[dict] = None, include_kwargs: bool = True
) -> dict:
    """
    Create a mapping from the id of each node to their unique variable names.
    Args:
        root: the root node of the graph.
        incomplete_map: a possibly incomplete dictionary mapping the id of a node to its variable name.
        include_kwargs: whether to include the nodes in kwargs at each node as its children.
    Returns:
        a complete mapping from the id of each node to their unique variable names.
    """
    name_gen = name_generator()
    incomplete_map = incomplete_map or {}
    _named_nodes = [
        node for node in topsort(root, include_kwargs) if node.name is not None
    ]
    _name_map = {}
    _used_names = set()
    for node in _named_nodes:
        if node.name not in _used_names:
            _name_map[id(node)] = node.name
            _used_names.add(node.name)

    def _map_nodes(node: GraphNode):
        """
        Priority:
        0. if the node is already in _name_map, skip.
        1. if the node has a name, use it.
        2. if the node is in the incomplete_map, use the name in the incomplete_map.
        3. otherwise, generate a new name.

        Jump to the next whenever there is a name conflict.
        """
        if id(node) not in _name_map:
            if (
                id(node) in incomplete_map
                and incomplete_map[id(node)] not in _used_names
            ):
                _name_map[id(node)] = incomplete_map[id(node)]
            else:
                name = next(name_gen)
                while name in _used_names:
                    name = next(name_gen)
                _name_map[id(node)] = name
            _used_names.add(_name_map[id(node)])

        for operand in node.operands:
            _map_nodes(operand)

        if include_kwargs and not node.is_leaf():
            _assure_kwargs(node)
            for operand in node.kwargs.values():
                if isinstance(operand, GraphNode):
                    _map_nodes(operand)

    _map_nodes(root)

    return _name_map
