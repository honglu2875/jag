from typing import List, Set

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
