from jag import ops
from jag import graph

ops.TraceableOp._global_node_cls.append(graph.Node)
ops.TraceableOp._to_traceable_fn.append(graph.to_traceable)
