from jag import graph, ops

ops.TraceableOp._global_node_cls.append(graph.Node)
ops.TraceableOp._global_leaf_cls.append(graph.TracedArray)
ops.TraceableOp._to_traceable_fn.append(graph.to_traceable)
