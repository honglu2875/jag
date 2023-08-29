# JAG
**J**ust **A**nother deep learnin**G** framework.
or 

**J**ust **A**nother **G**raph-based deep learning framework.
or

almost like **JA**X except that we have **G**raphs as central objects lol.

# Huh?
I want to write a custom deep learning framework with **the computation graphs** being
the first class citizens. Everything else goes around that.

Why:
1. For my own learning
2. A computation graph is universal: it doesn't care about the language as long as you implement each node operation.
3. It should easily extend to generalized differential programming, such as semi-ring algebra like in [2307.03056](https://arxiv.org/pdf/2307.03056.pdf).

More concretely, I want to write a framework where the following can be done:
- Imagine you have a model `M`, to port it to a different programming language, all you need to do is to 
  1. Trace the computational graph `graph = trace(M, *args)`.
  2. JSON-ify the graph and save it `graph.save('graph.json')`. Each node in the dict represents a node in the graph.
  3. In your favorite language (C, javascript, or whatever), you write some codes that:
     1. reads json files;
     2. implements the 5-10 basic matrix ops that show up in the JSON nodes.
  4. Execute the graph like a champ.
- And imagine you can do this:
  1. You write a model `M` and trace it for a graph `graph = trace(M, *args)`.
  2. You want to port the implementations to the three popular frameworks: PyTorch, Tensorflow and JAX.
  3. All you do is to start with `graph` and do `graph.to_torch()`, `graph.to_tf()`, `graph.to_jax()` to generate their
  Python codes.
- Also imagine you can do this:
  1. Start with a model `M`.
  2. Obtain the computation graph of its gradient function `grad_graph = trace(grad(M), *args)`.
  3. Execute the graph with `graph_log = grad_graph.execute(*args, visualize=True)`.
  4. Use the graph log object to inspect the flow of gradients, identify the max flow path, save to images, etc.

## The content of the repo is currently WIP...