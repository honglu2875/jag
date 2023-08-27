# jag
Just Another deep learninG framework.

# Huh?
Ok, the G in the name also emphasizes the graphs. I want to write a custom deep learning framework with **the computational graphs** being
the first class citizens. Everything else goes around that.

Why:
1. For my own learning
2. A computation graph is universal: it doesn't care about the deep learning framework as long as you implement each node operation.
3. It should easily extend to generalized differential programming, such as semi-ring algebra like in [2307.03056](https://arxiv.org/pdf/2307.03056.pdf).

Note that:
- the consequence of 2 indicates a cross-framework interoperability of deep learning models:
  imagine you have a model `M`, all you need to do is to 
  1. Trace the computational graph `graph = trace(M)`.
  2. JSON-ify the graph and save it `graph.save('graph.json')`. Each node in the dict represents a node in the graph.
  3. In your favorite language (C, javascript, or whatever), you write some codes that:
     1. reads json files;
     2. implements the 5-10 basic matrix ops that show up in the JSON nodes.
  4. Execute the graph like a champ (e.g., running LLaMA 2 on your favorite language)

## The content of the repo is currently WIP...