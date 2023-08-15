# jag
Just Another deep learninG framework

# Yea?
Ok, the G in the name also means graph. I want to write a custom deep learning framework with the computational graph 
the first class citizen and everything else goes around that.

Why:
1. For my own learning
2. PyTorch is something that pretends to be your friend and does all the magic behind. It puts tensors under the
spotlight and does all the magics around that (allows eager, builds graphs on the fly, etc). JAX screams every time 
you step a wrong foot, but saves your time in the long run. I would say that the functions and functional programming 
is its design highlight. What if we make the computational graph the first class citizen in a deep learning framework? 
Not saying I know but I just want to find out.
3. I want to be able to port my model to any coding language. If I have a properly abstracted computational graph,
I would just need to implement them and reconstruct it iteratively in the new language, isn't it?

# If you see this message, the repo isn't working yet. I have some thoughts and let's see if I can pull it through.