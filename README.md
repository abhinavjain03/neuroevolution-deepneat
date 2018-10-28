# NeuroEvolution using DeepNEAT
The idea is to use Genetic algorithms to automatically learn Neural Network topologies as opposed to learning using Grid Search. We start with the NEAT algorithm which is depicted.
![Not Loading](/imgs/1.PNG "Test")

We extend the NEAT algorithm to be able to learn arbitrary Deep Neural Topologies using Genetic Algorithm which we call DeepNEAT. The following modifications are needed for this to work.
![Not Loading](/imgs/2.PNG "Test")

A relatively simple easier task of character recognition is selected for performance evaluation. We chose Devanagari Script Character Recognition whose SOTA is shown.
https://ieeexplore.ieee.org/document/7400041
![Not Loading](/imgs/3.PNG "Test")

The results we got were very promising. We got a network topology which was less complex than the SOTA mentioned in the paper but performed better on the splits.
![Not Loading](/imgs/4.PNG "Test")
