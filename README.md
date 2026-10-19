# Graph state entanglement explorer

In this work, we investigate how does changing a graph affect the entanglement. We implement functions returning state vector of a quantum state, graph states, corresponding stabilizers, and conditions of checking entanglement. Through this project, we are looking for an answer for questions of "is this graph state AME(n ,q), k-uniform state, maximally entangled".

## Graph states

Given a simple graph $G = (V, E)$, a graph state is defined as $$ \ket{G} = \prod_{{i,j} \in E} CZ_{ij} \ket{+}^{\otimes n}, where $n$ is the number of physical qubits. 

