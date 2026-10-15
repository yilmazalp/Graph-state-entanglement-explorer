import numpy as np
import basis_states


def validate_edges(n, edges):

    for edge in edges:
        if not ((0 <= edge[0] and 0 <= edge[1]) and (edge[0] < n and edge[1] < n)):
            return ValueError("Invalid graph!")

        if edge[0] == edge[1]:
            return ValueError("The graph has no self-loop")

        reversed_edge = edge[::-1]

        if reversed_edge in edges:
            return ValueError("The graph cannot have duplicate edges")

    return None

def build_adjacency(n, edges):
    adj_matrix = np.zeros((n, n))

    for edge in edges:
        adj_matrix[edge[0], edge[1]] = 1
        adj_matrix[edge[1], edge[0]] = 1

    if validate_edges(n, edges) is None:
        return adj_matrix

    return validate_edges(n, edges)


def graph_states(n, edges):
    initial_state = basis_states.Statevector.basis_state([0]*n)
    state = basis_states.Statevector(initial_state)

    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    for q in range(n):
        state.apply_1q(H, q)

    for edge in edges:
        state.apply_cz(edge[0], edge[1])

    return state.psi

def generate_stabilizers(n, edges):
    adj_matrix = build_adjacency(n, edges)

    I = np.array([[1, 0], [0, 1]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)


    stabilizers = []

    for i in range(n):
        initial_op = [I] * n
        initial_op[i] = X

        for j in range(n):
            if adj_matrix[i][j] == 1:
                initial_op[j] = Z

        k_i = basis_states.Statevector.tensor_product(initial_op)
        stabilizers.append(k_i)

    return stabilizers


def graph_syndrome(psi, generators):

    syndromes = []

    for generator in generators:
        syndromes.append(int(round((np.conj(psi) @ generator @ psi).real.item(), 10)))

    return syndromes


if __name__ == "__main__":
    edges = [(0,1), (0,2), (2, 1)]
    adj_matrix = build_adjacency(3, edges)
    psi = graph_states(3, edges)
    generators = generate_stabilizers(3, edges)
    syndromes = graph_syndrome(psi, generators)


    print("Adjacency matrix: \n", adj_matrix)
    print("Graph state: \n", psi)
    print("Generators of the graph state: \n", generators)
    print("Syndrome of the graph stabilizers: \n", syndromes)
