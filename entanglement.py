import numpy as np
from numpy.matrixlib.defmatrix import matrix
from basis_states import basis_state, tensor_product
from scipy.linalg import logm
from graph_state import graph_states

I = np.array([[1, 0], [0, 1]], dtype=complex)

def build_basis_set(k):
    basis_set = [[]]

    i= 0
    while i < k:
        new_basis_set = []

        for bits in basis_set:
            new_basis_set.append(bits + [0])
            new_basis_set.append(bits + [1])

        basis_set = new_basis_set
        i += 1

    return basis_set

def reduced_density(psi, keep):
    n = int(round(np.log2(len(psi))))
    k = len(keep)
    r = n - k

    #reordering the psi based on the keep qubits for tracing out
    traced = [q for q in range(n) if q not in keep]
    order = list(keep) + traced

    psi = psi.reshape([2] * n).transpose(order).reshape(-1)

    reduced_identity = tensor_product([I]*k)

    # begin with 2^k x 2^k zero matrix
    summation = np.zeros((2**k, 2**k), dtype=complex)

    # implement the reduced density matrix formula
    for bits in build_basis_set(r):
        b = basis_state(bits)
        bra_b = np.conj(b).reshape(1, -1)
        v = np.kron(reduced_identity, bra_b) @ psi
        summation += np.outer(v, v.conj())

    return summation

psi = basis_state([1,0,0])
print(reduced_density(psi, [2,0]))


def entropy(rho):
    eigenvalues = np.linalg.eigvalsh(rho)

    ent = 0
    for eig_val in eigenvalues:
        if eig_val < -1e-10:
            raise ValueError("Density matrix has a negative eigenvalue")
        elif eig_val > 0:
            ent += -1 * eig_val * np.log2(eig_val)

    return ent

def purity(rho):
    return np.trace(rho @ rho).real

'''
print(entropy(np.array([[1, 0],[0, 0]], dtype=complex)))
print(purity(np.array([[1, 0],[0, 0]], dtype=complex)))

bell = (1/np.sqrt(2)) * (basis_state([0,0]) + basis_state([1,1]))
print(entropy(reduced_density(bell, [0])))
print(purity(reduced_density(bell, [0])))

print(purity(np.outer(bell, bell.conj())))
'''

bell = (1/np.sqrt(2)) * (basis_state([0,0]) + basis_state([1,1]))

def cut_profile(psi, n):
    entropy_list = []

    keep_set = [[0]]

    i = 1
    while i < n:
        new_keep_set = []

        for keeps in keep_set:
            new_keep_set.append(keeps)
            new_keep_set.append(keeps + [i])


        keep_set = new_keep_set
        i += 1


    for keep in keep_set:
        if len(keep) < n: # do not include the whole quantum system
            ent_values = [keep, float(entropy(reduced_density(psi, keep=keep)))]
            entropy_list.append(ent_values)

    return entropy_list

'''
ghz_4 = (1/np.sqrt(2)) * (basis_state([0,0,0,0]) + basis_state([1,1,1,1]))
p_4 = graph_states(4, edges=[(0,1),(1,2),(2,3)])

print(cut_profile(ghz_4, 4))
print(cut_profile(p_4, 4))
'''

def check_max_mixed(density_matrix, k):
    reduced_identity = tensor_product([I]*k)

    return np.allclose(reduced_identity/(2**k), density_matrix, atol=1e-10)


def is_k_uniform(psi, n, k):

    matrix_set = []

    for j in range(n):
        keep_set = [[j]]

        i = j+1
        while i < n:
            new_keep_set = []

            for keeps in keep_set:
                new_keep_set.append(keeps)
                new_keep_set.append(keeps + [i])

            keep_set = new_keep_set
            i += 1

        for keep in keep_set:
            if len(keep) == k:  # looking at only k-qubit reduced systems
                # reduced_matrices = [keep, reduced_density(psi, keep=keep)]
                matrix_set.append(reduced_density(psi, keep=keep))


    for dens_matrix in matrix_set:
        if not check_max_mixed(dens_matrix, k):
            print(f"It is not {k}-uniform state")
            return False

    print(f"It is {k}-uniform state")

    return True

def is_AME(psi, n):
    k = int(np.floor(n/2))

    if is_k_uniform(psi, n, k):
        print(f"It is AME({n},2)")
        return True
    else:
        return False


ghz_4 = (1/np.sqrt(2)) * (basis_state([0,0,0,0]) + basis_state([1,1,1,1]))
p_4 = graph_states(4, edges=[(0,1),(1,2),(2,3)])

graph_6 = graph_states(6, edges=[(1,2), (2,3), (3,4), (4,5), (1,0), (5,0), (0,3), (1,5), (2,4)])
is_k_uniform(graph_6, 6, 3)
is_AME(graph_6, 6)