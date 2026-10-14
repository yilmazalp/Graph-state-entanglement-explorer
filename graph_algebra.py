import numpy as np

def gf2_rank(m):

    pivot_list = []
    m = m.copy()

    for i in range(len(m)):

        positions = np.flatnonzero(m[i] == 1)

        # zero row: no pivot
        if positions.size == 0:
            continue

        # determine pivots
        pivot = positions[0]

        pivot_list.append(pivot)
        for j in range(i+1, len(m)):
            if m[j, pivot] == 1:
                m[j] = (m[j] + m[i]) % 2

    return len(pivot_list)

print(gf2_rank(np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])))

def graph_cut_entropy(adj, keep):

    # construct the partition except the keep one
    other_partition = [q for q in range(len(adj)) if q not in keep]

    # build the reduced adjacency matrix regarding the partition on qubits
    reduced_adj = adj[np.ix_(keep, other_partition)]

    return gf2_rank(reduced_adj)

def local_complement(adj, v):
    # get column indices with nonzero values at row v
    distinct_neighbours = np.flatnonzero(adj[v])

    # preserve the adjacency matrix
    adj = adj.copy()

    for i in range(len(distinct_neighbours)):
        for j in range(i+1, len(distinct_neighbours)):
            u = distinct_neighbours[i]
            w = distinct_neighbours[j]

            # connect the edge u-w, and removes existing edge
            adj[u, w] = 1 - adj[u, w]
            # the matrix should be symmetric
            adj[w, u] = adj[u, w]

    return adj
