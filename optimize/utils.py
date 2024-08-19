import numpy as np
import scipy.sparse as sp


def build_spt_matrix(G, edge_dists, centers, region):
    n_blocks = len(region)
    node_map = {n: nix for nix, n in enumerate(region)}
    rows = []
    columns = []
    for cix, center in enumerate(centers):
        rows.append(cix * n_blocks + node_map[center])
        columns.append(node_map[center])
        for bix, block in enumerate(region):
            shortest_path_hop_dist = edge_dists[center][block]
            for nbor in G[block]:
                if edge_dists[center][nbor] == shortest_path_hop_dist - 1:
                    rows.append(cix * n_blocks + bix)
                    columns.append(node_map[nbor])
    rows = np.array(rows)
    columns = np.array(columns)
    data = np.ones(columns.shape, dtype=int)

    spt_matrix = sp.coo_matrix((data, (rows, columns)))
    return sp.csr_matrix(spt_matrix, shape=(n_blocks * len(centers), n_blocks))