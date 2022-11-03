import random
import networkx as nx
import numpy as np

def random_order_expansion(county_G, seed):
    n = len(county_G)
    exploration_order = [seed]
    explored_set = np.zeros(n)
    explored_set[seed] = 1
    adjacency_matrix = nx.adjacency_matrix(county_G)
    while len(exploration_order) < len(county_G):
        adjacent_nodes = adjacency_matrix @ explored_set
        unexplored_nodes = np.clip(np.clip(adjacent_nodes, 0, 1) - explored_set, 0, 1)
        expansion = np.random.choice(n, p=unexplored_nodes / unexplored_nodes.sum())
        exploration_order.append(expansion)
        explored_set[expansion] = 1
    return exploration_order


def weighted_random_order_expansion(county_G, seed):
    n = len(county_G)
    exploration_order = [seed]
    unexplored_set = np.ones(n)
    unexplored_set[seed] = 0
    adjacency_matrix = nx.adjacency_matrix(county_G)
    while len(exploration_order) < len(county_G):
        n_adjacent_nodes = adjacency_matrix @ (1 - unexplored_set)
        unexplored_nodes = n_adjacent_nodes * unexplored_set
        expansion = np.random.choice(n, p=unexplored_nodes/unexplored_nodes.sum())
        exploration_order.append(expansion)
        unexplored_set[expansion] = 0
    return exploration_order


def convert_to_digraph(county_G, seed):
    def convert_edge(n1, n2, path_lengths):
        if path_lengths[n1] < path_lengths[n2]:
            return (n1, n2)
        elif path_lengths[n2] < path_lengths[n1]:
            return (n2, n1)

    path_lengths = nx.shortest_path_length(county_G, source=seed)
    county_digraph = nx.DiGraph()
    county_digraph.add_nodes_from(county_G.nodes)

    county_digraph.add_edges_from([convert_edge(n1, n2, path_lengths)
                                   for n1, n2 in county_G.edges
                                   if path_lengths[n1] != path_lengths[n2]])
    return county_digraph


def random_topological_ordering(county_G, seed):
    bounary_digraph = convert_to_digraph(county_G, seed)
    in_degree = np.array(bounary_digraph.in_degree)[:, 1]
    ordering = []
    zero_in_degree = [seed]
    while (zero_in_degree):
        random.shuffle(zero_in_degree)
        next_node = zero_in_degree.pop()
        ordering.append(next_node)
        for n in county_G[next_node]:
            in_degree[n] -= 1
            if in_degree[n] == 0:
                zero_in_degree.append(n)
    return ordering

def bfs(county_G, seed):
    edge_dists = nx.shortest_path_length(county_G, source=seed)
    return np.array([hop_distance for _, hop_distance
                     in sorted(list(edge_dists.items()), key=lambda x: x[0])])


def random_translation(county_G, seed):
    path_lengths = nx.shortest_path_length(county_G, source=seed)
    distance_array = np.zeros(len(county_G))
    distance_array[list(path_lengths.keys())] = list(path_lengths.values())
    return distance_array ** (np.random.rand(len(distance_array)) + .5)