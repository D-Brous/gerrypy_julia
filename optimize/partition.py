from gurobipy import *
import numpy as np

from gurobipy import quicksum
from gurobipy import Model
from gurobipy import GRB

def make_partition_IP(costs, connectivty_sets, population, pop_bounds):
    """
    Creates the Gurobi model to partition a region.
    Args:
        costs: (dict) {center: {tract: cost}} The keys of the outer dict
            are the only centers that are considered in the partition.
             I.e. len(lengths) = z.
        edge_dists: (dict) {center: {tract: hop_distance}} Same as lengths but
            value is the shortest path hop distance (# edges between i and j)
        G: (nx.Graph) The block adjacency graph
        population: (dict) {tract int id: population}
        pop_bounds: (dict) {center int id: {'lb': district population lower
            bound, 'ub': tract population upper bound}
        alpha: (int) The exponential cost term

    Returns: (tuple (Gurobi partition model, model variables as a dict)

    """
    partition_problem = Model('partition')
    districts = {}
    # Create the variables
    for center, tracts in costs.items():
        districts[center] = {}
        for tract in tracts:
            districts[center][tract] = partition_problem.addVar(
                vtype=GRB.BINARY,
                obj=costs[center][tract]
            )
    # Each tract belongs to exactly one district
    for j in population:
        partition_problem.addConstr(quicksum(districts[i][j] for i in districts
                                    if j in districts[i]) == 1,
                           name='exactlyOne')
    # Population tolerances
    for i in districts:
        partition_problem.addConstr(quicksum(districts[i][j] * population[j]
                                    for j in districts[i])
                           >= pop_bounds[i]['lb'],
                           name='x%s_minsize' % i)

        partition_problem.addConstr(quicksum(districts[i][j] * population[j]
                                    for j in districts[i])
                           <= pop_bounds[i]['ub'],
                           name='x%s_maxsize' % i)

    for center, sp_sets in connectivty_sets.items():
        for node, sp_set in sp_sets.items():
            if center == node:
                continue
            partition_problem.addConstr(districts[center][node] <=
                               quicksum(districts[center][nbor]
                                        for nbor in sp_set))

    #TODO Add constraint that each neighborhood is in no more than 2 districts

    # partition_problem.setObjective(quicksum(districts[i][j] * costs[i][j]
    #                                for i in costs for j in costs[i]),
    #                       GRB.MINIMIZE)
    partition_problem.Params.LogToConsole = 0
    partition_problem.Params.TimeLimit = len(population) / 200
    partition_problem.update()

    return partition_problem, districts


def edge_distance_connectivity_sets(edge_distance, G):
    connectivity_set = {}
    for center in edge_distance:
        connectivity_set[center] = {}
        for node in edge_distance[center]:
            constr_set = []
            dist = edge_distance[center][node]
            for nbor in G[node]:
                if edge_distance[center][nbor] < dist:
                    constr_set.append(nbor)
            connectivity_set[center][node] = constr_set
    return connectivity_set


def make_partition_IP_vectorized(cost_coeffs, spt_matrix, population, pop_bounds):
    """
    Creates the Gurobi model to partition a region.
    Args:
        cost_coeffs: (dict) {center: {tract: distance}} The keys of the outer dict
            are the only centers that are considered in the partition.
             I.e. len(lengths) = z.
        spt_matrix: (np.array) nonzero elements of row (i * B + j) are
            equal to the set S_ij
        G: (nx.Graph) The block adjacency graph
        population: (dict) {tract int id: population}
        pop_bounds: (dict) {center int id: {'lb': district population lower
            bound, 'ub': tract population upper bound}

    Returns: (tuple (Gurobi partition model, Gurobi MVar)

    """
    partition_problem = Model('partition')
    n_centers, n_blocks = cost_coeffs.shape

    # Assigment variables
    assignment = partition_problem.addMVar(shape=n_centers * n_blocks,
                                           vtype=GRB.BINARY,
                                           obj=cost_coeffs.flatten())

    population_matrix = np.zeros((n_centers, n_blocks * n_centers))
    for i in range(n_centers):
        population_matrix[i, i * n_blocks: (i + 1) * n_blocks] = population

    # Population balance
    partition_problem.addConstr(population_matrix @ assignment <= pop_bounds[:, 1])
    partition_problem.addConstr(population_matrix @ assignment >= pop_bounds[:, 0])

    # Strict covering
    partition_problem.addConstr(np.tile(np.eye(n_blocks), (1, n_centers)) @ assignment == 1)

    # Subtree of shortest path tree
    partition_problem.addConstrs(spt_matrix[n_blocks * c: n_blocks * (c + 1), :]
                                 @ assignment[n_blocks * c: n_blocks * (c + 1)]
                                 >= assignment[n_blocks * c: n_blocks * (c + 1)]
                                 for c in range(n_centers))

    partition_problem.Params.LogToConsole = 0
    partition_problem.Params.TimeLimit = len(population) / 20
    partition_problem.update()

    return partition_problem, assignment
