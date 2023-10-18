from gurobipy import *
import numpy as np

from gurobipy import quicksum
from gurobipy import Model
from gurobipy import GRB

def make_partition_IP_MajMinReq(costs, connectivity_sets, population, pop_bounds, counties, split_lim, nonwhite_per_block, req_mm):
    """
    Creates the Gurobi model to partition a region.
    Includes constraints that each county is in no more than 2 districts,
        and that the total # of split counties is less than a set limit
    Includes a requirement that the each district contains req_mm MajMin dists
    For now, remove the MajMin bonus from the objective
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
        counties: (dict) {tract int id: county} County FIPS for each tract
        split_lim: (int) Maximum number of counties that are allowed to be split in each district
        nonwhite_per_block: (np.array) The NUMBER of nonwhite people in each block
        req_mm: (np.array) For each center, the number of mm dists that must be created from it.
            Equal to 0 if center contains more than 1 district

    Returns: (tuple (Gurobi partition model, model variables as a dict)

    """

    partition_problem = Model('partition')
    districts = {}
    BinCounts={}
    #make list and dict of counties in this region
    countlist = []
    for j in counties:
        if counties[j] not in countlist:
            countlist.append(counties[j]) #TODO make this faster
    countdict = {v:k for v, k in enumerate(countlist)}
    # Create the variables
    for center, tracts in costs.items():
        districts[center] = {}
        for tract in tracts:
            districts[center][tract] = partition_problem.addVar(
                vtype=GRB.BINARY,
                obj=costs[center][tract]
            )

    #Create binary variables to track if county k is in center i
    for i in districts:
        BinCounts[i] = {}
        for k in countdict:
            BinCounts[i][k] = partition_problem.addVar(
                vtype=GRB.BINARY
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
    # Set up binary county variables
    for k in countdict:
        for i in districts:
            partition_problem.addConstr(quicksum(districts[i][j] for j in districts[i]
                                        if counties[j]==countdict[k])
                                        <= BinCounts[i][k]*10000,
                                        name='binarycounts_%s' % k)
    
    # Each county in no more than 2 districts
    for k in countdict:
        partition_problem.addConstr(quicksum(BinCounts[i][k] for i in districts) <= 2, #TODO
                          name='count_%s' % k)
        
    # No more than split_lim total split counties
    partition_problem.addConstr(quicksum(BinCounts[i][k] for i in districts for k in countdict)
                            <= len(countdict)+ split_lim,
                            name='split_lim')
    
    #Note: This only tracks if a county is in two districts, BOTH OF WHICH are in the partition
        #instead, see if 100% of county population is in district? Lower the cutoff?

    # connectivity
    for center, sp_sets in connectivity_sets.items():
        for node, sp_set in sp_sets.items():
            if center == node:
                continue
            partition_problem.addConstr(districts[center][node] <=
                               quicksum(districts[center][nbor]
                                        for nbor in sp_set))

    #handling majority-minority districts
    #add binary variable to say if a district is majority minority
    m={}
    for i in districts:
            m[i] = partition_problem.addVar(
                vtype=GRB.BINARY
            )
    #now set these variable values
    for i in districts:
        partition_problem.addConstr(quicksum(districts[i][j]*nonwhite_per_block[j] for j in districts[i])>=0.5*pop_bounds[i]['ub']*m[i],
                                    name='maj_min_%s' % i)
    
    #constraint to say that the total # of mm dists in each center is >= req_mm for each center
    #note that "total # mm dists" is tracked by binary variable m, but req_mm will only ever be 0 or 1 so this is fine.
    #req_mm is 0 if the center has more than 2 dists, so this is basically ignored
    for i in districts:
        partition_problem.addConstr(m[i]>=req_mm[i], name='req_maj_min_%s' % i)

    #obj: compactness + bonus for MajMin #TODO
    partition_problem.setObjective(quicksum(districts[i][j] * costs[i][j]
                                    for i in costs for j in costs[i])-1000000*quicksum(m[j] for j in districts),
                           GRB.MINIMIZE)
    partition_problem.Params.LogToConsole = 0
    partition_problem.Params.TimeLimit = len(population) / 200
    partition_problem.update()

    return partition_problem, districts, BinCounts

def make_partition_IP_County_MajMin(costs, connectivity_sets, population, pop_bounds, counties, split_lim, nonwhite_per_block):
    """
    Creates the Gurobi model to partition a region.
    Includes constraints that each county is in no more than 2 districts,
        and that the total # of split counties is less than a set limit
    Includes a huge bonus for creating majority minority districts
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
        counties: (dict) {tract int id: county} County FIPS for each tract
        split_lim: (int) Maximum number of counties that are allowed to be split in each district
        nonwhite_per_block: (np.array) The NUMBER of nonwhite people in each block

    Returns: (tuple (Gurobi partition model, model variables as a dict)

    """

    partition_problem = Model('partition')
    districts = {}
    BinCounts={}
    #make list and dict of counties in this region
    countlist = []
    for j in counties:
        if counties[j] not in countlist:
            countlist.append(counties[j]) #TODO make this faster
    countdict = {v:k for v, k in enumerate(countlist)}
    # Create the variables
    for center, tracts in costs.items():
        districts[center] = {}
        for tract in tracts:
            districts[center][tract] = partition_problem.addVar(
                vtype=GRB.BINARY,
                obj=costs[center][tract]
            )
    #Create binary variables to track if county k is in center i
    for i in districts:
        BinCounts[i] = {}
        for k in countdict:
            BinCounts[i][k] = partition_problem.addVar(
                vtype=GRB.BINARY
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
    # Set up binary county variables
    for k in countdict:
        for i in districts:
            partition_problem.addConstr(quicksum(districts[i][j] for j in districts[i]
                                        if counties[j]==countdict[k])
                                        <= BinCounts[i][k]*10000,
                                        name='binarycounts_%s' % k)
    
    # Each county in no more than 2 districts
    for k in countdict:
        partition_problem.addConstr(quicksum(BinCounts[i][k] for i in districts) <= 2, #TODO
                          name='count_%s' % k)
        
    # No more than split_lim total split counties
    partition_problem.addConstr(quicksum(BinCounts[i][k] for i in districts for k in countdict)
                            <= len(countdict)+ split_lim,
                            name='split_lim')
    
    #TODO: This only tracks if a county is in two districts, BOTH OF WHICH are in the partition
        #instead, see if 100% of county population is in district? Lower the cutoff?

    # connectivity
    for center, sp_sets in connectivity_sets.items():
        for node, sp_set in sp_sets.items():
            if center == node:
                continue
            partition_problem.addConstr(districts[center][node] <=
                               quicksum(districts[center][nbor]
                                        for nbor in sp_set))

    #handling majority-minority districts
    #add binary variable to say if a district is majority minority
    m={}
    for i in districts:
            m[i] = partition_problem.addVar(
                vtype=GRB.BINARY
            )
    #now set these variable values
    for i in districts:
        partition_problem.addConstr(quicksum(districts[i][j]*nonwhite_per_block[j] for j in districts[i])>=0.5*pop_bounds[i]['ub']*m[i],
                                    name='maj_min_%s' % i)
    #TODO we only care about maj min when it is a single district center, not ie at the first partition stage

    #test
    partition_problem.addConstr(quicksum(m[j] for j in districts) >=0,
                     name="test1")

    #finally, the objective has a bonus for every majority minority district
    partition_problem.setObjective(quicksum(districts[i][j] * costs[i][j]
                                    for i in costs for j in costs[i])-1000000*quicksum(m[j] for j in districts),
                           GRB.MINIMIZE)
    partition_problem.Params.LogToConsole = 0
    partition_problem.Params.TimeLimit = len(population) / 200
    partition_problem.update()

    return partition_problem, districts, BinCounts,m

def make_partition_IP_County(costs, connectivity_sets, population, pop_bounds, counties, split_lim):
    """
    Creates the Gurobi model to partition a region.
    Includes constraints that each county is in no more than 2 districts,
        and that the total # of split counties is less than a set limit
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
        counties: (dict) {tract int id: county} County FIPS for each tract
        split_lim: (int) Maximum number of counties that are allowed to be split in each district

    Returns: (tuple (Gurobi partition model, model variables as a dict)

    """

    partition_problem = Model('partition')
    districts = {}
    BinCounts={}
    #make list and dict of counties in this region
    countlist = []
    for j in counties:
        if counties[j] not in countlist:
            countlist.append(counties[j]) #TODO make this faster
    countdict = {v:k for v, k in enumerate(countlist)}
    # Create the variables
    for center, tracts in costs.items():
        districts[center] = {}
        for tract in tracts:
            districts[center][tract] = partition_problem.addVar(
                vtype=GRB.BINARY,
                obj=costs[center][tract]
            )
    #Create binary variables to track if county k is in center i
    for i in districts:
        BinCounts[i] = {}
        for k in countdict:
            BinCounts[i][k] = partition_problem.addVar(
                vtype=GRB.BINARY
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
    # Set up binary county variables
    for k in countdict:
        for i in districts:
            partition_problem.addConstr(quicksum(districts[i][j] for j in districts[i]
                                        if counties[j]==countdict[k])
                                        <= BinCounts[i][k]*10000,
                                        name='binarycounts_%s' % k)
    
    # Each county in no more than 2 districts
    for k in countdict:
        partition_problem.addConstr(quicksum(BinCounts[i][k] for i in districts) <= 2, #TODO
                          name='count_%s' % k)
        
    # No more than split_lim total split counties
    partition_problem.addConstr(quicksum(BinCounts[i][k] for i in districts for k in countdict)
                            <= len(countdict)+ split_lim,
                            name='split_lim')
    
    #TODO: This only tracks if a county is in two districts, BOTH OF WHICH are in the partition
        #instead, see if 100% of county population is in district? Lower the cutoff?

    # connectivity
    for center, sp_sets in connectivity_sets.items():
        for node, sp_set in sp_sets.items():
            if center == node:
                continue
            partition_problem.addConstr(districts[center][node] <=
                               quicksum(districts[center][nbor]
                                        for nbor in sp_set))

    partition_problem.setObjective(quicksum(districts[i][j] * costs[i][j]
                                    for i in costs for j in costs[i]),
                           GRB.MINIMIZE)
    partition_problem.Params.LogToConsole = 0
    partition_problem.Params.TimeLimit = len(population) / 200
    partition_problem.update()

    return partition_problem, districts, BinCounts

def make_partition_IP_Buffalo(costs, connectivity_sets, population, pop_bounds, neighborhoods, split_lim):
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
        neighborhoods: (dict) {tract int id: neighborhood} Nbhd name for each tract
        split_lim: (int) Maximum number of counties that are allowed to be split in each district

    Returns: (tuple (Gurobi partition model, model variables as a dict)

    """
    partition_problem = Model('partition')
    districts = {}
    BinNbds={}
    #make list and dict of neighborhoods in this region
    nbdlist = []
    for j in neighborhoods:
        if neighborhoods[j] not in nbdlist:
            nbdlist.append(neighborhoods[j])
    nbddict = {v:k for v, k in enumerate(nbdlist)}
    # Create the variables
    for center, tracts in costs.items():
        districts[center] = {}
        for tract in tracts:
            districts[center][tract] = partition_problem.addVar(
                vtype=GRB.BINARY,
                obj=costs[center][tract]
            )
    #Create binary variables to track if nbhd k is in center i
    for i in districts:
        BinNbds[i] = {}
        for k in nbddict:
            BinNbds[i][k] = partition_problem.addVar(
                vtype=GRB.BINARY
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
    # Set up binary neighborhood variables
    for k in nbddict:
        for i in districts:
            partition_problem.addConstr(quicksum(districts[i][j] for j in districts[i]
                                        if neighborhoods[j]==nbddict[k])
                                        <= BinNbds[i][k]*10000,
                                        name='binarynbhd_%s' % k)
    
    # Each neighborhood in no more than 2 districts
    for k in nbddict:
        partition_problem.addConstr(quicksum(BinNbds[i][k] for i in districts) <= 2, #TODO
                          name='nbhd_%s' % k)
        
    # No more than split_lim total split nbds
    partition_problem.addConstr(quicksum(BinNbds[i][k] for i in districts for k in nbddict)
                            <= len(nbddict)+ split_lim,
                            name='split_lim')

    # connectivity
    for center, sp_sets in connectivity_sets.items():
        for node, sp_set in sp_sets.items():
            if center == node:
                continue
            partition_problem.addConstr(districts[center][node] <=
                               quicksum(districts[center][nbor]
                                        for nbor in sp_set))

    partition_problem.setObjective(quicksum(districts[i][j] * costs[i][j]
                                    for i in costs for j in costs[i]),
                           GRB.MINIMIZE)
    partition_problem.Params.LogToConsole = 0
    partition_problem.Params.TimeLimit = len(population) / 200
    partition_problem.update()

    return partition_problem, districts, BinNbds

def make_partition_IP(costs, connectivity_sets, population, pop_bounds):
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
                           
    # connectivity
    for center, sp_sets in connectivity_sets.items():
        for node, sp_set in sp_sets.items():
            if center == node:
                continue
            partition_problem.addConstr(districts[center][node] <=
                               quicksum(districts[center][nbor]
                                        for nbor in sp_set))

    partition_problem.setObjective(quicksum(districts[i][j] * costs[i][j]
                                    for i in costs for j in costs[i]),
                           GRB.MINIMIZE)
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

def majority_minority_partition(bdm, state_df):
    """
    Works at the pre-bdm partition level, not the master problem level
    Args:
        bdm: (np.array) binary matrix a_ij = 1 if block i is in district j
        state_df: (pd.DataFrame) original dataframe. Must include race data

    Returns: (np.array) of an entry for each district where 1 means the district IS majority-minority
        and 0 means it's not

    """
    maj_min=[]
    white_per_block=np.multiply(state_df['p_white'], state_df['population'])/100
    print(bdm.shape)
    print(white_per_block.shape)
    for d in bdm.T:
        dist_white=np.sum(np.multiply(d, white_per_block))
        dist_pop=np.sum(np.multiply(d, state_df['population']))
        dist_p_white=np.divide(dist_white, dist_pop)
        #print(dist_p_white)
        if dist_p_white<0.5:
            maj_min.append(1)
        else:
            maj_min.append(0)
    return np.array(maj_min)
