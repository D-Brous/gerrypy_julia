from gurobipy import *
import numpy as np
from scipy.stats import t
import pandas as pd

from gurobipy import quicksum
from gurobipy import Model
from gurobipy import GRB

def make_master(k, block_district_matrix, costs,
                relax=False, opt_type='minimize'):
    """
    Constructs the master selection problem.
    Args:
        k: (int) the number of districts in a plan
        block_district_matrix: (np.array) binary matrix a_ij = 1 if block i is in district j
        costs: (np.array) cost coefficients of districts
        relax: (bool) construct relaxed linear master problem
        opt_type: (str) {"minimize", "maximize", "abs_val"

    Returns: (Gurobi.model, (dict) of master selection problem variables)

    """
    n_blocks, n_columns = block_district_matrix.shape

    master = Model("master LP")

    x = {}
    D = range(n_columns)
    vtype = GRB.CONTINUOUS if relax else GRB.BINARY
    for j in D:
        x[j] = master.addVar(vtype=vtype, name="x(%s)" % j)

    master.addConstrs((quicksum(x[j] * block_district_matrix[i, j] for j in D) == 1
                       for i in range(n_blocks)), name='exactlyOne')

    master.addConstr(quicksum(x[j] for j in D) == k,
                     name="totalDistricts")

    if opt_type == 'minimize':
        master.setObjective(quicksum(costs[j] * x[j] for j in D), GRB.MINIMIZE)
    elif opt_type == 'maximize':
        master.setObjective(quicksum(costs[j] * x[j] for j in D), GRB.MAXIMIZE)
    elif opt_type == 'abs_val':
        w = master.addVar(name="w", lb=-k, ub=k)
        master.addConstr(quicksum(costs[j] * x[j] for j in D) <= w,
                         name='absval_pos')
        master.addConstr(quicksum(costs[j] * x[j] for j in D) >= -w,
                         name='absval_neg')

        master.setObjective(w, GRB.MINIMIZE)
    else:
        raise ValueError('Invalid optimization type')

    return master, x


def make_master_vectorized(k, block_district_matrix, costs, opt_type='abs_val'):
    """
    Constructs the master selection problem.
    Args:
        k: (int) the number of districts in a plan
        block_district_matrix: (np.array) binary matrix a_ij = 1 if block i is in district j
        costs: (np.array) cost coefficients of districts
        opt_type: (str) {"minimize", "maximize", "abs_val"

    Returns: (Gurobi.model, (dict) of master selection problem variables)

    """
    n_blocks, n_columns = block_district_matrix.shape

    master_selection_problem = Model("master LP")

    selection = master_selection_problem.addMVar(shape=n_columns,
                                                   vtype=GRB.BINARY)
    master_selection_problem.addConstr(block_district_matrix @ selection == 1)
    master_selection_problem.addConstr(selection.sum() == k)

    if opt_type == 'minimize':
        master_selection_problem.setObjective(costs @ selection, GRB.MINIMIZE)
    elif opt_type == 'maximize':
        master_selection_problem.setObjective(costs @ selection, GRB.MAXIMIZE)
    elif opt_type == 'abs_val':
        w = master_selection_problem.addMVar(shape=1, name="w", lb=-k, ub=k)
        master_selection_problem.addConstr(costs @ selection <= w)
        master_selection_problem.addConstr(costs @ selection >= -w)
        master_selection_problem.setObjective(w, GRB.MINIMIZE)
    else:
        raise ValueError('Invalid optimization type')

    return master_selection_problem, selection


def efficiency_gap_coefficients(district_df, state_vote_share):
    """

    Args:
        district_df: (pd.DataFrame) selected district statistics
            (requires "mean", "std_dev", "DoF")
        state_vote_share: (float) average state vote share across historical elections.

    Returns: (np.array) of efficiency gap cost coefficients

    """
    mean = district_df['mean'].values
    std_dev = district_df['std_dev'].values
    DoF = district_df['DoF'].values
    expected_seats = 1 - t.cdf(.5, DoF, mean, std_dev)
    # https://www.brennancenter.org/sites/default/files/legal-work/How_the_Efficiency_Gap_Standard_Works.pdf
    # Efficiency Gap = (Seat Margin – 50%) – 2 (Vote Margin – 50%)
    return (expected_seats - .5) - 2 * (state_vote_share - .5)


def nbd_coefficients(bdm, state_df):
    """

    Args:
        bdm: (np.array) binary matrix a_ij = 1 if block i is in district j
        state_df: (pd.DataFrame) original dataframe. Must include 'nbhdname' field

    Returns: (np.array) of neighborhood counts per district

    """
    nbhds = state_df['nbhdname']
    nbhd_count=[]
    for d in bdm.T:
        dist_nbhds = []
        for index, i in enumerate(d):
            if i==1: dist_nbhds.append(nbhds[index])
        dist_nbhds_unique=list(set(dist_nbhds))
        nbhd_count.append(len(dist_nbhds_unique))
    return(np.array(nbhd_count))


def make_root_partition_to_leaf_map(leaf_nodes, internal_nodes):
    """
    Shard the sample tree leaf nodes by root partition.

    Args:
        leaf_nodes: (SHPNode list) with node capacity equal to 1 (has no child nodes).
        internal_nodes: (SHPNode list) with node capacity >1 (has child nodes).

    Returns: (dict) {root partition index: array of leaf node indices}

    """
    def add_children(node, root_partition_id):
        if node.n_districts > 1:
            for partition in node.children_ids:
                for child in partition:
                    add_children(node_dict[child], root_partition_id)
        else:
            node_to_root_partition[id_to_ix[node.id]] = root_partition_id

    # Create mapping from leaf ix to root partition ix
    node_to_root_partition = {}
    node_dict = {**internal_nodes, **leaf_nodes}
    id_to_ix = {nid: ix for ix, nid in enumerate(sorted(leaf_nodes))}
    root = internal_nodes[0]
    for ix, root_partition in enumerate(root.children_ids):
        for child in root_partition:
            add_children(node_dict[child], ix)

    # Create inverse mapping
    partition_map = {}
    for node_ix, partition_ix in node_to_root_partition.items():
        try:
            partition_map[partition_ix].append(node_ix)
        except KeyError:
            partition_map[partition_ix] = [node_ix]
    partition_map = {ix: np.array(leaf_list) for ix, leaf_list in partition_map.items()}

    return partition_map