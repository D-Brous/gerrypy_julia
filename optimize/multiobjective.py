import sys
sys.path.append('../gerrypy')

import time
from datetime import datetime
from optimize.master import *
from analyze.tree import *
from analyze.districts import *


def compute_pareto_front_shards(k, bdm, partition_map, primary_objective, secondary_objective,
                                opt_secondary_per_shard, epsilon=1, opt_type='abs_val'):
    pareto_front_shards = {}
    for shard_ix, shard_nodes in partition_map.items():
        if shard_ix % 50 == 0:
            current_time = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            print(f'Completing shard {shard_ix} at {current_time} for state with {k} districts')
        pareto_front_shards[shard_ix] = {
            'primary_objective': [],
            'secondary_objective': [],
            'solutions': []
        }
        bdm_shard = bdm[:, shard_nodes]
        p_obj_shard = primary_objective[shard_nodes]
        msp, selection = make_master_vectorized(k, bdm_shard, p_obj_shard, opt_type=opt_type)
        msp.Params.LogToConsole = 0

        secondary_objective_shard_constraint = max(secondary_objective) * k
        msp.addConstr(secondary_objective[shard_nodes] @ selection <= secondary_objective_shard_constraint,
                      name='second_obj')
        msp.update()
        while secondary_objective_shard_constraint > opt_secondary_per_shard[shard_ix]:
            msp.setAttr('RHS', msp.getConstrByName('second_obj'), secondary_objective_shard_constraint)
            msp.optimize()
            solution = [shard_nodes[j] for j, v in enumerate(selection.tolist()) if v.X > .5]
            p_obj = primary_objective[solution].sum()
            s_obj = secondary_objective[solution].sum()
            pareto_front_shards[shard_ix]['primary_objective'].append(p_obj)
            pareto_front_shards[shard_ix]['secondary_objective'].append(s_obj)
            pareto_front_shards[shard_ix]['solutions'].append(solution)
            secondary_objective_shard_constraint = s_obj - epsilon

    return pareto_front_shards


def compute_pareto_front(pareto_front_shards, primary_objective_mode='abs_val'):
    if primary_objective_mode == 'abs_val':
        sort_key = lambda x: abs(x[0])
        reverse = False
    elif primary_objective_mode == 'maximize':
        sort_key = lambda x: x[0]
        reverse = True
    elif primary_objective_mode == 'minimize':
        sort_key = lambda x: x[0]
        reverse = False
    else:
        raise ValueError('Unknown primary_objective_mode')

    fused_pareto_candidates = [candidate_pareto_point for pareto_shard in pareto_front_shards.values()
                               for candidate_pareto_point in list(zip(pareto_shard['primary_objective'],
                                                                      pareto_shard['secondary_objective'],
                                                                      pareto_shard['solutions']))]
    fused_pareto_candidates.sort(key=sort_key, reverse=reverse)
    merged_pareto_front = [fused_pareto_candidates[0]]
    for p_obj, s_obj, sol in fused_pareto_candidates[1:]:
        if s_obj < merged_pareto_front[-1][1]:
            merged_pareto_front.append((p_obj, s_obj, sol))

    pareto_front = list(zip(*merged_pareto_front))
    pareto_primary = np.array(pareto_front[0])
    pareto_secondary = np.array(pareto_front[1])
    pareto_solutions = list(pareto_front[2])
    return pareto_primary, pareto_secondary, pareto_solutions


def create_secondary_objective(objective_df, objective_formula):
    objective_coeffs = np.zeros(len(objective_df))
    for column, weight in objective_formula.items():
        objective_coeffs += weight * objective_df[column].values
    return objective_coeffs


def run_multi_objective(leaf_nodes, internal_nodes, primary_objective, secondary_objective,
                        epsilon=1, opt_type='abs_val'):
    opt_secondary_per_shard, _ = query_per_root_partition(leaf_nodes, internal_nodes, -secondary_objective)
    opt_secondary_per_shard = -np.array(opt_secondary_per_shard)

    k = internal_nodes[0].n_districts
    n = len(internal_nodes[0].area)
    bdm = make_bdm(leaf_nodes, n)
    partition_map = make_root_partition_to_leaf_map(leaf_nodes, internal_nodes)
    shards = compute_pareto_front_shards(k, bdm, partition_map, primary_objective, secondary_objective,
                                         opt_secondary_per_shard, epsilon=epsilon, opt_type=opt_type)

    pareto_primary, pareto_secondary, pareto_solutions = compute_pareto_front(shards, opt_type)
    return pareto_primary, pareto_secondary, pareto_solutions, shards

