import sys
sys.path.append('../gerrypy_julia')

#from optimize.generate_mm import ColumnGenerator
from optimize.generate import ColumnGenerator
from analyze.districts import *
from optimize.dir_processing import district_df_of_tree_dir
import constants
from copy import deepcopy
import time
import os
import numpy as np
import json
from optimize.master import *
from data.data2020.load import *
from optimize.improvement import *

class Experiment:
    """
    Experiment class to test different generation configurations.
    """
    def __init__(self, base_config):
        """
        Args:
            base_config: the config shared across as experiments.
            experiment_config: the config specific to a trial.
        """
        self.base_config = base_config

    def run(self):
        """Performs all generation trials.

        Saves a file with the tree as well as a large number of ensemble level metrics."""
        name = 'la1'
        experiment_dir = '%s_results_%s' % (name, str(int(time.time())))
        save_dir = os.path.join(constants.RESULTS_PATH, experiment_dir)
        os.mkdir(save_dir)

        print('Starting trial', base_config)
        cg = ColumnGenerator(base_config)
        generation_start_t = time.thread_time()
        cg.generate()
        generation_t = time.thread_time() - generation_start_t
        analysis_start_t = time.time()
        #metrics = generation_metrics(cg)
        analysis_t = time.thread_time() - analysis_start_t

        trial_results = {
            'generation_time': generation_t,
            'analysis_time': analysis_t,
            'leaf_nodes': cg.leaf_nodes,
            'internal_nodes': cg.internal_nodes,
            #'metrics': metrics,
            'n_plans': number_of_districtings(cg.leaf_nodes, cg.internal_nodes)
        }
        print(f"Tree generation time: {trial_results['generation_time']}")
        print(number_of_districtings(cg.leaf_nodes, cg.internal_nodes))

        def process(val):
            if isinstance(val, dict):
                return ''.join([c for c in str(val) if c.isalnum()])
            else:
                return str(val)

        save_name = '_'.join(['la1', str(int(time.time()))]) + '.npy'
        csv_save_name = '_'.join(['la1', str(int(time.time()))]) + '.csv'
        print(type(trial_results))
        np.save(os.path.join(save_dir, save_name), trial_results)

        #print(cg.internal_nodes)

        bdm = make_cdm(cg.leaf_nodes)
        bdm_df = pd.DataFrame(bdm)
        bdm_df.to_csv(os.path.join(save_dir, csv_save_name), index=False)
        #district_df_of_tree_dir(save_dir)

        state_df, G, lengths, edge_dists = load_opt_data(state='LA', special_input=self.base_config['optimization_data'])

        maj_min=majority_minority(bdm, state_df)
        print(maj_min)
        print(sum(maj_min))

        #county_split_coefficients(bdm,state_df,G)

        solutions, sol_tree = master_solutions(bdm,cg.leaf_nodes, cg.internal_nodes, state_df, lengths,G, maj_min) 
        print(solutions)
        solutions_df = export_solutions(solutions, state_df, bdm, sol_tree, cg.internal_nodes)

        results_save_name = '_'.join(['la1', str(int(time.time()))]) + 'assignments.csv'
        solutions_df.to_csv(os.path.join(save_dir, results_save_name), index=False)

def master_solutions(bdm,leaf_nodes, internal_nodes, state_df, lengths, G, maj_min):
    """
    Solves the master selection problem optimizing for fairness on all root partitions.
    Args:
        leaf_nodes: (SHPNode list) with node capacity equal to 1 (has no child nodes).
        internal_nodes: (SHPNode list) with node capacity >1 (has child nodes).
        district_df: (pd.DataFrame) selected statistics of generated districts.
        state: (str) two letter state abbreviation
        state_vote_share: (float) the expected Republican vote-share of the state.
        lengths: (np.array) Pairwise block distance matrix.
        G: (nx.Graph) The block adjacency graph

    Returns: (dict) solution data for each optimal solution.

    """
    #bdm = make_bdm(leaf_nodes) #TODO don't need to make the bdm twice
    #cost_coeffs = compactness_coefficients(bdm, state_df, lengths)
    #cost_coeffs = county_split_coefficients(bdm, state_df,G)
    cost_coeffs = compactness_coefficients(bdm, state_df, lengths)
    bb = np.zeros((len(cost_coeffs)))
    cost_coeffs=np.array(cost_coeffs)
    root_map, ix_to_id = make_root_partition_to_leaf_map(leaf_nodes, internal_nodes)
    #print(root_map)
    sol_dict = {}

    for partition_ix, leaf_slice in root_map.items():
        start_t = time.time()
        model, dvars = make_master(base_config['n_districts'], bdm[:, leaf_slice], cost_coeffs[leaf_slice], maj_min[leaf_slice], bb[leaf_slice])
        construction_t = time.time()

        model.Params.LogToConsole = 0
        model.Params.MIPGapAbs = 1e-4
        model.Params.TimeLimit = len(leaf_nodes) / 10
        model.optimize()
        opt_cols = [j for j, v in dvars.items() if v.x > .5]
        solve_t = time.time()

        sol_dict[partition_ix] = {
            'construction_time': construction_t - start_t,
            'solve_time': solve_t - construction_t,
            'n_leaves': len(leaf_slice),
            'solution_ixs': root_map[partition_ix][opt_cols],
            'optimal_objective': cost_coeffs[leaf_slice][opt_cols]
        }
        print(opt_cols)
        print(sol_dict[partition_ix]['solution_ixs'])
        print(maj_min[sol_dict[partition_ix]['solution_ixs']])
        #constraint = model.getConstrByName("majorityMinority")
        #print(constraint.slack)

        sol_tree= get_solution_tree(leaf_nodes, internal_nodes, ix_to_id, sol_dict[partition_ix]['solution_ixs'])

        status=model.status
        if status==2:
            print("Optimal solution found")
        elif status==3:
            print("WARNING: no optimal solution is possible. Solving relaxation.")

        # constraintm_slacks=[]
        # for k in range(len(cost_coeffs)):
        #     constraintm=model.getConstrByName('testm_%s' % k)
        #     constraintm_slacks.append(constraintm.slack)
        #     if constraintm.slack!=0:
        #         print(str(k)+": "+str(int(constraintm.slack)))
        #         print(dvars[k])
    return {'master_solutions': sol_dict}, sol_tree

def export_solutions(solutions, state_df, bdm, sol_tree, internal_nodes):
    """
    Creates a dataframe with each block matched to a district based on the IP solution
    Args:
        solutions: (dict) of solutions outputted by IP
        state_df: (pd DataFrame) with state data
        bdm: (np.array) n x d matrix where a_ij = 1 when block i appears in district j.
        sol_tree: list of SHPNodes representing the leaf nodes and their ancestors #TODO make this work for multiple trials

    Returns: Dataframe mapping GEOID to district assignments

    """
    solutions_df = pd.DataFrame()
    solutions_df['GEOID'] = state_df['GEOID']
    selected_dists = np.zeros(state_df.shape[0])

    print('begin export solutions')

    for sol_idx in range(len(solutions['master_solutions'])):
        solution_ixs = solutions['master_solutions'][sol_idx]['solution_ixs']
        for i in solution_ixs:
            for index, j in enumerate(bdm.T[i]):
                if j==1: selected_dists[index]=i
        print(selected_dists)
        col_title = 'District'+str(sol_idx)
        solutions_df[col_title] = selected_dists

    #add a column parent which tells us this block's parent's center IF it is a center for the final tree, or -1 if it is a root for the final tree
    solutions_df['Parent'] = np.nan
    solutions_df['ID'] = np.nan
    for node in sol_tree:
        if node.parent_id is not None:
            solutions_df.loc[node.center, 'ID']=node.id
            parent_center=internal_nodes[node.parent_id].center
            solutions_df.loc[node.center, 'Parent']=parent_center
            if parent_center is None:
                solutions_df.loc[node.center, 'Parent']=-1

    return solutions_df

if __name__ == '__main__':
    center_selection_config = {
        'selection_method': 'uniform_random',  # one of
        'perturbation_scale': 1,
        'n_random_seeds': 1,
        'capacities': 'match',
        'capacity_weights': 'voronoi',
    }
    tree_config = {
        'parent_resample_trials': 5, #TODO 5-10
        'max_sample_tries': 25,
        'n_samples': 12, #TODO 10-20
        'n_root_samples': 1,
        'max_n_splits': 2,
        'min_n_splits': 2, 
        'max_split_population_difference': 1.5,
        'event_logging': False,
        'verbose': False,
    }
    gurobi_config = {
        'IP_gap_tol': 1e-3,
        'IP_timeout': 10,
    }
    pdp_config = {
        'state': 'LA',
        'n_districts': 6,
        'population_tolerance': .01,
        'required_mm': 0, #TODO if this is 0, partition stage works
        #'population_tolerance': population_tolerance()*12,
        'optimization_data': 'tract'
    }
    base_config = {**center_selection_config,
                   **tree_config,
                   **gurobi_config,
                   **pdp_config}

    experiment = Experiment(base_config)
    experiment.run()