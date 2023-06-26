import sys
sys.path.append('../gerrypy')

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
        name = 'al1'
        experiment_dir = '%s_results_%s' % (name, str(int(time.time())))
        save_dir = os.path.join(constants.RESULTS_PATH, experiment_dir)
        os.mkdir(save_dir)

        print('Starting trial', base_config)
        cg = ColumnGenerator(base_config)
        generation_start_t = time.time()
        cg.generate()
        generation_t = time.time() - generation_start_t
        analysis_start_t = time.time()
        #metrics = generation_metrics(cg)
        analysis_t = time.time() - analysis_start_t

        trial_results = {
            'generation_time': generation_t,
            'analysis_time': analysis_t,
            'leaf_nodes': cg.leaf_nodes,
            'internal_nodes': cg.internal_nodes,
            #'metrics': metrics,
            'n_plans': number_of_districtings(cg.leaf_nodes, cg.internal_nodes)
        }

        print(number_of_districtings(cg.leaf_nodes, cg.internal_nodes))

        def process(val):
            if isinstance(val, dict):
                return ''.join([c for c in str(val) if c.isalnum()])
            else:
                return str(val)

        save_name = '_'.join(['al1', str(int(time.time()))]) + '.npy'
        csv_save_name = '_'.join(['al1', str(int(time.time()))]) + '.csv'
        print(type(trial_results))
        np.save(os.path.join(save_dir, save_name), trial_results)

        bdm = make_bdm(cg.leaf_nodes)
        bdm_df = pd.DataFrame(bdm)
        bdm_df.to_csv(os.path.join(save_dir, csv_save_name), index=False)
        #district_df_of_tree_dir(save_dir)

        state_df, G, lengths, edge_dists = load_opt_data(state_abbrev='AL')
        print(state_df['GEOID'][0])
        print(type(state_df['GEOID'][0]))

        maj_min=majority_minority(bdm, state_df)
        print(maj_min)

        #county_split_coefficients(bdm,state_df,G)

        state_df=load_state_df('AL')
        solutions = master_solutions(cg.leaf_nodes, cg.internal_nodes, state_df, lengths,G, maj_min)
        print(solutions)
        solutions_df = export_solutions(solutions, state_df, bdm)

        results_save_name = '_'.join(['al1', str(int(time.time()))]) + 'assignments.csv'
        solutions_df.to_csv(os.path.join(save_dir, results_save_name), index=False)

def master_solutions(leaf_nodes, internal_nodes, state_df, lengths, G, maj_min):
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
    bdm = make_bdm(leaf_nodes)
    #cost_coeffs = compactness_coefficients(bdm, state_df, lengths)
    cost_coeffs = county_split_coefficients(bdm, state_df,G)
    cost_coeffs=np.array(cost_coeffs)
    root_map = make_root_partition_to_leaf_map(leaf_nodes, internal_nodes)
    sol_dict = {}
    for partition_ix, leaf_slice in root_map.items():
        start_t = time.time()
        model, dvars = make_master(base_config['n_districts'], bdm[:, leaf_slice], cost_coeffs[leaf_slice], False, maj_min)
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

        print(maj_min[sol_dict[partition_ix]['solution_ixs']])
    return {'master_solutions': sol_dict}

def export_solutions(solutions, state_df, bdm):
    """
    Creates a dataframe with each block matched to a district based on the IP solution
    Args:
        solutions: (dict) of solutions outputted by IP
        state_df: (pd DataFrame) with state data
        bdm: (np.array) n x d matrix where a_ij = 1 when block i appears in district j.

    Returns: Dataframe mapping GEOID to district assignments

    """
    solutions_df = pd.DataFrame()
    solutions_df['GEOID'] = state_df['GEOID']
    print(state_df['GEOID'][0])
    selected_dists = np.zeros(state_df.shape[0])

    for sol_idx in range(len(solutions['master_solutions'])):
        solution_ixs = solutions['master_solutions'][sol_idx]['solution_ixs']
        for i in solution_ixs:
            for index, j in enumerate(bdm.T[i]):
                if j==1: selected_dists[index]=i
        print(selected_dists)
        col_title = 'District'+str(sol_idx)
        solutions_df[col_title] = selected_dists
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
        'parent_resample_trials': 5,
        'max_sample_tries': 25,
        'n_samples': 20, #TODO 2
        'n_root_samples': 3, #TODO 5
        'max_n_splits': 5, 
        'min_n_splits': 2,
        'max_split_population_difference': 1.5,
        'event_logging': False,
        'verbose': True,
    }
    gurobi_config = {
        'IP_gap_tol': 1e-3,
        'IP_timeout': 10,
    }
    pdp_config = {
        'state': 'AL',
        'n_districts': 7,
        'population_tolerance': .01,
        #'population_tolerance': population_tolerance()*12,
    }
    base_config = {**center_selection_config,
                   **tree_config,
                   **gurobi_config,
                   **pdp_config}

    experiment = Experiment(base_config)
    experiment.run()