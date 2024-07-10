import sys
sys.path.append('../gerrypy_julia')

import os
import constants
import pandas as pd
from data.load import *

def check_feasibility(config, results_time_str, assignment_time_str, num_plans=None, num_districts=None):
    results_df_path = os.path.join(config['results_dir'], 
                                   'results_' + results_time_str, 
                                   'assignments_' + assignment_time_str + '.csv')
    state_df_path = os.path.join(constants.OPT_DATA_PATH, 
                                 config['granularity'], 
                                 config['state'], 
                                 'state_df.csv')
    state_df = pd.read_csv(state_df_path)
    results_df = pd.read_csv(results_df_path)
    
    if num_plans is None:
        num_plans = 0
        for column in results_df.columns.tolist():
            if column[:8] == 'District':
                num_plans += 1
    if num_districts is None:
        num_districts = config['n_districts']
    
    bvap_per_district = np.zeros((num_plans, num_districts))
    vap_per_district = np.zeros((num_plans, num_districts))
    pop_per_district = np.zeros((num_plans, num_districts))
    
    for i, row in enumerate(results_df.values):
        for j in range(num_plans):
            district = int(results_df.loc[i, f'District{j}'])
            try:
                bvap_per_district[j, district] += int(state_df.loc[i, 'BVAP'])
                vap_per_district[j, district] += int(state_df.loc[i, 'VAP'])
                pop_per_district[j, district] += int(state_df.loc[i, 'population'])
            except IndexError:
                print(results_df_path)
    
    for plan in range(num_plans):
        tot_pop_districts = int(pop_per_district[plan].sum())
        tot_pop_state_df = int(state_df['population'].sum())
        if tot_pop_state_df != tot_pop_districts:
            raise ValueError(f'Districts from plan {plan} using all of state population')
        ideal_district_pop = tot_pop_districts / num_districts
        for district in range(num_districts):
            if pop_per_district[plan][district] > (1 + config['population_tolerance']) * ideal_district_pop:
                raise ValueError(f'The population of district {district} from plan {plan} is too large')
            if pop_per_district[plan][district] < (1 - config['population_tolerance']) * ideal_district_pop:
                raise ValueError(f'The population of district {district} from plan {plan} is too small')
    '''
    for center, sp_sets in connectivity_sets.items():
        for node, sp_set in sp_sets.items():
            if center == node:
                continue
            partition_problem.addLConstr(districts[center][node] <=
                               quicksum(districts[center][nbor]
                                        for nbor in sp_set))
    '''
    
    avgs = bvap_per_district / vap_per_district
if __name__ == '__main__':
    center_selection_config = {
        'selection_method': 'uniform_random',  # one of
        'perturbation_scale': 1,
        'n_random_seeds': 1,
        'capacities': 'match',
        'capacity_weights': 'voronoi',
    }
    tree_config = {
        'parent_resample_trials': 5, #5 before #TODO 5-10
        'max_sample_tries': 5, # 25 before
        'n_samples': 3, #Should be 3-5 #TODO 10-20
        'n_root_samples': 10,
        'max_n_splits': 2,
        'min_n_splits': 2, 
        'max_split_population_difference': 1.5,
        'granularity': 'block_group',
        'event_logging': False,
        'verbose': False,
        'debug_file': 'debug_file.txt',
        'save_tree': True,
        'exact_partition_range': [2,3,4,5],
        'maj_black_partition_IP': 'make_partition_IP_MajBlack_approximate',
        'alpha': 0,
        'epsilon': 0.01
    }
    master_config = {
        'IP_gap_tol': 1e-3,
        'IP_timeout': 10,
        'callback_time_interval': None,
        'cost_coeffs': 'maj_black',
        'tree_time_str': str(1719991175), #'1718831492',
        'save_cdms': True,
        'linear_objective': True
    }
    state_config = {
        'state': 'LA',
        'n_districts': 105,
        'population_tolerance': .045,
        'required_mm': 0, #TODO if this is 0, partition stage works
        #'population_tolerance': population_tolerance()*12,
        'results_dir': constants.LOUISIANA_HOUSE_RESULTS_PATH,
        #'partition_IP': 'make_partition_IP',
        'mode': 'both'
    }
    config = {**center_selection_config,
              **tree_config,
              **master_config,
              **state_config}
    
    results_time_str = '1719459163'
    assignment_time_str = '1719586752'
    check_feasibility(config, results_time_str, assignment_time_str)