import sys
sys.path.append('../gerrypy_julia')

import os
import constants
import pandas as pd
from data.load import *

def check_feasibility(config, assignments_df, state_df, G):
    print('\n--------------------Running Feasibility Check--------------------\n')
    num_plans = config['n_root_samples']
    if len(config['maj_black_partition_IPs']) > 0:
        num_plans *= len(config['maj_black_partition_IPs'])
    num_districts = config['n_districts']
    assignments_df = assignments_df.sort_values(by=['GEOID'])
    state_df = state_df.sort_values(by=['GEOID'])
    
    #bvap_per_district = np.zeros((num_plans, num_districts), dtype=int)
    #vap_per_district = np.zeros((num_plans, num_districts), dtype=int)
    pop_per_district = np.zeros((num_plans, num_districts), dtype=int)

    try:
        for plan in range(num_plans):
            assignments_df[f'District{plan}'] = assignments_df[f'District{plan}'].astype(int)
            for district in range(num_districts):
                district_cgus = list(assignments_df.loc[assignments_df[f'District{plan}'] == district].index)
                if not nx.is_connected(nx.subgraph(G, district_cgus)):
                    raise RuntimeError(f'District {district} from plan {plan} is not connected')
                pop_per_district[plan, district] = state_df.loc[assignments_df[f'District{plan}'] == district]['population'].sum()
        
        for plan in range(num_plans):
            tot_pop_districts = int(pop_per_district[plan].sum())
            tot_pop_state_df = int(state_df['population'].sum())
            if tot_pop_state_df != tot_pop_districts:
                raise RuntimeError(f'Districts from plan {plan} not using all of state population')
            ideal_district_pop = tot_pop_districts / num_districts
            for district in range(num_districts):
                if pop_per_district[plan][district] > (1 + config['population_tolerance']) * ideal_district_pop:
                    raise RuntimeError(f'The population of district {district} from plan {plan} is too large')
                if pop_per_district[plan][district] < (1 - config['population_tolerance']) * ideal_district_pop:
                    raise RuntimeError(f'The population of district {district} from plan {plan} is too small')
        
        print('\nAll solutions are feasible\n')
    except RuntimeError as error:
        print(f'\nError: {str(error)}\n')

    
    #avgs = bvap_per_district / vap_per_district

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
        'n_samples': 2, #Should be 3-5 #TODO 10-20
        'n_root_samples': 10,
        'max_n_splits': 2,
        'min_n_splits': 2, 
        'max_split_population_difference': 1.5,
        'granularity': 'block_group',
        'event_logging': False,
        'verbose': False,
        'debug_file': 'debug_file.txt',
        'debug_file_2': 'debug_file_2.txt',
        'save_tree': True,
        'exact_partition_range': [2,3,4,5],
        'maj_black_partition_IPs': [],
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
        'results_path': constants.LOUISIANA_HOUSE_RESULTS_PATH,
        #'partition_IP': 'make_partition_IP',
        'mode': 'both'
    }
    config = {**center_selection_config,
              **tree_config,
              **master_config,
              **state_config}
    
    
    results_time_str = '1721813164' #'1721243348' #'1719459163'
    assignment_time_str = '0' #'1721243348' #'1719586752'
    
    state_df = load_state_df(config['state'], config['year'], config['granularity'])
    G = load_adjacency_graph(config['state'], config['year'], config['granularity'])
    assignments_df = load_assignments_df(config['results_path'], results_time_str, assignment_time_str)
    check_feasibility(config, assignments_df, state_df, G)