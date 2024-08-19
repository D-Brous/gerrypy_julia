import sys
sys.path.append('../gerrypy_julia')

import constants
from optimize.shp import SHP
from optimize.postprocess import *
from functools import partial

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
    'debug_file': 'debug_file.txt',
    'debug_file_2': 'debug_file_2.txt',
    'save_tree': True,
    'exact_partition_range': [2,3,4,5],
    'maj_black_partition_IPs': ['make_partition_IP_MajBlack_approximate', 'make_partition_IP_MajBlack'],
    'alpha': 0,
    'epsilon': 0.03,
    'use_black_maj_warm_start': False,
    'use_time_limit': True
}
master_config = {
    'IP_gap_tol': 1e-3,
    'IP_timeout': 10,
    'callback_time_interval': None,
    'cost_coeffs': 'maj_black',
    'tree_time_str': str(1719991175), #'1718831492',
    'save_cdms': True,
    'linear_objective': True,
    'save_assignments': True
}
state_config = {
    'state': 'LA',
    'year': 2010,
    'n_districts': 105,
    'population_tolerance': 0.045,
    'required_mm': 0, #TODO if this is 0, partition stage works
    #'population_tolerance': population_tolerance()*12,
    'results_path': constants.LOUISIANA_HOUSE_RESULTS_PATH,
    #'partition_IP': 'make_partition_IP',
    'mode': 'both',
    'subregion': None,
    'verbose': True,
    'save_config': True
}
config = {**center_selection_config,
            **tree_config,
            **master_config,
            **state_config}
config['ideal_pop'] = load_state_df(config['state'], config['year'], config['granularity'])['population'].sum() / config['n_districts']

if __name__ == '__main__':
    #shp = SHP(config)
    #assignments_df = shp.run()
    
    results_time_str = '1721243348'
    assignments_time_str = '1721243348'
    local_search_func = partial(average_geq_threshold_random, config['state'], config['year'], config['granularity'])
    partition_func = shp_partition
    save_rule_func = lambda step : True

    config['n_root_samples'] = 1
    config['debug_file'] = None
    config['debug_file_2'] = None
    config['save_tree'] = False
    config['maj_black_partition_IPs'] = ['make_partition_IP_MajBlack']
    config['use_black_maj_warm_start'] = False
    config['use_time_limit'] = False
    config['save_cdms'] = False
    config['save_assignments'] = False
    config['n_districts'] = 2
    config['verbose'] = False
    config['save_config'] = False
    post_processing = PostProcessing(config, local_search_func, partition_func, save_rule_func, results_time_str, assignments_time_str)
    n_steps = 3
    plans = [0]
    post_processing.one_chain(n_steps, plans=plans)
    #post_processing.short_bursts(n_steps=10, burst_length=2, plans=[0])
    #G = load_district_adjacency_graph(config['state'], config['year'], config['granularity'], assignments_df['District1'])
    #average_geq_threshold(config['state'], config['year'], config['granularity'], assignments_df['District1'])
    #assignments_dict = assignments_df_to_dict(assignments_df)
    #assignment_dict = assignments_dict[0]

    #assignment_ser = assignments_df['District0']
    #assignment_dict = assignment_ser_to_dict(assignment_ser)
    #assignment_ser_2 = assignment_dict_to_ser(assignment_dict)
    #total_subregion = average_geq_threshold_random(config['state'], config['year'], config['granularity'], assignment_ser)
    #print(total_subregion)
    