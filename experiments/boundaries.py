import copy
import time
import numpy as np
import pandas as pd
from gerrypy import constants
from gerrypy.analyze.districts import *
from gerrypy.optimize.generate import ColumnGenerator
from gerrypy.rules.political_boundaries.preservation_constraint import *


class Experiment:
    def __init__(self, default_config, boundary_type, base_dir='boundary_experiment'):
        self.default_config = default_config
        self.save_path = os.path.join(constants.RESULTS_PATH, base_dir, boundary_type)
        self.default_config['boundary_type'] = boundary_type
        os.makedirs(self.save_path, exist_ok=True)

    def k_to_w(self, k):
        """Compute the sample width as a function of the number of districts."""
        w_root = int(round((1500 / k) ** 1.2))
        w_internal = int(round((200 / k) ** .7))
        return w_root, w_internal

    def run(self, states=None):
        if states is None:
            states = [state for state in constants.seats]
        for state in states:
            if constants.seats[state]['house'] < 2:
                continue
            trial_config = copy.deepcopy(self.default_config)
            k = constants.seats[state]['house']
            w_root, w = self.k_to_w(k)

            trial_config['state'] = state
            trial_config['n_districts'] = k
            trial_config['n_root_samples'] = w_root
            trial_config['n_samples'] = w

            cg = ColumnGenerator(trial_config)
            start_t = time.time()
            cg.generate()
            end_t = time.time()
            n_plans = number_of_districtings(cg.leaf_nodes, cg.internal_nodes)
            result_dict = {
                'generation_time': end_t - start_t,
                'leaf_nodes': cg.leaf_nodes,
                'internal_nodes': cg.internal_nodes,
                'trial_config': trial_config,
                'n_plans': n_plans
            }
            file_name = '%s_%d.p' % (state, n_plans)
            file_path = os.path.join(self.save_path, file_name)
            pickle.dump(result_dict, open(file_path, 'wb'))


if __name__ == '__main__':
    center_selection_config = {
        'selection_method': 'random_iterative',  # one of
        'perturbation_scale': 0,
        'n_random_seeds': 0,
        'capacities': 'match',
        'capacity_weights': 'voronoi',
    }
    tree_config = {
        'n_samples': 3,
        'n_root_samples': 3,
        'max_sample_tries': 25,
        'parent_resample_trials': 3,
        'max_n_splits': 5,
        'min_n_splits': 2,
        'max_split_population_difference': 1.5,
        'event_logging': False,
        'verbose': True,
    }
    gurobi_config = {
        'IP_gap_tol': 1e-5,
        'IP_timeout': 10,
    }
    pdp_config = {
        'state': 'IA',
        'n_districts': 4,
        'population_tolerance': .01,
        'county_discount_factor': 0.5,
        'boundary_type': 'baseline'
    }
    base_config = {**center_selection_config,
                   **tree_config,
                   **gurobi_config,
                   **pdp_config}

    boundary_type = 'baseline'  # or 'county' or 'municipal'
    exp = Experiment(base_config, boundary_type, base_dir='boundary_experiment')
    exp.run()
