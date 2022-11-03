from gerrypy.optimize.generate import ColumnGenerator
from gerrypy.analyze.districts import *
from gerrypy import constants
from copy import deepcopy
import time
import os
import numpy as np
import json


class Experiment:
    """
    Experiment class to test different generation configurations.
    """
    def __init__(self, base_config, experiment_config):
        """
        Args:
            base_config: the config shared across as experiments.
            experiment_config: the config specific to a trial.
        """
        self.base_config = base_config
        self.experiment_config = experiment_config

    def run(self):
        """Performs all generation trials.

        Saves a file with the tree as well as a large number of ensemble level metrics."""
        name = self.experiment_config['name']
        save_dir = os.path.join(constants.RESULTS_PATH, name)
        os.makedirs(save_dir, exist_ok=True)
        for state in self.experiment_config['states']:
            print('############## Starting %s trials ##############' % state)
            for trial_values in self.experiment_config['trial_parameters']:
                trial_config = deepcopy(self.base_config)
                for (k, v) in trial_values:
                    if len(k) == 2:
                        trial_config[k[0]][k[1]] = v
                    else:
                        trial_config[k] = v
                trial_config['state'] = state

                print('Starting trial', trial_config)
                cg = ColumnGenerator(trial_config)
                generation_start_t = time.time()
                cg.generate()
                generation_t = time.time() - generation_start_t
                n_plans = number_of_districtings(cg.leaf_nodes, cg.internal_nodes)

                trial_results = {
                    'generation_time': generation_t,
                    'leaf_nodes': cg.leaf_nodes,
                    'internal_nodes': cg.internal_nodes,
                    'trial_config': trial_config,
                    'trial_values': trial_values,
                    'n_plans': n_plans
                }

                def process(val):
                    if isinstance(val, dict):
                        return ''.join([c for c in str(val) if c.isalnum()])
                    else:
                        return str(val)

                config_str = '_'.join([process(v) for k, v in trial_values])
                file_name = '_'.join([state, config_str, 'plans', str(n_plans)]) + '.p'
                file_path = os.path.join(save_dir, file_name)
                pickle.dump(trial_results, open(file_path, 'wb'))



if __name__ == '__main__':
    center_selection_config = {
        'selection_method': 'random_method',
        'perturbation_scale': 1,
        'n_random_seeds': 1,
        'capacities': 'match',
        'capacity_weights': 'voronoi',
    }
    tree_config = {
        'n_samples': 5,
        'n_root_samples': 100,
        'parent_resample_trials': 3,
        'max_sample_tries': 30,
        'max_n_splits': 5,
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
        'state': 'NC',
        'n_districts': 13,
        'population_tolerance': .01,
        # 'optimization_data': 'gamo',
    }
    base_config = {**center_selection_config,
                   **tree_config,
                   **gurobi_config,
                   **pdp_config}
    experiment_config = {
        'name': 'nc_trials',
        'states': [pdp_config['state']],
        'trial_parameters': [
            [('boundary_type', 'baseline'),
             ('n_samples', 3),
             ('n_root_samples', 300)],
            [('boundary_type', 'county'),
             ('n_samples', 3),
             ('n_root_samples', 200)],
        ]
    }

    experiment = Experiment(base_config, experiment_config)
    experiment.run()

