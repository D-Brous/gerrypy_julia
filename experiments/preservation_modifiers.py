import copy
import time
import numpy as np
import pandas as pd
from gerrypy import constants
from gerrypy.analyze.districts import *
from gerrypy.optimize.generate import ColumnGenerator
from gerrypy.rules.political_boundaries.preservation_cost_function import *
from gerrypy.rules.political_boundaries.preservation_metrics import *


class Experiment:
    def __init__(self, default_config, modifiers_configs, save_path, c=200, alpha=0.6):
        self.default_config = default_config
        self.modifiers_configs = modifiers_configs
        self.c = c
        self.alpha = alpha
        self.county_tract_matrix = None
        self.save_path = os.path.join(constants.RESULTS_PATH, save_path)
        os.makedirs(self.save_path, exist_ok=True)

    def k_to_w(self, k):
        return max(1, np.round(self.c / (k ** self.alpha)).astype(int)), 2

    def parse_results(self, cg):
        internal_nodes = cg.internal_nodes
        leaf_nodes = cg.leaf_nodes
        
        b_county_m = self.county_tract_matrix
        b_district_m = make_bdm(leaf_nodes, len(cg.state_df))
        b_region_m = make_bdm(internal_nodes[1:], len(cg.state_df))

        # Preservation metrics
        population = cg.state_df.population.values
        preservation_metrics = {
            'district_splits': splits(b_district_m, b_county_m).mean(),
            'district_pieces': pieces(b_district_m, b_county_m).mean(),
            'district_entropy': boundary_entropy(b_district_m, b_county_m, population).mean(),
            'region_splits': splits(b_region_m, b_county_m).mean(),
            'region_pieces': pieces(b_region_m, b_county_m).mean(),
            'region_entropy': boundary_entropy(b_region_m, b_county_m, population).mean()
        }

        ensemble_metrics = generation_metrics(cg)

        return {**preservation_metrics, **ensemble_metrics}

    def run(self, states=None):
        results = {}
        if states is None:
            states = [state for state in constants.seats]
        for state in states:
            results[state] = {}
            if constants.seats[state]['house'] < 2:
                continue
            config = copy.deepcopy(self.default_config)
            k = constants.seats[state]['house']
            w_root, w = self.k_to_w(k)
            self.county_tract_matrix = make_block_boundary_matrix(load_tract_shapes(state))

            config['state'] = state
            config['n_districts'] = k
            config['n_root_samples'] = w_root
            config['n_samples'] = w
    
            baseline_cg = ColumnGenerator(config)
            baseline_cg.generate()

            results[state]['baseline'] = self.parse_results(baseline_cg)
            for cm_fn in self.modifiers_configs:
                cg = ColumnGenerator(config, cm_fn)
                cg.generate()

                results[state][cm_fn] = self.parse_results(cg)

        result_df = pd.DataFrame({
            (state, method): metrics for state, method_dict in results.items()
            for method, metrics in method_dict.items()
        }).T.astype(float)

        save_name = f'preservation_metrics_{len(self.modifiers_configs)}_' \
                    f'{self.c}_{self.alpha}_{round(time.time())}.csv'
        result_df.to_csv(os.path.join(self.save_path, save_name))

        return results


if __name__ == '__main__':
    center_selection_config = {
        'selection_method': 'random_iterative',  # one of
        'perturbation_scale': 0,
        'n_random_seeds': 0,
        'capacities': 'match',
        'capacity_weights': 'voronoi',
    }
    tree_config = {
        'max_sample_tries': 25,
        'n_samples': 3,
        'n_root_samples': 3,
        'max_n_splits': 5,
        'min_n_splits': 2,
        'max_split_population_difference': 1.5,
        'event_logging': False,
        'verbose': False,
    }
    gurobi_config = {
        'IP_gap_tol': 1e-5,
        'IP_timeout': 10,
    }
    pdp_config = {
        'state': 'IA',
        'n_districts': 4,
        'population_tolerance': .01,
    }
    base_config = {**center_selection_config,
                   **tree_config,
                   **gurobi_config,
                   **pdp_config}

    modifier_configs = ['weighted_random_order_expansion',
                        'random_topological_ordering',
                        'bfs_translation',
                        'random_translation']

    exp = Experiment(base_config, modifier_configs, 'preservation_metrics', 200, 0.8)
    exp.run(states=['IA', 'NC'])
