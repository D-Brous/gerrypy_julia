import copy
import time
import numpy as np
import pandas as pd
from gerrypy import constants
from gerrypy.analyze.districts import *
from gerrypy.optimize.generate_fault_tolerant import ColumnGenerator
from gerrypy.rules.political_boundaries.preservation_constraint import *
from gerrypy.rules.political_boundaries.preservation_metrics import *


class Experiment:
    def __init__(self, default_config, discount_factors, save_path, c=200, alpha=0.6):
        self.default_config = default_config
        self.discount_factors = discount_factors
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
        b_region_m = make_bdm(internal_nodes, len(cg.state_df))

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

            for factor in self.discount_factors:
                config['county_discount_factor'] = factor
                cg = ColumnGenerator(config)
                cg.generate()

                results[state][factor] = self.parse_results(cg)

            result_df = pd.DataFrame({
                (state, discount_factor): metrics
                for discount_factor, metrics in results[state].items()
            }).T.astype(float)

            save_name = f'{state}_lambda_{len(self.discount_factors)}_' \
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
    }
    base_config = {**center_selection_config,
                   **tree_config,
                   **gurobi_config,
                   **pdp_config}

    discount_factors = np.arange(0, 1.1, 0.1)
    exp = Experiment(base_config, discount_factors, 'lagrange_discount_factors')
    exp.run(states=['FL'])
