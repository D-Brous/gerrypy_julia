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
#from data.data2020.load import *
from gurobipy import GRB
from data.load import *
from optimize.improvement import *

def callback(model, where):
    if where == GRB.Callback.MIP:
        time = model.cbGet(GRB.Callback.RUNTIME)
        if model._last_callback_time + model._callback_time_interval <= time:
            model._last_callback_time = time
            num_feasible = model.cbGet(GRB.Callback.MIP_SOLCNT)
            #num_unexplored = model.cbGet(GRB.Callback.MIP_NODLFT)
            best_obj = model.cbGet(GRB.Callback.MIP_OBJBST)
            best_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
            phase = model.cbGet(GRB.Callback.MIP_PHASE)
            print(f'{time} s: In phase {phase}, {num_feasible} feasible sols found, curr best obj is {best_obj}, curr best bound is {best_bound}\n')

class Experiment:
    """
    Experiment class to test different generation configurations.
    """
    def __init__(self, config):
        """
        Initialized with configuration dict
        Args:
            config: (dict) the following are the required keys
                state: (str) 2 letter abbreviation
                n_districts: (int)
                population_tolerance: (float) ideal population +/- factor epsilon
                max_sample_tries: (int) number of attempts at each node
                n_samples: (int) the fan-out split width
                n_root_samples: (int) the split width of the root node w
                max_n_splits: (int) max split size z_max
                min_n_splits: (int) min split size z_min
                max_split_population_difference: (float) maximum
                    capacity difference between 2 sibling nodes
                event_logging: (bool) log events for visualization
                verbose: (bool) print runtime information
                selection_method: (str) seed selection method to use
                perturbation_scale: (float) pareto distribution parameter
                n_random_seeds: (int) number of fixed seeds in seed selection
                capacities: (str) style of capacity matching/computing
                capacity_weights: (str) 'voronoi' or 'fractional'
                IP_gap_tol: (float) partition IP gap tolerance
                IP_timeout: (float) maximum seconds to spend solving IP
                callback_time_interval: (int) interval in seconds between printouts to console with some information during master problem(s) or
                                        (NoneType) None if we don't want master problem console printouts
                'granularity': (str) level of granularity of the census data on which to operate
                

        """
        self.config = config
        
    def run(self):
        """Performs all generation trials.

        Saves a file with the tree as well as a large number of ensemble level metrics."""
        name = 'la_house'
        time_str = str(int(time.time()))
        experiment_dir = '%s_results_%s' % (name, time_str)
        save_dir = os.path.join(self.config['results_dir'], experiment_dir)
        os.mkdir(save_dir)  

        with open(os.path.join(save_dir, 'config.json'), 'w') as file:
            #file.write('\n' + experiment_dir + ':\n\n')
            json.dump(self.config, file, indent=0)
            #for key, value in self.config.items():
            #    file.write('%s: %s\n' % (key, value))
        
        print('\n-------------Starting tree generation-------------\n')
        if self.config['debug_file'] is not None:
            with open(os.path.join(save_dir, self.config['debug_file']), 'a') as debug_file:
                self.config['debug_file'] = debug_file
                cg = ColumnGenerator(self.config)
                n_units = len(cg.state_df['GEOID'])
                generation_start_t = time.thread_time()
                cg.generate()
                generation_t = time.thread_time() - generation_start_t
                concluding_str = '\n-------------------------------------------\n'
                concluding_str += f'Number of leaf nodes: {len(cg.leaf_nodes)}\n'
                concluding_str += f'Number of nodes: {1+len(cg.internal_nodes)+len(cg.leaf_nodes)}'
                self.config['debug_file'].write(concluding_str)
        else:
            cg = ColumnGenerator(self.config)
            n_units = len(cg.state_df['GEOID'])
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
        print(f"\nTree generation time: {trial_results['generation_time']}")
        print(f'Number of districtings = {number_of_districtings(cg.leaf_nodes, cg.internal_nodes)}')

        def process(val):
            if isinstance(val, dict):
                return ''.join([c for c in str(val) if c.isalnum()])
            else:
                return str(val)

        save_name = 'trial_results.npy'
        #save_name = '_'.join(['la_house', str(int(time.time()))]) + '.npy'
        csv_save_name = 'bdm.csv'
        #csv_save_name = '_'.join(['la_house', str(int(time.time()))]) + '.csv'
        #print(type(trial_results))
        np.save(os.path.join(save_dir, save_name), trial_results)

        #print(cg.internal_nodes)

        bdm = make_bdm(cg.leaf_nodes, n_blocks=n_units)
        #print(bdm[0, :]-bdm[0, :])
        bdm_df = pd.DataFrame(bdm)
        bdm_df.to_csv(os.path.join(save_dir, csv_save_name), index=False)
        #district_df_of_tree_dir(save_dir)

        state_df, G, lengths, edge_dists = load_opt_data('LA', self.config['granularity'])
        self.state_df = state_df
        #maj_min=majority_minority(bdm, state_df)
        #print(maj_min)
        #print(sum(maj_min))
        maj_min = np.zeros((len(bdm.T)))

        #county_split_coefficients(bdm,state_df,G)

        print('\n-------------Starting master problems-------------\n')
        self.master_solutions(bdm, cg.leaf_nodes, cg.internal_nodes, state_df, lengths, G, maj_min, save_dir) 
        #print(sol_dict)

    def master_solutions(self, bdm, leaf_nodes, internal_nodes, state_df, lengths, G, maj_min, save_dir):
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
        """
        #bdm = make_bdm(leaf_nodes) #TODO don't need to make the bdm twice
        #cost_coeffs = compactness_coefficients(bdm, state_df, lengths)
        #cost_coeffs = county_split_coefficients(bdm, state_df,G)
        #cost_coeffs=np.array(cost_coeffs)
        #cost_coeffs = compactness_coefficients(bdm, state_df, lengths)
        cost_coeffs = majority_black(bdm, state_df)
        bb = np.zeros((len(cost_coeffs)))
        root_map, ix_to_id = make_root_partition_to_leaf_map(leaf_nodes, internal_nodes)
        #print(root_map)
        sol_dict = {}

        for partition_ix, leaf_slice in root_map.items():
            start_t = time.thread_time()
            model, dvars = make_master(self.config['n_districts'], 
                                        bdm[:, leaf_slice], 
                                        cost_coeffs[leaf_slice], 
                                        maj_min[leaf_slice], 
                                        bb[leaf_slice], 
                                        callback_time_interval=self.config['callback_time_interval'])
            construction_t = time.thread_time()

            model.Params.LogToConsole = 0
            model.Params.MIPGapAbs = 1e-4
            model.Params.TimeLimit = len(leaf_slice) / 10
            model.Params.ImproveStartTime = 0
            if self.config['callback_time_interval'] is not None:
                model.optimize(callback=callback)
            else:
                model.optimize()
            status = model.status
            if status == GRB.INF_OR_UNBD:
                print('model is infeasible or unbounded')
                model.reset()
                if self.config['callback_time_interval'] is not None:
                    model.optimize(callback=callback)
                else:
                    model.optimize()
                status = model.status
            if status == GRB.INFEASIBLE:
                print('computing IIS')
                model.computeIIS()
                model.write('model.ilp')
                print('done with IIS')
            else:
                print(f'Model status = {status}')
            opt_cols = [j for j, v in dvars.items() if v.X > .5]
            solve_t = time.thread_time()

            sol_dict[partition_ix] = {
                'construction_time': construction_t - start_t,
                'solve_time': solve_t - construction_t,
                'n_leaves': len(leaf_slice),
                'solution_ixs': root_map[partition_ix][opt_cols],
                'optimal_objective': cost_coeffs[leaf_slice][opt_cols]
            }
            print(f"\nConstruction time: {sol_dict[partition_ix]['construction_time']}")
            print(f"Solve time: {sol_dict[partition_ix]['solve_time']}")
            #print(opt_cols)
            #print(sol_dict[partition_ix]['solution_ixs'])
            #print(maj_min[sol_dict[partition_ix]['solution_ixs']])
            #constraint = model.getConstrByName("majorityMinority")
            #print(constraint.slack)

            sol_tree = get_solution_tree(leaf_nodes, internal_nodes, ix_to_id, sol_dict[partition_ix]['solution_ixs'])
            #TODO fix sol_tree so that it gives us the right data structure and then write a function to print this to an output file, probably json

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
            solutions_df = self.export_solutions(sol_dict, state_df, bdm, sol_tree, internal_nodes)
            results_save_name = 'assignments.csv'
            solutions_df.to_csv(os.path.join(save_dir, results_save_name), index=False)

    def export_solutions(self, sol_dict, state_df, bdm, sol_tree, internal_nodes):
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

        for sol_idx in range(len(sol_dict)):
            solution_ixs = sol_dict[sol_idx]['solution_ixs']
            for i in range(len(solution_ixs)):
                for index, j in enumerate(bdm.T[solution_ixs[i]]):
                    if j==1: selected_dists[index]=i
            #print(selected_dists)
            col_title = 'District'+str(sol_idx)
            solutions_df[col_title] = selected_dists

        #add a column parent which tells us this block's parent's center IF it is a center for the final tree, or -1 if it is a root for the final tree
        # solutions_df['Parent'] = np.nan
        # solutions_df['ID'] = np.nan
        # for node in sol_tree:
        #     if node.parent_id is not None:
        #         solutions_df.loc[node.center, 'ID']=node.id
        #         parent_center=internal_nodes[node.parent_id].center
        #         solutions_df.loc[node.center, 'Parent']=parent_center
        #         if parent_center is None:
        #             solutions_df.loc[node.center, 'Parent']=-1

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
        'parent_resample_trials': 5, #5 before #TODO 5-10
        'max_sample_tries': 5, # 25 before
        'n_samples': 3, #Should be 3-5 #TODO 10-20
        'n_root_samples': 1,
        'max_n_splits': 2,
        'min_n_splits': 2, 
        'max_split_population_difference': 1.5,
        'granularity': 'block_group',
        'event_logging': False,
        'verbose': False,
        'debug_file': 'debug_file.txt'
    }
    gurobi_config = {
        'IP_gap_tol': 1e-3,
        'IP_timeout': 10,
        'callback_time_interval': 10
    }
    pdp_config = {
        'state': 'LA',
        'n_districts': 105,
        'population_tolerance': .045,
        'required_mm': 0, #TODO if this is 0, partition stage works
        #'population_tolerance': population_tolerance()*12,
        'results_dir': constants.LOUISIANA_HOUSE_RESULTS_PATH,
        'cost_coeffs': 'maj_black',
        'partition_IP': 'make_partition_IP'
    }
    config = {**center_selection_config,
                   **tree_config,
                   **gurobi_config,
                   **pdp_config}

    test()
    experiment = Experiment(config)
    experiment.run()