import sys
sys.path.append('../gerrypy_julia')

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
from gurobipy import GRB
from data.load import *
from optimize.improvement import *
import pickle
import scipy as sp
import pandas as pd
from analyze.maj_black import majority_black, maj_black_logging_info
from analyze.feasibility import check_feasibility

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

def save_object(obj, filepath):
    with open(filepath, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def save_matrix(mtx, filepath):
    sparse_mtx = sp.sparse.csc_matrix(mtx)
    sp.sparse.save_npz(filepath, sparse_mtx)


class SHP:
    """
    SHP class to test different generation configurations.
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
        self.state_df = load_state_df(config['state'], config['year'], config['granularity'])
        if self.config['subregion'] is not None:
            self.state_df = self.state_df.loc[self.config['subregion']]
        self.save_path = self.get_save_path()
        if self.config['save_assignments']:
            self.assignments_file = 'assignments_%s.csv' % str(int(time.time()))
        self.n_cgus = len(self.state_df['GEOID'])
        self.assignments_df = pd.DataFrame()
        self.assignments_df['GEOID'] = self.state_df['GEOID']
    
    def run(self):
        if self.config['mode'] == 'partition' or self.config['mode'] == 'both':
            if self.config['save_assignments']:
                os.mkdir(self.save_path)
            if self.config['save_tree']:
                os.mkdir(os.path.join(self.save_path, 'tree'))
            if self.config['save_cdms']:
                os.mkdir(os.path.join(self.save_path, 'cdms'))
            self.shp()
        elif self.config['mode'] == 'master':
            self.save_path = self.get_save_path(time_str=self.config['tree_time_str'])
            self.assignments_df = self.master_solutions()
        else:
            raise ValueError('mode value is invalid')
        if self.config['mode'] == 'both' or self.config['mode'] == 'master':
            G = load_adjacency_graph(self.config['state'], self.config['year'], self.config['granularity'])
            if self.config['subregion'] is not None:
                G = G.subgraph(self.config['subregion'])
            check_feasibility(self.config, self.assignments_df, self.state_df, G)
            return self.assignments_df
    
    def get_save_path(self, time_str=str(int(time.time()))):
        """
        If time_str=None, creates directory save_path and returns it, and otherwise
        assembles it using time_str
        Args:
            time_str: (str or None) time string of desired save directory
        """
        return os.path.join(self.config['results_path'], 'results_%s' % time_str)
    
    def shp(self):
        """Performs all generation trials.

        Saves a file with the tree as well as a large number of ensemble level metrics."""
        if self.config['save_config']:
            with open(os.path.join(self.save_path, 'config.json'), 'w') as file:
                json.dump(self.config, file, indent=0)
        if self.config['verbose']:
            print('\n<><><><><><><><><><><> SHP Algorithm Start <><><><><><><><><><><>\n')
        if self.config['debug_file'] is not None:
            self.config['debug_file'] = open(os.path.join(self.save_path, self.config['debug_file']), 'a')
        debug_2 = False
        if self.config['debug_file_2'] is not None:
            debug_2 = True
            self.config['debug_file_2'] = open(os.path.join(self.save_path, self.config['debug_file_2']), 'a')
        cg = ColumnGenerator(self.config)
        generation_times = np.zeros((self.config['n_root_samples']))
        master_times = np.zeros((self.config['n_root_samples']))
        cdms = {}
        for root_partition_ix in range(self.config['n_root_samples']):
            generation_start_t = time.thread_time()
            sample_internal_nodes, sample_leaf_nodes = cg.generate_root_sample()
            generation_times[root_partition_ix] = time.thread_time() - generation_start_t
            if self.config['verbose']:
                print(f'\nGeneration time: {generation_times[root_partition_ix]}')
                print(f'Number of deleted nodes: {cg.n_deleted_nodes[root_partition_ix]}')
                print(f'Number of infeasible partitions: {cg.n_infeasible_partitions[root_partition_ix]}')
                print(f'Number of successful partitions: {cg.n_successful_partitions[root_partition_ix]}')
                print(f'Number of partitions that found an optimal solution: {cg.n_optimal_partitions[root_partition_ix]}')
                print(f'Number of partitions that reached their time limit: {cg.n_time_limit_partitions[root_partition_ix]}')
            if debug_2:
                maj_black_logging_info(cg.root,
                                       sample_internal_nodes, 
                                       sample_leaf_nodes,
                                       self.config['debug_file_2'],
                                       self.config['exact_partition_range'],
                                       self.config['maj_black_partition_IPs'],
                                       self.state_df)
            if self.config['save_tree']:
                save_object((sample_internal_nodes, sample_leaf_nodes), os.path.join(self.save_path, f'tree/root_sample_{root_partition_ix}.pkl'))
            if self.config['save_cdms']:
                cdm = make_cdm(sample_leaf_nodes, n_cgus=self.n_cgus)
                cdms[root_partition_ix] = cdm
                save_matrix(cdm, os.path.join(self.save_path, f'cdms/cdm_{root_partition_ix}.npz'))
            if self.config['mode'] == 'both':
                master_start_t = time.thread_time()
                root_partition = cg.root.children_ids[-1]
                n_maj_black_partitions = len(self.config['maj_black_partition_IPs'])
                for maj_black_partition_ix in range(n_maj_black_partitions):
                    col_title = 'District'+str(n_maj_black_partitions * root_partition_ix + maj_black_partition_ix)
                    maj_black_partitioned_nodes = {}
                    sample_trial_leaf_nodes = {}
                    for node in sample_leaf_nodes.values():
                        parent_id = node.parent_id
                        if parent_id == 0 and cg.root.n_districts in self.config['exact_partition_range']:
                            maj_black_partitioned_nodes[parent_id] = cg.root
                        elif sample_internal_nodes[parent_id].n_districts in self.config['exact_partition_range']:
                            maj_black_partitioned_nodes[parent_id] = sample_internal_nodes[parent_id]
                        else:
                            sample_trial_leaf_nodes[node.id] = node
                    for node in maj_black_partitioned_nodes.values():
                        ix = 0
                        for maj_black_partitions_ixs in node.partitions_used:
                            for mbp_ix in maj_black_partitions_ixs:
                                if mbp_ix == maj_black_partition_ix:
                                    sample_trial_leaf_nodes.update({id: sample_leaf_nodes[id] for id in node.children_ids[ix]})
                                ix += 1
                    self.assignments_df[col_title] = self.get_solution_dp(cg.root, 
                                                                        sample_internal_nodes, 
                                                                        sample_trial_leaf_nodes,
                                                                        root_partition_ix, 
                                                                        root_partition)
                    if self.config['save_assignments']:
                        self.assignments_df.to_csv(os.path.join(self.save_path, self.assignments_file), index=False)
                master_times[root_partition_ix] = time.thread_time() - master_start_t
                if self.config['verbose']:
                    print(f'Master solutions time: {master_times[root_partition_ix]}')
        if self.config['save_tree']:
            save_object(cg.root, os.path.join(self.save_path, f'tree/root.pkl'))
        if self.config['debug_file'] is not None:
            concluding_str = '\n-------------------------------------------\n'
            concluding_str += f'Number of leaf nodes: {len(cg.leaf_nodes)}\n'
            concluding_str += f'Number of nodes: {1+len(cg.internal_nodes)+len(cg.leaf_nodes)}'
            self.config['debug_file'].write(concluding_str)
            self.config['debug_file'].close()
        if self.config['debug_file_2'] is not None:
            self.config['debug_file_2'].close()
        if self.config['verbose']:    
            print('\n<><><><><><><><><><><> SHP Algorithm End <><><><><><><><><><><>\n')
            
            print(f'Number of leaf_nodes: {len(cg.leaf_nodes)}')
            print(f'Number of nodes: {1+len(cg.internal_nodes)+len(cg.leaf_nodes)}\n')
            print(f'Tree generation times: {generation_times.astype("int32")}')
            print(f'--> Total generation time: {np.sum(generation_times):0.2f}\n')
            print(f'Master solutions times: {np.round(master_times, 2)}')
            print(f'--> Total master solutions time: {np.sum(master_times):0.2f}\n')
            print(f'Number of districtings = {number_of_districtings(cg.leaf_nodes, cg.internal_nodes)}')
            print(f'Numbers of deleted nodes: {cg.n_deleted_nodes}')
            print(f'Numbers of infeasible partitions: {cg.n_infeasible_partitions}')
            print(f'Numbers of successful partitions: {cg.n_successful_partitions}')
            print(f'Numbers of partitions that found an optimal solution: {cg.n_optimal_partitions}')
            print(f'Numbers of partitions that reached their time limit: {cg.n_time_limit_partitions}')
            print(f'Number of failed root samples: {cg.failed_root_samples}')
        if self.config['verbose'] and self.config['mode'] == 'both':
            n_maj_black_partitions = len(self.config['maj_black_partition_IPs'])
            nums_maj_black = majority_black(self.assignments_df, self.state_df, self.config['n_districts'], num_plans=self.config['n_root_samples'] * n_maj_black_partitions).sum(axis=1)
            for maj_black_partition_ix in range(n_maj_black_partitions):
                print(f'Numbers of maj black districts using {self.config["maj_black_partition_IPs"][maj_black_partition_ix]}: {nums_maj_black[maj_black_partition_ix::n_maj_black_partitions]}')

    def master_solutions(self):
        try:
            tree = load_tree(self.save_path)
            root = tree[-1]
        except:
            raise FileExistsError('tree_time_str is invalid or given directory has no tree files')
        try:
            cdms = load_cdms(self.save_path)
        except:
            cdms = {}
            for root_partition_ix in range(self.config['n_root_samples']):
                cdms[root_partition_ix] = make_cdm(tree[root_partition_ix][1], n_cgus=self.n_cgus)
        master_times = np.zeros((self.config['n_root_samples']))
        print(len(root.children_ids))
        for root_partition_ix, root_partition in enumerate(root.children_ids):
            master_start_t = time.thread_time()
            col_title = 'District'+str(root_partition_ix)
            self.assignments_df[col_title] = self.get_solution_dp(root, 
                                                                tree[root_partition_ix][0], 
                                                                tree[root_partition_ix][1], 
                                                                cdms[root_partition_ix], 
                                                                root_partition_ix, 
                                                                root_partition)
            if self.config['save_assignments']:
                self.assignments_df.to_csv(os.path.join(self.save_path, self.assignments_file), index=False)
            master_times[root_partition_ix] = time.thread_time() - master_start_t
        print(f'Master solutions times: {np.round(master_times, 2)}')
        print(f'Total master solutions time: {np.sum(master_times):0.2f}')
        
    
    def get_solution_dp(self, root, internal_nodes, leaf_nodes, root_partition_ix, root_partition):
        if self.config['verbose']:
            print('\n-------------Solving master for root sample number %d-------------\n' % root_partition_ix)
        nodes = {**internal_nodes, **leaf_nodes}
        dp_queue = []
        parent_layer = [root]
        children_layer = [internal_nodes[id] for id in root_partition if id in nodes and nodes[id].n_districts != 1]
        #children_layer = [internal_nodes[id] for partition in root.children_ids for id in partition if nodes[id].n_districts != 1]
        while len(children_layer) > 0:
            dp_queue += children_layer
            parent_layer = children_layer
            children_layer = [nodes[id] for node in parent_layer for partition in node.children_ids for id in partition if id in nodes and nodes[id].n_districts != 1]
        
        for node in leaf_nodes.values():
            total_bvap = sum(self.state_df.loc[index, 'BVAP'] for index in node.area)
            total_vap = sum(self.state_df.loc[index, 'VAP'] for index in node.area)
            node.best_subtree = ([node.id], total_bvap / total_vap > 0.5)
        
        for i in range(len(dp_queue)-1, -1, -1):
            current_node = dp_queue[i]
            best_subtree_ids = []
            best_subtree_score = -1 
            for partition in current_node.children_ids:
                try:
                    sample_score = sum(nodes[id].best_subtree[1] for id in partition)
                    if best_subtree_score < sample_score:
                        best_subtree_ids = [subtree_id for id in partition for subtree_id in nodes[id].best_subtree[0]]
                        best_subtree_score = sample_score
                except:
                    continue
            current_node.best_subtree = (best_subtree_ids, best_subtree_score)
        
        id_to_ix = {node_id: node_ix for node_ix, node_id in enumerate(leaf_nodes)}
        #solution_ixs = [id_to_ix[subtree_id] for node_id in root_partition for subtree_id in nodes[node_id].best_subtree[0]]
        solution_ixs = [subtree_id for node_id in root_partition for subtree_id in nodes[node_id].best_subtree[0]]
        assignment_ser = pd.Series(index=self.assignments_df.index)
        for district_ix in range(len(solution_ixs)):
            assignment_ser.loc[leaf_nodes[solution_ixs[district_ix]].area] = district_ix
        return assignment_ser
        #return self.selected_districts(solution_ixs, cdm)
    
    def master_solutions_nonlinear(self, args=None):
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
        if args is None:
            save_path = self.get_save_path(time_str=self.config['tree_time_str'])
            try:
                tree = load_object(os.path.join(save_path, 'tree.pkl'))
            except:
                raise FileExistsError('tree_time_str is invalid')
            state_df, G, lengths, edge_dists = load_opt_data(self.config['state'], self.config['year'], self.config['granularity'])
            n_census_shapes = len(state_df['GEOID'])
            cdm = make_cdm(tree[1], n_cgus=n_census_shapes)
        else:
            tree = args['tree']
            cdm = args['cdm']
            state_df = args['state_df']
            save_path = args['save_path']
        internal_nodes = tree[0]
        leaf_nodes = tree[1]
        #district_df_of_tree_dir(save_path)
        #maj_min=majority_minority(cdm, state_df)
        #print(maj_min)
        #print(sum(maj_min))
        cost_coeffs = majority_black(cdm, state_df)
        maj_min = np.zeros((len(cdm.T)))
        bb = np.zeros((len(cost_coeffs)))
        
        root_map, ix_to_id = make_root_partition_to_leaf_map(leaf_nodes, internal_nodes)
        sol_dict = {}
        initial_t = time.thread_time()
        print('\n-------------Starting master problems-------------\n')
        for partition_ix, leaf_slice in root_map.items():
            '''
            relax_start_t = time.thread_time()
            model_relaxed, dvars_relaxed = make_master(self.config['n_districts'], 
                                        cdm[:, leaf_slice], 
                                        cost_coeffs[leaf_slice], 
                                        maj_min[leaf_slice], 
                                        bb[leaf_slice], 
                                        callback_time_interval=self.config['callback_time_interval'],
                                        relax=True)
            relax_construction_t = time.thread_time()
            model_relaxed.optimize()
            relax_solve_t = time.thread_time()
            print(f"\nRelaxed construction time: {relax_construction_t-relax_start_t}")
            print(f"Relaxed solve time: {relax_solve_t-relax_construction_t}")
            df = pd.DataFrame()
            xs = [v.X for v in dvars_relaxed.values()]
            df['X'] = xs
            print(f'Number of nonzero elements in xs = {sum([x != 0 for x in xs])}')
            #print([v.X for v in dvars_relaxed.values()])
            probabilities = [v.X * (1.0 >= v.X >= 0.5) + 1.0 * (v.X > 1.0) for v in dvars_relaxed.values()]
            df['probs'] = probabilities
            print(f'Number of nonzero elements in probabilities = {sum([x != 0 for x in probabilities])}')
            included_districts = np.random.binomial(n=1, p=probabilities)
            df['included'] = included_districts
            df.to_csv(os.path.join(save_path, 'test.csv'))
            print(f'Number of nonzero elements in included_districts = {sum([x != 0 for x in included_districts])}')
            #print(included_districts)
            '''
            start_t = time.thread_time()
            model, dvars = make_master(self.config['n_districts'], 
                                        cdm[:, leaf_slice], 
                                        cost_coeffs[leaf_slice], 
                                        maj_min[leaf_slice], 
                                        bb[leaf_slice], 
                                        callback_time_interval=self.config['callback_time_interval'])
            construction_t = time.thread_time()

            #model.Params.LogToConsole = 0
            model.Params.MIPGapAbs = 1e-4
            model.Params.TimeLimit = len(leaf_slice) / 10
            print(f'MIPFocus: {model.Params.MIPFocus}')
            print(f'Threads: {model.Params.Threads}')
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
            assignments_df = self.export_solutions(sol_dict, state_df, cdm, sol_tree, internal_nodes)
            results_save_name = 'assignments_%s.csv' % str(int(time.time()))
            assignments_df.to_csv(os.path.join(save_path, results_save_name), index=False)
        print(f'\nTotal time for master problems: {time.thread_time()-initial_t}')
    
    def selected_districts(self, solution_ixs, cdm):
        selected_districts = np.zeros(self.n_cgus, dtype=int)
        for district in range(len(solution_ixs)):
            for cgu, cgu_in_district in enumerate(cdm.T[solution_ixs[district]]):
                if cgu_in_district: selected_districts[cgu]=district
        return selected_districts
        
    def export_solutions(self, sol_dict, state_df, cdm, sol_tree, internal_nodes):
        """
        Creates a dataframe with each block matched to a district based on the IP solution
        Args:
            solutions: (dict) of solutions outputted by IP
            state_df: (pd DataFrame) with state data
            cdm: (np.array) n x d matrix where a_ij = 1 when block i appears in district j.
            sol_tree: list of SHPNodes representing the leaf nodes and their ancestors #TODO make this work for multiple trials

        Returns: Dataframe mapping GEOID to district assignments

        """
        assignments_df = pd.DataFrame()
        assignments_df['GEOID'] = state_df['GEOID']

        print('begin export solutions')

        for sol_idx in range(len(sol_dict)):
            solution_ixs = sol_dict[sol_idx]['solution_ixs']
            col_title = 'District'+str(sol_idx)
            assignments_df[col_title] = self.selected_districts(solution_ixs, cdm)

        #add a column parent which tells us this block's parent's center IF it is a center for the final tree, or -1 if it is a root for the final tree
        # assignments_df['Parent'] = np.nan
        # assignments_df['ID'] = np.nan
        # for node in sol_tree:
        #     if node.parent_id is not None:
        #         assignments_df.loc[node.center, 'ID']=node.id
        #         parent_center=internal_nodes[node.parent_id].center
        #         assignments_df.loc[node.center, 'Parent']=parent_center
        #         if parent_center is None:
        #             assignments_df.loc[node.center, 'Parent']=-1

        return assignments_df