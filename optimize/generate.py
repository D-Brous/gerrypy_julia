import sys
sys.path.append('../gerrypy_julia')

import os
import time
import json
import numpy as np
import networkx as nx
from collections import OrderedDict
import constants as consts
from analyze.districts import *
from data.load import load_opt_data
from optimize.center_selection import *
from optimize.partition import *
from optimize.tree import SHPNode
from rules.political_boundaries.preservation_constraint import *
from rules.political_boundaries.preservation_cost_function import *

class DefaultCostFunction:
    def __init__(self, lengths):
        self.lengths = lengths

    def get_costs(self, area_df, centers):
        population = area_df.population.values
        index = list(area_df.index)
        costs = self.lengths[np.ix_(centers, index)] * ((population / 1000) + 1)

        costs **= (1 + random.random())
        return {center: {index[bix]: cost for bix, cost in enumerate(costs[cix])}
                for cix, center in enumerate(centers)}


def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

def flatten2(lis_of_lis):
    return [element for lis in lis_of_lis for element in lis]

class ColumnGenerator:
    """
    Generates columns with the Stochastic Hierarchical Paritioning algorithm.
    Maintains samples tree and logging data.
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

        """
        state_abbrev = config['state']
        #print("")
        #if optimization_data_location=='':
        #    print("it's empty string")
        #print("")
        state_df, G, lengths, edge_dists = load_opt_data(state_abbrev, config['granularity'])
        lengths /= 1000

        self.state_abbrev = state_abbrev

        ideal_pop = state_df['population'].values.sum() / config['n_districts']
        max_pop_variation = ideal_pop * config['population_tolerance']

        config['max_pop_variation'] = max_pop_variation
        config['ideal_pop'] = ideal_pop

        self.config = config
        self.G = G
        self.state_df = state_df
        self.lengths = lengths
        self.edge_dists = edge_dists

        self.sample_queue = []
        self.internal_nodes = {}
        self.leaf_nodes = {}
        self.max_id = 0
        self.root = None

        self.failed_regions = []
        self.failed_root_samples = 0
        self.n_infeasible_partitions = 0
        self.n_successful_partitions = 0

        self.event_list = []

        if self.config.get('boundary_type') == 'county':
            tracts = load_tract_shapes(state_abbrev, custom_path=config.get('custom_shape_path', ''))
            tracts = tracts.rename(columns={'COUNTYFP20': 'COUNTYFP'})
            self.model_factory = BoundaryPreservationConstraints(
                tracts, config.get('county_discount_factor', 0.5))

        self.cost_fn = DefaultCostFunction(lengths)
        #random.seed(0)
        #np.random.seed(0)

    def _assign_id(self):
        self.max_id += 1
        return self.max_id

    def retry_sample(self, problem_node, sample_internal_nodes, sample_leaf_nodes, debug=False):
        def get_descendents(node_id):
            if self.config['verbose']:
                print('executing get_descendents()')
            direct_descendents = [child for partition in
                                  sample_internal_nodes[node_id].children_ids
                                  for child in partition]
            indirect_descendants = [get_descendents(child) for child in direct_descendents
                                    if (child in sample_internal_nodes)]
            return direct_descendents + flatten2(indirect_descendants)
        if problem_node.id == 0:
            raise RuntimeError('Root partition failed')
        if problem_node.parent_id == 0:
            raise RuntimeError('Root partition region not subdivisible')
        parent = sample_internal_nodes[problem_node.parent_id]

        parent.infeasible_children += 1
        if parent.infeasible_children > self.config['parent_resample_trials']:
            # Failure couldn't be corrected
            if self.config['verbose']:
                print(problem_node.area)
            return self.retry_sample(parent, sample_internal_nodes, sample_leaf_nodes, debug=debug)
            #raise RuntimeError('Unable to sample tree')

        branch_ix, branch_ids = parent.get_branch(problem_node.id)
        nodes_to_delete = set()
        for node_id in branch_ids:
            nodes_to_delete.add(node_id)
            if node_id in sample_internal_nodes:
                for child_id in get_descendents(node_id):
                    nodes_to_delete.add(child_id)

        for node_id in nodes_to_delete:
            if node_id in sample_leaf_nodes:
                if debug:
                    self.config['debug_file'].write(f'{node_id}, ')
                del sample_leaf_nodes[node_id]
            elif node_id in sample_internal_nodes:
                if debug:
                    self.config['debug_file'].write(f'{node_id}, ')
                del sample_internal_nodes[node_id]

        parent.delete_branch(branch_ix)
        self.sample_queue = [parent] + [n for n in self.sample_queue if n.id not in nodes_to_delete]
        if self.config['verbose']:
            print(f'Pruned branch from {parent.id} with {len(nodes_to_delete)}'
                  f' deleted descendents.')

    def generate(self):
        """
        Main method for running the generation process.

        Returns: None

        """
        completed_root_samples = 0
        n_root_samples = self.config['n_root_samples']

        root = SHPNode(self.config['n_districts'],
                       list(self.state_df.index),
                       0, is_root=True)
        self.root = root
        self.internal_nodes[root.id] = root
        #this root is the entire state_df

        debug = False
        if self.config['debug_file'] is not None:
            debug = True
            self.config['debug_file'].write(f'Logged root node 0 at time {time.thread_time()}\n')

        while completed_root_samples < n_root_samples:
            # For each root partition, we attempt to populate the sample tree
            # If failure in particular root, prune all work from that root
            # partition. If successful, commit subtree to whole tree.
            self.sample_queue = [root]
            sample_leaf_nodes = {}
            sample_internal_nodes = {}
            try:
                print('Root sample number', completed_root_samples)
                while len(self.sample_queue) > 0:
                    #node = self.sample_queue.pop() #DFS
                    node = self.sample_queue.pop(0) #BFS
                    child_samples = self.sample_node(node, debug)
                    if len(child_samples) == 0:  # Failure detected
                        self.failed_regions.append(node.area)
                        # Try to correct failure
                        
                        if debug:
                            self.config['debug_file'].write(f'Failed split of node {node.id}.\n    Deleted nodes:\n    [')
                        self.retry_sample(node, sample_internal_nodes, sample_leaf_nodes, debug=debug)
                        if debug:
                            #num_deleted_nodes = num_completed_samples - len(sample_internal_nodes)
                            #self.config['debug_file'].write(f'Failed split: {num_deleted_nodes} nodes deleted. Remaining sample queue:\n[')
                            self.config['debug_file'].write(f']\n    Remaining sample queue:\n    [')
                            for n in self.sample_queue:
                                self.config['debug_file'].write(f'{n.id}, ')
                            self.config['debug_file'].write(']\n')
                            num_nodes = len(sample_internal_nodes) + len(sample_leaf_nodes) + len(self.sample_queue)
                            self.config['debug_file'].write(f'    Total number of nodes in tree after deletion: {num_nodes}\n')
                        continue
                    for child in child_samples:
                        if child.n_districts == 1:
                            sample_leaf_nodes[child.id] = child
                            if debug:
                                self.config['debug_file'].write(f'Logged leaf node {child.id} at time {time.thread_time()}\n')
                        else:
                            self.sample_queue.append(child)
                            #if debug:
                            #    self.config['debug_file'].write(f'{child.id}, internal, {child.n_districts}, {len(child.area)}, {time.thread_time()}\n')
                    if not node.is_root:
                        sample_internal_nodes[node.id] = node
                        if debug:
                            self.config['debug_file'].write(f'Logged internal node {node.id} at time {time.thread_time()}\n')
                self.internal_nodes.update(sample_internal_nodes)
                self.leaf_nodes.update(sample_leaf_nodes)
                completed_root_samples += 1
            except RuntimeError as error:  # Stop trying on root partition
                print(f'Root sample failed: {str(error)}')
                if debug:
                    self.config['debug_file'].write('\n-------------------------------Root sample failed-------------------------------\n')
                self.root.children_ids = self.root.children_ids[:-1]
                self.root.partition_times = self.root.partition_times[:-1]
                self.failed_root_samples += 1

    def sample_node(self, node, debug=False):
        """
        Generate children partitions of a region contained by [node].

        Args:
            node: (SHPnode) Node to be samples

        Returns: A flattened list of child regions from multiple partitions.

        """
        initial_time = time.thread_time()
        area_df = self.state_df.loc[node.area]
        samples = []
        n_trials = 0
        n_samples = 1 if node.is_root else self.config['n_samples']
        if not isinstance(n_samples, int):
            n_samples = int((n_samples // 1) + (random.random() < n_samples % 1))
        while len(samples) < n_samples and n_trials < self.config['max_sample_tries']:
            partition_start_t = time.time()
            child_nodes = self.make_partition(area_df, node) 
            #print(child_nodes)
            partition_end_t = time.time()
            if child_nodes:
                self.n_successful_partitions += 1
                samples.append(child_nodes)
                node.children_ids.append([child.id for child in child_nodes])
                node.partition_times.append(partition_end_t - partition_start_t)
            else:
                self.n_infeasible_partitions += 1
                node.n_infeasible_samples += 1
            n_trials += 1
        #if len(node.partition_times) > 0:
        #    print(sum(node.partition_times)/len(node.partition_times))
        children = [node for sample in samples for node in sample]
        if debug:
            self.config['debug_file'].write(f'Split node {node.id} with {n_trials} trials in {time.thread_time()-initial_time} sec. Children are:\n')
            for child in children:
                if child.n_districts == 1:
                    self.config['debug_file'].write(f'    {child.id}, leaf, {child.n_districts}, {len(child.area)}\n')
                else:
                    self.config['debug_file'].write(f'    {child.id}, internal, {child.n_districts}, {len(child.area)}\n')
        return children

    def make_partition(self, area_df, node):
        """
        Using a random seed, attempt one split from a sample tree node.
        Args:
            area_df: (DataFrame) Subset of rows of state_df for the node region
            node: (SHPnode) the node to sample from

        Returns: (list) of shape nodes for each sub-region in the partition.

        """
        children_sizes = node.sample_n_splits_and_child_sizes(self.config) #TODO

        # dict : {center_ix : child size}
        children_centers = OrderedDict(self.select_centers(area_df, children_sizes))
        #print(children_centers)

        connectivity = self.config.get('connectivity_constraint', None)
        if not node.is_root:
            G = nx.subgraph(self.G, node.area)
            edge_dists = {center: nx.shortest_path_length(G, source=center, weight=connectivity)
                          for center in children_centers}
        else:
            G = self.G
            edge_dists = {center: self.edge_dists[center] for center in children_centers}

        pop_bounds = self.make_pop_bounds(children_centers)

        costs = self.cost_fn.get_costs(area_df, list(children_centers.keys()))
        connectivity_sets = edge_distance_connectivity_sets(edge_dists, G)

        #counties = area_df.CountyCode.to_dict()
        #split_lim=3 #TODO trial and error

        #nonwhite_per_block=np.multiply((100-area_df['p_white']), area_df['population'])/100

        #TODO do this for every possible maj-min split
        """
        partition_IP, xs, BinCounts, m = make_partition_IP_County_MajMin(costs,
                                             connectivity_sets,
                                             area_df.population.to_dict(),
                                             pop_bounds, counties, split_lim, nonwhite_per_block)
        """
        partition_IP, xs = make_partition_IP(costs, 
                                                          connectivity_sets, 
                                                          area_df['population'].to_dict(),
                                                          pop_bounds)
        
        #partition_IP, xs, = make_partition_IP(costs,
        #                                     connectivity_sets,
        #                                     area_df.population.to_dict(),
        #                                     pop_bounds)

        #if self.config['boundary_type'] == 'county':
        #    self.model_factory.augment_model(partition_IP, xs, area_df.index.values, costs)

        partition_IP.Params.MIPGap = self.config['IP_gap_tol']
        partition_IP.update()
        partition_IP.optimize()
        try:
            districting = {i: [j for j in xs[i] if xs[i][j].X > .5]
                           for i in children_centers}
            #bins = {i: [k for k in BinCounts[i] if BinCounts[i][k].X > .5]
            #               for i in children_centers}
            #print(bins)
            feasible = all([nx.is_connected(nx.subgraph(self.G, distr)) for
                            distr in districting.values()])
            if not feasible:
                print('WARNING: PARTITION NOT CONNECTED')
        except AttributeError:
            feasible = False

        #print(bins)

        if self.config['event_logging']:
            if feasible:
                self.event_list.append({
                    'partition': districting,
                    'sizes': pop_bounds,
                    'feasible': True,
                })
            else:
                self.event_list.append({
                    'area': node.area,
                    'centers': children_centers,
                    'feasible': False,
                })

        if self.config['verbose']:
            if feasible:
                #constraintm=partition_IP.getConstrByName('test1')
                #slack=constraintm.slack
                #if(slack!=0):
                #    print(constraintm.slack)
                objective_value = partition_IP.objVal
                #print(objective_value)
                print('successful sample')
            else:
                print('infeasible')
        
        if feasible:
            #print([len(area) for _, area in districting.items()])
            return [SHPNode(pop_bounds[center]['n_districts'], area, self._assign_id(), node.id, False, center)
                    for center, area in districting.items()]
        else:
            node.n_infeasible_samples += 1
            return []

    def select_centers(self, area_df, children_sizes):
        """
        Routes arguments to the right seed selection function.
        Args:
            area_df: (DataFrame) Subset of rows of state_df of the node region
            children_sizes: (int list) Capacity of the child regions
        Returns: (dict) {center index: # districts assigned to that center}

        """
        method = self.config['selection_method']
        if method == 'random_method':
            key = random.random()
            if key < 0.5:
                method = 'random_iterative'
            elif key < 0.95:
                method = 'uncapacitated_kmeans'
            else:
                method = 'uniform_random'

        if method == 'random_iterative':
            centers = iterative_random(area_df, len(children_sizes), self.lengths)
        elif method == 'capacitated_random_iterative':
            pop_capacity = self.config['ideal_pop'] * np.array(children_sizes)
            centers = iterative_random(area_df, pop_capacity, self.lengths)
        elif method == 'uncapacitated_kmeans':
            weight_perturbation_scale = self.config['perturbation_scale']
            n_random_seeds = self.config['n_random_seeds']
            centers = kmeans_seeds(area_df, len(children_sizes),
                                   n_random_seeds, weight_perturbation_scale)
        elif method == 'uniform_random':
            centers = uniform_random(area_df, len(children_sizes))
        else:
            raise ValueError('center selection_method not valid')

        center_capacities = get_capacities(centers, children_sizes,
                                           area_df, self.config)

        return center_capacities

    def make_pop_bounds(self, children_centers):
        """
        Finds the upper and lower population bounds of a dict of center sizes
        Args:
            children_centers: (dict) {center index: # districts}

        Returns: (dict) center index keys and upper/lower population bounds
            and # districts as values in nested dict

        """
        pop_deviation = self.config['max_pop_variation']
        pop_bounds = {}
        # Make the bounds for an area considering # area districts and tree level
        for center, n_child_districts in children_centers.items():
            levels_to_leaf = max(math.ceil(math.log2(n_child_districts)), 1)
            distr_pop = self.config['ideal_pop'] * n_child_districts

            ub = distr_pop + pop_deviation / levels_to_leaf
            lb = distr_pop - pop_deviation / levels_to_leaf

            pop_bounds[center] = {
                'ub': ub,
                'lb': lb,
                'n_districts': n_child_districts
            }

        return pop_bounds

    def make_viz_list(self):
        """
        Saves logging information useful for the SHP viz flask app

        Returns: None
        """
        n_districtings = 'nd' + str(number_of_districtings(self.leaf_nodes, self.internal_nodes))
        n_leaves = 'nl' + str(len(self.leaf_nodes))
        n_interior = 'ni' + str(len(self.internal_nodes))
        width = 'w' + str(self.config['n_samples'])
        n_districts = 'ndist' + str(self.config['n_districts'])
        save_time = str(int(time.time()))
        save_name = '_'.join([self.state_abbrev, n_districtings, n_leaves,
                              n_interior, width, n_districts, save_time])

        json.dump(self.event_list, open(save_name + '.json', 'w'))


if __name__ == '__main__':
    center_selection_config = {
        'selection_method': 'uniform_random',  # one of
        'perturbation_scale': 1,
        'n_random_seeds': 1,
        'capacities': 'match',
        'capacity_weights': 'voronoi',
    }
    tree_config = {
        'max_sample_tries': 25,
        'n_samples': 2,
        'n_root_samples': 5,
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
        'state': 'NY',
        'n_districts': 26,
        'population_tolerance': .01,
    }
    base_config = {**center_selection_config,
                   **tree_config,
                   **gurobi_config,
                   **pdp_config}
    cg = ColumnGenerator(base_config)
    cg.generate()
