import sys
sys.path.append('../gerrypy_julia')

import pandas as pd
from optimize.shp import SHP
import numpy as np
from analyze.maj_black import *
from data.load import *
import networkx as nx
import random
import libpysal
import os

class PostProcessing:
    def __init__(self, config, local_search_func, partition_func, save_rule_func, results_time_str, assignments_time_str):
        """
        Args:
            config: (dict) Check documentation for description of all key value pairs
            assignments_df: (pd.DataFrame) Contains the districting plans produced by SHP
            local_search_func: (pd.Series -> subregion) Takes in an assignment series and 
                outputs a subregion (list of cgu indices) over which to reoptimize
            partition_func: (dict -> pd.Series) Takes in a config which includes the subregion
                and outputs a redistricting of the subregion
        """
        self.config = config
        self.assignments_df = load_assignments_df(config['results_path'], results_time_str, assignments_time_str)
        self.saved_assignments_df_dict = {} # Necessary?
        self.local_search_func = local_search_func
        self.partition_func = partition_func
        self.save_rule_func = save_rule_func
        self.state_df = load_state_df(config['state'], config['year'], config['granularity'])
        self.save_path = os.path.join(config['results_path'], 
                                      'results_%s' % results_time_str)
        try:
            os.mkdir(os.path.join(self.save_path, 'post_processing'))
        except FileExistsError:
            pass


    def short_bursts(self, n_steps, burst_length, plans): #TODO
        """
        Args:
            n_steps: (int) Number of repartitioning steps
            burst_length: (int) Number of steps per burst
            plans: (list(int)) Plans to reoptimize
        """
        n_bursts = n_steps // burst_length
        if plans is None:
            n_plans = sum(1 for column in self.assignments_df.columns if column.startswith('District'))
            plans = np.arange(n_plans)

        burst_assignments_df = pd.DataFrame()
        zeros = np.zeros(len(self.assignments_df['GEOID']), dtype=int)
        for burst_step in range(burst_length):
            burst_assignments_df[f'District{burst_step}'] = zeros
        
        for plan in plans:
            curr_best_assignment_ser = self.assignments_df[f'District{plan}']
            curr_best_n_maj_black = n_maj_black(assignment_ser_to_dict(curr_best_assignment_ser))
            step = 0
            for burst in range(n_bursts):
                # Run the burst
                curr_assignment = curr_best_assignment_ser
                #curr_assignment = self.assignments_df[f'District{plan}']
                for burst_step in range(burst_length):
                    subregion = self.local_search_func(curr_assignment)
                    chosen_districts = curr_assignment[subregion].unique()
                    self.config['subregion'] = subregion
                    self.config['n_districts'] = len(chosen_districts)
                    reassignment_dict = assignment_ser_to_dict(self.partition_func(self.config)['District0'])
                    for chosen_district_ix, district_subregion in reassignment_dict.items():
                        curr_assignment.iloc[district_subregion] = chosen_districts[chosen_district_ix]
                    burst_assignments_df[f'District{burst_step}'] = curr_assignment.copy()
                    if self.save_rule_func(step):
                        self.saved_assignments_df[step] = curr_assignment
                    step += 1
                # Find the best assignment in the chain
                max_maj_black = 0
                curr_best_assignment_ser = burst_assignments_df['District0']
                nums_maj_black = majority_black(burst_assignments_df, self.state_df, self.config['n_districts'], num_plans=burst_length).sum(axis=1)
                for burst_step in range(1, burst_length):
                    if nums_maj_black[burst_step] > max_maj_black:
                        max_maj_black_burst_step = burst_step
                self.assignments_df[f'District{plan}'] = burst_assignments_df[f'District{max_maj_black_burst_step}']
        return self.assignments_df
    
    def one_chain(self, n_steps, plans):
        if plans is None:
            n_plans = sum(1 for column in self.assignments_df.columns if column.startswith('District'))
            plans = np.arange(n_plans)
        for plan in plans:
            curr_assignment = self.assignments_df[f'District{plan}']
            saved_assignments_df = pd.DataFrame()
            saved_assignments_df['GEOID'] = self.state_df['GEOID']
            for step in range(n_steps):
                subregion = self.local_search_func(curr_assignment)
                chosen_districts = curr_assignment[subregion].unique()
                self.config['subregion'] = subregion
                self.config['n_districts'] = len(chosen_districts)
                reassignment_dict = assignment_ser_to_dict(self.partition_func(self.config)['District0'])
                for chosen_district_ix, district_subregion in reassignment_dict.items():
                    curr_assignment.loc[district_subregion] = chosen_districts[chosen_district_ix]
                if self.save_rule_func(step):
                    saved_assignments_df[f'District{step}'] = curr_assignment
            saved_assignments_df.to_csv(os.path.join(self.save_path, 'post_processing', f'assignments_{plan}.csv'), index=False)
            self.assignments_df[f'District{plan}'] = curr_assignment
        return self.assignments_df

    def one_chain_pq(self, n_steps, plans, geq):

def shp_partition(config):
    experiment = SHP(config)
    return experiment.run()

'''
def assignments_df_to_dict(assignments_df, n_districts=None, n_plans=None):
    """
    Args:
        assignments_df: (pd.DataFrame) assignments of cgus to districts for several plans
        n_districts: (U(int, None)) number of districts
        n_plans: (U(int, None)) number of plans

    Returns:
        assignments_dict: (dict) assignments of cgus to districts for several plans
    """
    if n_districts is None:
        n_districts = len(assignments_df['District0'].unique())
    if n_plans is None:
        n_plans = sum(1 for column in assignments_df.columns if column.startswith('District'))
    assignments_dict = {}
    for plan in range(n_plans):
        assignments_dict[plan] = {district_id: assignments_df[assignments_df[f'District{plan}']==district_id].index.tolist() for district_id in range(n_districts)}
    return assignments_dict
'''

def average_geq_threshold_random(state, year, granularity, assignment_ser):
    state_df = load_state_df(state, year, granularity)
    G_d = load_district_adjacency_graph(state, year, granularity, assignment_ser)
    assignment_dict = assignment_ser_to_dict(assignment_ser)
    while True:
        random_district = random.sample(G_d.nodes, 1)[0]
        random_neighboring_district = random.sample(list(G_d.neighbors(random_district)), 1)[0]
        district_1_subregion = assignment_dict[random_district]
        district_2_subregion = assignment_dict[random_neighboring_district]
        bvap_prop_1 = bvap_prop(district_1_subregion, state_df)
        bvap_prop_2 = bvap_prop(district_2_subregion, state_df)
        n_black_maj = int(bvap_prop_1 > 0.5) + int(bvap_prop_2 > 0.5)
        if (n_black_maj == 0 and bvap_prop_1 + bvap_prop_2 > 0.8) or (n_black_maj == 1 and bvap_prop_1 + bvap_prop_2 > 1.2):
            return district_1_subregion + district_2_subregion
        
def average_geq_threshold