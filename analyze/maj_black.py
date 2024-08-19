import sys
sys.path.append('../gerrypy_julia')

import numpy as np
import pandas as pd
import constants
import os
import fnmatch
from data.load import *
import heapq

def bvap_prop(subregion, state_df):
    """
    Args:
        subregion: (list(int)) list of indices of cgus
        state_df: (pd.DataFrame) dataframe with census demographic data
    
    Returns: (float) total BVAP / total VAP in the subregion
    """
    subregion_df = state_df.loc[subregion]
    return subregion_df['BVAP'].sum() / subregion_df['VAP'].sum()

def n_maj_black(assignment_dict, state_df):
    """
    Args:
        assignment_dict: (dict) assignment of cgus to districts
        state_df: (pd.DataFrame) dataframe with census demographic data
    
    Returns: (int) total number of maj black districts
    """
    return sum(bvap_prop(district_subregion, state_df) > 0.5 for district_subregion in assignment_dict.values())

def majority_black(assignments_df, state_df, num_districts, num_plans=None):
    if num_plans is None:
        num_plans = 0
        for column in assignments_df.columns.tolist():
            if column[:8] == 'District':
                num_plans += 1
    
    black_pop_per_district = np.zeros((num_plans, num_districts))
    tot_pop_per_district = np.zeros((num_plans, num_districts))

    for i in list(assignments_df.index):
        for j in range(num_plans):
            district = int(assignments_df.loc[i, f'District{j}'])
            black_pop_per_district[j, district] += int(state_df.loc[i, 'BVAP'])
            tot_pop_per_district[j, district] += int(state_df.loc[i, 'VAP'])
    
    avgs = black_pop_per_district / tot_pop_per_district
    return (avgs > 0.5)

def maj_black_logging_info(root, internal_nodes, leaf_nodes, debug_file, exact_partition_range, maj_black_partition_IPs, state_df):
    maj_black_partitioned_nodes = {}
    for node in leaf_nodes.values():
        parent_id = node.parent_id
        if parent_id == 0 and root.n_districts in exact_partition_range:
            maj_black_partitioned_nodes[parent_id] = root
        elif internal_nodes[parent_id].n_districts in exact_partition_range:
            maj_black_partitioned_nodes[parent_id] = internal_nodes[parent_id]
    num_wins_per_partition = np.zeros((len(maj_black_partition_IPs)), dtype=int)
    total_maj_black_per_partition = np.zeros((len(maj_black_partition_IPs)), dtype=int)
    num_examples_per_partition = np.zeros((len(maj_black_partition_IPs)), dtype=int)
    statuses = {2 : 0, 9 : 0}
    for node in maj_black_partitioned_nodes.values():
        samples = []
        sample = []
        trial_ix = 0
        sample_ix = 0
        n_trials = len(node.partitions_used[sample_ix])
        ix = 0
        for maj_black_partition_ixs in node.partitions_used:
            for i in range(len(maj_black_partition_ixs)):
                sample.append(node.children_ids[ix])
                ix += 1
            samples.append(sample)
            sample = []
        '''
        for ix in range(len(node.children_ids)):
            sample.append(node.children_ids[ix])
            trial_ix += 1
            if trial_ix >= n_trials:
                samples.append(sample)
                sample = []
                sample_ix += 1
                trial_ix = 0
                n_trials = len(node.partitions_used[sample_ix])
        '''
        ideal_vap = state_df.loc[node.area]['VAP'].sum() / node.n_districts
        for sample_ix in range(len(samples)):
            sample = samples[sample_ix]
            n_trials = len(sample)
            n_districts = len(sample[0])
            bvaps = np.zeros((n_trials, n_districts), dtype=int)
            vaps = np.zeros((n_trials, n_districts), dtype=int)
            pops = np.zeros((n_trials, n_districts), dtype=int)
            for trial_ix in range(n_trials):
                trial = sample[trial_ix]
                bvaps[trial_ix] = np.array([state_df.loc[leaf_nodes[id].area]['BVAP'].sum() for id in trial])
                vaps[trial_ix] = np.array([state_df.loc[leaf_nodes[id].area]['VAP'].sum() for id in trial])
                pops[trial_ix] = np.array([state_df.loc[leaf_nodes[id].area]['population'].sum() for id in trial])
            maj_black_per_trial = (bvaps / vaps > 0.5).sum(axis=1)
            different = not all(num_maj_black == maj_black_per_trial[0] for num_maj_black in maj_black_per_trial)
            if different:
                debug_file.write(f'\n---------------------Partitions of node {node.id}, sample {sample_ix}---------------------\n\n')
                debug_file.write(f'ideal VAP: {ideal_vap}\n\n')
                winner_ix = 0
                for trial_ix in range(n_trials):
                    maj_black_partition_ix = node.partitions_used[sample_ix][trial_ix]
                    num_examples_per_partition[maj_black_partition_ix] += 1
                    total_maj_black_per_partition[maj_black_partition_ix] += maj_black_per_trial[trial_ix]
                    if maj_black_per_trial[trial_ix] > maj_black_per_trial[winner_ix]:
                        winner_ix = trial_ix
                    debug_file.write(f'partitition_IP {trial_ix}: {maj_black_partition_IPs[maj_black_partition_ix]}\n')
                    debug_file.write(f'> num maj black = {maj_black_per_trial[trial_ix]}\n')
                    debug_file.write(f'> obj value = {node.partition_obj_values[sample_ix][trial_ix]}\n')
                    status = node.partition_statuses[sample_ix][trial_ix]
                    debug_file.write(f'> model status = {status}\n')
                    statuses[status] += 1
                    bvap_str = '    BVAP    '
                    vap_str = '    VAP     '
                    bprop_str = '    BVAP/VAP'
                    pop_str = '    pop     '
                    for district_ix in range(n_districts):
                        bvap_str += f'{bvaps[trial_ix][district_ix]:8d}'
                        vap_str += f'{vaps[trial_ix][district_ix]:8d}'
                        bprop_str += f'{bvaps[trial_ix][district_ix]/vaps[trial_ix][district_ix]:8.4f}'
                        pop_str += f'{pops[trial_ix][district_ix]:8d}'
                    debug_file.write(bvap_str + '\n')
                    debug_file.write(vap_str + '\n')
                    debug_file.write(bprop_str + '\n')
                    debug_file.write(pop_str + '\n')
                num_wins_per_partition[node.partitions_used[sample_ix][winner_ix]] += 1      
    debug_file.write('------------------------------Summary Info------------------------------\n')
    debug_file.write(f'Num wins per partition: {num_wins_per_partition}\n')
    debug_file.write(f'Avg maj black per partition: {total_maj_black_per_partition / num_examples_per_partition}\n')   
    debug_file.write(f'Model status distribution: {statuses}\n')  

def moon_upper_bound(area_df):
    margins = 2 * area_df['BVAP'].to_numpy() - area_df['VAP'].to_numpy()
    pops = area_df['population'].to_numpy()
    margins_and_pops = [(margins[i], pops[i]) for i in range(len(margins))]
    margins_and_pops_sorted = sorted(margins_and_pops, key=lambda tup : tup[0] / tup[1], reverse=True)
    tot_margin = 0
    tot_pop = 0
    for i in range(len(margins_and_pops_sorted)):
        tot_margin += margins_and_pops_sorted[i][0]
        if tot_margin <= 0:
            tot_pop += margins_and_pops_sorted[i][0]

if __name__ == '__main__':
    
    results_path = constants.LOUISIANA_HOUSE_RESULTS_PATH
    state_df_path = os.path.join(constants.OPT_DATA_PATH, 'block_group/LA/state_df.csv')
    state_df = load_state_df('LA', 2010, 'block_group')
    state_df = state_df.sort_values('GEOID')
    """
    for dir in sorted(os.listdir(results_path)):
        dir_path = os.path.join(results_path, dir)
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                if file[:11] == 'assignments' and file[-4:] == '.csv':
                    assignments_df_path = os.path.join(dir_path, file)
                    assignments_df = pd.read_csv(assignments_df_path)
                    assignments_df = assignments_df.sort_values('GEOID')
                    try:
                        print(os.path.join(dir, file), majority_black(assignments_df, state_df, 105).sum(axis=1))
                    except IndexError:
                        print(assignments_df_path)
    #debug_file = open(os.path.join(save_path, debug_filename), 'a')
    '''
    results_time_str = '1721243348' #'1719459163'
    assignment_time_str = '1721243348' #'1719586752'

    from experiments.louisiana_house import config
    save_path = os.path.join(config['results_path'], 'results_' + results_time_str)
    tree = load_tree(save_path)
    debug_file = open(os.path.join(save_path, 'debug_file_2_test.txt'), 'a')
    state_df = load_state_df(config['state'], config['granularity'])
    internal_nodes = {}
    leaf_nodes = {}
    for key, value in tree.items():
        if key!=-1:
            internal_nodes.update(value[0])
            leaf_nodes.update(value[1])
    maj_black_logging_info(internal_nodes, leaf_nodes, debug_file, config['exact_partition_range'], config['maj_black_partition_IPs'], state_df)
    debug_file.close()
    '''
    """
    print(moon_upper_bound(state_df))