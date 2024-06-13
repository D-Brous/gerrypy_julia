import sys
sys.path.append('../gerrypy_julia')

import constants
import os

# Configuration
################################################################################
debug_filename = 'debug_file.txt'
name = 'la_house'
time = 1716842385 #1716235478
################################################################################

'''
experiment_dir = '%s_results_%s' % (name, str(time))
save_dir = os.path.join(constants.LOUISIANA_HOUSE_RESULTS_PATH, experiment_dir)
with open(os.path.join(save_dir, debug_filename), 'r') as debug_file:
    num_leaf_nodes = 0
    num_internal_nodes = 0
    for line in debug_file:
        
        params = line[:-1].split(',')
        for i in range(len(params)):
            params[i] = params[i].strip()
        if params[1] == 'leaf':
            num_leaf_nodes += 1
        elif params[1] == 'internal':
            num_internal_nodes += 1
    print(f'Number of leaf nodes: {num_leaf_nodes}')
    print(f'Number of internal nodes: {num_internal_nodes}')   
'''
'''
def flatten2(lis_of_lis):
    return [element for lis in lis_of_lis for element in lis]

a = [2,3]
b = [4,5,6]
c = [6,7]
d = [a,b,c]
e = flatten2(d)
print(e)
'''

class DebugNode:
    def __init__(self, n_districts, id, parent_id=None):
        self.n_districts = n_districts
        self.id = id
        self.parent_id = parent_id


experiment_dir = '%s_results_%s' % (name, str(time))
save_dir = os.path.join(constants.LOUISIANA_HOUSE_RESULTS_PATH, experiment_dir)
with open(os.path.join(save_dir, debug_filename), 'r') as debug_file:
    num_leaf_nodes = 0
    num_internal_nodes = 0
    num_nodes_in_tree = 0
    num_successful_splits = 0
    num_failed_splits = 0
    total_split_time = 0
    num_failed_line = 0
    split_complete = True
    parent_id = 0
    sample_queue = [DebugNode(105, 0)]
    node = None
    sample_internal_nodes = {}
    sample_leaf_nodes = {}
    max_id = 0
    for line in debug_file:
        if line[0] == 'S': # Successful split
            split_complete = False
            num_successful_splits += 1
            words = line.strip().split(' ')
            total_split_time += float(words[7])
            parent_id = int(words[2])
            node = sample_queue.pop(0)
        elif line[0] == 'F' or num_failed_line > 0: # Failed split
            split_complete = True
            num_failed_splits += 1
            num_failed_line += 1
            if num_failed_line == 3: # Delete deleted nodes
                ids = line.strip().strip('[]').split(', ')[:-1]
                for id in ids:
                    if id in sample_internal_nodes:
                        del sample_internal_nodes[int(id)]
                    elif id in sample_leaf_nodes:
                        del sample_leaf_nodes[int(id)]
                    else:
                        raise ValueError('Deleted nodes not present in sample_internal_nodes or sample_leaf_nodes')
            elif num_failed_line == 5: # Modify sample queue
                ids = line.strip().strip('[]').split(', ')[:-1]
                sample_queue = ids
            elif num_failed_line == 6: # Count number of nodes in tree
                num_nodes_in_tree = int(line.strip().split(' ')[8])
                if num_nodes_in_tree != num_leaf_nodes + num_internal_nodes + 1:
                    raise ValueError('Number of nodes in tree inconsistent')
                num_failed_line = 0
        elif line[0] == 'L': # Logging an internal or leaf node
            split_complete = True
            words = line.split(' ')
            if words[1] == 'internal':
                sample_internal_nodes[node.id] = node
        elif not split_complete: # Appending children to sample queue
            params = line.strip()[:-1].split(', ')
            if params[1] == 'leaf':
                num_leaf_nodes += 1
                sample_leaf_nodes[int(params[0])] = DebugNode(int(params[2]), int(params[0]), parent_id=parent_id)
            elif params[1] == 'internal':
                num_internal_nodes += 1
                sample_queue.append(DebugNode(int(params[2]), int(params[0]), parent_id=parent_id))
        else:
            raise ValueError('Unrecognized line in debug file')
    print(f'Number of leaf nodes: {num_leaf_nodes}\n')
    print(f'Number of internal nodes: {num_internal_nodes}\n')   
    print(f'Number of nodes in tree: {num_internal_nodes + num_leaf_nodes + 1}\n')
    print(f'Number of successful splits: {num_successful_splits}\n')
    print(f'Number of failed splits: {num_failed_splits}\n')
    print()
