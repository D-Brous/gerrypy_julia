import sys
sys.path.append('../gerrypy_julia')

import numpy as np
import networkx as nx
import os
# import pygeos
# import pysal
import libpysal
import pickle
import pandas as pd
import constants
from data.adjacency import connect_components
from data.data2020.load import *
from data.columns import CENSUS_VARIABLE_TO_NAME
from scipy.spatial.distance import pdist, squareform
# import dill

def preprocess_tracts(state_abbrev):
    """
    Create and save adjacency, pairwise dists, construct state_df
    Args:
        state_abbrev: (str) two letter state abbreviation
    """

    tract_shapes = load_tract_shapes(state_abbrev, constants.ACS_BASE_YEAR)
    
    state_df = pd.DataFrame({

        'x': tract_shapes.centroid.x,
        'y': tract_shapes.centroid.y,
        'area': tract_shapes.area / 1000**2,  # sq km
        'perimeter': tract_shapes.length,
        'GEOID': tract_shapes.GEOID.apply(lambda x: str(x).zfill(11)),
    })

    # Join location data with demographic data TODO: get demographic data
    demo_data = pd.read_csv(os.path.join(constants.TRACT_DATA_PATH_2020,
                                         '%d_acs5' % constants.ACS_BASE_YEAR,
                                         '%s_tract.csv' % state_abbrev),
                            low_memory=False)
    demo_data['GEOID'] = demo_data['GEOID'].astype(str).apply(lambda x: x.zfill(11))
    demo_data = demo_data.set_index('GEOID')
    demo_data = demo_data[list(CENSUS_VARIABLE_TO_NAME[str(constants.ACS_BASE_YEAR)])]
    demo_data = demo_data.rename(columns=CENSUS_VARIABLE_TO_NAME[str(constants.ACS_BASE_YEAR)])
    demo_data[demo_data < 0] = 0

    state_df = state_df.set_index('GEOID')
    state_df = state_df.join(demo_data)
    state_df = state_df.reset_index()
    state_df['CountyCode'] = state_df.GEOID.str[2:5]

    #create an indicator that is 1 if the block is in the Black Belt, 0 if not
    def set_BB_indicator(x):
        return 1 if x in ['063', '013', '131', '119', '113', '109', '107', '105', '101', '091', '087', '085', '065', '047', '041', '023', '011', '005'] else 0
    state_df['BlackBelt'] = state_df['CountyCode'].apply(set_BB_indicator)

    shape_list = tract_shapes.geometry.to_list()
    adj_graph = libpysal.weights.Rook.from_iterable(shape_list).to_networkx()

    if not nx.is_connected(adj_graph):
        adj_graph = connect_components(tract_shapes)
    adj_mat = nx.linalg.graphmatrix.adjacency_matrix(adj_graph).toarray()
    border_dists=get_border_dists(adj_mat,state_df,tract_shapes)

    edge_dists = dict(nx.all_pairs_shortest_path_length(adj_graph))

    centroids = state_df[['x', 'y']].values
    plengths = squareform(pdist(centroids))

    save_path = os.path.join(constants.OPT_DATA_PATH_2020, state_abbrev)
    os.makedirs(save_path, exist_ok=True)

    state_df.to_csv(os.path.join(save_path, 'state_df.csv'), index=False)
    np.save(os.path.join(save_path, 'lengths.npy'), plengths)
    nx.write_gpickle(adj_graph, os.path.join(save_path, 'G.p'))
    pickle.dump(edge_dists, open(os.path.join(save_path, 'edge_dists.p'), 'wb'))
    np.savetxt(save_path+'border_dists.csv', border_dists, delimiter=',')

def get_border_dists(adj_mat, state_df,tract_shapes):
    """
    Generates a matrix with the length of the shared border between every pair of adjacent blocks
    (0 if not adjacent)
    Args:
        state_df: (pd.DataFrame) contains info on each block, including perimeter
        adj_mat: (np.array) nxn array, where n is # blocks, which has (i,j)=1 if blocks i and j are
            adjecent, 0 if not
    """
    border_mat=np.empty([len(state_df.index),len(state_df.index)])
    for index, row in state_df.iterrows():
        for j in range(index+1,len(state_df.index)):
            if adj_mat[index,j]==0:
                border_mat[index,j]=0
            else:
                union = tract_shapes.loc[[index,j],'geometry'].unary_union
                border=state_df.loc[index,'perimeter']+state_df.loc[j,'perimeter']-union.length
                border_mat[index,j]=border
                border_mat[j,index]=border
                
    return border_mat

def preprocess_all_states():
    """
    Calls preprocess_tracts for all states
    """
    #NOTE: raw VTD data is not available for CA, OR, or HI for some reason
    states = [abbr for _, abbr, _ in constants.STATE_IDS]

    #only download ones we don't already have
    opt_dir = constants.OPT_DATA_PATH_2020
    cached = os.listdir(opt_dir)
    states = [state for state in states if state not in cached]

    print(states)

    for state in states:
        if state != 'CA' and state != 'HI' and state != 'OR':
            preprocess_tracts(state)

if __name__ == "__main__":
    #preprocess_all_states()
    preprocess_tracts('AL')