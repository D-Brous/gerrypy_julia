import sys
sys.path.append('../gerrypy')

import numpy as np
import networkx as nx
import os
import pygeos
import pysal
import pickle
import pandas as pd
import constants
from data.adjacency import connect_components
from data.buffalo_data.load import *
from data.columns import CENSUS_VARIABLE_TO_NAME
from scipy.spatial.distance import pdist, squareform
import dill

def preprocess_tracts():
    """
    Create and save adjacency, pairwise dists, construct state_df
    Args:
        state_abbrev: (str) two letter state abbreviation
    """

    tract_shapes = load_tract_shapes()

    state_df = pd.DataFrame({

        'x': tract_shapes.centroid.x,
        'y': tract_shapes.centroid.y,
        'area': tract_shapes.area / 1000**2,  # sq km
        'GEOID': tract_shapes.GEOID20.apply(lambda x: str(x).zfill(11)),
    })

    # Join location data with demographic data TODO: get demographic data
    demo_data = pd.read_csv(os.path.join(constants.TRACT_DATA_PATH_BUFFALO,
                                         'Demo_data.csv'))
    demo_data['GEOID'] = demo_data['GEOID20'].astype(str).apply(lambda x: x.zfill(11))
    demo_data = demo_data.set_index('GEOID')
    #demo_data = demo_data[list(CENSUS_VARIABLE_TO_NAME[str(constants.ACS_BASE_YEAR)])]
    #demo_data = demo_data.rename(columns=CENSUS_VARIABLE_TO_NAME[str(constants.ACS_BASE_YEAR)])
    #demo_data[demo_data < 0] = 0

    state_df = state_df.set_index('GEOID')
    state_df = state_df.join(demo_data)
    state_df = state_df.reset_index()

    shape_list = tract_shapes.geometry.to_list()
    adj_graph = pysal.lib.weights.Rook.from_iterable(shape_list).to_networkx()

    if not nx.is_connected(adj_graph):
        adj_graph = connect_components(tract_shapes)

    edge_dists = dict(nx.all_pairs_shortest_path_length(adj_graph))

    centroids = state_df[['x', 'y']].values
    plengths = squareform(pdist(centroids))

    save_path = constants.OPT_DATA_PATH_BUFFALO
    os.makedirs(save_path, exist_ok=True)

    state_df.to_csv(os.path.join(save_path, 'state_df.csv'), index=False)
    np.save(os.path.join(save_path, 'lengths.npy'), plengths)
    nx.write_gpickle(adj_graph, os.path.join(save_path, 'G.p'))
    pickle.dump(edge_dists, open(os.path.join(save_path, 'edge_dists.p'), 'wb'))

if __name__ == "__main__":
    preprocess_tracts()
