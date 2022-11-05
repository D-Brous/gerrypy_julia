"""This module is used to interface with all downloaded data.

You must use this to load all data or else indices may become inconsistent."""

import sys
sys.path.append('../gerrypy')

import pickle
import constants
import networkx as nx
import os
import glob
import numpy as np
import scipy.sparse as sp
import pandas as pd
import geopandas as gpd

def load_tract_shapes(year=None, custom_path=''):
    """
    Args:
        state_abbrev: (str) two letter state abbreviation
        year: (int) the year of the TIGERLINE shapefiles

    Returns: (gpd.GeoDataFrame) of tract shapes
    """
    if custom_path:
        tract_shapes = gpd.read_file(custom_path)
        return tract_shapes.sort_values(by='GEOID20').reset_index(drop=True)
    if not year:
        year = constants.ACS_BASE_YEAR
    tract_shapes = gpd.read_file(constants.CENSUS_SHAPE_PATH_BUFFALO)
    tract_shapes = tract_shapes.to_crs("EPSG:3078")  # meters
    #tract_shapes = tract_shapes[tract_shapes.ALAND20 > 0] #TODO
    return tract_shapes.sort_values(by='GEOID20').reset_index(drop=True)

def load_opt_data(special_input='', use_spt_matrix=False):
    """
    Args:
        state_abbrev: (str) two letter state abbreviation
        special_input: (str) subdirectory containing specialized inputs
        use_spt_matrix: (bool) load shortest path tree matrix instead of
            shortest path dict

    Returns: (pd.DataFrame, nx.Graph, np.array, dict) tuple of optimization
        data structures
    """
    data_base_path = os.path.join(constants.OPT_DATA_PATH_BUFFALO, special_input)
    adjacency_graph_path = os.path.join(data_base_path, 'G.p')
    state_df_path = os.path.join(data_base_path, 'state_df.csv')

    state_df = pd.read_csv(state_df_path)
    G = nx.read_gpickle(adjacency_graph_path)

    if os.path.exists(os.path.join(data_base_path, 'lengths.npy')):
        lengths_path = os.path.join(data_base_path, 'lengths.npy')
        lengths = np.load(lengths_path)
    else:
        from scipy.spatial.distance import pdist, squareform
        lengths = squareform(pdist(state_df[['x', 'y']].values))

    if not use_spt_matrix:
        if os.path.exists(os.path.join(data_base_path, 'edge_dists.p')):
            edge_dists_path = os.path.join(data_base_path, 'edge_dists.p')
            edge_dists = pickle.load(open(edge_dists_path, 'rb'))
        else:
            edge_dists = dict(nx.all_pairs_shortest_path_length(G))
    else:
        if os.path.exists(os.path.join(data_base_path, 'spt_matrix.npy')):
            spt_matrix_path = os.path.join(data_base_path, 'spt_matrix.npz')
            spt_matrix = sp.load_npz(spt_matrix_path)
        else:
            raise Exception
            # edge_dists = dict(nx.all_pairs_shortest_path_length(G))
    tree_encoding = spt_matrix if use_spt_matrix else edge_dists

    return state_df, G, lengths, tree_encoding

if __name__ == "__main__":
    load_tract_shapes()