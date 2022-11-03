import numpy as np
import networkx as nx
import os
import pysal
import json
import pickle
import pandas as pd
import geopandas as gpd
from gerrypy import constants
from gerrypy.data.adjacency import connect_components
from gerrypy.data.precinct_state_wrappers import wrappers
from gerrypy.data.load import *
from scipy.spatial.distance import pdist, squareform


def preprocess_vtds(state):
    """
    Create and save adjacency, pairwise dists, construct state_df
    Args:
        state: (str) two letter state abbreviation
    """
    state_fips = constants.ABBREV_DICT[state][constants.FIPS_IX]

    base_path = os.path.join(constants.GERRYPY_BASE_PATH, 'data', '2020_redistricting')
    block_population_path = os.path.join(base_path, 'block_population_2019_estimate', f'pop19_{state_fips}.csv')
    block_to_vtd_path = os.path.join(base_path, 'block_to_vtd_mapping', f'BlockAssign_ST{state_fips}_{state}_VTD.txt')
    vtd_shape_path = os.path.join(base_path, 'vtd_shapes', state)

    # Get vtd level population estimates
    block_populations = pd.read_csv(block_population_path, index_col=0, names=['fips', 'population'])
    block_mapping = pd.read_csv(block_to_vtd_path, sep='|', dtype=object, index_col=0)
    block_mapping['VTD_ID'] = state_fips + block_mapping.COUNTYFP + block_mapping.DISTRICT
    block_mapping['population'] = block_populations
    vtd_populations = block_mapping.groupby('VTD_ID').agg({'population': 'sum'})

    vtd_shapes = gpd.read_file(vtd_shape_path).sort_values('GEOID20').reset_index(drop=True)
    vtd_2d_shapes = vtd_shapes.to_crs(epsg=constants.CRS)

    election_df = wrappers[state]().get_data(vtd_2d_shapes, False)

    vtd_2d_shapes = vtd_2d_shapes.set_index('GEOID20')
    state_df = pd.DataFrame({
        'population': vtd_populations.population,
        'x': vtd_2d_shapes.centroid.x,
        'y': vtd_2d_shapes.centroid.y,
        'area': vtd_2d_shapes.area / 1000**2,  # sq km
    }).sort_index()
    state_df.index.name = "GEOID"

    shape_list = vtd_shapes.geometry.to_list()
    adj_graph = pysal.lib.weights.Rook.from_iterable(shape_list).to_networkx()

    if not nx.is_connected(adj_graph):
        adj_graph = connect_components(vtd_shapes)

    edge_dists = dict(nx.all_pairs_shortest_path_length(adj_graph))

    centroids = state_df[['x', 'y']].values
    plengths = squareform(pdist(centroids))

    save_path = os.path.join(constants.OPT_DATA_PATH, 'gamo', state)
    os.makedirs(save_path, exist_ok=True)

    state_df.to_csv(os.path.join(save_path, 'state_df.csv'), index=True)
    election_df.to_csv(os.path.join(save_path, 'election_df.csv'), index=False)
    np.save(os.path.join(save_path, 'lengths.npy'), plengths)
    nx.write_gpickle(adj_graph, os.path.join(save_path, 'G.p'))
    pickle.dump(edge_dists, open(os.path.join(save_path, 'edge_dists.p'), 'wb'))


if __name__ == '__main__':
    states_to_process = ['CO', 'WI', 'OH', 'IL', 'FL']
    for state in states_to_process:
        print('Processing', state)
        preprocess_vtds(state)