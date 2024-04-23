import sys
sys.path.append('../gerrypy_julia')

import numpy as np
import networkx as nx
import os
import libpysal
import json
import pickle
import pandas as pd
import geopandas as gpd
import constants
from data.adjacency import connect_components
from data.load import *
from data.adjacency import create_adjacency_graph
from data.columns import CENSUS_VARIABLE_TO_NAME
from optimize.utils import build_spt_matrix
from scipy.spatial.distance import pdist, squareform


def load_county_political_data():
    """Helper function to preprocess county voting data to get paritsan vote totals
    for all counties."""
    path = os.path.join(constants.GERRYPY_BASE_PATH, 'data', 'countypres_2000-2016.tab')
    county_results = pd.read_csv(path, sep='\t')
    county_results = county_results.query('party == "democrat" or party == "republican"')
    county_results['FIPS'] = county_results['FIPS'].fillna(0).astype(int).astype(str).apply(lambda x: x.zfill(5))
    calt = county_results.groupby(['year', 'FIPS', 'party']).sum()
    county_year = county_results.groupby(['year', 'FIPS'])
    county_year_votes = county_year['candidatevotes'].sum()
    county_year_vote_p = pd.DataFrame(calt['candidatevotes'] / county_year_votes).query('party == "democrat"')
    county_year_vote_p = county_year_vote_p.reset_index(level=[2], drop=True)
    county_year_vote_p = (1 - county_year_vote_p)
    df = county_year_vote_p.unstack('year')
    county_year_vote_p = df.apply(lambda row: row.fillna(row.mean()), axis=1).stack('year')
    return county_year_vote_p


def preprocess_tracts(state_abbrev, opt_data_path=constants.OPT_DATA_PATH, year=constants.ACS_BASE_YEAR, granularity='tract', use_name_map=True):
    """
    Create and save adjacency, pairwise dists, construct state_df
    Args:
        state_abbrev: (str) two letter state abbreviation
    """

    tract_shapes = load_tract_shapes(state_abbrev, year=year, granularity=granularity)
    if 'GEOID10' in tract_shapes.columns:
        tract_shapes.rename(columns={'GEOID10' : 'GEOID'}, inplace=True)
    state_df = pd.DataFrame({
        'x': tract_shapes.centroid.x,
        'y': tract_shapes.centroid.y,
        'area': tract_shapes.area / 1000**2,  # sq km
        'GEOID': tract_shapes.GEOID.apply(lambda x: str(x).zfill(11)),
    })

    # Join location data with demographic data
    granularity_path = ''
    if granularity == 'block':
        granularity_path = constants.BLOCK_DATA_PATH
    elif granularity == 'block_group':
        granularity_path = constants.BLOCK_GROUP_DATA_PATH
    elif granularity == 'tract':
        granularity_path = constants.TRACT_DATA_PATH
    elif granularity == 'county':
        granularity_path = constants.COUNTY_DATA_PATH

    demo_data = pd.read_csv(os.path.join(granularity_path,
                                         '%d_acs5' % year,
                                         '%s_tract.csv' % state_abbrev),
                            low_memory=False)
    demo_data['GEOID'] = demo_data['GEOID'].astype(str).apply(lambda x: x.zfill(11)) # might need to modify this to accommadte other granularities
    demo_data = demo_data.set_index('GEOID')
    if use_name_map:
        demo_data = demo_data[list(CENSUS_VARIABLE_TO_NAME[str(year)])]
        demo_data = demo_data.rename(columns=CENSUS_VARIABLE_TO_NAME[str(year)])
    else:
        demo_data = demo_data[['TOTPOP', 'VAP', 'BVAP']]
        demo_data.rename(columns={'TOTPOP' : 'population'}, inplace=True)
    demo_data[demo_data < 0] = 0

    state_df = state_df.set_index('GEOID')
    state_df = state_df.join(demo_data)
    state_df = state_df.reset_index()

    shape_list = tract_shapes.geometry.to_list()
    adj_graph = libpysal.weights.Rook.from_iterable(shape_list).to_networkx()

    if not nx.is_connected(adj_graph):
        adj_graph = connect_components(tract_shapes)

    edge_dists = dict(nx.all_pairs_shortest_path_length(adj_graph))

    centroids = state_df[['x', 'y']].values
    plengths = squareform(pdist(centroids))

    save_path = os.path.join(opt_data_path, state_abbrev)
    os.makedirs(save_path, exist_ok=True)

    state_df.to_csv(os.path.join(save_path, 'state_df.csv'), index=False)
    np.save(os.path.join(save_path, 'lengths.npy'), plengths)
    nx.write_gpickle(adj_graph, os.path.join(save_path, 'G.p'))
    pickle.dump(edge_dists, open(os.path.join(save_path, 'edge_dists.p'), 'wb'))

if __name__ == '__main__':
    preprocess_tracts('LA', year=2010, granularity='block_group', use_name_map=False)