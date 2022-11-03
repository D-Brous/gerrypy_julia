"""This module is used to interface with all downloaded data.

You must use this to load all data or else indices may become inconsistent."""


import pickle
from gerrypy import constants
import networkx as nx
import os
import glob
import numpy as np
import scipy.sparse as sp
import pandas as pd
import geopandas as gpd


def load_state_df(state_abbrev):
    """
    Args:
        state_abbrev: (str) two letter state abbreviation

    Returns: (pd.DataFrame) of selected tract level metrics
    """
    state_df_path = os.path.join(constants.OPT_DATA_PATH,
                                 state_abbrev,
                                 'state_df.csv')
    df = pd.read_csv(state_df_path)
    return df.sort_values(by='GEOID').reset_index(drop=True)


def load_election_df(state_abbrev, custom_mapping='', custom_path=''):
    """
    Args:
        state_abbrev: (str) two letter state abbreviation
        custom_mapping: (str) location of tract aggregation mapping
            (subdir within OPT_DATA)

    Returns: (pd.DataFrame) of estimated votes by election and party for all tracts
    """
    election_df_path = os.path.join(constants.OPT_DATA_PATH,
                                    custom_path, state_abbrev,
                                    'election_df.csv')
    try:
        df = pd.read_csv(election_df_path)
        if custom_mapping:
            new_to_old, old_to_new = load_custom_mapping(state_abbrev, custom_mapping)
            df['custom_mapping'] = pd.Series(old_to_new)
            df = df.groupby('custom_mapping').sum()
    except FileNotFoundError:
        df = None
    return df  # Indices are equal to state_df integer indices


def load_acs(state_abbrev, year=None, county=False):
    """
    Args:
        state_abbrev: (str) two letter state abbreviation
        year: (int) year of ACS survey
        county: (bool) load ACS at the county or tract level

    Returns:
    """
    base_path = constants.COUNTY_DATA_PATH if county else constants.TRACT_DATA_PATH
    name_extension = 'county' if county else 'tract'
    year = year if year else constants.ACS_BASE_YEAR
    state_path = os.path.join(base_path,
                              '%s_acs5' % str(year),
                              '%s_%s.csv' % (state_abbrev, name_extension))
    return pd.read_csv(state_path, low_memory=False).sort_values('GEOID').reset_index(drop=True)


def load_tract_shapes(state_abbrev, year=None, custom_path=''):
    """
    Args:
        state_abbrev: (str) two letter state abbreviation
        year: (int) the year of the TIGERLINE shapefiles

    Returns: (gpd.GeoDataFrame) of tract shapes
    """
    if custom_path:
        tract_shapes = gpd.read_file(os.path.join(custom_path, state_abbrev))
        return tract_shapes.sort_values(by='GEOID20').reset_index(drop=True)
    if not year:
        year = constants.ACS_BASE_YEAR
    shape_fname = state_abbrev + '_' + str(year)
    tract_shapes = gpd.read_file(os.path.join(constants.CENSUS_SHAPE_PATH,
                                              shape_fname))
    tract_shapes = tract_shapes.to_crs("EPSG:3078")  # meters
    tract_shapes = tract_shapes[tract_shapes.ALAND > 0]
    return tract_shapes.sort_values(by='GEOID').reset_index(drop=True)


def load_adjacency_graph(state_abbrev):
    adjacency_graph_path = os.path.join(constants.OPT_DATA_PATH,
                                        state_abbrev, 'G.p')
    return nx.read_gpickle(adjacency_graph_path)


def load_district_shapes(state_abbrev=None, year=2018):
    """
    Args:
        state_abbrev: (str) two letter state abbreviation
        year: (int) districts of the desired year

    Returns: (gpd.GeoDataFrame) of district shapes
    """
    path = os.path.join(constants.GERRYPY_BASE_PATH, 'data',
                        'district_shapes', 'cd_' + str(year))
    gdf = gpd.read_file(path).sort_values('GEOID').to_crs("EPSG:3078")  # meters
    if state_abbrev is not None:
        state_geoid = str(constants.ABBREV_DICT[state_abbrev][constants.FIPS_IX])
        return gdf[gdf.STATEFP == state_geoid]
    else:
        return gdf


def load_opt_data(state_abbrev, special_input='', use_spt_matrix=False):
    """
    Args:
        state_abbrev: (str) two letter state abbreviation
        special_input: (str) subdirectory containing specialized inputs
        use_spt_matrix: (bool) load shortest path tree matrix instead of
            shortest path dict

    Returns: (pd.DataFrame, nx.Graph, np.array, dict) tuple of optimization
        data structures
    """
    data_base_path = os.path.join(constants.OPT_DATA_PATH, special_input, state_abbrev)
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


def load_ensemble(ensemble_path, state):
    state_path = glob.glob(os.path.join(ensemble_path, f'{state}_*.p'))
    file_name = os.path.basename(state_path[0])
    ensemble_name = file_name[:-2]
    ensemble = pickle.load(open(os.path.join(ensemble_path, file_name), 'rb'))
    ddf_path = os.path.join(ensemble_path, 'district_dfs', ensemble_name + '_district_df.csv')
    district_df = pd.read_csv(ddf_path)
    return ensemble, district_df


def load_census_places(state, year=constants.ACS_BASE_YEAR):
    path = os.path.join(constants.PLACES_PATH, f'{state}_{year}')
    return gpd.read_file(path).to_crs("EPSG:3078")


def load_custom_mapping(state, location):
    file_path = os.path.join(constants.OPT_DATA_PATH, location, state)
    new_to_old = pickle.load(open(os.path.join(file_path, 'new_ix_to_old_ix.p'), 'rb'))
    old_to_new = pickle.load(open(os.path.join(file_path, 'old_ix_to_new_ix.p'), 'rb'))
    return new_to_old, old_to_new


