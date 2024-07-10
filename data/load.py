"""This module is used to interface with all downloaded data.

You must use this to load all data or else indices may become inconsistent."""

import sys
sys.path.append('../gerrypy_julia')

import pickle
import networkx as nx
import os
import glob
import numpy as np
import scipy.sparse as sp
import pandas as pd
import geopandas as gpd

import constants


def load_state_df(state_abbrev, granularity):
    """
    Args:
        state_abbrev: (str) two letter state abbreviation
        granularity: (str) granularity of cgus

    Returns: (pd.DataFrame) of selected tract level metrics
    """
    state_df_path = os.path.join(constants.OPT_DATA_PATH,
                                 granularity,
                                 state_abbrev,
                                 'state_df.csv')
    return pd.read_csv(state_df_path)


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


def load_cgus(state_abbrev, year=None, custom_path='', granularity='tract'):
    """
    Args:
        state_abbrev: (str) two letter state abbreviation
        year: (int) the year of the TIGERLINE shapefiles

    Returns: (gpd.GeoDataFrame) of tract shapes
    """
    if custom_path:
        cgus = gpd.read_file(os.path.join(custom_path, state_abbrev))
        return cgus.sort_values(by='GEOID20').reset_index(drop=True)
    if not year:
        year = constants.ACS_BASE_YEAR
    shape_fname = state_abbrev + '_' + str(year)
    cgus = gpd.read_file(os.path.join(constants.CENSUS_SHAPE_PATH,
                                              granularity,
                                              shape_fname))
    cgus = cgus.to_crs("EPSG:3078")  # meters
    #cgus = cgus[tract_shapes.ALAND > 0]
    if 'GEOID10' in cgus.columns:
        cgus.rename(columns={'GEOID10' : 'GEOID'}, inplace=True)
    return cgus.sort_values(by='GEOID').reset_index(drop=True)


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


def load_opt_data(state_abbrev, granularity, use_spt_matrix=False, opt_data_path=constants.OPT_DATA_PATH):
    """
    Args:
        state_abbrev: (str) two letter state abbreviation
        special_input: (str) subdirectory containing specialized inputs
        use_spt_matrix: (bool) load shortest path tree matrix instead of
            shortest path dict

    Returns: (pd.DataFrame, nx.Graph, np.array, dict) tuple of optimization
        data structures
    """
    data_base_path = os.path.join(opt_data_path, granularity, state_abbrev)
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

def load_matrix(filepath):
    sparse_mtx = sp.sparse.load_npz(filepath)
    return sparse_mtx.toarray()

def load_object(filepath):
    with open(filepath, 'rb') as inp:
        obj = pickle.load(inp)
    return obj

def load_tree(save_dir):
    tree = {}
    for file in os.listdir(os.path.join(save_dir, 'tree')):
        if file == 'root.pkl':
            tree[-1] = load_object(os.path.join(save_dir, 'tree', file))
        else:
            tree[int(file[12:-4])] = load_object(os.path.join(save_dir, 'tree', file))   
    return tree

def load_cdms(save_dir):
    cdms = {}
    for file in os.listdir(os.path.join(save_dir, 'cdms')):
        cdms[int(file[5:-4])] = load_object(os.path.join(save_dir, 'cdms', file))   
    return cdms

def test():
    print("we're in the correct module")