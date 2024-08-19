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
import libpysal
import constants


def load_state_df(state, year, granularity):
    """
    Args:
        state: (str) two letter state abbreviation
        granularity: (str) granularity of cgus

    Returns: (pd.DataFrame) of selected tract level metrics
    """
    state_df_path = os.path.join(constants.OPT_DATA_PATH,
                                 granularity,
                                 state,
                                 str(year),
                                 'state_df.csv')
    return pd.read_csv(state_df_path).sort_values('GEOID')


def load_election_df(state, custom_mapping='', custom_path=''):
    """
    Args:
        state: (str) two letter state abbreviation
        custom_mapping: (str) location of tract aggregation mapping
            (subdir within OPT_DATA)

    Returns: (pd.DataFrame) of estimated votes by election and party for all tracts
    """
    election_df_path = os.path.join(constants.OPT_DATA_PATH,
                                    custom_path, state,
                                    'election_df.csv')
    try:
        df = pd.read_csv(election_df_path)
        if custom_mapping:
            new_to_old, old_to_new = load_custom_mapping(state, custom_mapping)
            df['custom_mapping'] = pd.Series(old_to_new)
            df = df.groupby('custom_mapping').sum()
    except FileNotFoundError:
        df = None
    return df  # Indices are equal to state_df integer indices


def load_acs(state, year, county=False):
    """
    Args:
        state: (str) two letter state abbreviation
        year: (int) year of ACS survey
        county: (bool) load ACS at the county or tract level

    Returns:
    """
    base_path = constants.COUNTY_DATA_PATH if county else constants.TRACT_DATA_PATH
    name_extension = 'county' if county else 'tract'
    state_path = os.path.join(base_path,
                              '%s_acs5' % str(year),
                              '%s_%s.csv' % (state, name_extension))
    return pd.read_csv(state_path, low_memory=False).sort_values('GEOID').reset_index(drop=True)


def load_cgus(state, year, granularity):
    """
    Args:
        state: (str) two letter state abbreviation
        year: (int) the year of the TIGERLINE shapefiles
        granularity: (str) the granularity of census graphical units

    Returns: (gpd.GeoDataFrame) of tract shapes
    """
    cgus = gpd.read_file(os.path.join(constants.CENSUS_SHAPE_PATH,
                                              granularity,
                                              state,
                                              str(year)))
    cgus = cgus.to_crs("EPSG:3078")  # meters
    #cgus = cgus[cgus.ALAND > 0]
    if 'GEOID10' in cgus.columns:
        cgus.rename(columns={'GEOID10' : 'GEOID'}, inplace=True)
    return cgus.sort_values(by='GEOID').reset_index(drop=True)


def load_adjacency_graph(state, year, granularity):
    """
    Args:
        state: (str) two letter state abbreviation
        granularity: (str) granularity of cgus
        assignment: (pd.DataFrame) assignment of cgus to districts

    Returns: (nx.Graph) adjacency graph of districts
    """
    adjacency_graph_path = os.path.join(constants.OPT_DATA_PATH,
                                        granularity,
                                        state, 
                                        str(year),
                                        'G.p')
    return nx.read_gpickle(adjacency_graph_path)

def load_district_adjacency_graph(state, year, granularity, assignment_ser):
    """
    Creates adjacency graph for given set of districts

    Args:
        state: (str) two letter state abbreviation
        year: (int) year of desired census
        granularity: (str) granularity of cgus
        assignment_ser: (pd.Series) assignment of cgus to districts

    Returns: (nx.Graph) adjacency graph of districts
    """
    cgus = load_cgus(state, year, granularity)
    cgus['District0'] = assignment_ser
    district_shapes = cgus.dissolve(by='District0')
    shape_list = district_shapes.geometry.to_list()
    return libpysal.weights.Rook.from_iterable(shape_list).to_networkx()

def load_district_shapes(state=None, year=2018):
    """
    Args:
        state: (str) two letter state abbreviation
        year: (int) districts of the desired year

    Returns: (gpd.GeoDataFrame) of district shapes
    """
    path = os.path.join(constants.GERRYPY_BASE_PATH, 'data',
                        'district_shapes', 'cd_' + str(year))
    gdf = gpd.read_file(path).sort_values('GEOID').to_crs("EPSG:3078")  # meters
    if state is not None:
        state_geoid = str(constants.ABBREV_DICT[state][constants.FIPS_IX])
        return gdf[gdf.STATEFP == state_geoid]
    else:
        return gdf


def load_opt_data(state, year, granularity, use_spt_matrix=False, opt_data_path=constants.OPT_DATA_PATH):
    """
    Args:
        state: (str) two letter state abbreviation
        special_input: (str) subdirectory containing specialized inputs
        use_spt_matrix: (bool) load shortest path tree matrix instead of
            shortest path dict

    Returns: (pd.DataFrame, nx.Graph, np.array, dict) tuple of optimization
        data structures
    """
    data_base_path = os.path.join(opt_data_path, granularity, state, str(year))
    state_df = load_state_df(state, year, granularity)
    G = load_adjacency_graph(state, year, granularity)

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

def load_assignments_df(results_path, results_time_str, assignments_time_str, results_subdir=None):
    if results_subdir is None:
        assignments_df_path = os.path.join(results_path, 
                                           'results_' + results_time_str, 
                                           'assignments_' + assignments_time_str + '.csv')
    else:
        assignments_df_path = os.path.join(results_path, 
                                           'results_' + results_time_str,
                                           results_subdir, 
                                           'assignments_' + assignments_time_str + '.csv')
    
    return pd.read_csv(assignments_df_path)

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

def load_tree(save_path):
    tree = {}
    for file in os.listdir(os.path.join(save_path, 'tree')):
        if file == 'root.pkl':
            tree[-1] = load_object(os.path.join(save_path, 'tree', file))
        else:
            tree[int(file[12:-4])] = load_object(os.path.join(save_path, 'tree', file))   
    return tree

def load_cdms(save_path):
    cdms = {}
    for file in os.listdir(os.path.join(save_path, 'cdms')):
        cdms[int(file[5:-4])] = load_object(os.path.join(save_path, 'cdms', file))   
    return cdms

def assignment_ser_to_dict(assignment_ser, n_districts=None):
    if n_districts is None:
        n_districts = len(assignment_ser.unique())
    return {district_id: assignment_ser[assignment_ser==district_id].index.tolist() for district_id in range(n_districts)}

def assignment_dict_to_ser(assignment_dict, n_cgus=None):
    if n_cgus is None:
        n_cgus = sum(len(district) for district in assignment_dict.values())
    assignment_arr = np.zeros(n_cgus, dtype=int)
    for district_id, district in assignment_dict.items():
        assignment_arr[district] = district_id
    return pd.Series(assignment_arr, np.arange(n_cgus))