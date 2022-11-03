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

if __name__ == "__main__":
    load_tract_shapes()