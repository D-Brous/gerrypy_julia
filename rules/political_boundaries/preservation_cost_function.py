import sys
sys.path.append('../gerrypy')

import random
import networkx as nx
import numpy as np
import geopandas as gpd
from scipy.spatial.distance import pdist, squareform
from rules.political_boundaries.translations import *
from rules.political_boundaries.preservation_metrics import make_block_boundary_matrix


class CountyPreservationRandomizedCostFunction:
    def __init__(self, cm_config, tracts, adj_graph):
        # Normalize the names of counties
        county_fips_to_ix = {fips: ix for ix, fips
                             in enumerate(sorted(tracts.COUNTYFP.unique()))}

        # County polygons
        county_shapes = gpd.GeoSeries(tracts.groupby('COUNTYFP').apply(
            lambda x: x.geometry.unary_union))
        county_shapes.rename(county_fips_to_ix)

        # Mappings to and from tract indices
        tract_ix_to_county_ix = {ix: county_fips_to_ix[cid] for ix, cid
                                 in zip(tracts.index, tracts.COUNTYFP)}
        county_fips_to_tract_ix = tracts.groupby('COUNTYFP').apply(
            lambda x: list(x.index)).to_dict()

        county_ix_to_tract_list = {county_fips_to_ix[cfips]: tract_list
                                 for cfips, tract_list in county_fips_to_tract_ix.items()}

        # Construct county adjacency graph
        county_graph = nx.Graph()
        county_graph.add_nodes_from(county_ix_to_tract_list)
        for county, tract_list in county_ix_to_tract_list.items():
            adjacent_counties = set(tract_ix_to_county_ix[neighbor]
                                    for node in tract_list
                                    for neighbor in adj_graph[node])
            county_graph.add_edges_from([(county, cid) for cid in adjacent_counties
                                         if county != cid])

        self.county_ix_to_tract_list = county_ix_to_tract_list
        self.tract_to_county_ix = tract_ix_to_county_ix
        self.county_graph = county_graph
        self.county_centroids = np.array([county_shapes.values.centroid.x,
                                          county_shapes.values.centroid.y]).T
        self.county_pdists = squareform(pdist(self.county_centroids))

        self.county_tract_matrix = make_block_boundary_matrix(tracts)

        self.cm_config = cm_config

    def get_costs(self, area_df, centers):
        tract_centroids = area_df[['x', 'y']].values
        cost_coefficients = np.zeros((len(centers), len(area_df)))
        for cix, center in enumerate(centers):

            county_seed = self.tract_to_county_ix[center]
            if self.cm_config['sample_mode'] == 'none':
                return None
            elif self.cm_config['sample_mode'] == 'random_order_expansion':
                sample = random_order_expansion(self.county_graph, county_seed)
            elif self.cm_config['sample_mode'] == 'weighted_random_order_expansion':
                sample = weighted_random_order_expansion(self.county_graph, county_seed)
            elif self.cm_config['sample_mode'] == 'random_topological_ordering':
                sample = random_topological_ordering(self.county_graph, county_seed)
            elif self.cm_config['sample_mode'] == 'bfs_translation':
                sample = bfs(self.county_graph, county_seed)
            elif self.cm_config['sample_mode'] == 'random_translation':
                sample = random_translation(self.county_graph, county_seed)
            elif self.cm_config['sample_mode'] == 'constant_translation':
                sample = 1+ np.random.rand(len(self.county_centroids)) * 100
            else:
                raise ValueError('Unknown sample mode')

            if self.cm_config['sample_mode'][-len('translation'):] == 'translation':
                county_scale_factor = sample + np.random.rand(len(sample))
            else:
                county_scale_factor = np.argsort(np.array(sample)) ** 2

            center_centroid = area_df.loc[center, ['x', 'y']].values
            centered_tracts = tract_centroids - center_centroid

            centered_counties = self.county_centroids - center_centroid
            scaled_counties = (centered_counties.T * county_scale_factor).T
            county_translation_matrix = scaled_counties - centered_counties

            tract_translation = self.county_tract_matrix[area_df.index] @ county_translation_matrix

            tract_locations = centered_tracts + tract_translation
            population = area_df.population.values
            cost_coefficients[cix, :] = np.sqrt(np.sum(tract_locations ** 2, axis=1)) * population

        index = list(area_df.index)
        return {center: {index[bix]: cost for bix, cost in enumerate(cost_coefficients[cix])}
                for cix, center in enumerate(centers)}



