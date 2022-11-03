import sys
sys.path.append('../gerrypy')

import random
import networkx as nx
import numpy as np
import geopandas as gpd
from gurobipy import *
from rules.political_boundaries.translations import *
from rules.political_boundaries.preservation_metrics import make_block_boundary_matrix


class BoundaryPreservationConstraints:
    def __init__(self, tracts, discount_factor=0.5):
        # Normalize the names of counties
        county_fips_to_ix = {fips: ix for ix, fips
                             in enumerate(sorted(tracts.COUNTYFP.unique()))}

        # Mappings to and from tract indices
        tract_ix_to_county_ix = {ix: county_fips_to_ix[cid] for ix, cid
                                 in zip(tracts.index, tracts.COUNTYFP)}
        county_fips_to_tract_ix = tracts.groupby('COUNTYFP').apply(
            lambda x: list(x.index)).to_dict()

        county_ix_to_tract_list = {county_fips_to_ix[cfips]: tract_list
                                   for cfips, tract_list in county_fips_to_tract_ix.items()}

        self.county_ix_to_tract_list = county_ix_to_tract_list
        self.tract_to_county_ix = tract_ix_to_county_ix

        self.county_tract_matrix = make_block_boundary_matrix(tracts)
        self.county_sizes = self.county_tract_matrix.sum(axis=0)

        self.discount_factor = discount_factor

    def augment_model(self, model, vars, region, costs):
        active_units = self.county_tract_matrix[region].sum(axis=0) == self.county_sizes
        active_units = np.nonzero(active_units)[0]
        complete_geographic_units = {}
        for center in vars:
            complete_geographic_units[center] = {}
            for unit_ix in active_units:
                unit_cost = sum(costs[center][tract_id] for tract_id
                                 in self.county_ix_to_tract_list[unit_ix])
                complete_geographic_units[center][unit_ix] = model.addVar(
                    vtype=GRB.BINARY,
                    obj=-self.discount_factor * unit_cost
                )

        model.addConstrs(quicksum(vars[i][j] for j in self.county_ix_to_tract_list[k])
                         >= self.county_sizes[k] * complete_geographic_units[i][k]
                         for i in complete_geographic_units
                         for k in complete_geographic_units[i])

        return model, complete_geographic_units
