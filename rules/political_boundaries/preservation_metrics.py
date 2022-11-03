import numpy as np


def make_block_boundary_matrix(tracts):
    tract_ix_to_county_id = {ix: cid for ix, cid in zip(tracts.index, tracts.COUNTYFP)}
    county_id_to_tract_ix = tracts.groupby('COUNTYFP').apply(lambda x: list(x.index)).to_dict()

    bbm = np.zeros((len(tract_ix_to_county_id), len(county_id_to_tract_ix)))

    for cix, ckey in enumerate(sorted(list(county_id_to_tract_ix.keys()))):
        bbm[np.ix_(county_id_to_tract_ix[ckey], [cix])] = 1
    return bbm


def splits(block_district_matrix, block_boundary_matrix):
    overlap_matrix = block_boundary_matrix.T @ block_district_matrix
    boundary_sizes = block_boundary_matrix.sum(axis=0)
    return np.equal(overlap_matrix.T, boundary_sizes).sum(axis=1)


def pieces(block_district_matrix, block_boundary_matrix):
    overlap_matrix = block_boundary_matrix.T @ block_district_matrix
    return np.count_nonzero(overlap_matrix, axis=0)


def boundary_entropy(block_district_matrix, block_boundary_matrix, block_population):
    county_pops = block_boundary_matrix.T @ block_population
    state_population = block_population.sum()
    population_overlap = (block_boundary_matrix.T * block_population) @ block_district_matrix
    log_county_probability = (np.log(county_pops) - np.ma.log(population_overlap).filled(0).T).T
    population_overlap_fraction = population_overlap / state_population
    entropy = np.sum(population_overlap_fraction * log_county_probability, axis=0)
    return entropy
