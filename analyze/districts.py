import sys
sys.path.append('../gerrypy')

import pandas as pd
import numpy as np
from scipy.stats import t
import math
import itertools
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, cdist
from data.buffalo_data.load import *
from analyze.tree import *
from analyze.subsample import *
from analyze.tree import *
from rules.political_boundaries.preservation_metrics import *

def average_entropy(conditional_p):
    """
    Compute average entropy of conditional probability of block cooccurence.
    Args:
        conditional_p: (np.array) n x n matrix where a_ij is P(i in D | j in D)

    Returns: (float) average entropy

    """
    return (- conditional_p * np.ma.log(conditional_p).filled(0) - (1 - conditional_p) *
            np.ma.log(1 - conditional_p).filled(0)).sum() / (conditional_p.shape[0] * conditional_p.shape[1])


def svd_entropy(sigma):
    """
    Compute the SVD entropy of the block district matrix.
    Args:
        sigma: (np.array) the singular values of the block district matrix.

    Returns: (float) SVD entropy

    """
    sigma_hat = sigma / sigma.sum()
    entropy = - (sigma_hat * (np.ma.log(sigma_hat).filled(0) / np.log(math.e))).sum()
    return entropy / (math.log(len(sigma)) / math.log(math.e))


def make_bdm(leaf_nodes, n_blocks=None):
    """
    Generate the block district matrix given by a sample trees leaf nodes.
    Args:
        leaf_nodes: SHPNode list, output of the generation routine
        n_blocks: (int) number of blocks in the state

    Returns: (np.array) n x d matrix where a_ij = 1 when block i appears in district j.

    """
    districts = [d.area for d in sorted(leaf_nodes.values(),
                 key=lambda x: x.id)]
    if n_blocks is None:
        n_blocks = max([max(d) for d in districts]) + 1
    block_district_matrix = np.zeros((n_blocks, len(districts)))
    for ix, d in enumerate(districts):
        block_district_matrix[d, ix] = 1
    return block_district_matrix


def bdm_metrics(block_district_matrix, k):
    """
    Compute selected diversity metrics of a district ensemble.

    WARNING: this function is O(d^2) in memory; this function should not be
        called with large ensembles as this will likely cause an OOM error.
    Args:
        block_district_matrix: (np.array)
        k: (int) number of seats in the plan

    Returns: (dict) of selected ensemble diversity metrics.

    """
    ubdm = np.unique(block_district_matrix, axis=1)
    max_rank = min(ubdm.shape)
    U, Sigma, Vt = np.linalg.svd(ubdm)

    Dsim = 1 - pdist(ubdm.T, metric='jaccard')

    precinct_coocc = ubdm @ ubdm.T
    precinct_conditional_p = precinct_coocc / ubdm.sum(axis=1)

    L = (np.diag(precinct_coocc.sum(axis=1)) - precinct_coocc)
    D_inv = np.diag(precinct_coocc.sum(axis=1) ** -.5)
    e = np.linalg.eigvals(D_inv @ L @ D_inv)

    conditional_entropy = average_entropy(precinct_conditional_p)

    return {
        'conditional_entropy': conditional_entropy,
        'average_district_sim': k * np.average(Dsim),
        'svd_entropy': svd_entropy(Sigma),
        '50p_approx_rank': sum(np.cumsum(Sigma / Sigma.sum()) < .5) / max_rank,
        '95p_approx_rank': sum(np.cumsum(Sigma / Sigma.sum()) < .95) / max_rank,
        '99p_approx_rank': sum(np.cumsum(Sigma / Sigma.sum()) < .99) / max_rank,
        'lambda_2': e[1],
        'lambda_k': e[k],
    }


def generation_metrics(cg, low_memory=False):
    """
    Compute ensemble generation summary statistics.
    Args:
        cg: (ColumnGenerator) that generated the ensemble
        low_memory: (bool) if bdm diversity metrics should be computed.

    Returns: (dict) of district ensemble summary statistics.

    """
    p_infeasible = cg.n_infeasible_partitions / \
                   (cg.n_infeasible_partitions + cg.n_successful_partitions)
    n_internal_nodes = len(cg.internal_nodes)
    districts = [d.area for d in cg.leaf_nodes.values()]
    duplicates = len(districts) - len(set([frozenset(d) for d in districts]))

    block_district_matrix = make_bdm(cg.leaf_nodes)
    district_df = create_district_df(cg.config['state'], block_district_matrix)

    edge_cuts = district_df.edge_cuts.values
    min_compactness, _ = query_tree(cg.leaf_nodes, cg.internal_nodes, edge_cuts)
    max_compactness, _ = query_tree(cg.leaf_nodes, cg.internal_nodes, -edge_cuts)
    compactness_disparity = - min_compactness / max_compactness

    expected_seats = party_advantage_query_fn(district_df)
    max_seats, _ = query_tree(cg.leaf_nodes, cg.internal_nodes, expected_seats)
    min_seats, _ = query_tree(cg.leaf_nodes, cg.internal_nodes, -expected_seats)
    seat_disparity = max_seats + min_seats  # Min seats is negative

    metrics = {
        'n_root_failures': cg.failed_root_samples,
        'p_infeasible': p_infeasible,
        'n_internal_nodes': n_internal_nodes,
        'n_districts': len(districts),
        'p_duplicates': duplicates / len(districts),
        'dispersion': district_df.dispersion.values.mean(),
        'cut_edges': np.array(edge_cuts).mean(),
        'compactness_disparity': compactness_disparity,
        'seat_disparity': seat_disparity
    }
    if low_memory:
        return metrics
    else:
        return {**metrics, **bdm_metrics(block_district_matrix, cg.config['n_districts'])}


def roeck_compactness(districts, state_df, lengths):
    """
    Calculate Roeck compactness approximation based on block centroids
    Args:
        districts: (list of lists) inner list contains block integer ixs of the district
        state_df: (pd.DataFrame) selected block statistics (requires "area" field")
        lengths: (np.array) Pairwise block distance matrix.

    Returns: (list) approximate Roeck compactness

    """
    compactness_scores = []
    for d in districts:
        area = state_df.loc[d]['area'].sum()
        radius = lengths[np.ix_(d, d)].max() / 2
        circle_area = radius ** 2 * math.pi
        roeck = area / circle_area
        compactness_scores.append(roeck)
    return compactness_scores


def roeck_more_exact(districts, state_df, tracts, lengths):
    """
    Calculate a more precise version of the Roeck compactness metric.
    Args:
        districts: (list of lists) inner list contains block integer ixs of the district
        state_df: (pd.DataFrame) selected block statistics (requires "area" field")
        tracts: (gpd.GeoSeries) tract polygons
        lengths: (np.array) Pairwise block distance matrix.

    Returns: List of district Roeck compactness scores.

    """
    def unwind_coords(poly):
        try:
            return np.array(poly.exterior.coords)
        except AttributeError:
            return np.concatenate([p.exterior.coords for p in poly])
    compactness_scores = []
    for d in districts:
        area = state_df.loc[d]['area'].sum()
        pairwise_dists = lengths[np.ix_(d, d)]
        max_pts = np.unravel_index(np.argmax(pairwise_dists), pairwise_dists.shape)
        t1, t2 = max_pts
        p1 = unwind_coords(tracts.loc[d[t1]].geometry)
        p2 = unwind_coords(tracts.loc[d[t2]].geometry)
        radius = np.max(cdist(p1, p2)) / 2000
        circle_area = radius ** 2 * math.pi
        roeck = area / circle_area
        compactness_scores.append(roeck)
    return compactness_scores


def dispersion_compactness(districts, state_df):
    """
    Compute the dispersion measure of compactness.
    Args:
        districts: (list of lists) inner list contains block integer ixs of the district
        state_df: (pd.DataFrame) selected block statistics (requires "population", "x", "y")

    Returns: (list) dispersion compactness for given districts.

    """
    compactness_scores = []
    for d in districts:
        population = state_df.loc[d]['population'].values
        dlocs = state_df.loc[d][['x', 'y']].values
        centroid = np.average(dlocs, weights=population, axis=0)
        geo_dispersion = (np.subtract(dlocs, centroid) ** 2).sum(axis=1) ** .5 / 1000
        dispersion = np.average(geo_dispersion, weights=population)
        compactness_scores.append(dispersion)
    return compactness_scores


def vectorized_edge_cuts(bdm, G):
    adjacency_matrix = nx.linalg.graphmatrix.adjacency_matrix(G)
    degree_vector = adjacency_matrix.sum(axis=1).flatten()
    all_edges = degree_vector @ bdm
    district_edges = ((adjacency_matrix @ bdm) * bdm).sum(axis=0)
    return np.asarray(all_edges - district_edges)[0]


def vectorized_polsby_popper(bdm, G, shape_gdf, land_threshold=-1):
    if land_threshold > -1:
        land_column = [c for c in shape_gdf.columns if c.startswith('ALAND')][0]
        water_column = [c for c in shape_gdf.columns if c.startswith('AWATER')][0]
        land_ratio = shape_gdf[land_column] / (shape_gdf[land_column] + shape_gdf[water_column])
        water_blocks = np.where(land_ratio <= land_threshold)[0]

    geometry = shape_gdf.geometry.to_crs(epsg=constants.CRS)
    edge_lengths = [(n1, n2, geometry[n1].intersection(geometry[n2]).length)
                    for n1, n2 in list(G.edges)]
    edge_lengths_zip = list(zip(*edge_lengths))
    n1s = np.array(edge_lengths_zip[0])
    n2s = np.array(edge_lengths_zip[1])
    edge_weights = np.array(edge_lengths_zip[2]) / 1000
    if land_threshold > -1:
        mask = ~(np.isin(n1s, water_blocks) | np.isin(n2s, water_blocks))
        n1s = n1s[mask]
        n2s = n2s[mask]
        edge_weights = edge_weights[mask]
    edge_weight_matrix = coo_matrix((edge_weights, (n1s, n2s)), shape=(len(G), len(G))).tocsr()
    edge_weight_matrix += edge_weight_matrix.T  # Make symmetric

    adjacency_matrix = nx.linalg.graphmatrix.adjacency_matrix(G)
    weighted_adjacency_matrix = adjacency_matrix.multiply(edge_weight_matrix)

    block_perimeters = geometry.length / 1000
    block_areas = geometry.area / 1000 ** 2
    if land_threshold > -1:
        block_perimeters[water_blocks] = 0
        block_areas[water_blocks] = 0

    total_interior_perimeter = block_perimeters @ bdm
    district_interior_perimeters = ((weighted_adjacency_matrix @ bdm) * bdm).sum(axis=0)
    district_perimeters = total_interior_perimeter - district_interior_perimeters
    district_areas = block_areas @ bdm
    polsby_popper = (4 * math.pi * district_areas) / district_perimeters**2
    return polsby_popper, district_perimeters


def vectorized_dispersion(bdm, state_df, bdm_shard_size=100_000):
    x_locs = state_df['x'].values
    y_locs = state_df['y'].values
    population = state_df.population.values

    n_cols = bdm.shape[1]
    dispersions = np.zeros(n_cols)
    for ix in range(0, n_cols, bdm_shard_size):
        bdm_shard = bdm[:, ix: ix + bdm_shard_size]
        district_pop = bdm_shard.T @ population
        bdm_p = bdm_shard.T * population

        district_centroid_x = bdm_p @ x_locs / district_pop
        district_centroid_y = bdm_p @ y_locs / district_pop

        centroid_distance_matrix = np.sqrt((((bdm_shard.T * x_locs).T - district_centroid_x)**2 +
                                            ((bdm_shard.T * y_locs).T - district_centroid_y)**2) * bdm_shard)
        shard_dispersion = (centroid_distance_matrix.T @ population) / district_pop / 1000
        dispersions[ix: ix + bdm_shard_size] = shard_dispersion
    return dispersions


def calculate_statewide_average_voteshare(election_df):
    partisan_totals = election_df.sum(axis=0).to_dict()
    elections = set(e[2:] for e in partisan_totals)
    election_results = {e: partisan_totals['R_' + e] /
                        (partisan_totals['R_' + e] + partisan_totals['D_' + e])
                    for e in elections}
    return np.mean(np.array(list(election_results.values())))


def aggregate_district_election_results(bdm, election_df):
    election_columns = {e: ix for ix, e in enumerate(election_df.columns)}
    elections = list(set([e[2:] for e in election_columns]))
    election_ixs = {e: {'D': election_columns['D_' + e], 'R': election_columns['R_' + e]}
                    for e in elections}

    election_vote_totals = bdm.T @ election_df.values

    result_df = pd.DataFrame({
        election: election_vote_totals[:, column_ixs['R']] /
                  (election_vote_totals[:, column_ixs['R']] + election_vote_totals[:, column_ixs['D']])
        for election, column_ixs in election_ixs.items()
    })
    return result_df


def aggregate_sum_metrics(bdm, metric_df):
    return pd.DataFrame(
        bdm.T @ metric_df.values,
        columns=metric_df.columns
    )


def aggregate_average_metrics(bdm, metric_df, weights):
    district_weight_normalizer = bdm.T @ weights
    return pd.DataFrame(
        (((bdm.T * weights) @ metric_df.values).T / district_weight_normalizer).T,
        columns=metric_df.columns
    )


def create_district_df(state, bdm, special_input=''):
    election_df = load_election_df(state, custom_path=special_input)
    state_df, G, _, _ = load_opt_data(state, special_input)

    sum_metrics = state_df[['area', 'population']]
    average_metrics = state_df.drop(columns=['area', 'population', 'GEOID'])

    sum_metric_df = aggregate_sum_metrics(bdm, sum_metrics)
    average_metric_df = aggregate_average_metrics(bdm, average_metrics,
                                                  state_df.population.values)
    vote_share_df = aggregate_district_election_results(bdm, election_df)

    compactness_df = pd.DataFrame({
        'edge_cuts': vectorized_edge_cuts(bdm, G),
        'dispersion': vectorized_dispersion(bdm, state_df)
    })

    vote_share_distribution_df = pd.DataFrame({
        'mean': vote_share_df.mean(axis=1),
        'std_dev': vote_share_df.std(ddof=1, axis=1),
        'DoF': len(vote_share_df.columns) - 1
    })

    return pd.concat([
        sum_metric_df,
        average_metric_df,
        compactness_df,
        vote_share_df,
        vote_share_distribution_df
    ], axis=1)


def metric_suboptimality_factor(metric, leaf_nodes, internal_nodes):
    optimal_value, _ = query_tree(leaf_nodes, internal_nodes, metric)
    optimal_value /= internal_nodes[0].n_districts
    return metric / optimal_value


def metric_zscore(metric, leaf_nodes, internal_nodes):
    metric_distribution = enumerate_distribution(leaf_nodes, internal_nodes, metric)
    metric_distribution = metric_distribution / internal_nodes[0].n_districts
    return (metric - metric_distribution.mean()) / metric_distribution.std()


def create_objective_df(state, leaf_nodes, internal_nodes, fixed_std=None):
    state_df, G, _, _ = load_opt_data(state)

    bdm = make_bdm(leaf_nodes, len(state_df))
    ddf = create_district_df(state, bdm)

    vs_mean = ddf['mean'].values
    vs_std_dev = ddf['std_dev'].values if fixed_std is None else fixed_std
    DoF = ddf['DoF'].values if fixed_std is None else 1000
    ess = 1 - t.cdf(.5, DoF, vs_mean, vs_std_dev)
    competitiveness = np.nan_to_num(-ess * np.log2(ess) - ((1-ess) * np.log2(1-ess)), 0)

    election_df = load_election_df(state)
    average_voteshare = calculate_statewide_average_voteshare(election_df)
    proportionality_coeffs = ess - average_voteshare
    efficiency_gap_coeffs = (ess - .5) - 2 * (average_voteshare - .5)

    shapes = load_tract_shapes(state)
    pp_scores, perimeters = vectorized_polsby_popper(bdm, G, shapes)

    bbm = make_block_boundary_matrix(shapes)#.rename(columns={'COUNTYFP20': 'COUNTYFP'}))
    n_splits = splits(bdm, bbm)
    n_pieces = pieces(bdm, bbm)
    entropy = boundary_entropy(bdm, bbm, state_df.population.values)

    raw_metric_dict = {
        'ess': ess,
        'proportionality': proportionality_coeffs,
        'efficiency_gap': efficiency_gap_coeffs,
        'competitiveness': competitiveness,
        'polsby_popper': pp_scores,
        'perimeter': perimeters,
        'county_splits': n_splits,
        'county_pieces': n_pieces,
        'county_entropy': entropy,
    }
    raw_metric_df = pd.DataFrame(raw_metric_dict)
    subopt_factor_df = pd.DataFrame({
        k + '_sub_factor': metric_suboptimality_factor(metric, leaf_nodes, internal_nodes)
        for k, metric in raw_metric_dict.items()
    })
    solution_count, parent_nodes = get_node_info(leaf_nodes, internal_nodes)
    pruned_internal_nodes = prune_sample_space(internal_nodes, solution_count, parent_nodes, 10_000)
    zscore_df = pd.DataFrame({
        k + '_zscore': metric_zscore(metric, leaf_nodes, pruned_internal_nodes)
        for k, metric in raw_metric_dict.items()
    })
    objective_df = pd.concat([ddf, raw_metric_df, subopt_factor_df, zscore_df], axis=1)
    objective_df.index = pd.Index(sorted(list(leaf_nodes.keys())), name='node_id')
    return objective_df
