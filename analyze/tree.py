import numpy as np
from scipy.stats import t
import itertools


def number_of_districtings(leaf_nodes, internal_nodes):
    """
    Dynamic programming method to compute the total number of district plans.
    Args:
        leaf_nodes: (SHPnode list) with node capacity equal to 1 (has no child nodes).
        internal_nodes: (SHPnode list) with node capacity >1 (has child nodes).

    Returns: (int) the total number of distinct district plans.

    """
    nodes = {**leaf_nodes,  **internal_nodes}
    root = internal_nodes[0]

    def recursive_compute(current_node, all_nodes):
        if not current_node.children_ids:
            return 1

        total_districtings = 0
        for sample in current_node.children_ids:
            sample_districtings = 1
            for child_id in sample:
                child_node = nodes[child_id]
                sample_districtings *= recursive_compute(child_node, all_nodes)

            total_districtings += sample_districtings
        return total_districtings

    return recursive_compute(root, nodes)


def enumerate_partitions(leaf_nodes, internal_nodes):
    """
    Enumerate all feasible plans stored in the sample tree.
    Args:
        leaf_nodes: (SHPnode list) with node capacity equal to 1 (has no child nodes).
        internal_nodes: (SHPnode list) with node capacity >1 (has child nodes).

    Returns: A list of lists, each inner list is a plan comprised of leaf node ids.

    """
    def feasible_partitions(node, node_dict):
        if not node.children_ids:
            return [[node.id]]

        partitions = []
        for disjoint_sibling_set in node.children_ids:
            sibling_partitions = []
            for child in disjoint_sibling_set:
                sibling_partitions.append(feasible_partitions(node_dict[child],
                                                              node_dict))
            combinations = [list(itertools.chain.from_iterable(combo))
                            for combo in itertools.product(*sibling_partitions)]
            partitions.append(combinations)

        return list(itertools.chain.from_iterable(partitions))

    root = internal_nodes[0]

    node_dict = {**internal_nodes, **leaf_nodes}
    return feasible_partitions(root, node_dict)


def enumerate_distribution(leaf_nodes, internal_nodes, leaf_values):
    """
    Compute a given linear metric for all feasible plans.

    Use this function to achieve O(k) memory savings when computing
    exact distribution metrics of the plan ensemble.

    Args:
        leaf_nodes: (SHPnode list) with node capacity equal to 1 (has no child nodes).
        internal_nodes: (SHPnode list) with node capacity >1 (has child nodes).
        leaf_values: (dict) keyed by leaf node ID with value equal to district metric value.

    Returns:

    """
    def feasible_partitions(node, node_dict):
        if not node.children_ids:
            return [[leaf_dict[node.id]]]

        partitions = []
        for disjoint_sibling_set in node.children_ids:
            sibling_partitions = []
            for child in disjoint_sibling_set:
                sibling_partitions.append(feasible_partitions(node_dict[child],
                                                              node_dict))
            combinations = [list(itertools.chain.from_iterable(combo))
                            for combo in itertools.product(*sibling_partitions)]
            partitions.append(combinations)
        return [[sum(c)] for c in list(itertools.chain.from_iterable(partitions))]

    root = internal_nodes[0] if internal_nodes[0].is_root \
        else [n for n in internal_nodes if n.is_root][0]

    leaf_dict = {n.id: leaf_values[ix] for ix, n in enumerate(leaf_nodes.values())}
    node_dict = {**leaf_nodes, **internal_nodes}
    plan_values = feasible_partitions(root, node_dict)
    return np.array([item for sublist in plan_values for item in sublist])


def query_tree(leaf_nodes, internal_nodes, query_vals):
    """
    Dynamic programming method to find plan which maximizes linear district metric.
    Args:
        leaf_nodes: (SHPnode list) with node capacity equal to 1 (has no child nodes).
        internal_nodes: (SHPnode list) with node capacity >1 (has child nodes).
        query_vals: (list) of metric values per node.

    Returns: (float, list) tuple of optimal objective value and optimal plan.

    """
    nodes = {**leaf_nodes,  **internal_nodes}
    id_to_ix = {nid: ix for ix, nid in enumerate(sorted(leaf_nodes))}
    root = internal_nodes[0]

    def recursive_query(current_node, all_nodes):
        if not current_node.children_ids:
            return query_vals[id_to_ix[current_node.id]], [current_node.id]

        node_opts = []
        for sample in current_node.children_ids:  # Node partition
            sample_value = 0
            sample_opt_nodes = []
            for child_id in sample:  # partition slice
                child_node = nodes[child_id]
                child_value, child_opt = recursive_query(child_node, all_nodes)
                sample_value += child_value
                sample_opt_nodes += child_opt

            node_opts.append((sample_value, sample_opt_nodes))

        return max(node_opts, key=lambda x: x[0])

    return recursive_query(root, nodes)


def query_per_root_partition(leaf_nodes, internal_nodes, query_vals):
    """
    Dynamic programming method to find plan per root partition which
    maximizes linear district metric.
    Args:
        leaf_nodes: (SHPnode list) with node capacity equal to 1 (has no child nodes).
        internal_nodes: (SHPnode list) with node capacity >1 (has child nodes).
        query_vals: (list) of metric values per node.

    Returns: (float list, list list) tuple of optimal objective values and optimal plans
        per root partition.

    """
    nodes = {**leaf_nodes,  **internal_nodes}
    id_to_ix = {nid: ix for ix, nid in enumerate(sorted(leaf_nodes))}
    root = internal_nodes[0]

    def recursive_query(current_node, all_nodes):
        if not current_node.children_ids:
            return query_vals[id_to_ix[current_node.id]], [current_node.id]

        node_opts = []
        partitions = current_node.children_ids
        if current_node.is_root:
            partitions = [partitions[ROOT_PARTITION_IX]]
        for sample in partitions:  # Node partition
            sample_value = 0
            sample_opt_nodes = []
            for child_id in sample:  # partition slice
                child_node = nodes[child_id]
                child_value, child_opt = recursive_query(child_node, all_nodes)
                sample_value += child_value
                sample_opt_nodes += child_opt

            node_opts.append((sample_value, sample_opt_nodes))

        return max(node_opts, key=lambda x: x[0])

    partition_optimal_plans = []
    partition_optimal_values = []
    for ROOT_PARTITION_IX in range(0, len(root.children_ids)):
        val, plan = recursive_query(root, nodes)
        partition_optimal_values.append(val)
        partition_optimal_plans.append(plan)
    return partition_optimal_values, partition_optimal_plans



def party_step_advantage_query_fn(district_df, minimize=False):
    """
    Compute the expected seat share as a step function.
    Args:
        district_df: (pd.DataFrame) selected district statistics (requires "mean")
        minimize: (bool) negates values if true

    Returns: (np.array) leaf node query values

    """
    mean = district_df['mean'].values
    return mean < .50 * (-1 if minimize else 1)


def party_advantage_query_fn(district_df, minimize=False):
    """
    Compute the expected seat share as a t distribution.
    Args:
        district_df: (pd.DataFrame) selected district statistics
            (requires "mean", "std_dev", "DoF")
        minimize: (bool) negates values if true

    Returns: (np.array) leaf node query values

    """
    mean = district_df['mean'].values
    std_dev = district_df['std_dev'].values
    DoF = district_df['DoF'].values
    return (1 - t.cdf(.5, DoF, mean, std_dev)) * (-1 if minimize else 1)


def competitive_query_fn(district_df, minimize=False):
    """
    Compute the expected seat swaps.
    Args:
        district_df: (pd.DataFrame) selected district statistics
            (requires "mean", "std_dev", "DoF")
        minimize: (bool) negates values if true

    Returns: (np.array) leaf node query values

    """
    mean = district_df['mean'].values
    std_dev = district_df['std_dev'].values
    DoF = district_df['DoF'].values
    lose_p = t.cdf(.5, DoF, mean, std_dev)
    expected_flips = 2 * (1 - lose_p) * lose_p
    return expected_flips * (-1 if minimize else 1)
