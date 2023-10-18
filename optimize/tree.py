import math
import random
import numpy as np


class SHPNode:
    def __init__(self, n_districts, area, id,
                 parent_id=None, is_root=False, center=None, num_mm=0): #TODO changed to include center, num_mm
        """
        SHPNodes store information needed to reconstruct the tree and gathers
        metadata from the generation process.

        Args:
            n_districts: (int) the capacity of the node
            area: (list) of block indices associated with the region
            id: (int) unique integer to identify this node
            parent_id: (int) id of parent node
            is_root: (bool) if this node is the root of the sample tree
        """
        self.is_root = is_root
        self.parent_id = parent_id
        self.n_districts = n_districts  # size

        self.area_hash = hash(frozenset(area))
        self.id = id

        self.area = area
        self.children_ids = []
        self.partition_times = []

        self.n_infeasible_samples = 0
        self.infeasible_children = 0

        self.center = center
        self.num_mm = num_mm

    def get_branch(self, child_id):
        for branch_ix, branch in enumerate(self.children_ids):
            if child_id in branch:
                return branch_ix, branch
        raise ValueError(f'Node {child_id} does not exist within node {self.id}')

    def delete_branch(self, branch_ix):
        del self.children_ids[branch_ix]
        del self.partition_times[branch_ix]

    def sample_n_splits_and_child_sizes(self, config):
        """
        Samples both the split size and the capacity of all children.

        Args:
            config: (dict) ColumnGenerator configuration

        Returns: (int list) of child node capacities.

        """
        n_distrs = self.n_districts
        n_splits = random.randint(min(config['min_n_splits'], n_distrs),
                                  min(config['max_n_splits'], n_distrs))

        ub = max(math.ceil(config['max_split_population_difference']
                           * n_distrs / n_splits), 2)
        lb = max(math.floor((1 / config['max_split_population_difference'])
                            * n_distrs / n_splits), 1)

        child_n_distrs = np.zeros(n_splits) + lb
        while int(sum(child_n_distrs)) != n_distrs:
            ix = random.randint(0, n_splits - 1)
            if child_n_distrs[ix] < ub:
                child_n_distrs[ix] += 1

        return child_n_distrs

    def __repr__(self):
        """Utility function for printing a SHPNode."""
        print_str = "Node %d \n" % self.id
        internals = self.__dict__
        for k, v in internals.items():
            if k == 'area':
                continue
            print_str += k + ': ' + v.__repr__() + '\n'
        return print_str


class ExampleTree:
    """
    For illustration purposes.
    """
    def __init__(self, config, n_districts, level=0):
        self.n_districts = n_districts
        self.level = 0

        if n_districts > 1:
            children_n_distrs = SHPNode.sample_n_splits_and_child_sizes()
            self.children = [ExampleTree(config, n, level + 1)
                             for n in children_n_distrs]
        else:
            self.children = None

        self.max_levels_to_leaf = 0
        self.max_layer()

    def __repr__(self, level=0):
        ret = "\t" * level + repr(self.n_districts) + "\n"
        if self.children is not None:
            for child in self.children:
                ret += child.__repr__(level + 1)
        return ret

    def max_layer(self):
        try:
            to_leaf = 1 + max([child.max_layer() for child in self.children])
            self.max_levels_to_leaf = to_leaf
            return to_leaf
        except TypeError:
            return 0
