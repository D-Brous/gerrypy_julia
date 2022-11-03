import gerrypy.constants
from gerrypy.data.load import *
from gerrypy.pipelines.recom import convert_opt_data_to_gerrychain_input
from gerrypy.analyze.viz import *
from gerrypy.analyze.tree import *
from gerrypy.data.precinct_state_wrappers import wrappers
import numpy as np
import time
import random
from functools import partial
import argparse

from gerrychain import (GeographicPartition, MarkovChain,
                        updaters, constraints, accept, Election)
from gerrychain.updaters import cut_edges
from gerrychain.proposals import recom
from gerrychain.tree import recursive_tree_part



def run_chain(adj_graph, total_steps=100):
    my_updaters = {
        "population": updaters.Tally("population"),
    }

    initial_partition = GeographicPartition(adj_graph, assignment="initial_plan",
                                            updaters=my_updaters)
    ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)
    proposal = partial(recom,
                       pop_col="population",
                       pop_target=ideal_population,
                       epsilon=0.01,
                       node_repeats=2
                       )
    pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.01)
    chain = MarkovChain(
        proposal=proposal,
        constraints=[
            pop_constraint,
        ],
        accept=accept.always_accept,
        initial_state=initial_partition,
        total_steps=total_steps
    )

    districts = []
    plans = []
    for ix, partition in enumerate(chain):
        try:
            try:
                swap1, swap2 = list(partition.flows.keys())
            except ValueError:
                print('Attempted Infeasible')
                continue
            part1 = min(swap1, swap2)
            part2 = max(swap1, swap2)
            new_plan = plans[-1][:]
            new_plan[int(part1)] = len(districts)
            new_plan[int(part2)] = len(districts) + 1
            plans.append(new_plan)
            districts.append(list(partition.assignment.parts[part1]))
            districts.append(list(partition.assignment.parts[part2]))
        except AttributeError:  # First step in the chain
            districts = [list(part) for _, part in sorted(partition.assignment.parts.items())]
            plans.append(list(range(len(districts))))

    return districts, plans


def create_pools(n_sample_fn, n_splits):
    multi_district_states = {state: constants.seats[state]['house']
                             for state in constants.seats
                             if constants.seats[state]['house'] > 1}

    estimated_times = {k: n_sample_fn(v) for k, v in multi_district_states.items()}
    pool = [[] for _ in range(n_splits)]
    pool_times = np.zeros(n_splits)
    for state, time in sorted(estimated_times.items(), key=lambda x: x[1], reverse=True):
        pool_index = np.argmin(pool_times)
        pool[pool_index].append(state)
        pool_times[pool_index] += time
    return pool


def run_experiment(save_dir, n_samples_fn, n_splits=1, split_index=0):
    pool = create_pools(n_samples_fn, n_splits)
    trial = pool[split_index]
    random.shuffle(trial)

    for state in trial:
        n_samples = n_samples_fn(constants.seats[state]['house'])
        print(f'Running {state} chain with {n_samples} samples in process {split_index}')
        adj_graph = convert_opt_data_to_gerrychain_input(state)
        districts, plans = run_chain(adj_graph, n_samples)
        districts_file_path = os.path.join(save_dir,
                                           f'{state}_districts_{round(time.time())}.p')
        plans_file_path = os.path.join(save_dir,
                                           f'{state}_plans_{round(time.time())}.p')
        pickle.dump(districts, open(districts_file_path, 'wb'))
        pickle.dump(plans, open(plans_file_path, 'wb'))

if __name__=='__main__':
    SAVE_DIR = os.path.join(constants.RESULTS_PATH,
                            "geographic_rules", "baseline")
    os.makedirs(SAVE_DIR, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_splits', default=1, type=int)
    parser.add_argument('-i', '--index', default=0, type=int)
    parser.add_argument('--test', default=False, action='store_true')
    args = parser.parse_args()
    if args.test:
        n_samples_fn = lambda x: 2
    else:
        n_samples_fn = lambda x: int(10000 * x**1.5)
    run_experiment(SAVE_DIR, n_samples_fn, args.n_splits, args.index)


