import os
import pickle
import time
import argparse
from gerrypy.analyze.viz import *
from gerrypy.analyze.districts import *
from gerrypy.optimize.multiobjective import *


def make_objective_dfs_for_experiment(experiment_dir):
    trials = os.listdir(experiment_dir)
    os.makedirs(os.path.join(experiment_dir, 'objective_dfs'), exist_ok=True)
    for trial in trials:
        if trial[-2:] != '.p':
            continue
        state = trial[:2]
        tree = pickle.load(open(os.path.join(experiment_dir, trial), 'rb'))
        leaf_nodes = tree['leaf_nodes']
        internal_nodes = tree['internal_nodes']
        std = STD_CONSTANTS.get(state, None)
        objective_df = create_objective_df(state, leaf_nodes, internal_nodes, std)
        save_path = os.path.join(experiment_dir, 'objective_dfs', trial[:-2] + '.csv')
        objective_df.to_csv(save_path)
        print('Finished objective_df for trial', trial[:-2])


def node_plan_to_tract_dict(node_plan, leaf_nodes):
    return {node_id: leaf_nodes[node_id].area for node_id in node_plan}


def node_plan_to_tract_assignment(node_plan, leaf_nodes):
    plan = node_plan_to_tract_dict(node_plan, leaf_nodes)
    assignment = {}
    for dix, tracts in list(enumerate(plan.values())):
        for tract in tracts:
            assignment[tract] = dix + 1
    return assignment


def create_dra_assignment(vtd_assignment, state_df):
    return pd.DataFrame({
        'GEOID20': state_df.GEOID.apply(lambda x: str(x).zfill(11)),
        'District': pd.Series(vtd_assignment)
    }, dtype=object)


def gamo_pipeline(pipeline_config):
    trial_path = pipeline_config['trial_path']
    trial_file = pipeline_config['trial_file']
    trial = trial_file[:-2]
    state = trial[:2]

    tree = pickle.load(open(os.path.join(trial_path, trial_file), 'rb'))
    leaf_nodes = tree['leaf_nodes']
    internal_nodes = tree['internal_nodes']

    odf_path = os.path.join(trial_path, 'objective_dfs', trial + '.csv')
    odf = pd.read_csv(odf_path, index_col='node_id')
    state_df, _, _, _ = load_opt_data(state)

    p_obj_type = pipeline_config['primary_objective_type']
    p_obj = odf[pipeline_config['primary_objective']].values
    s_obj = create_secondary_objective(odf, pipeline_config['secondary_objective_formula'])

    r = run_multi_objective(leaf_nodes, internal_nodes, p_obj, s_obj,
                        epsilon=pipeline_config['epsilon'], opt_type=p_obj_type)
    pareto_primary, pareto_secondary, pareto_solutions, shards = r

    leaf_node_list = sorted(list(leaf_nodes.values()), key=lambda x: x.id)
    pareto_plans = []
    pareto_assignment_dfs = []
    for solution_ix, solution_point in enumerate(pareto_solutions):
        plan = {n: leaf_node_list[n].area for n in solution_point}
        assignment = node_plan_to_tract_assignment(plan, leaf_node_list)
        dra_df = create_dra_assignment(assignment, state_df)
        pareto_plans.append(plan)
        pareto_assignment_dfs.append(dra_df)

    pareto_dict = {
        'primary_objective': pareto_primary,
        'secondary_objective': pareto_secondary,
        'all_shards': shards,
        'plans': pareto_plans
    }

    result_path = os.path.join(trial_path, 'pareto_results', trial + pipeline_config['obj_name'])
    os.makedirs(result_path, exist_ok=True)
    pickle.dump(pareto_dict, open(os.path.join(result_path, 'pareto_front.p'), 'wb'))

    if pipeline_config['save_dra_assignments']:
        assignment_path = os.path.join(result_path, 'assignments')
        os.makedirs(assignment_path, exist_ok=True)
        for ix, df in enumerate(pareto_assignment_dfs):
            df.to_csv(os.path.join(assignment_path, f'{ix}.csv'), index=False)

    if pipeline_config['plot_pareto_solutions']:
        plot_path = os.path.join(result_path, 'plots')
        os.makedirs(plot_path, exist_ok=True)

        # vtd_shape_path = os.path.join(constants.GERRYPY_BASE_PATH,
        #                               'data', '2020_redistricting', 'vtd_shapes', state)
        shapes = load_tract_shapes(state)
        shapes = shapes.sort_values('GEOID').reset_index(drop=True)

        for ix, plan in enumerate(pareto_plans):
            district_vote_shares = {k: odf['mean'].values[k] for k in pareto_plans[ix]}
            _, ax = politics_map(shapes, district_vote_shares, pareto_plans[ix])
            p = round(pareto_primary[ix], 2)
            s = round(pareto_secondary[ix], 2)
            ax.set_title(
                f'Solution {ix} | Primary objective: {p} | Secondary objective: {s}',
                size=20)
            plt.savefig(os.path.join(plot_path, f'{ix}.png'), bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    CREATE_ODFS = True

    STD_CONSTANTS = {
        'CO': 0.03,
        'OH': 0.03,
        'IL': 0.03,
        'VA': 0.03,
    }
    EXPERIMENT_DIR = os.path.join(constants.RESULTS_PATH, 'va_trials')
    if CREATE_ODFS:
        make_objective_dfs_for_experiment(experiment_dir=EXPERIMENT_DIR)

    baseline_pipeline_config = {
        'trial_path': EXPERIMENT_DIR,
        'save_dra_assignments': True,
        'plot_pareto_solutions': True
    }
    trial_configs = [
        {
            'obj_name': 'ess',
            'trial_file': 'VA_baseline_6_400_plans_38898195.p',
            'primary_objective_type': 'maximize',
            'primary_objective': 'proportionality',
            'secondary_objective_formula': {
                'polsby_popper': -1,
            },
            'epsilon': 0.05,
        },
        {
            'obj_name': 'ess',
            'trial_file': 'VA_baseline_6_400_plans_38898195.p',
            'primary_objective_type': 'minimize',
            'primary_objective': 'proportionality',
            'secondary_objective_formula': {
                'polsby_popper': -1,
            },
            'epsilon': 0.05,
        },
        {
            'obj_name': 'ess',
            'trial_file': 'VA_county_6_300_plans_29356732.p',
            'primary_objective_type': 'maximize',
            'primary_objective': 'proportionality',
            'secondary_objective_formula': {
                'polsby_popper': -1,
            },
            'epsilon': 0.05,
        },
        {
            'obj_name': 'ess',
            'trial_file': 'VA_county_6_300_plans_29356732.p',
            'primary_objective_type': 'minimize',
            'primary_objective': 'proportionality',
            'secondary_objective_formula': {
                'polsby_popper': -1,
            },
            'epsilon': 0.05,
        },
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', default=0, type=int)
    parser.add_argument('-n', '--n_parallel', default=1, type=int)
    args = parser.parse_args()

    for trial_config in trial_configs[args.index::args.n_parallel]:
        pipeline_config = {**baseline_pipeline_config, **trial_config}
        start_t = time.time()
        gamo_pipeline(pipeline_config)
        end_t = time.time()

        run_t = round((end_t - start_t) / 60, 2)
        t_name = pipeline_config['trial_file']
        o_name = pipeline_config['obj_name']
        print(f'Finished {o_name} objective for {t_name} in {run_t} mins')