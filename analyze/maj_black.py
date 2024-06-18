import sys
sys.path.append('../gerrypy_julia')

import numpy as np
import pandas as pd
import constants
import os
import fnmatch

def majority_black(results_df_path, state_df_path, num_districts):
    state_df = pd.read_csv(state_df_path)
    results_df = pd.read_csv(results_df_path)
    state_df = state_df.sort_values('GEOID')
    results_df = results_df.sort_values('GEOID')

    num_plans = 0
    for column in results_df.columns.tolist():
        if column[:8] == 'District':
            num_plans += 1
    
    black_pop_per_district = np.zeros((num_plans, num_districts))
    tot_pop_per_district = np.zeros((num_plans, num_districts))
    for i, row in enumerate(results_df.values):
        for j in range(num_plans):
            district = int(results_df.loc[i, f'District{j}'])
            try:
                black_pop_per_district[j, district] += int(state_df.loc[i, 'BVAP'])
                tot_pop_per_district[j, district] += int(state_df.loc[i, 'VAP'])
            except IndexError:
                print(results_df_path)
    
    avgs = black_pop_per_district / tot_pop_per_district
    return (avgs > 0.5)

if __name__ == '__main__':

    results_path = constants.LOUISIANA_HOUSE_RESULTS_PATH
    state_df_path = os.path.join(constants.OPT_DATA_PATH, 'block_group/LA/state_df.csv')
    for dir in os.listdir(results_path):
        dir_path = os.path.join(results_path, dir)
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                if file == 'assignments.csv':
                    results_df_path = os.path.join(dir_path, file)
                    print(os.path.join(dir, file), majority_black(results_df_path, state_df_path, 105).sum(axis=1))