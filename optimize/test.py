import sys
sys.path.append('../gerrypy_julia')

import numpy as np
from data.load import *
from experiments.louisiana_house import config

#state_df, G, lengths, tree_encoding = load_opt_data(config['state'], config['year'], config['granularity'])
#view = state_df.loc[[0,4,5]]
#print(list(view.index))
#view.loc[4, 'GEOID'] = 2
#print(view)

a = 0.6
b = 0.7
print((a>0.5) + (b>0.5))
# results_time_str = '1721243348'
# assignments_time_str = '0'
# assignments_df = load_assignments_df(constants.LOUISIANA_HOUSE_RESULTS_PATH, results_time_str, assignments_time_str, results_subdir='post_processing')
# print(assignments_df.loc[966])
# for i in range(len(assignments_df['District0'])):
#     if assignments_df.loc[i, 'District0']!=assignments_df.loc[i, 'District1']:
#         print(f'0-1:{i}')
#     if assignments_df.loc[i, 'District1']!=assignments_df.loc[i, 'District2']:
#         print(f'1-2:{i}')
#print(state_df['VAP'].min(), state_df['VAP'].mean(), state_df['VAP'].max(), state_df['VAP'].sum()/config['n_districts'])
#print(state_df['BVAP'].min(), state_df['BVAP'].mean(), state_df['BVAP'].max(), state_df['BVAP'].sum()/config['n_districts'])
#print(state_df['population'].min(), state_df['population'].mean(), state_df['population'].max(), state_df['population'].sum()/config['n_districts'])
#print(state_df['BVAP'].sum()/state_df['VAP'].sum())