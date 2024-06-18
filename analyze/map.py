import sys
sys.path.append('../gerrypy_julia')

import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import constants
import numpy as np
from data.load import load_census_shapes
from matplotlib.colors import ListedColormap
import libpysal
import random


def six_coloring(district_adj_mtx):
    def get_min_degree(adj_mtx):
        min_degree_v = 0
        min_degree = adj_mtx[0].sum()
        for i in range(1, len(adj_mtx)):
            curr_degree = adj_mtx[i].sum()
            if curr_degree < min_degree:
                min_degree_v = i
                min_degree = curr_degree
        return min_degree_v
    
    def remove_v(v, adj_mtx):
        upper = np.append(adj_mtx[:v,:v], adj_mtx[:v,v+1:], axis=1)
        lower = np.append(adj_mtx[v+1:,:v], adj_mtx[v+1:,v+1:], axis=1)
        return np.append(upper, lower, axis=0)
    
    n = len(district_adj_mtx)
    if n <= 6:
        return list(np.arange(n))
    coloring = list(np.arange(6))

    deletion_order = np.zeros((n-6), dtype='int')
    adj_mtxs = [district_adj_mtx]
    for i in range(n-7, -1, -1):
        min_degree_v = get_min_degree(adj_mtxs[0])
        deletion_order[i] = min_degree_v
        adj_mtxs = [remove_v(min_degree_v, adj_mtxs[0])] + adj_mtxs
    adj_mtxs = adj_mtxs[1:]

    for i in range(0, n-6):
        v = deletion_order[i]
        coloring = coloring[:v] + [-1] + coloring[v:]
        available_colors = np.ones((6))
        for j in range(i+6):
            if adj_mtxs[i][v][j] != 0:
                available_colors[coloring[j]] = 0
        available_colors_list = []
        for k in range(6):
            if available_colors[k] == 1:
                available_colors_list.append(k)
        random.seed(i)
        coloring[v] = random.choice(available_colors_list)
    
    return coloring

            

    
def cmap_colored(color_list, num_districts, district_adj_mtx):
    #ordered_colors = np.zeros((num_districts, 4))
    coloring = six_coloring(district_adj_mtx)
    return ListedColormap([color_list[coloring[i]] for i in range(num_districts)])
        #for i in range(len(ordered_colors)):
        #    ordered_colors[i] = color_list[coloring[i]]
    #return ListedColormap(ordered_colors)

def cmap_grayscale(num_districts, black_proportion):
    return ListedColormap([mpl.colormaps['binary'](black_proportion[i]) for i in range(num_districts)])

def draw_maps(state_abbrev, year, granularity, num_districts, num_plans, results_df_path, color_list, map_func):
    census_shapes = load_census_shapes(state_abbrev, year=year, granularity=granularity)
    assignment_exists = False
    for file in os.listdir(results_path):
        if file == 'assignments.csv':
            results_df_path = os.path.join(results_path, file)
            results_df = pd.read_csv(results_df_path)
            results_df['GEOID'] = results_df['GEOID'].astype(str).apply(lambda x: x.zfill(11))
            census_shapes = census_shapes.merge(results_df, on='GEOID', how='left')
            assignment_exists = True
    if not assignment_exists:
        raise FileExistsError('No file named assignments.csv')
    
    fig, ax = plt.subplots()
    ax.set_axis_off()
    state_df_path = os.path.join(constants.OPT_DATA_PATH, os.path.join(granularity, state_abbrev, 'state_df.csv'))
    state_df = pd.read_csv(state_df_path)
    
    for plan in range(num_plans):
        pdf_path = os.path.join(results_path, f'districting{plan}_{map_func.__name__}.pdf')
        black_pop_per_district = np.zeros(num_districts)
        tot_pop_per_district = np.zeros(num_districts)
        for i, row in enumerate(results_df.values):
            district = int(results_df.loc[i, f'District{plan}'])
            try:
                black_pop_per_district[district] += int(state_df.loc[i, 'BVAP'])
                tot_pop_per_district[district] += int(state_df.loc[i, 'VAP'])
            except IndexError:
                print(f'District indices for plan {plan} are incorrect')
                return
        black_proportion = black_pop_per_district/tot_pop_per_district
        map_func(plan, ax, black_proportion, census_shapes, color_list)
        fig.savefig(pdf_path, bbox_inches='tight', format='pdf', dpi=300)

def six_colored_bm_outlined(plan, ax, black_proportion, census_shapes, color_list):
    maj_black = black_proportion > 0.5
    district_shapes = census_shapes.dissolve(by=f'District{plan}')
    district_neighbors = libpysal.weights.Queen.from_dataframe(district_shapes)
    district_adj_mtx, _ = district_neighbors.full()
    cmap = cmap_colored(color_list, num_districts, district_adj_mtx)
    census_shapes.plot(ax=ax, column=f'District{plan}', cmap=cmap, linewidth=1.0)

    maj_black_districts = []
    i=0
    for district in district_shapes.index:
        if not maj_black[int(district)]:
            i += 1
            #district_shapes[district_shapes.index == district].boundary.plot(ax=ax, edgecolor='black', linewidth=0.4)
        else:
            maj_black_districts.append(district)
    for district in maj_black_districts:
        district_shapes[district_shapes.index == district].boundary.plot(ax=ax, edgecolor='black', linewidth=0.2)

def black_proportion_grayscale(plan, ax, black_proportion, census_shapes, color_list):
    #district_shapes = census_shapes.dissolve(by=f'District{plan}')
    cmap = cmap_grayscale(num_districts, black_proportion)
    census_shapes.plot(ax=ax, column=f'District{plan}', cmap=cmap, linewidth=1.0)

if __name__ == '__main__':
    time_str = '1718597026' #'1717700560'
    num_districts = 105
    num_plans = 1
    results_path = os.path.join(constants.LOUISIANA_HOUSE_RESULTS_PATH, 'la_house_results_' + time_str)
    colors = [[232, 23, 23, 256], [23, 131, 232, 256], [232, 138, 23, 256], [252, 226, 25, 256], [40, 138, 45, 256], [114, 66, 245, 256]]
    color_list = [np.array(color, dtype=float)/256 for color in colors]
    draw_maps('LA', '2010', 'block_group', num_districts, num_plans, results_path, color_list, black_proportion_grayscale)