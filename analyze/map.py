import sys
sys.path.append('../gerrypy_julia')

import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import constants
import numpy as np
from data.load import load_cgus
from matplotlib.colors import ListedColormap
import libpysal
import random
import math

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
    coloring = six_coloring(district_adj_mtx)
    return ListedColormap([color_list[coloring[i]] for i in range(num_districts)])

def cmap_colored_shaded(color_list, num_districts, district_adj_mtx, shading_factors):
    coloring = six_coloring(district_adj_mtx)
    return ListedColormap([color_list[coloring[i]] * np.array([1,1,1,shading_factors[i]]) for i in range(num_districts)])

def cmap_grayscale(num_districts, black_proportions):
    return ListedColormap([mpl.colormaps['binary'](black_proportions[i]) for i in range(num_districts)])

def districts_six_colored_bm_outlined(plan, ax, num_districts, black_proportions, cgus, color_list):
    maj_black = black_proportions > 0.5
    district_shapes = cgus.dissolve(by=f'District{plan}')
    district_neighbors = libpysal.weights.Queen.from_dataframe(district_shapes)
    district_adj_mtx, _ = district_neighbors.full()
    cmap = cmap_colored(color_list, num_districts, district_adj_mtx)
    cgus.plot(ax=ax, column=f'District{plan}', cmap=cmap, linewidth=1.0)

    maj_black_districts = []
    i=0
    for district in district_shapes.index:
        if not maj_black[int(district)]:
            i += 1
        else:
            maj_black_districts.append(district)
    for district in maj_black_districts:
        district_shapes[district_shapes.index == district].boundary.plot(ax=ax, edgecolor='black', linewidth=0.2)

def districts_six_colored_bm_shaded(plan, ax, num_districts, black_proportions, cgus, color_list, num_shades=3):
    district_shapes = cgus.dissolve(by=f'District{plan}')
    district_neighbors = libpysal.weights.Queen.from_dataframe(district_shapes)
    district_adj_mtx, _ = district_neighbors.full()
    def shading_factor(black_proportion):
        if 0 <= black_proportion < 0.4:
            return 1 / 3
        elif 0.4 <= black_proportion < 0.5:
            return 2 / 3
        elif 0.5 <= black_proportion:
            return 1
    shading_factors = [shading_factor(black_proportions[i]) for i in range(num_districts)]
    #shading_factors = [1 / ((num_shades - math.floor(black_proportions[i] * 2 * (num_shades - 1))) * (black_proportions[i] < 0.5) + (black_proportions[i] >= 0.5)) for i in range(num_districts)]
    cmap = cmap_colored_shaded(color_list, num_districts, district_adj_mtx, shading_factors)
    cgus.plot(ax=ax, column=f'District{plan}', cmap=cmap, linewidth=1.0)

def cgus_bm_shaded(plan, ax, num_districts, black_proportions, cgus, color_list):
    cgus.plot(ax=ax)

def cgus_six_colored(plan, ax, num_districts, black_proportions, cgus, color_list):
    census_shape_neighbors = libpysal.weights.Queen.from_dataframe(cgus)
    census_shape_adj_mtx, _ = census_shape_neighbors.full()
    print("before cmap")
    #cmap = cmap_colored(color_list, len(census_shape_adj_mtx), census_shape_adj_mtx)
    print("after cmap")
    cgus.plot(ax=ax, linewidth=1.0)
    for census_shape in cgus.index:
        cgus[cgus.index == census_shape].boundary.plot(ax=ax, edgecolor='black', linewidth=0.2)
    
def black_proportions_grayscale(plan, ax, num_districts, black_proportions, cgus, color_list):
    cmap = cmap_grayscale(num_districts, black_proportions)
    cgus.plot(ax=ax, column=f'District{plan}', cmap=cmap, linewidth=1.0)
    plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['binary']), ax=ax)

def draw_maps(state_abbrev, year, granularity, results_df_path, assignment_time_str, color_list, map_func):
    cgus = load_cgus(state_abbrev, year=year, granularity=granularity)
    assignment_exists = False
    assignment_file = 'assignments_%s.csv' % assignment_time_str
    for file in os.listdir(results_path):
        if file==assignment_file:
            results_df_path = os.path.join(results_path, file)
            results_df = pd.read_csv(results_df_path)
            results_df['GEOID'] = results_df['GEOID'].astype(str).apply(lambda x: x.zfill(11))
            cgus = cgus.merge(results_df, on='GEOID', how='left')
            num_districts = int(max(results_df['District0'].to_list()))+1
            num_plans = 0
            for column_name in results_df.columns.values:
                if column_name[:8] == 'District':
                    num_plans += 1
            assignment_exists = True
    if not assignment_exists:
        raise FileExistsError('No file named assignments.csv')
    
    state_df_path = os.path.join(constants.OPT_DATA_PATH, os.path.join(granularity, state_abbrev, 'state_df.csv'))
    state_df = pd.read_csv(state_df_path)
    
    for plan in range(num_plans):
        fig, ax = plt.subplots()
        ax.set_axis_off()
        pdf_path = os.path.join(results_path, f'districting{plan}_{assignment_time_str}_{map_func.__name__}.pdf')
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
        black_proportions = black_pop_per_district / tot_pop_per_district
        map_func(plan, ax, num_districts, black_proportions, cgus, color_list)
        fig.savefig(pdf_path, bbox_inches='tight', format='pdf', dpi=300)

if __name__ == '__main__':
    results_time_str = '1718831492'
    assignment_time_str = '1719260871' #'1718830061'
    num_districts = 105
    num_plans = 1
    results_path = os.path.join(constants.LOUISIANA_HOUSE_RESULTS_PATH, 'results_' + results_time_str)
    colors = [[232, 23, 23, 256], [23, 131, 232, 256], [232, 138, 23, 256], [252, 226, 25, 256], [40, 138, 45, 256], [114, 66, 245, 256]]
    color_list = [np.array(color, dtype=float)/256 for color in colors]
    map_funcs = {
        0: districts_six_colored_bm_outlined,
        1: districts_six_colored_bm_shaded,
        2: cgus_bm_shaded,
        3: cgus_six_colored,
        4: black_proportions_grayscale
    }
    draw_maps('LA', '2010', 'block_group', results_path, assignment_time_str, color_list, map_funcs[4])