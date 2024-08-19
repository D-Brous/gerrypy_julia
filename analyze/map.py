import sys
sys.path.append('../gerrypy_julia')

import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import constants
import numpy as np
from data.load import *
from matplotlib.colors import ListedColormap
import libpysal
import random
import math
from analyze.maj_black import bvap_prop

# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors):
        self.colors = colors
        
# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(plt.Rectangle([width/len(orig_handle.colors) * i - handlebox.xdescent, 
                                          -handlebox.ydescent],
                           width / len(orig_handle.colors),
                           height, 
                           facecolor=c, 
                           edgecolor='k',
                           linewidth=0.2))

        patch = mpl.collections.PatchCollection(patches,match_original=True)

        handlebox.add_artist(patch)
        return patch

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

def cmap_grayscale(num_proportions, proportions):
    return ListedColormap([mpl.colormaps['binary'](proportions[i]) for i in range(num_proportions)])

def cgus_bm_shaded(ax, state_df, cgus, color_list):
    black_proportions = state_df['BVAP'].to_numpy() / state_df['VAP'].to_numpy()
    cmap = cmap_grayscale(len(cgus['GEOID']), black_proportions)
    cgus.plot(ax=ax, cmap=cmap)
    cgus.boundary.plot(ax=ax, edgecolor='red', linewidth=0.05)
    plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['binary']), ax=ax)

def cgus_pop_shaded(ax, state_df, cgus, color_list):
    ideal_pop = state_df['population'].sum() / config['n_districts']
    pop_proportions = state_df['population'].to_numpy() / ideal_pop
    cmap = cmap_grayscale(len(cgus['GEOID']), pop_proportions)
    cgus.plot(ax=ax, cmap=cmap)
    cgus.boundary.plot(ax=ax, edgecolor='red', linewidth=0.05)
    plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['binary']), ax=ax)

def cgus_six_colored(plan, ax, num_districts, black_proportions, cgus, color_list):
    cgu_neighbors = libpysal.weights.Queen.from_dataframe(cgus)
    cgu_adj_mtx, _ = cgu_neighbors.full()
    print("before cmap")
    #cmap = cmap_colored(color_list, len(cgu_adj_mtx), cgu_adj_mtx)
    print("after cmap")
    cgus.plot(ax=ax, linewidth=1.0)
    for cgu in cgus.index:
        cgus[cgus.index == cgu].boundary.plot(ax=ax, edgecolor='black', linewidth=0.2)

def cgus_highlight_disconnected(plan, ax, num_districts, black_proportions, cgus, color_list):
    coloring = np.full((3471, 4), 1, dtype=float)
    geomtypes = cgus.geometry.geom_type.values
    multipolygon_cgus = [i for i in range(len(geomtypes)) if geomtypes[i]=='MultiPolygon']
    coloring[multipolygon_cgus] = np.array([227, 25, 25, 256]) / 256
    cmap = ListedColormap(coloring)
    cgus.plot(ax=ax, cmap=cmap)
    cgus.boundary.plot(ax=ax, edgecolor='black', linewidth=0.05)
    for point, label in zip(cgus.geometry.representative_point(), cgus.index):
        #circle = plt.Circle((point.x, point.y), 0.2, color='black')
        #ax.add_patch(circle)
        if label in multipolygon_cgus:
            ax.annotate(label, xy=(point.x, point.y), xytext=(0.5, 0.5), textcoords="offset points", fontsize=0.5)

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

def districts_six_colored_bp_shaded(plan, ax, num_districts, black_proportions, cgus, color_list, num_shades=3):
    district_shapes = cgus.dissolve(by=f'District{plan}')
    district_neighbors = libpysal.weights.Queen.from_dataframe(district_shapes)
    district_adj_mtx, _ = district_neighbors.full()
    def shading_factor(black_proportion):
        if 0 <= black_proportion < 0.5:
            return 1 / 4
        else:
            return 1
    shading_factors = [shading_factor(black_proportions[i]) for i in range(num_districts)]
    #shading_factors = [1 / ((num_shades - math.floor(black_proportions[i] * 2 * (num_shades - 1))) * (black_proportions[i] < 0.5) + (black_proportions[i] >= 0.5)) for i in range(num_districts)]
    cmap = cmap_colored_shaded(color_list, num_districts, district_adj_mtx, shading_factors)
    cgus.plot(ax=ax, column=f'District{plan}', cmap=cmap, linewidth=5.0)
    cgus.boundary.plot(ax=ax, edgecolor='black', linewidth=0.05)
    for district in district_shapes.index:
        district_shapes[district_shapes.index == district].boundary.plot(ax=ax, edgecolor='black', linewidth=0.2)
    '''
    assignment_dict = assignment_ser_to_dict(cgus[f'District{plan}'])
    for point, district_id in zip(cgus.geometry.representative_point(), cgus[f'District{plan}']):
        #circle = plt.Circle((point.x, point.y), 0.2, color='black')
        #ax.add_patch(circle)
        label = bvap_prop(assignment_dict[district_id], state_df)
        ax.annotate(label, xy=(point.x, point.y), xytext=(0.5, 0.5), textcoords="offset points", fontsize=0.5)
    
    for point in district_shapes.geometry.representative_point():
        circle = plt.Circle((point.x, point.y), 0.2, color='black')
        ax.add_patch(circle)
    '''
    handles = [MulticolorPatch(color_list), MulticolorPatch([c * np.array([1,1,1,1/4]) for c in color_list])]
    labels = ['Majority Black', 'Not Majority Black']
    ax.legend(handles=handles, 
              labels=labels, 
              handler_map={MulticolorPatch: MulticolorPatchHandler()},
              loc='upper right', 
              labelspacing=0, 
              fontsize=3,
              handlelength=15, 
              handleheight=3, 
              bbox_to_anchor=(0.95, 0.95))

def districts_bp_grayscale(plan, ax, num_districts, black_proportions, cgus, color_list):
    cmap = cmap_grayscale(num_districts, black_proportions)
    cgus.plot(ax=ax, column=f'District{plan}', cmap=cmap, linewidth=1.0)
    plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['binary']), ax=ax)

def draw_cgu_maps(config, color_list, cgu_map_func):
    cgus = load_cgus(config['state'], config['year'], config['granularity'])
    state_df = load_state_df(config['state'], config['year'], config['granularity'])
    fig, ax = plt.subplots()
    ax.set_axis_off()
    pdf_path = os.path.join(config['results_path'], f'{cgu_map_func.__name__}.pdf')
    cgu_map_func(ax, state_df, cgus, color_list)
    fig.savefig(pdf_path, bbox_inches='tight', format='pdf', dpi=300)

def draw_district_maps(config, assignments_df, color_list, district_map_func, plans=None, highlight_diff=False):
    cgus = load_cgus(config['state'], config['year'], config['granularity'])
    save_path = os.path.join(config['results_path'], 'results_' + results_time_str)
    assignments_df['GEOID'] = assignments_df['GEOID'].astype(str).apply(lambda x: x.zfill(11))
    cgus = cgus.merge(assignments_df, on='GEOID', how='left')
    num_districts = int(max(assignments_df['District0'].to_list()))+1
    num_plans = 0
    for column_name in assignments_df.columns.values:
        if column_name[:8] == 'District':
            num_plans += 1
    
    state_df = load_state_df(config['state'], config['year'], config['granularity'])
    
    if plans is None:
        plans = np.arange(num_plans)
    
    prev_assignment_dict = None
    for plan in plans:
        fig, ax = plt.subplots()
        ax.set_axis_off()
        pdf_path = os.path.join(save_path, f'districting{plan}_{assignments_time_str}_{district_map_func.__name__}.pdf')
        bvap_per_district = np.zeros(num_districts)
        vap_per_district = np.zeros(num_districts)
        for i, row in enumerate(assignments_df.values):
            district = int(assignments_df.loc[i, f'District{plan}'])
            try:
                bvap_per_district[district] += int(state_df.loc[i, 'BVAP'])
                vap_per_district[district] += int(state_df.loc[i, 'VAP'])
            except IndexError:
                print(f'District indices for plan {plan} are incorrect')
                return
        black_proportions = bvap_per_district / vap_per_district
        if highlight_diff:
            diff = []
            curr_assignment_dict = assignment_ser_to_dict(assignments_df[f'District{plan}'])
            if prev_assignment_dict is not None:
                for district_id, district_region in curr_assignment_dict.items():
                    if prev_assignment_dict[district_id] != district_region:
                        diff.append(district_id)
            prev_assignment_dict = curr_assignment_dict
            district_map_func(plan, ax, num_districts, black_proportions, cgus, color_list)
            district_shapes = cgus.dissolve(by=f'District{plan}')
            for point in district_shapes.loc[diff].geometry.representative_point():
                circle = plt.Circle((point.x, point.y), 0.2, color='black')
                ax.add_patch(circle)
            '''
            for point, district_id in zip(district_shapes.geometry.representative_point(), district_shapes.index):
                #circle = plt.Circle((point.x, point.y), 0.2, color='black')
                #ax.add_patch(circle)
                label = bvap_prop(curr_assignment_dict[district_id], state_df)
                ax.annotate(label, xy=(point.x, point.y), xytext=(0.5, 0.5), textcoords="offset points", fontsize=0.5)
            '''
        else:
            district_map_func(plan, ax, num_districts, black_proportions, cgus, color_list)
        fig.savefig(pdf_path, bbox_inches='tight', format='pdf', dpi=300)

if __name__ == '__main__':
    cgu_map_funcs = {
        0: cgus_bm_shaded,
        1: cgus_pop_shaded,
        2: cgus_six_colored,
        3: cgus_highlight_disconnected,
    }
    district_map_funcs = {
        0: districts_six_colored_bm_outlined,
        1: districts_six_colored_bp_shaded,
        2: districts_bp_grayscale
    }

    results_time_str = '1721243348'  #'1719459163'
    assignments_time_str = '0' #'1719586752'
    assignments_df = load_assignments_df(constants.LOUISIANA_HOUSE_RESULTS_PATH, results_time_str, assignments_time_str, results_subdir='post_processing')

    colors = [[232, 23, 23, 256], [23, 131, 232, 256], [232, 138, 23, 256], [252, 226, 25, 256], [40, 138, 45, 256], [114, 66, 245, 256]]
    color_list = [np.array(color, dtype=float)/256 for color in colors]

    from experiments.louisiana_house import config
    draw_district_maps(config, assignments_df, color_list, district_map_funcs[1], highlight_diff=True)
    #draw_cgu_maps(config, color_list, cgu_map_funcs[1])