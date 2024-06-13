import sys
sys.path.append('../gerrypy_julia')

import os
#os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import constants
import fnmatch
import numpy as np
from data.load import load_census_shapes
from analyze.maj_black import majority_black
from matplotlib.colors import ListedColormap
import libpysal


def create_cmap(color_list, num_districts, district_adj_mtx):
    ordered_colors = np.zeros((num_districts, 4))
    for i in range(len(ordered_colors)):
        ordered_colors[i] = color_list[i%len(color_list)]
    ordered_colors[26] = np.array([1.0, 1.0, 1.0, 1.0])
    return ListedColormap(ordered_colors)

def draw_maps(state_abbrev, year, granularity, num_districts, num_plans, shape_file, results_df_path, color_list):
    census_shapes = load_census_shapes(state_abbrev, year=year, granularity=granularity)
    #shape_path = os.path.join('data', 'shapes', granularity, state_abbrev + '_' + year, shape_file)
    #census_shapes = gpd.read_file(shape_path)
    assignment_exists = False
    for file in os.listdir(results_path):
        #print(file)
        if fnmatch.fnmatch(file, '*assignments.csv'):
            results_df_path = os.path.join(results_path, file)
            results_df = pd.read_csv(results_df_path)
            results_df['GEOID'] = results_df['GEOID'].astype(str).apply(lambda x: x.zfill(11))
            census_shapes = census_shapes.merge(results_df, on='GEOID', how='left')
            assignment_exists = True
    if not assignment_exists:
        raise FileExistsError('No file matching *assignments.csv')
    
    fig, ax = plt.subplots()
    ax.set_axis_off()
    #plt.show(block=True)
    #print(census_shapes.columns.tolist())
    state_df_path = os.path.join(constants.OPT_DATA_PATH, os.path.join(granularity, state_abbrev, 'state_df.csv'))
    state_df = pd.read_csv(state_df_path)
    
    for plan in range(num_plans):
        pdf_path = os.path.join(results_path, f'districting{plan}.pdf')
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
        maj_black = black_pop_per_district/tot_pop_per_district > 0.5
        district_shapes = census_shapes.dissolve(by='District0')
        district_neighbors = libpysal.weights.Queen.from_dataframe(district_shapes)
        district_adj_mtx, district_adj_mtx_indices = district_neighbors.full()
        #print(district_adj_mtx.shape)
        cmap = create_cmap(color_list, num_districts, district_adj_mtx)
        census_shapes.plot(ax=ax, column=f'District{plan}', cmap=cmap, linewidth=1.0)
        #colors = np.ones((num_districts-1, 3),dtype=float)
        #colors[0] = np.array([0.5,0.5,0.5])
        #census_shapes.plot(ax=ax, column=f'District{plan}', color=colors, linewidth=1.0)

        state_df_path = os.path.join(constants.OPT_DATA_PATH, os.path.join(granularity, state_abbrev, 'state_df.csv'))
        #majority_black(results_df_path, state_df_path, num_districts, num_plans)
        
        #print(district_shapes['GEOID'])
        for district in district_shapes.index:
            if maj_black[int(district)]:
                district_shapes[district_shapes.index == district].boundary.plot(ax=ax, edgecolor='yellow', linewidth=1, alpha=0.8)
            else:
                district_shapes[district_shapes.index == district].boundary.plot(ax=ax, edgecolor='black', linewidth=0.5, alpha=0.8)
        
        #for district, shape in district_shapes.iterrows():
        #    centroid = shape.geometry.centroid
        #    ax.text(centroid.x, centroid.y, str(district), ha='center', va='center', fontsize=8, color='black', weight='bold')
        
        fig.savefig(pdf_path, bbox_inches='tight', format='pdf', dpi=300)

if __name__ == '__main__':
    time_str = '1718214768' #'1717700560'
    num_districts = 105
    num_plans = 1
    results_path = os.path.join(constants.LOUISIANA_HOUSE_RESULTS_PATH, 'la_house_results_' + time_str)
    colors = [[232, 23, 23, 256], [23, 131, 232, 256], [232, 138, 23, 256], [252, 226, 25, 256], [40, 138, 45, 256]]
    color_list = [np.array(color, dtype=float)/256 for color in colors]
    draw_maps('LA', '2010', 'block_group', num_districts, num_plans, 'tl_2010_22_bg10.shp', results_path, color_list)