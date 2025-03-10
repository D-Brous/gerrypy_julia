import geopandas as gpd
import networkx as nx
import os
import numpy as np
import pandas as pd
from pysal.lib.weights import Queen
from matplotlib.colors import LinearSegmentedColormap as LSC
import matplotlib.pyplot as plt
import seaborn as sns
from gerrypy.analyze.plan import *


def color_map(gdf, districting):
    # Takes a few seconds

    # block : distr num
    inv_map = {block: k for k, district in districting.items()
               for block in district}

    gdf['district'] = pd.Series(inv_map)
    shapes = []
    for name, group in gdf.groupby('district'):
        shapes.append(group.geometry.unary_union)
    shape_series = gpd.GeoSeries(shapes)
    G = Queen(shapes).to_networkx()
    color_series = pd.Series(nx.greedy_color(G))
    n_colors = len(set(color_series.values))
    cmap = LSC.from_list("", ["red", "green", "dodgerblue",
                              'yellow', 'darkviolet', 'chocolate'][:n_colors])

    map_gdf = gpd.GeoDataFrame({'geometry': shape_series,
                                'color': color_series})
    ax = map_gdf.plot(column='color', figsize=(15, 15), cmap=cmap, edgecolor='black', lw=.5)
    gdf.plot(ax=ax, facecolor='none', edgecolor='white', lw=.05)
    ax.axis('off')
    return map_gdf, ax


def draw_adjacency_graph(gdf, G, include_tract_shapes=True, include_graph=True, size=(200, 150)):
    edgecolor = 'black' if include_tract_shapes else 'white'
    base = gdf.plot(color='white', edgecolor=edgecolor, figsize=size, lw=.5)
    edge_colors = ['green' if G[u][v].get('inferred', False) else 'red'
                   for u, v in G.edges]
    pos = {i: (geo.centroid.x, geo.centroid.y)
           for i, geo in gdf.geometry.iteritems()}
    if len(G) == len(gdf) + 1:  # If adj graph with dummy node
        pos[len(gdf)] = (min(gdf.centroid.x), min(gdf.centroid.y))
    nx.draw_networkx(G,
                     pos=pos,
                     ax=base,
                     node_size=1 if include_graph else 0,
                     width=.5,
                     linewidths=.5,
                     with_labels=False,
                     edge_color=edge_colors if include_graph else 'none')
    base.axis('off')
    return base


def color_synthetic_map(config, districting):
    h, w = config['synmap_config']['height'], config['synmap_config']['width']
    tmap = np.zeros((h, w))
    for ix, district in enumerate(list(districting.values())):
        for tract in district:
            tmap[tract // w, tract % w] += ix
    plt.matshow(tmap)


# helper function to visualize data
def plot_percentiles(xs, ys):

    probs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    percentiles = [np.percentile(ys, prob, axis=0) for prob in probs]

    ultra_light = '#ede1e1'
    light = "#DCBCBC"
    light_highlight = "#C79999"
    mid = "#B97C7C"
    mid_highlight = "#A25050"
    dark = "#8F2727"
    dark_highlight = "#7C0000"
    green = "#00FF00"

    plt.fill_between(xs, percentiles[0], percentiles[10],
                     facecolor=light, color=ultra_light)
    plt.fill_between(xs, percentiles[1], percentiles[9],
                     facecolor=light, color=light)
    plt.fill_between(xs, percentiles[2], percentiles[8],
                     facecolor=light_highlight, color=light_highlight)
    plt.fill_between(xs, percentiles[3], percentiles[7],
                     facecolor=mid, color=mid)
    plt.fill_between(xs, percentiles[4], percentiles[6],
                     facecolor=mid_highlight, color=mid_highlight)
    plt.plot(xs, percentiles[5], color=dark)


def politics_map(gdf, politics, districting):
    # Takes a few seconds

    # block : distr num
    inv_map = {block: k for k, district in districting.items()
               for block in district}

    gdf['district'] = pd.Series(inv_map)

    shapes = []
    colors = []
    for name, group in gdf.groupby('district'):
        shapes.append(group.geometry.unary_union)
        colors.append(politics[name])
    shape_series = gpd.GeoSeries(shapes)

    map_gdf = gpd.GeoDataFrame({'geometry': shape_series,
                                'color': pd.Series(colors)})
    ax = map_gdf.plot(column='color', figsize=(15, 15), edgecolor='black', lw=1.5,
                      cmap='seismic', vmin=.35, vmax=.65)
    gdf.plot(ax=ax, facecolor='none', edgecolor='white', lw=.05)
    ax.axis('off')
    return map_gdf, ax


def plot_politics_comparison(plan_gdfs, save_name, fig_dir, tract_gdf=None):
    # Takes a few seconds
    n_plots = len(plan_gdfs)
    fig, axs = plt.subplots(n_plots, 1)
    for fig_ix, plan in enumerate(plan_gdfs):
        ax = plan.plot(ax=axs[fig_ix], column='politics', figsize=(15, 15), edgecolor='black', lw=1,
                          cmap='seismic', vmin=.2, vmax=.8)
        if tract_gdf is not None:
            tract_gdf.plot(ax=axs[fig_ix], facecolor='none', edgecolor='white', lw=.05)
        axs[fig_ix].axis('off')
    zoom = 2
    w, h = fig.get_size_inches()
    fig.colorbar(ax.collections[0], ax=axs.ravel().tolist(), shrink=0.4, pad=.025,
                 label='Republican vote-share', orientation='horizontal')
    fig.set_size_inches(w * zoom, h * zoom)
    plt.savefig(os.path.join(fig_dir, '%s.eps' % save_name),
                format='eps', bbox_inches='tight')


def plot_seat_vote_curve(plan_df, n_samples=1000, height=10):
    xs, ys, stds = seat_vote_curve_t_estimate_with_seat_std(plan_df)
    seats, votes = sample_elections(plan_df, n=n_samples, p_seats=True)
    g = sns.jointplot(votes, seats, kind='kde', space=0, height=height)

    g.ax_joint.plot(xs, ys, color='red', linestyle=':', label='E[S]')
    g.ax_joint.fill_between(xs, np.maximum(ys - stds, 0), np.minimum(ys + stds, 1), alpha=0.2, color='red',
                            label='$E[S] \pm \sigma$')
    g.ax_joint.fill_between(xs, np.maximum(ys - 2 * stds, 0), np.minimum(ys + 2 * stds, 1), alpha=0.08, color='red',
                            label='$E[S] \pm 2\sigma$')
    g.ax_joint.axvline(x=.5, linestyle='--', color='black', lw=1)
    g.ax_joint.axhline(y=.5, linestyle='--', color='black', lw=1)


def plot_result_distribution(plan_df, n_samples=5000, symmetry=True):
    seats, votes = sample_elections(plan_df, n=n_samples, p_seats=True)
    plt.figure(figsize=(10, 10))
    sns.kdeplot(votes, seats, cmap='Reds', shade_lowest=False, shade=True)
    if symmetry:
        ax = sns.kdeplot(1 - votes, 1 - seats, cmap='Blues', shade_lowest=False, shade=True)
    ax.axvline(x=.5, linestyle='--', color='black', lw=1)
    ax.axhline(y=.5, linestyle='--', color='black', lw=1)