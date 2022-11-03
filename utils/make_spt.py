from gerrypy import constants
import numpy as np
import os
import pickle
import networkx as nx
import scipy.sparse as sp
from gerrypy.optimize.utils import  build_spt_matrix

for state in os.listdir(constants.OPT_DATA_PATH):
    print(state)
    data_base_path = os.path.join(constants.OPT_DATA_PATH, state)
    state_df_path = os.path.join(data_base_path, 'state_df.csv')
    adjacency_graph_path = os.path.join(data_base_path, 'G.p')
    G = nx.read_gpickle(adjacency_graph_path)
    edge_dists = pickle.load(open(os.path.join(data_base_path, 'edge_dists.p'), 'rb'))
    spt_matrix = build_spt_matrix(G, edge_dists, list(range(len(G))), list(range(len(G))))
    sp.save_npz(os.path.join(data_base_path, 'spt_matrix.npz'), spt_matrix)