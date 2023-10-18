import sys
sys.path.append('../gerrypy')
import numpy as np
import pandas as pd
from analyze.districts import vectorized_edge_cuts
from data.data2020.load import load_opt_data
import networkx as nx
from scipy import sparse
import os

def improvement(border_dists, assignments,state_df):
    """
    Given a solution set and the pairwise lengths of shared borders,
    return a solution set iteratively improved

    Args:
        solutions: (pandas dataframe) contains GEOID and district assignment
        perimeters: (np.array) n x n array showing the shared border length between district i and j
    """

    #shared_perims=cut_perimeter(border_dists,assignments,state_df)
    shared_perims=pd.read_csv(r"results/buffalo/buffalo1_results_1688399971/shared_perims.csv")
    to_fix=shared_perims[shared_perims['Percent Shared Perimeter']>0.6]
    print(to_fix)

    for index,row in to_fix.iterrows():
        i=assignments.index[assignments['GEOID20'] == row['GEOID20']].tolist()[0]
        assignments.loc[i,'District0']=row['Shared District']
    
    save_path=r"C:\Users\krazy\gerrypy\gerrypy\results\buffalo"
    assignments.to_csv(os.path.join(save_path, 'improved_assignments.csv'), index=False)


def cut_perimeter(border_dists, assignments,state_df):
    """
    Returns an array containing, for each block, the length of the perimeter it shares with
    districts it is not in

    Args:
        border_dists: (np.array) n x n array with the length of the shared border between every
        pair of adjacent blocks (0 if not adjacent)
        assignments: (np.array) array containing n district assignments (in same order as state_df)
    """

    districts = np.unique(assignments)
    print(districts)

    cx = sparse.coo_matrix(border_dists)
    for i,j,v in zip(cx.row, cx.col, cx.data):
        if i==16:
            print(str(j)+" "+str(v))

    p=np.zeros((len(assignments),len(districts)))

    for i,j,v in zip(cx.row, cx.col, cx.data):
        if assignments[i]!=assignments[j]:
            dist=np.where(districts==assignments[j])[0][0]
            t=v/state_df.loc[i,'perimeter']
            if t>1:
                print(state_df.loc[i,'GEOID20'])
                print(t)
            p[i,dist]=p[i,dist]+v/state_df.loc[i,'perimeter']

    #print(sparse.coo_matrix(p))
    #p2=-np.sort(-p)
    #print(p2)
    #print(sparse.coo_matrix(p))

    p2=sparse.coo_matrix(p)
    sorted_indices = np.lexsort((p2.row, -p2.data))
    sorted_p = sparse.coo_matrix((p2.data[sorted_indices], (p2.row[sorted_indices], p2.col[sorted_indices])))
    print(sorted_p)

    shared_perims=pd.DataFrame()
    shared_perims['GEOID20']=[state_df.loc[i,'GEOID20'] for i in sorted_p.row]
    shared_perims['Shared District']=[districts[j] for j in sorted_p.col]
    shared_perims['Percent Shared Perimeter']=sorted_p.data
    save_path=r"results/buffalo/buffalo1_results_1688399971/"
    shared_perims.to_csv(os.path.join(save_path, 'shared_perims.csv'), index=False)

    return(shared_perims)

    #assign_arr = np.zeros((len(assignments), len(districts)))
    #for i, district in enumerate(districts):
    #    assign_arr[:, i] = (assignments == district).astype(int)

    #print('assign_arr')
    #print(assign_arr)
    #print(sparse.csr_matrix(assign_arr))
    
    #edge_cuts=vectorized_edge_cuts(assign_arr,G)
    #adjacency_matrix = nx.linalg.graphmatrix.adjacency_matrix(G)
    
    #edge_cuts=(adjacency_matrix @ assign_arr) * assign_arr
    #print('edge cuts')
    #print(edge_cuts)
    #print(edge_cuts.shape)
    #print(sparse.coo_matrix(edge_cuts))



    #edge_cut_arr=np.zeros((len(assignments), len(assignments)))
    #for i in range(len(assignments)):
    #    for j in range(len(assignments)):
    #        edge_cut_arr[i, j] = (assignments[i] == assignments[j] and border_dists[i,j]>0).astype(int)

    #print(sparse.csr_matrix(edge_cut_arr))


if __name__ == '__main__':
    #adjacency_graph_path=r"C:\Users\krazy\gerrypy\gerrypy\data\buffalo_data\optimization_data\G.p"
    #G = nx.read_gpickle(adjacency_graph_path)
    #print('G')

    #border_dists = np.genfromtxt(r"data/buffalo_data/optimization_data/border_dists.csv", delimiter=',')
    assignments= pd.read_csv(r"results/buffalo/buffalo1_results_1688399971/buffalo1_1688400559assignments.csv")

    #state_df=pd.read_csv(r"data/buffalo_data/optimization_data/state_df.csv")
    print('S')

    #cut_perimeter(border_dists,assignments['District0'],state_df)
    improvement(1,assignments,3)