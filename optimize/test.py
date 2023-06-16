import sys
sys.path.append('../gerrypy')

import numpy as np
from data.data2020.load import *

#state_df, G, lengths, edge_dists = load_opt_data(state_abbrev='NY')

n=np.array([[1,1,0,1],[0,0,1,0],[0,1,1,0],[1,0,0,0]])
print(n)
n1=n.T[0]
print(n1)
n2= np.outer(n1,n1)
print(n2)
print(np.sum(n2))