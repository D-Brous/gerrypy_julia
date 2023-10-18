import sys
sys.path.append('../gerrypy')

import os
import time
import json
import numpy as np
from data.data2020.load import load_state_df
#from data.buffalo_data.load import load_state_df
import pandas as pd

from gurobipy import quicksum
from gurobipy import Model
from gurobipy import GRB
import constants
import os

def population_tolerance():
    state_df = load_state_df()
    populations = state_df.groupby(['Council2'])['TOTAL_ADJ'].sum()
    pop_dif = max(populations)-min(populations)
    pop_tolerance = pop_dif/np.mean(populations)
    return pop_tolerance

def gurobitest():
    bools=[1,0,0,1,0]
    costs=[5,1,1,4,2]
    x={}
    D=range(5)
    master = Model("master LP")
    for j in D:
        x[j] = master.addVar(vtype=GRB.BINARY, name="x(%s)" % j)
    master.addConstr(quicksum(x[j] * bools[j] for j in D)>=1, name="bools")
    master.addConstr(quicksum(x[j] for j in D)==2, name="exactly2")
    master.setObjective(quicksum(costs[j] * x[j] for j in D), GRB.MINIMIZE)

    master.addConstr(6>=0,name='basictest')

    master.optimize()

    print(master.getConstrByName("basictest").slack)
    print(x)

def get_p_white():
    state_df = load_state_df('AL')
    p_white_df = state_df[['GEOID', 'p_white']]

    save_path = os.path.join(constants.OPT_DATA_PATH_2020, 'AL')
    p_white_df.to_csv(os.path.join(save_path, 'p_white.csv'), index=False)

if __name__ == "__main__":
    get_p_white()