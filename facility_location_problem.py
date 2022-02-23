# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:06:25 2022

@author: Mattéo
"""

import numpy as np
import cplex
from collections import namedtuple


#-----------------------------------------------------------------------------
# Input variables
#-----------------------------------------------------------------------------

Client = namedtuple('Client', ('num',           # Client number, i
                               'demand',        # Demand of the client, d_{i}
                               'gain'))         # Unitary gain to satistfy the demand, r_{i}
                                    
Facility = namedtuple('Facility', ('num',         # Facility number, j
                                   'capacity',    # Supply of the facility, S_{j}
                                   'cost'))       # Facility opening cost, c_{j}

# List of facilities
Facilities = (Facility("1", 3, 480),
              Facility("2", 1, 200),
              Facility("3", 2, 320),
              Facility("4", 4, 340),
              Facility("5", 1, 300))
n = len(Facilities)

# List of clients
Clients = (Client("1", 2, 200),
           Client("2", 5, 150),
           Client("3", 3, 175))
m = len(Clients)

# Unitary transport cost between Client i and Facility j, t_{i,j}
t = [[20, 30, 50, 35, 60],
     [10, 60, 45, 55, 25],
     [45, 30, 55, 80, 20]]
t = np.array(t)

# Total unitary transport gain between Client i and Facility j, q_{i,j}
q = [[(Clients[i].gain-t[i,j])*Clients[i].demand for j in range(n)] for i in range(m)]
q = np.array(q).astype("float")

#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------


## Create CPO model
flp = cplex.Cplex()


## Create the decisions variables

# Whether or not facility j must be opened, x_{j}
x = list(flp.variables.add(obj = [f.cost for f in Facilities],
                           lb = [0]*n,
                           ub = [1]*n,
                           types = ["B"]*n))

y = [[] for i in range(m)]

for i in range(m):
    y[i] = list(flp.variables.add(obj = -q[i],
                                  lb = [0]*n,
                                  ub = [1]*n))


## Create the constraints

# Facility j cannot supply more than S_{j} :
    
for j in range(n):
    ind = [y[i][j] for i in range(m)] + [1.0]
    val = [c.demand for c in Clients] + [-Facilities[j].capacity]
    flp.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = ind, val = val)],
                                senses = ["L"],
                                rhs = [0.0])
    
# The demands must be fullfiled
for i in range(m):
    flp.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = y[i], val = [1.0]*n)],
                                senses = ["L"],
                                rhs = [1.0])
    
# No supply can come from a closed facility
for i in range(m):
    for j in range(n):
        flp.linear_constraints.add(lin_expr = cplex.SparsePair(ind = [y[i,j], x[j]], val = [1.0, -1.0]),
                                    senses = ["L"],
                                    rhs = [0.0])
        
# y_{i,j} are rates
for i in range(m):
    for j in range(n):
        flp.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = y[i,j], val = [1.0])],
                                    senses = ["L"],
                                    rhs = [1.0])

flp.solve()

