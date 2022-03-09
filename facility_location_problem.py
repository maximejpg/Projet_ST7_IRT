import numpy as np
import cplex
from collections import namedtuple
import random


#-----------------------------------------------------------------------------
# Input variables for example
#-----------------------------------------------------------------------------


# List of facilities                                   
Facility = namedtuple('Facility', ('num',         # Facility number, j
                                   'capacity',    # Supply of the facility, S_{j}
                                   'cost'))       # Facility opening cost, c_{j}
n = random.randint(10,20)
Facilities = ()
for i in range(n):
    Facilities += (Facility(str(i),random.randint(4,8), random.randint(10,50) ),)


# List of clients
Client = namedtuple('Client', ('num',           # Client number, i
                               'demand',        # Demand of the client, d_{i}
                               'gain'))         # Unitary gain to satistfy the demand, r_{i}
m = random.randint(10,20)
Clients = ()
for i in range(m):
    Clients += (Client(str(i),random.randint(1,5), random.randint(10,50) ),)


# Unitary transport cost between Client i and Facility j, t_{i,j}
t = np.random.randint(low = 15, high = 80, size = (m,n))


# Total unitary transport gain between Client i and Facility j, q_{i,j}
q = [[(Clients[i].gain-t[i,j])*Clients[i].demand for j in range(n)] for i in range(m)]
q = np.array(q).astype("float")


#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

def facility_location_problem(Facilities, Clients, q, eq = False) :

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
    flp.linear_constraints.add(lin_expr = [[[y[i][j] for i in range(m)],[c.demand for c in Clients]] for j in range(n)],
                               senses = ["L"]*n,
                               rhs = [f.capacity for f in Facilities]
                               )
        
    # Demands are fullfiled
    if eq == False :
        flp.linear_constraints.add(lin_expr = [[y[i],[1.0]*n] for i in range(m)],
                                senses = ["L"]*m,                               
                                rhs = [1.0]*m)
    
    else : 
        flp.linear_constraints.add(lin_expr = [[y[i],[1.0]*n] for i in range(m)],
                                    senses = ["E"]*m,                               
                                    rhs = [1.0]*m)
                
    # No supply can come from a closed facility
    for i in range(m):
        for j in range(n):
            flp.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = [y[i][j], x[j]], val = [1.0, -1.0])],
                                       senses = ["L"],
                                       rhs = [0.0])
    
    ### ADD a constraint to control the minimum global delivery rate
    
    # tol = 0.8
    # flp.linear_constraints.add(lin_expr = [[[sum(y[i][j] for j in range(n)) for i in range(m)], [c.demand for c in Clients]]], 
    #                            senses = ["G"],
    #                            rhs = [tol*sum(c.demand for c in Clients)])
    
    
    
    ## Solve the model
    
    flp.solve()
    return flp


#-----------------------------------------------------------------------------
# Get and print results
#-----------------------------------------------------------------------------

flp = facility_location_problem(Facilities, Clients, q)
    
x_sol = flp.solution.get_values()[:n]
y_sol = np.array([flp.solution.get_values()[n+(i*n):n+(i+1)*n] for i in range(m)])
distr = np.array([y_sol[i]*Clients[i].demand for i in range(m)])


print(distr)

for j in range(n):
    if x_sol[j] == 0:
        print("Facility " + str(j+1) + " is closed")
    else :
        print("Facility " + str(j+1) + " supplies : (" + str(100*round(sum(distr[:,j])/Facilities[j].capacity,3)) + "% stock used)")
        for i in [i for i in range(m) if y_sol[i,j] > 0.0]:
            print("   - Client " + str(i+1) + " at a rate of " + str(100*round(y_sol[i,j], 3)) + "%")
print("\n"
      + str(100*round(sum(sum(distr))/sum([Facilities[j].capacity for j in range(n)]), 3))
      + "% capacity used\n"
      + str(100*round(sum(sum(distr))/sum([Clients[i].demand for i in range(m)]), 3))
      + "% delivery rate"
      + "\nTotal profit: " + str(round(-flp.solution.get_objective_value(),2)))
