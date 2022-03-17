import numpy as np
import cplex
from collections import namedtuple
import random


#-----------------------------------------------------------------------------
# Input variables for example
#-----------------------------------------------------------------------------


# List of servers                                   
Server = namedtuple('Server', ('num',         # Facility number, j
                               'capacity',    # Computing capacity of the server (in terms of CPU cores), S_{j}
                               'cost'))       # Unitary cost per cpu, c_{j}
n = random.randint(5,15)
Servers = ()
for i in range(n):
    Servers += (Server(str(i),random.randint(4,8), random.randint(10,50) ),)


# List of clients
Client = namedtuple('Client', ('num',           # Client number, i
                               'demand',        # Need of computing power of a part of the algorithm (in terms of CPU cores), d_{i}
                               'gain'))         
m = random.randint(2,5)
Clients = ()
for i in range(m):
    Clients += (Client(str(i),random.randint(5,30), random.randint(0,0)),)




# Total unitary transport gain between Client i and Facility j, q_{i,j}
q = [[-((Servers[j].cost)*Clients[i].demand) for j in range(n)] for i in range(m)]
q = np.array(q).astype("float")


#Génréation de la liste des tolérances des clients
tol=[random.random() for i in range(m)]


#En résumé, on a négligé tout les phénomènes de transport, et on a changé le problème en répondant à la question, comment faire tourner l'algorithme en allumant le moins de serveurs possibles. Les seuls coûts sont ceux de l'allumage des serveurs et du traitement des données par le serveur. Le serveur est capable de traiter tant de données, le client représente un volume de données à traiter 

#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

def facility_location_problem(Facilities, Clients, q, tol, eq = False) :

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
    flp.linear_constraints.add(lin_expr = [[[y[i][j] for i in range(m)]+[x[j]],[c.demand for c in Clients]+[-Facilities[j].capacity]] for j in range(n)],
                               senses = ["L"]*n,
                               rhs = [0.0]*n
                               )
    
# Facility j cannot supply more than S_{j} : 
#    flp.linear_constraints.add(lin_expr = [[[y[i][j] for i in range(m)],[c.demand for c in Clients]] for j in range(n)],
#                               senses = ["L"]*n,
#                               rhs = [f.capacity for f in Facilities]
#                               )
        
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
#    for i in range(m):
 #       for j in range(n):
  #          flp.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = [y[i][j], x[j]], val = [1.0, -1.0])],
   #                                    senses = ["L"],
    #                                   rhs = [0.0])
 
    
    
    flp.linear_constraints.add(lin_expr = [[[y[i][j] for j in range(n)], [1.0]*n] for i in range(m)], 
                               senses = ["G"]*m,
                               rhs = tol)    
    
    
    ## Solve the model
    
    flp.solve()
    return flp


#-----------------------------------------------------------------------------
# Get and print results
#-----------------------------------------------------------------------------
if sum(Clients[i].demand for i in range(m)) <= sum(Servers[j].capacity for j in range(n)):
    flp = facility_location_problem(Servers, Clients, q, tol,True)
else :
    flp = facility_location_problem(Servers, Clients, q, tol,)

x_sol = flp.solution.get_values()[:n]
y_sol = np.array([flp.solution.get_values()[n+(i*n):n+(i+1)*n] for i in range(m)])
distr = np.array([y_sol[i]*Clients[i].demand for i in range(m)])


print(distr)

for j in range(n):
    if x_sol[j] == 0:
        a=1
   #     print("Server " + str(j+1) + " is shutdown")
    else :
        print("Server " + str(j) + " uses " + str(100*round(sum(distr[:,j])/Servers[j].capacity,3)) + "% of computing capacity")
        for i in [i for i in range(m) if y_sol[i,j] > 0.0]:
            a=2
            #print("   - Client " + str(i+1) + " which has " + str(100*round(y_sol[i,j], 3)) + "% of its data treated by this server")
print("\n"
      + str(100*round(sum(sum(distr))/sum([Servers[j].capacity for j in range(n)]), 3))
      + "% computing capacity used\n"
      + str(100*round(sum(sum(distr))/sum([Clients[i].demand for i in range(m)]), 3))
      + "% data treated"
      + "\nTotal cost : " + str(round(flp.solution.get_objective_value(),2)))

