import numpy as np
import cplex
from collections import namedtuple
import random
from scipy.sparse.csgraph import shortest_path
import time


#-----------------------------------------------------------------------------
# Input variables for example
#-----------------------------------------------------------------------------



# List of servers                                   
Server = namedtuple('Server', ('num',         # Facility number, j
                               'capacity',    # Computing capacity of the server (in terms of CPU cores), S_{j}
                               'cost'))       # Unitary time to compute, c_{j}
n = random.randint(100,100)
Servers = ()
for i in range(n):
    Servers += (Server(str(i),random.randint(4,8), random.randint(10,50) ),)
    

# Generates the network
th = 0.2
net = np.zeros(shape = (n,n))
for j in range(n):
    for jj in range(j+1,n):
        p = random.random()
        if p >= th:
            net[j,jj] = random.randint(1,100)

dist_matrix = shortest_path(csgraph=net, directed = False)
    


# List of clients
Client = namedtuple('Client', ('num',           # On which node are the data initially 
                               'demand',        # Need of computing power of a part of the algorithm (in terms of CPU cores), d_{i}
                               'gain',
                               'tol'))          # Minimum delivery rate desired each client, alpha_{i}
m = random.randint(30,30)

if m > n :
    print("More clients")
Clients = ()
for i in range(m):
    Clients += (Client(str(random.randint(1,n)),random.randint(10,15), random.randint(0,0), random.random()),)

# Total unitary transport gain between Client i and Facility j, q_{i,j}
q = [[(Servers[j].cost+dist_matrix[int(Clients[i].num)-1][j])*Clients[i].demand for j in range(n)] for i in range(m)]
q = np.array(q).astype("float")





#En résumé, on a négligé tout les phénomènes de transport, et on a changé le problème en répondant à la question, comment faire tourner l'algorithme en allumant le moins de serveurs possibles. Les seuls coûts sont ceux de l'allumage des serveurs et du traitement des données par le serveur. Le serveur est capable de traiter tant de données, le client représente un volume de données à traiter 

#-----------------------------------------------------------------------------
# Build the model
#-----------------------------------------------------------------------------

def facility_location_problem(Servers, Clients, q) :

    ## Create CPO model
    flp = cplex.Cplex()
    
    
    ## Create the decisions variables
    
    # Whether or not server j must be used, x_{j}
    x = list(flp.variables.add(obj = [Servers[j].cost for j in range(n)],
                               lb = [0]*n,
                               ub = [1]*n,
                               types = ["B"]*n))
    
    # Rate of client's i demand fullfilled by server j, y_{i,j}
    y = [[] for i in range(m)]
    for i in range(m):
        y[i] = list(flp.variables.add(obj = q[i],
                                      lb = [0]*n,
                                      ub = [1]*n))
    
    
    ## Create the constraints
    
    # Facility j cannot supply more than x_{j}*S_{j}: 
        
    flp.linear_constraints.add(lin_expr = [[[y[i][j] for i in range(m)]+[x[j]],[Clients[i].demand for i in range(m)]+[-Servers[j].capacity]] for j in range(n)],
                               senses = ["L"]*n,
                               rhs = [0.0]*n
                               )
    
        
    # Demands are fullfiled : if all demands can be fullfilled, they are, and if not then we insure that client's demand are satisfied with a certain tolerance
    
    if sum(Clients[i].demand for i in range(m)) <= sum(Servers[j].capacity for j in range(n)): 
        flp.linear_constraints.add(lin_expr = [[y[i],[1.0]*n] for i in range(m)],
                                    senses = ["E"]*m,                               
                                    rhs = [1.0]*m)
    
    else :
        flp.linear_constraints.add(lin_expr = [[y[i],[1.0]*n] for i in range(m)],
                                senses = ["L"]*m,                               
                                rhs = [1.0]*m)
        flp.linear_constraints.add(lin_expr = [[y[i], [1.0]*n] for i in range(m)], 
                               senses = ["G"]*m,
                               rhs = [Clients[i].tol for i in range(m)])  
        
    ## Solve the model
    
    flp.solve()
    return flp


#-----------------------------------------------------------------------------
# Get and print results
#-----------------------------------------------------------------------------

start = time.time()
flp = facility_location_problem(Servers, Clients, q)
end = time.time()

print("runtime:", end-start)


x_sol = flp.solution.get_values()[:n]
y_sol = np.array([flp.solution.get_values()[(i+1)*n:(i+2)*n] for i in range(m)])
distr = np.array([y_sol[i]*Clients[i].demand for i in range(m)])


print(distr)

print(net)
print(dist_matrix)
print(Clients)
print([Servers[j].cost for j in range(n)])
print([Servers[j].capacity for j in range(n)])

for j in range(n):
    if x_sol[j] == 0:
        a=1
   #     print("Server " + str(j+1) + " is shutdown")
    else :
        print("Server " + str(j+1) + " uses " + str(100*round(sum(distr[:,j])/Servers[j].capacity,3)) + "% of computing capacity")
        for i in [i for i in range(m) if y_sol[i,j] > 0.0]:
            a=2
            # print("   - Client " + str(i+1) + " which has " + str(100*round(y_sol[i,j], 3)) + "% of its data treated by this server")
print("\n"
      + str(100*round(sum(sum(distr))/sum([Servers[j].capacity for j in range(n)]), 3))
      + "% computing capacity used\n"
      + str(100*round(sum(sum(distr))/sum([Clients[i].demand for i in range(m)]), 3))
      + "% data treated"
      + "\nTotal cost : " + str(round(flp.solution.get_objective_value(),2)))