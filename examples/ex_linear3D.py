import numpy as np 
from optwrapper import nlp, socp, snopt

#define your functions 
def instcost(x,u):
    Q = np.zeros( (3,3) )
    R = .01 
    return np.dot(np.dot(np.transpose(x),Q),x) + np.dot(np.dot(np.transpose(u),R),u)

instcost_ = [instcost, instcost, instcost]

def instcostgradst(x):
    Q = np.zeros( (3,3) )
    return 2*np.dot(np.transpose(x),Q)

instcostgradst_ = [instcostgradst, instcostgradst, instcostgradst]

def instcostgradinp(u):
    R = .01 
    return 2*np.dot(np.transpose(u),R)

instcostgradinp_ = [instcostgradinp, instcostgradinp, instcostgradinp]

def fincost(x):
    Q_terminalcost = np.identity(3)
    dist = x - np.ones(3)
    return np.dot(np.dot(np.transpose(dist),Q_terminalcost),dist)

def fincostgradst(x):
    Q_terminalcost = np.identity(3)
    dist = x - np.ones(3)
    return 2*np.dot(np.transpose(dist),Q_terminalcost)

def dynamics1(x, u):
    A = np.array( [ [1.0979, -.0105, .0167 ], [-.0105, 1.0481, .0825], [.0167, .0825, 1.1540] ] )
    B1 = np.array( [ [.9801], [-.1987], [0.0] ] )
    dyn1 = np.dot(A,x) + np.dot(B1,u)
    return dyn1 

def dynamics2(x, u):
    A = np.array( [ [1.0979, -.0105, .0167 ], [-.0105, 1.0481, .0825], [.0167, .0825, 1.1540] ] )
    B2 = np.array( [ [.1743], [.8601], [-.4794] ] )
    dyn2 = np.dot(A,x) + np.dot(B2,u)
    return dyn2

def dynamics3(x, u):
    A = np.array( [ [1.0979, -.0105, .0167 ], [-.0105, 1.0481, .0825], [.0167, .0825, 1.1540] ] )
    B3 = np.array( [ [.0952], [.4699], [0.8776] ] )
    dyn3 = np.dot(A,x) + np.dot(B3,u)
    return dyn3 

dynamics = [dynamics1, dynamics2, dynamics3]

def dynamics1gradst(x):
    A = np.array( [ [1.0979, -.0105, .0167 ], [-.0105, 1.0481, .0825], [.0167, .0825, 1.1540] ] )
    return A 

def dynamics2gradst(x):
    A = np.array( [ [1.0979, -.0105, .0167 ], [-.0105, 1.0481, .0825], [.0167, .0825, 1.1540] ] )
    return A 

def dynamics3gradst(x):
    A = np.array( [ [1.0979, -.0105, .0167 ], [-.0105, 1.0481, .0825], [.0167, .0825, 1.1540] ] )
    return A 

dynamicsgradst = [dynamics1gradst, dynamics2gradst, dynamics3gradst] 

def dynamics1gradinpcont(u):
    B1 = np.array( [ [.9801], [-.1987], [0.0] ] )
    return B1 

def dynamics2gradinpcont(u):
    B2 = np.array( [ [.1743], [.8601], [-.4794] ] )
    return B2

def dynamics3gradinpcont(u):
    B3 = np.array( [ [.0952], [.4699], [0.8776] ] )
    return B3 

dynamicsgradinpcont = [dynamics1gradinpcont, dynamics2gradinpcont, dynamics3gradinpcont] 

def cons(x):
    return 0 

def consgradst(x):
    return 0

#create an instance of socp:
prob = socp.Problem(Nst = 3, Ninpcont = 1, Nmodes = 3, Nineqcons = 0)
prob.initPoint( [0.0, 0.0, 0.0] )
prob.initialFinalTime( 0.0, 1.0 )
prob.instantCost( instcost_, instcostgradst_, instcostgradinp_ ) 
prob.finalCost( fincost, fincostgradst )
prob.dynamicsFctn( dynamics, dynamicsgradst, dynamicsgradinpcont )
prob.consFctn( cons, consgradst )
prob.consBox( -1.0 * np.ones( prob.Nst ), 2.0 * np.ones( prob.Nst ), -20 * np.ones( prob.Ninpcont ), 20 * np.ones( prob.Ninpcont ) )

nlpprob = nlp.Problem(socp = prob, Nsamples = 16 )

solver = snopt.Solver( nlpprob ) 
solver.debug = False
solver.printOpts[ "summaryFile" ] = "debugs.txt"
solver.printOpts[ "printFile" ] = "debugp.txt"
solver.printOpts[ "printLevel" ] = 10

solver.solve()
print( "Status: " + nlpprob.soln.getStatus() )
print( "Value: " + str( nlpprob.soln.value ) )
print( "Retval: " + str( nlpprob.soln.retval ) )

