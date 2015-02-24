import numpy as np
from optwrapper import nlp,ocp,snopt


def instcost(x,u):
    Q = np.identity(2)
    R = np.identity(2)
    return np.dot(np.dot(np.transpose(x),Q),x) + np.dot(np.dot(np.transpose(u),R),u)

def instcostgradst(x):
    Q = np.identity(2) 
    return 2*np.dot(np.transpose(x),Q)

def instcostgradin(u):
    R = np.identity(2) 
    return 2*np.dot(np.transpose(u),R)

def fincost(x):
    return x[1]

def fincostgradst(x):
    return np.array ( [0, 1 ] )

def dynamics(x,u):
    A = np.array( [ [1,4], [2,5] ] ) 
    B = np.array( [ [1,0], [0,1] ] )
    return np.dot(A,x) + np.dot(B,u)

def dynamicsgradst(x):
    return np.array( [ [1, 4], [2,5] ] )

def dynamicsgradin(u):
    return np.array( [ [1, 0], [0,1] ] )

def cons(x):
    return np.array( [ x[0] - 2 , x[1] - 4 ] )

def consgradst(x):
    return np.array( [ [1,0], [0,1] ] )


prob = ocp.Problem(Nst=2, Ninp=2, Nineqcons=2)
prob.initPoint( [1.0, 1.0] )
prob.initialFinalTime(0.0, 4.0)
prob.instantCost( instcost, instcostgradst, instcostgradin )
prob.finalCost( fincost, fincostgradst )
prob.dynamicsFctn(dynamics, dynamicsgradst, dynamicsgradin)
prob.consFctn(cons, consgradst)
prob.consBox(-50.0, 50.0, -50.0, 50.0)


nlp_prob = nlp.Problem(ocp=prob, Nsamples= 4)

#these vectors are all created in the nlp Problem -- printed to check that they're correct based off the OCP problem
#print nlp_prob.lb
#print nlp_prob.ub
#print nlp_prob.conslb
#print nlp_prob.consub

s =  np.array([1, 1, 3, 2, -1, 4, 0, 3, 2, -2, 0, 1, 1, 2, -1, 3, 2, 0 ] ) 

objf = nlp_prob.objf(s) 
objg = nlp_prob.objg(s)
consf = nlp_prob.consf(s)
consg = nlp_prob.consg(s)

print objf
print objg
print consf
print consg 

solver = snopt.Solver( nlp_prob )
solver.debug = True
solver.printOpts[ "summaryFile" ] = "debugs.txt"
solver.printOpts[ "printFile" ] = "debugp.txt"
solver.printOpts[ "printLevel" ] = 10


print( "First run..." )
solver.solve()
print( prob.soln.getStatus() )
print( "Value: " + str( prob.soln.value ) )
print( "Final point: " + str( prob.soln.final ) )
print( "Retval: " + str( prob.soln.retval ) )
        



