import ocp
import numpy as np
from python scripts import optwrapper 
from optwrapper import optwrapper 
from optwrapper import nlp 


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
	return np.array( [ [1, 0], [0,1] ] )

def dynamicsgradin(u):
	return np.array( [ [1], [0] ] )

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






