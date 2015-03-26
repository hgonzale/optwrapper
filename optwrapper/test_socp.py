import socp
import numpy as np

def instcost1(x,u):
    Q1 = np.identity(2)
    R1 = np.identity(2)
    L1 = np.dot(np.dot(np.transpose(x),Q1),x) + np.dot(np.dot(np.transpose(u),R1),u) 
    return L1 
    
def instcost2(x,u):
    Q2 = np.array( [ [2,0], [0,3] ] )
    R2 = np.array( [ [1,0], [0,4] ] )
    L2 = np.dot(np.dot(np.transpose(x),Q2),x) + np.dot(np.dot(np.transpose(u),R2),u)
    return L2 

def instcost3(x,u):
    Q3 = np.array( [ [1,2], [2,1] ] )
    R3 = np.array( [ [3,1], [1,2] ] )
    L3 = np.dot(np.dot(np.transpose(x),Q3),x) + np.dot(np.dot(np.transpose(u),R3),u)
    return L3

instcost = [instcost1, instcost2, instcost3] 

def instcost1gradst(x):
    Q1 = np.identity(2)
    dL1x = 2*np.dot(np.transpose(x),Q1)
    return dL1x

def instcost2gradst(x):
    Q2 = np.array( [ [2,0], [0,3] ] )
    dL2x = 2*np.dot(np.transpose(x),Q2)
    return dL2x

def instcost3gradst(x):
    Q3 = np.array( [ [1,2], [2,1] ] )
    dL3x = 2*np.dot(np.transpose(x),Q3)

    return dL3x

instcostgradst = [instcost1gradst, instcost2gradst, instcost3gradst] 

def instcost1gradinpcont(u):
    R1 = np.identity(2)
    dL1U = 2*np.dot(np.transpose(u),R1)
    return dL1U 

def instcost2gradinpcont(u):
    R2 = np.array( [ [1,0], [0,4] ] )
    dL2U = 2*np.dot(np.transpose(u),R2)
    return dL2U

def instcost3gradinpcont(u):
    R3 = np.array( [ [3,1], [1,2] ] )
    dL3U = 2*np.dot(np.transpose(u),R3)    
    return dL3U

instcostgradinpcont = [instcost1gradinpcont, instcost2gradinpcont, instcost3gradinpcont]

def fincost(x):
    return x[1]

def fincostgradst(x):
    return np.array ( [0, 1 ] )

def dynamics1(x,u):
    A1 = np.array( [ [1,2], [3,4] ] )
    B1 = np.array( [ [1,0], [0,1] ] )
    dyn1 = np.dot(A1,x) + np.dot(B1,u)
    return dyn1

def dynamics2(x,u):
    A2 = np.array( [ [3,1], [2,4] ] )
    B2 = np.array( [ [2,1], [0,2] ] )
    dyn2 = np.dot(A2,x) + np.dot(B2,u)
    return dyn2

def dynamics3(x,u):
    A3 = np.array( [ [2,4], [5,1] ] )
    B3 = np.array( [ [0,1], [1,4] ] )
    dyn3 = np.dot(A3,x) + np.dot(B3,u)
    return dyn3

dynamics = [dynamics1, dynamics2, dynamics3] 

def dynamics1gradst(x):
    A1 = np.array( [ [1,2], [3,4] ] )
    return A1 

def dynamics2gradst(x):
    A2 = np.array( [ [3,1], [2,4] ] )
    return A2

def dynamics3gradst(x):
    A3 = np.array( [ [2,4], [5,1] ] )
    return A3

dynamicsgradst = [dynamics1gradst, dynamics2gradst, dynamics3gradst] 

def dynamics1gradinpcont(u):
    B1 = np.array( [ [1,0], [0,1] ] )
    return B1

def dynamics2gradinpcont(u):
    B2 = np.array( [ [2,1], [0,2] ] )
    return B2 

def dynamics3gradinpcont(u):
    B3 = np.array( [ [0,1], [1,4] ] )
    return B3

dynamicsgradinpcont = [dynamics1gradinpcont, dynamics2gradinpcont, dynamics3gradinpcont] 

def cons(x):
    return np.array( [ x[0] - 2 , x[1] - 4 ] )

def consgradst(x):
    return np.identity(2)

prob = socp.Problem(Nst = 2, Ninpcont = 2, Nmodes = 3, Nineqcons = 2)
prob.initPoint( [1.0, 1.0] )
prob.initialFinalTime( 0.0, 4.0 )
prob.instantCost( instcost, instcostgradst, instcostgradinpcont ) 
prob.finalCost( fincost, fincostgradst )
prob.dynamicsFctn( dynamics, dynamicsgradst, dynamicsgradinpcont )
prob.consFctn( cons, consgradst )
prob.consBox( -50.0, 50.0, -50.0, 50.0 )


#this is the code I would include if I were to test my nlp code as well 
#nlp_prob = nlp.Problem(socp=prob, Nsamples= 4)

#these vectors are all created in the nlp Problem -- printed to check that they're correct based off the SOCP problem
#print nlp_prob.lb
#print nlp_prob.ub
#print nlp_prob.conslb
#print nlp_prob.consub

# s =  np.array([1, 1, 3, 2, -1, 4, 0, 3, 2, -2, 0, 1, 1, 2, -1, 3, 2, 0, 0.2, 0.6, 0.2, 0.4, 0.1, 0.5, 0.3, 0.3, 0.4, 0.2, 0.1, 0.7 ] )

# objf = nlp_prob.objf(s)
# objg = nlp_prob.objg(s)
# consf = nlp_prob.consf(s)
# consg = nlp_prob.consg(s)

# print objf
# print objg
# print consf
# print consg

