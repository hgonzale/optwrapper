import numpy as np 
from optwrapper import nlp, socp, snopt, wavelet_transform, pwm_transform 
import matplotlib.pyplot as plt 

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

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
Nsamples = 30
prob = socp.Problem(Nst = 3, Ninpcont = 1, Nmodes = 3, Nineqcons = 0)
prob.initPoint( [0.0, 0.0, 0.0] )
prob.initialFinalTime( 0.0, 1.0 )
prob.instantCost( instcost_, instcostgradst_, instcostgradinp_ ) 
prob.finalCost( fincost, fincostgradst )
prob.dynamicsFctn( dynamics, dynamicsgradst, dynamicsgradinpcont )
prob.consFctn( cons, consgradst )
prob.consBox( -1.0 * np.ones( prob.Nst ), 2.0 * np.ones( prob.Nst ), -20 * np.ones( prob.Ninpcont ), 20 * np.ones( prob.Ninpcont ) )

nlpprob = nlp.Problem(socp = prob, Nsamples = Nsamples )
#nlpprob.objf( nlpprob.init )
#nlpprob.objg( nlpprob.init )
#nlpprob.consf( nlpprob.init )
#nlpprob.consg( nlpprob.init )

solver = snopt.Solver( nlpprob ) 
solver.debug = False
solver.printOpts[ "summaryFile" ] = "debugs.txt"
solver.printOpts[ "printFile" ] = "debugp.txt"
solver.printOpts[ "printLevel" ] = 10

solver.solve()
print( "Status: " + nlpprob.soln.getStatus() )
print( "Value: " + str( nlpprob.soln.value ) )
# print( "Final point: " + str( nlpprob.soln.final ) )
print( "Retval: " + str( nlpprob.soln.retval ) )

def decode(s, Nst, Ninpcont, Nmodes, Nsamples):
    """
    this function creates the st, inpcont, and inpmode matrices from the optimizaion vector, s 

    arguments:
    s: the optimizaion vector of size N 

    """

    st = np.zeros( (Nst, Nsamples+1) )
    inpcont = np.zeros( (Ninpcont, Nsamples) )
    inpmode = np.zeros( (Nmodes, Nsamples) )

    for k in range(Nsamples):
        st[:,k] = s[ k*Nst : k*Nst+Nst]
        inpcont[:,k] = s[ Nst*(Nsamples+1)+k*Ninpcont : Nst*(Nsamples+1)+k*Ninpcont+Ninpcont ]
        inpmode[:,k] = s[ Nst*(Nsamples+1) + Ninpcont*Nsamples + k*Nmodes : Nst*(Nsamples+1) + Ninpcont*Nsamples + k*Nmodes + Nmodes]
 
    st[:,Nsamples] = s[ Nst * Nsamples : Nst * (Nsamples + 1) ]

    return (st, inpcont, inpmode)


(st, inpcont, inpmode) = decode(nlpprob.soln.final, 3, 1, 3, Nsamples)

#print inpmode 
#print inpmode.shape


#uniformly sampeled time vector from NLP discretization  
t = np.linspace( 0.0, 1.0, Nsamples + 1) 

# N corresponds to the Haar 
N = 8
N_Hsamples = 2**N

Nmodes = 3 
inpmode_haar = np.zeros( (Nmodes, N_Hsamples)  ) 
for k in range(Nmodes):
    coeff_d = wavelet_transform.haar_coeff(t, inpmode[k,:], N)
    d_inv = wavelet_transform.inv_haar(coeff_d)
    # print( "mode: {0}, d_inv: {1}".format( k, d_inv ) ) 
    inpmode_haar[k,:] = d_inv 

t_haar = np.linspace(0.0, 1.0, 2**N + 1) 

(t_pwm, d_pwm) = pwm_transform.pwm(t_haar, inpmode_haar) 

#print t_haar 
#print t_pwm 

#print t_haar.size
#print t_pwm.size 
#print d_pwm.shape 

u_inv = np.zeros( ( prob.Ninpcont, N_Hsamples ) )
# run the continuous input through Haar wavelet transform
for k in range( prob.Ninpcont ):
    coeff_u = wavelet_transform.haar_coeff(t, inpcont[k,:], N)
    u_inv[k,:] = wavelet_transform.inv_haar(coeff_u)

u_pwm = pwm_transform.resample_u( t_pwm, u_inv )

st_pwm = np.zeros( ( prob.Nst, t_pwm.size ) )
st_pwm[:,0] = prob.init

for k in range( t_pwm.size - 1 ):
    st_pwm[:,k+1] = st_pwm[:,k]
    for i in range(Nmodes):
        st_pwm[:,k+1] += d_pwm[i,k] * prob.dynamics[i](st_pwm[:,k], u_pwm[:,k]) * ( t_pwm[k+1] - t_pwm[k] )

#print( st_pwm )
#print (d_pwm)
fig = plt.figure( )
ax = fig.gca(projection='3d')
ax.plot( st_pwm[0,:], st_pwm[1,:], st_pwm[2,:]) 
#for k in range( d_pwm.size ):
#    for j in range( Nmodes ):
#        if d_pwm[j,k] == 1:
#            ax.scatter(st_pwm[0,k], st_pwm[1,k], st_pwm[2,k], 'o')
##plt.plot(t_pwm, st_pwm[0,:])
##plt.plot(t_pwm, st_pwm[1,:])
##plt.plot(t_pwm, st_pwm[2,:])
##plt.savefig( "test.pdf", format="pdf" )

idx0 = np.nonzero(d_pwm[0,:] > 1 - np.spacing(1) )
idx1 = np.nonzero(d_pwm[1,:] > 1 - np.spacing(1) ) 
idx2 = np.nonzero(d_pwm[2,:] > 1 - np.spacing(1) ) 

helper = np.ones( (1, ) )
for k in idx0[0]:
    if( t_pwm[k+1]-t_pwm[k] > np.spacing(1) ):
        ax.plot( st_pwm[0,k] * helper, st_pwm[1,k] * helper, st_pwm[2,k] * helper, 'ro') 
for k in idx1[0]:
    if( t_pwm[k+1]-t_pwm[k] > np.spacing(1) ):
        ax.plot( st_pwm[0,k] * helper, st_pwm[1,k] * helper, st_pwm[2,k] * helper, 'mv')
for k in idx2[0]:
    if( t_pwm[k+1]-t_pwm[k] > np.spacing(1) ):
        ax.plot( st_pwm[0,k] * helper, st_pwm[1,k] * helper, st_pwm[2,k] * helper, 'bx')


# ax.plot( hola, hola, hola, 'ro' )

#print t_pwm.shape
#print d_pwm[0,idx0].shape
#print t_pwm[idx0].shape

#ax.plot( t_pwm[idx0], np.transpose( d_pwm[0, idx0] ), marker = 'o', color ='blue')
#ax.plot( d_pwm[1, idx1], marker = 'x',  color = 'green')
#ax.plot( d_pwm[2, idx2], marker = 'v', color = 'magenta')

#ax.plot( d_pwm[0,idx0], d_pwm[1,idx1], d_pwm[2,idx2], marker = 'o')
#ax.plot( st_pwm[0,idx0], st_pwm[1,idx1], st_pwm[2,idx2])
plt.show()



#fig = plt.figure( )
#ax = fig.add_subplot(111, projection='3d') 
#for k in range( d_pwm.size - 1):
#    if d_pwm[0,k] == 1:
#        ax.scatter( st_pwm[0,k], st_pwm[1,k], st_pwm[2,k], 'o')
#    if d_pwm[1,k] == 1:
#        ax.scatter( st_pwm[0,k], st_pwm[1,k], st_pwm[2,k], 'x')
#    if d_pwm[2,k] == 1:
#        ax.scatter( st_pwm[0,k], st_pwm[1,k], st_pwm[2,k], 's')
#plt.show( )


 

