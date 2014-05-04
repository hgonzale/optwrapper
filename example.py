from optwrapper import *
from optwNpsol import *
import numpy as np
import math

def objf(x):
    return x[1]

def objg(x):
    return np.array( [ 0, 1 ] )

def consf(x):
    return np.array( [ x[0]*x[0] + 4*x[1]*x[1],
                       (x[0] - 2)*(x[0] - 2) + x[1]*x[1] ] )

def consg(x):
    return np.array( [ [ 2*x[0], 8*x[1] ],
                       [ 2*(x[0]-2), 2*x[1] ] ] )

prob = optwProblem( N=2, Ncons=2 )
prob.initCond( [10.0, 12.0] )
prob.consBox( [0, -10], [5, 2] )

prob.objFctn( objf )
prob.objGrad( objg )
prob.consFctn( consf, [ -1e6, -1e6 ], [ 4, 5 ] )
prob.consGrad( consg )

if( not prob.checkGrad() ):
    raise StandardError( "Gradient does not match function." )

solver = optwNpsol( prob )
solver.printOpts[ "printFile" ] = "debug2.txt"
solver.printOpts[ "printLevel" ] = 10

if( not solver.checkPrintOpts() ):
    raise StandardError( "Print options are invalid." )

solver.solve()
print( solver.getStatus() )
print( "Value: " + str( prob.value ) )
