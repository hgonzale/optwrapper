import numpy as np
import math
import sys
import optwrapper as ow

def objf( out, x ):
    out[0] = x[1]

def objg( out, x ):
    out[:] = [ 0, 1 ] ## this line is *not* the same as 'out = [0,1]'
                      ## the latter replaces the pointer

def consf( out, x ):
    out[0] = 4*x[1]*x[1] - x[0]
    out[1] = (x[0] - 2)*(x[0] - 2) + x[1]*x[1]

def consg( out, x ):
    out[0] = [ -1, 8*x[1] ]
    out[1] = [ 2*(x[0]-2), 2*x[1] ]

prob = ow.nlp.Problem( N=2, Ncons=2 )
prob.consBox( [0, -10], [5, 2] )

prob.objFctn( objf )
prob.objGrad( objg )
prob.consFctn( consf, lb=[ -np.inf, -np.inf ], ub=[ 4, 5 ] )
prob.consGrad( consg )

if( not prob.checkGrad( debug=True ) ):
    sys.exit( "Gradient check failed." )

solver = ow.ipopt.Solver( prob ) ## change this line to use another solver
# solver.options["printFile"] = "stdout"

print( "First run..." )
solver.initPoint( [1.0, -2.0] )
solver.solve()
print( prob.soln.getStatus() )
print( "Value: " + str( prob.soln.value ) )
print( "Final point: " + str( prob.soln.final ) )
print( "Retval: " + str( prob.soln.retval ) )

solver.warmStart()
print( "\nSecond run..." )
solver.solve()
print( prob.soln.getStatus() )
print( "Value: " + str( prob.soln.value ) )
print( "Final point: " + str( prob.soln.final ) )
print( "Retval: " + str( prob.soln.retval ) )
