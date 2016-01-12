import numpy as np
import math
import sys
import optwrapper as ow

def objf( out, x ):
    pass

def objg( out, x ):
    pass

def consf( out, x ):
    out[:] = [ 4*x[1]*x[1],
               (x[0] - 2)*(x[0] - 2) + x[1]*x[1] ]

def consg( out, x ):
    out[0,1] = 8*x[1]
    out[1,:] = [ 2*(x[0]-2), 2*x[1] ]

prob = ow.nlp.SparseProblem( N=2, Ncons=2 )
prob.initPoint( [10.0, 12.0] )
prob.consBox( [0,-10], [5,2] )

prob.objFctn( objf, A=[0,1] )
prob.objGrad( objg, pattern=[0,0] )
prob.consFctn( consf, lb=[-np.inf,-np.inf], ub=[4,5], A = [[-1,0],[0,0]] )
prob.consGrad( consg, pattern=[ [0,1], [1,1] ] )

if( not prob.checkGrad( debug=True ) ):
    sys.exit( "Gradient check failed." )

if( not prob.checkPattern( debug=True ) ):
    sys.exit( "Pattern check failed." )

solver = ow.ipopt.Solver( prob ) ## change this line to use another solver
# solver.debug = True
# solver.printOpts[ "summaryFile" ] = "debugs.txt"
# solver.printOpts[ "printFile" ] = "debugp.txt"
# solver.printOpts[ "printLevel" ] = 10

print( "First run..." )
solver.solve()
print( prob.soln.getStatus() )
print( "Value: " + str( prob.soln.value ) )
print( "Final point: " + str( prob.soln.final ) )
print( "Retval: " + str( prob.soln.retval ) )

prob.initPoint( [-10.0, -12.0] )
solver.warmStart()
print( "\nSecond run..." )
solver.solve()
print( prob.soln.getStatus() )
print( "Value: " + str( prob.soln.value ) )
print( "Final point: " + str( prob.soln.final ) )
print( "Retval: " + str( prob.soln.retval ) )
