import numpy as np
import math
from optwrapper import nlp, npsol, snopt

def objf( out, x ):
    out[0] = x[1]

def objg( out, x ):
    out[1] = 1

def consf( out, x ):
    out[:] = [ 4*x[1]*x[1] - x[0],
               (x[0] - 2)*(x[0] - 2) + x[1]*x[1] ]

def consg( out, x ):
    out[0,:] = [ -1, 8*x[1] ]
    out[1,:] = [ 2*(x[0]-2), 2*x[1] ]

prob = nlp.SparseProblem( N=2, Ncons=2 )
prob.initPoint( [10.0, 12.0] )
prob.consBox( [0, -10], [5, 2] )

prob.objFctn( objf )
prob.objGrad( objg, pattern=[ 0, 1 ] )
prob.consFctn( consf, lb=[ -np.inf, -np.inf ], ub=[ 4, 5 ] )
prob.consGrad( consg, pattern=[ [1,1], [1,1] ] )

if( not prob.checkGrad() ):
    print( "Gradient does not match function." )
    raise SystemExit

solver = snopt.Solver( prob )
solver.debug = True
solver.printOpts[ "summaryFile" ] = "debugs.txt"
solver.printOpts[ "printFile" ] = "debugp.txt"
solver.printOpts[ "printLevel" ] = 10

if( not solver.checkPrintOpts() or
    not solver.checkSolveOpts() ):
    print( "Options are invalid." )
    raise SystemExit

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
