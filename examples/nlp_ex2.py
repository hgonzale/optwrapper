from optwrapper import nlp, npsol, snopt
import numpy as np
import math

def objf( out, x ):
    pass

def objg( out, x ):
    pass

def consf( out, x ):
    out[0] = 4*x[1]*x[1]
    out[1] = (x[0] - 2)*(x[0] - 2) + x[1]*x[1]

def consg( out, x ):
    out[0] = [ 0, 8*x[1] ]
    out[1] = [ 2*(x[0]-2), 2*x[1] ]

prob = nlp.Problem( N=2, Ncons=2 )
prob.initPoint( [10.0, 12.0] )
prob.consBox( [0, -10], [5, 2] )

prob.objFctn( objf, A=[0,1] )
prob.objGrad( objg )
prob.consFctn( consf, lb=[ -np.inf, -np.inf ], ub=[ 4, 5 ] )
prob.consGrad( consg )

if( not prob.checkGrad() ):
    print( "Gradient does not match function." )
    raise SystemExit

solver = npsol.Solver( prob )
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
