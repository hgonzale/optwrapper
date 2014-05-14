import nlp
import npsol
import snopt
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

prob = nlp.Problem( N=2, Ncons=2 )
prob.initPoint( [10.0, 12.0] )
prob.consBox( [0, -10], [5, 2] )

prob.objFctn( objf )
prob.objGrad( objg )
prob.consFctn( consf, [ -np.inf, -np.inf ], [ 4, 5 ] )
prob.consGrad( consg )

if( not prob.checkGrad() ):
    print( "Gradient does not match function." )
    raise SystemExit

solver = snopt.Solver( prob )
# solver.printOpts[ "summaryFile" ] = ""
solver.printOpts[ "printFile" ] = "debugp.txt"
solver.printOpts[ "printLevel" ] = 1

if( not solver.checkPrintOpts() or
    not solver.checkSolveOpts() ):
    print( "Options are invalid." )
    raise SystemExit

solver.solve()
print( prob.soln.getStatus() )
print( "Value: " + str( prob.soln.value ) )
print( "Final point: " + str( prob.soln.final ) )
print( "Retval: " + str( prob.soln.retval ) )

prob.initPoint( [-10.0, -12.0] )
solver.warmStart()
solver.solve()
print( prob.soln.getStatus() )
print( "Value: " + str( prob.soln.value ) )
