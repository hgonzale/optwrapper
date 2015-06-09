import numpy as np
from optwrapper import qp, lssol

## pg. 25 of LSSOL's manual

Q = np.zeros( (9,9) )
Q[0:5,0:5] = 1
Q[ [0,1,2,3,4], [0,1,2,3,4] ] = 2

L = -1 * np.ones( (9,) )
L[0] = -4
L[7] = -.1
L[8] = -.3

C = np.ones( (3,9) )
C[0,-1] = 4
C[1,1] = 2
C[1,2] = 3
C[1,3] = 4
C[1,4] = -2
C[2,1] = -1
C[2,3] = -1

Clb = -2 * np.ones( (3,) )
Cub = 1.5 * np.ones( (3,) )
Cub[2] = 4

prob = qp.Problem( N=9, Nconslin=3 )
prob.initPoint( np.zeros( (9,) ) )
prob.consBox( -2 * np.ones( (9,) ),
              2 * np.ones( (9,) ) )
prob.objFctn( Q, L )
prob.consLinear( C, Clb, Cub )

solver = lssol.Solver( prob ) ## change this line to use another solver
solver.debug = True
solver.printOpts[ "printFile" ] = "debugp.txt"
solver.printOpts[ "summaryFile" ] = "debugs.txt"
solver.printOpts[ "printLevel" ] = 10

optimal_soln = np.array( [ 2, -.23333333, -.26666667, -.3, -.1, 2, 2, -1.77777778, -.45555556 ] )
optimal_value = -8.067777633666992

print( "The results of both runs should be..." )
print( "Value: " + str( optimal_value ) )
print( "Final point: " + str( optimal_soln ) )

print( "\nFirst run..." )
solver.solve()
print( prob.soln.getStatus() )
print( "Value: " + str( prob.soln.value ) )
print( "Final point: " + str( prob.soln.final ) )
print( "Retval: " + str( prob.soln.retval ) )

prob.initPoint( -5 * np.ones( (9,) ) )
solver.warmStart()
print( "\nSecond run..." )
solver.solve()
print( prob.soln.getStatus() )
print( "Value: " + str( prob.soln.value ) )
print( "Final point: " + str( prob.soln.final ) )
print( "Retval: " + str( prob.soln.retval ) )
