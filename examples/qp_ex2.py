import numpy as np
import optwrapper as ow

## pg. 25 of LSSOL's manual

Q = np.zeros( (9,9) )
Q[0:5,0:5] = 1
Q[ [0,1,2,3,4], [0,1,2,3,4] ] = 2

L = -1 * np.ones( (9,) )
L[0] = -4
L[7] = -.1
L[8] = -.3

# C = np.ones( (3,9) )
# C[0,-1] = 4
# C[1,1] = 2
# C[1,2] = 3
# C[1,3] = 4
# C[1,4] = -2
# C[2,1] = -1
# C[2,3] = -1

# Clb = -2 * np.ones( (3,) )
# Cub = 1.5 * np.ones( (3,) )
# Cub[2] = 4

prob = ow.qp.Problem( N=9 )
prob.consBox( lb = -2 * np.ones( (9,) ),
              ub =  2 * np.ones( (9,) ) )
prob.objFctn( quad=Q, lin=L )
# prob.consLinear( C, lb=Clb, ub=Cub )

solver = ow.qpoases.Solver( prob ) ## change this line to use another solver
solver.debug = True
# solver.options[ "printFile" ] = "debugp.txt"
# solver.options[ "summaryFile" ] = "debugs.txt"
solver.options[ "printLevel" ] = ow.qpoases.Params.PL_LOW

print( "\nFirst run..." )
solver.initPoint( np.zeros( (9,) ) )
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
