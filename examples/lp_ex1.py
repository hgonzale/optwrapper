import numpy as np
from optwrapper import qp, lssol

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
prob.consBox( lb = -2 * np.ones( (9,) ),
              ub =  2 * np.ones( (9,) ) )
prob.objFctn( lin=L )
prob.consLinear( C, lb=Clb, ub=Cub )

solver = lssol.Solver( prob ) ## change this line to use another solver
solver.debug = True
solver.printOpts[ "printFile" ] = "debugp.txt"
solver.printOpts[ "summaryFile" ] = "debugs.txt"
solver.printOpts[ "printLevel" ] = 10

print( "First run..." )
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