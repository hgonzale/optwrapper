from optwrapper import OptProblem
import numpy as np
import math

def objf(x):
    return x[1]

def objgrad(x):
    return np.array( [ 0, 1 ] )

def conf(x):
    return np.array( [ x[0]**2 + 4*x[1]**2,
                       (x[0] - 2)**2 + x[1]**2 ] )

def congrad(x):
    return np.array( [ 2*x[0], 8*x[1], 2*(x[0]-2), 2*x[1] ] )

prob = OptProblem( n=2, nconstraint=2, maximize=False )
prob.x_bounds( [0, -1e6], [1e6, 1e6] )
prob.constraint_bounds( [-1e6, -1e6], [4, 5] )
prob.set_start_x( [1.0, 1.0] )

prob.set_objective( objf )
prob.set_objective_gradient( objgrad )
prob.set_constraint( conf )
# prob.set_constraint_gradient( congrad )
## An alternative to set_constraint_gradient
prob.set_sparse_constraint_gradient( [0, 0, 1, 1], [0, 1, 0, 1], congrad )

stop = prob.get_stop_options()
stop['maxtime'] = 1.0
stop['maxeval'] = 1000
stop['ftol'] = None
stop['xtol'] = None
prob.set_stop_options( stop )

prt = prob.get_print_options()
prt['print_level'] = 9
prt['summary_level'] = 6
prob.set_print_options( prt )

if( prob.check_errors() and prob.check_gradient() ):
    # prob.print_options['print_file'] = 'snopt.out'
#    answer, finalX, status = prob.solve( 'SNOPT' )
#    print( "\nSNOPT answer: " + str( answer ) )
#    print( "x1: " + str( finalX[0] ) + " x2: " + str( finalX[1] ) )
#    print( "status: " + status )

    prob.print_options['print_level'] = 1
    prob.print_options['print_file'] = 'npsol.out'
    answer, finalX, status = prob.solve( 'NPSOL' )
    print( "\nNPSOL answer: " + str( answer ) )
    print( "x1: " + str( finalX[0] ) + " x2: " + str( finalX[1] ) )
    print( "status: " + status )

    # answer,finalX,status=prob.solve('NLOPT AUGLAG')
    # print("\nNLOPT answer: "+str(answer))
    # print("x1: "+str(finalX[0])+" x2: "+str(finalX[1]))
    # print("status: "+status)

    # prob.stop_options['ftol']=1e-7
    # prob.print_options['print_file']='ipopt.out'
    # answer,finalX,status=prob.solve('IPOPT ma57')
    # #note: IPOPT will still print a header
    # #     even if summary_level is set to 0
    # print("\nIPOPT answer: "+str(answer))
    # print("x1: "+str(finalX[0])+" x2: "+str(finalX[1]))
    # print("status: "+status)
