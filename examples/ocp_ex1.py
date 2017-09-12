import numpy as np
import sys
import optwrapper as ow

def instcost( x, u, grad=True ):
    Q = np.identity(2)
    R = np.identity(2)

    if( not grad ):
        return x.dot(Q).dot(x) + u.dot(R).dot(u)

    return ( x.dot(Q).dot(x) + u.dot(R).dot(u),
             2 * x.dot(Q),
             2 * u.dot(R) )

def finalcost( x, grad=True ):
    if( not grad ):
        return x[1]

    return ( x[1],
             np.array( [ 0, 1 ] ) )

def dynamics( x, u, grad=True ):
    A = np.array( [ [1,4], [2,5] ] )
    B = np.array( [ [1,0], [0,1] ] )

    if( not grad ):
        return A.dot(x) + B.dot(u)

    return ( A.dot(x) + B.dot(u),
             A,
             B )

def cons( x, grad=True ):
    out = np.array( [ x[0] - 2,
                      - x[1] - 4 ] )
    if( not grad ):
        return out

    return ( out,
             np.array( [ [ 1, 0 ],
                         [ 0, -1 ] ] ) )

## main ##
prob = ow.ocp.Problem( Nstates=2, Ninputs=2, Ncons=2 )
prob.initCond( [ 1, 1 ] )
prob.timeHorizon( 0, 5 )
prob.costInstant( instcost )
prob.costFinal( finalcost )
prob.vectorField( dynamics )
prob.consNonlinear( cons )
prob.consBoxState( -50 * np.ones( prob.Nstates ),
                   50 * np.ones( prob.Nstates ) )
prob.consBoxInput( -10 * np.ones( prob.Ninputs ),
                   10 * np.ones( prob.Ninputs ) )

( nlpprob, initencode, solndecode ) = prob.discForwardEuler( Nsamples=30 )

if( not nlpprob.checkGrad( debug=True ) ):
    sys.exit( "Gradient check failed." )

solver = ow.ipopt.Solver( nlpprob )
solver.initPoint( initencode( prob.init, [ 0, 0 ] ) )
solver.debug = True
# solver.options[ "summaryFile" ] = "debugs.txt"
# solver.options[ "printFile" ] = "debugp.txt"
# solver.options[ "printLevel" ] = 10

solver.solve()
print( "Status: " + nlpprob.soln.getStatus() )
print( "Value: " + str( nlpprob.soln.value ) )
print( "Retval: " + str( nlpprob.soln.retval ) )
( st, u, time ) = solndecode( nlpprob.soln.final )
print( "Optimal state:\n" + str( st ) )
print( "Optimal input:\n" + str( u ) )
