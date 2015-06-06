import numpy as np
import sys
from optwrapper import nlp, socp, snopt, npsol

A = np.array( [ [1.0979, -.0105, .0167 ], [-.0105, 1.0481, .0825], [.0167, .0825, 1.1540] ] )

def instcost( x, u, grad=True ):
    Q = np.zeros( (3,3) )
    R = .01

    if( not grad ):
        return x.dot(Q).dot(x) + u.dot(R).dot(u)

    return ( x.dot(Q).dot(x) + u.dot(R).dot(u),
             2 * x.dot(Q),
             2 * u.dot(R) )

def fcost( x, grad=True ):
    S = np.identity(3)
    dist = x - np.ones(3)

    if( not grad ):
        return dist.dot(S).dot(dist)

    return( dist.dot(S).dot(dist),
            2 * dist.dot(S) )

def dynamics_mode1( x, u, grad=True ):
    B1 = np.array( [ [.9801], [-.1987], [0.0] ] )

    if( not grad ):
        return A.dot(x) + B1.dot(u)

    return( A.dot(x) + B1.dot(u),
            A,
            B1 )

def dynamics_mode2( x, u, grad=True ):
    B2 = np.array( [ [.1743], [.8601], [-.4794] ] )

    if( not grad ):
        return A.dot(x) + B2.dot(u)

    return( A.dot(x) + B2.dot(u),
            A,
            B2 )

def dynamics_mode3( x, u, grad=True ):
    B3 = np.array( [ [.0952], [.4699], [0.8776] ] )

    if( not grad ):
        return A.dot(x) + B3.dot(u)

    return( A.dot(x) + B3.dot(u),
            A,
            B3 )


## create an instance of socp:
prob = socp.Problem( Nstates = 3, Ninputs = 1, Nmodes = 3, Ncons = 0 )
prob.initCond( [ 0.0, 0.0, 0.0 ] )
prob.timeHorizon( 0.0, 1.0 )
prob.costInstant( ( instcost, instcost, instcost ) )
prob.costFinal( fcost )
prob.vectorField( ( dynamics_mode1, dynamics_mode2, dynamics_mode3 ) )
prob.consBoxState( -1 * np.ones( (prob.Nstates,) ), 2 * np.ones( (prob.Nstates,) ) )
prob.consBoxInput( -20 * np.ones( (prob.Ninputs,) ), 20 * np.ones( (prob.Ninputs,) ) )

( nlpprob, solndecode ) = prob.discForwardEuler( Nsamples = 30 )

if( not nlpprob.checkGrad( debug=True ) ):
    sys.exit( "Gradient check failed." )

solver = snopt.Solver( nlpprob )
solver.debug = True
solver.printOpts[ "summaryFile" ] = "debugs.txt"
solver.printOpts[ "printFile" ] = "debugp.txt"
solver.printOpts[ "printLevel" ] = 10

solver.solve()
print( "Status: " + nlpprob.soln.getStatus() )
print( "Value: " + str( nlpprob.soln.value ) )
# print( "Final point: " + str( nlpprob.soln.final ) )
print( "Retval: " + str( nlpprob.soln.retval ) )

( st, u, d, time ) = solndecode( nlpprob.soln.final )
print( "Optimal state:\n" + str( st ) )
print( "Optimal continuous input:\n" + str( u ) )
print( "Optimal relaxed discrete input:\n" + str( d ) )

Npwm = 5
thaar = np.linspace( time[0], time[-1], 2**Npwm + 1 )
uhaar = socp.haarWaveletApprox( time, u, Npwm )
dhaar = socp.haarWaveletApprox( time, d, Npwm )
( tpwm, upwm, dpwm ) = socp.pwmTransform( thaar, uhaar, dhaar )
print( "PWM time samples:\n" + str( tpwm ) )
print( "Optimal continuous input:\n" + str( upwm ) )
print( "Optimal pure discrete input:\n" + str( dpwm ) )
