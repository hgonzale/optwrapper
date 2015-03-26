import numpy as np
from optwrapper import nlp, ocp, snopt

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
    if( not grad ):
        return np.array( [ x[0] - 2,
                           x[1] - 4 ] )

    return ( np.array( [ x[0] - 2,
                         x[1] - 4 ] ),
             np.array( [ [ 1, 0 ],
                         [ 0, 1 ] ] ) )

## main ##
prob = ocp.Problem( Nstates=2, Ninputs=2, Ncons=2 )
prob.initCond( [ 1, 1 ] )
prob.timeHorizon( 0, 4 )
prob.costInstant( instcost )
prob.costFinal( finalcost )
prob.vectorField( dynamics )
prob.consNonlinear( cons )
prob.consBoxState( -50 * np.ones( prob.Nstates ),
                   50 * np.ones( prob.Nstates ) )
prob.consBoxInput( -10 * np.ones( prob.Ninputs ),
                   10 * np.ones( prob.Ninputs ) )

nlpprob = prob.discForwardEuler( Nsamples=4 )
nlpprob.checkGrad( h=1e-6, etol=1e-4, point=None, debug=True )

# s =  np.array([1, 1, 3, 2, -1, 4, 0, 3, 2, -2, 0, 1, 1, 2, -1, 3, 2, 0 ] )

# objf = nlp_prob.objf(s)
# objg = nlp_prob.objg(s)
# consf = nlp_prob.consf(s)
# consg = nlp_prob.consg(s)

# print objf
# print objg
# print consf
# print consg

solver = snopt.Solver( nlpprob )
solver.debug = True
solver.printOpts[ "summaryFile" ] = "debugs.txt"
solver.printOpts[ "printFile" ] = "debugp.txt"
solver.printOpts[ "printLevel" ] = 10


print( "First run..." )
solver.solve()
print( nlpprob.soln.getStatus() )
print( "Value: " + str( nlpprob.soln.value ) )
print( "Final point: " + str( nlpprob.soln.final ) )
print( "Retval: " + str( nlpprob.soln.retval ) )
