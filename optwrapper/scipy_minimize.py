import numpy as np
import scipy.optimize
from functools import lru_cache

from optwrapper import base, utils, nlp


class Soln( base.Soln ):
    def __init__( self ):
        super().__init__()
        self.retval = -1
        self.message = "Uninitialized Solution"


    def getStatus( self ):
        return self.message



class Solver( base.Solver ):
    def __init__( self, prob=None ):
        super().__init__()

        self.prob = None
        self.warm_start = False

        self.options = utils.Options()

        if( prob ):
            self.setupProblem( prob )


    def setupProblem( self, prob ):
        if( not isinstance( prob, nlp.Problem ) ):
            raise TypeError( "Argument prob must be an instance of nlp.Problem" )

        if( not prob.checkSetup() ):
            raise ValueError( "Argument 'prob' has not been properly configured" )

        self.prob = prob


    def initPoint( self, init ):
        self.init = np.copy( init )


    def warmStart( self ):
        if( not isinstance( self.prob.soln, Soln ) ):
            return False

        self.initPoint( self.prob.soln.final )

        return True


    def solve( self ):
        def fun( x ):
            out = np.zeros( 1 )
            self.prob.objf( out, x )
            if( self.prob.objmixedA is not None ):
                out += self.prob.objmixedA.dot( x )
            return out

        def jac( x ):
            out = np.zeros( (self.prob.N,) )
            self.prob.objg( out, x )
            if( self.prob.objmixedA is not None ):
                out += self.prob.objmixedA
            return out

        @lru_cache( maxsize = 16 )
        def myconsf( x ):
            x = np.array( x ) ## undo tuple'ing of x
            out = np.zeros( (self.prob.Ncons,) )
            self.prob.consf( out, x )
            if( self.prob.consmixedA is not None ):
                out += self.prob.consmixedA.dot( x )
            return out

        @lru_cache( maxsize = 16 )
        def myconsg( x ):
            x = np.array( x ) ## undo tuple'ing of x
            out = np.zeros( ( self.prob.Ncons, self.prob.N ) )
            self.prob.consg( out, x )
            if( self.prob.consmixedA is not None ):
                out += self.prob.consmixedA
            return out

        def cons_dict():
            out = list()
            if( self.prob.Nconslin > 0 ):
                eq = ( self.prob.conslinlb == self.prob.conslinub )
                ineqlb = np.logical_and( ~eq, np.isfinite( self.prob.conslinlb ) )
                inequb = np.logical_and( ~eq, np.isfinite( self.prob.conslinub ) )

                out.extend(
                    map( lambda row, bnd: { "type": "eq",
                                            "fun": lambda x: row.dot( x ) - bnd,
                                            "jac": lambda x: row },
                         self.prob.conslinA[ eq, : ],
                         self.prob.conslinub[ eq ] )
                )
                out.extend(
                    map( lambda row, bnd: { "type": "ineq",
                                            "fun": lambda x: - row.dot( x ) + bnd,
                                            "jac": lambda x: - row },
                         self.prob.conslinA[ inequb, : ],
                         self.prob.conslinub[ inequb ] )
                )
                out.extend(
                    map( lambda row, bnd: { "type": "ineq",
                                            "fun": lambda x: row.dot( x ) - bnd,
                                            "jac": lambda x: row },
                         self.prob.conslinA[ ineqlb, : ],
                         self.prob.conslinlb[ ineqlb ] )
                )

            if( self.prob.Ncons > 0 ):
                eq = ( self.prob.conslb == self.prob.consub )
                ineqlb = np.logical_and( ~eq, np.isfinite( self.prob.conslb ) )
                inequb = np.logical_and( ~eq, np.isfinite( self.prob.consub ) )

                ## must pass tuple(x) as argument since it needs to be hashable by lru_cache
                out.extend(
                    map( lambda idx: { "type": "eq",
                                       "fun": lambda x: myconsf( tuple(x) )[idx] \
                                                        - self.prob.consub[idx],
                                       "jac": lambda x: myconsg( tuple(x) )[idx] },
                         eq.nonzero()[0] )
                )
                out.extend(
                    map( lambda idx: { "type": "ineq",
                                       "fun": lambda x: - myconsf( tuple(x) )[idx] \
                                                        + self.prob.consub[idx],
                                       "jac": lambda x: - myconsg( tuple(x) )[idx] },
                         inequb.nonzero()[0] )
                )
                out.extend(
                    map( lambda idx: { "type": "ineq",
                                       "fun": lambda x: myconsf( tuple(x) )[idx] \
                                                        - self.prob.conslb[idx],
                                       "jac": lambda x: myconsg( tuple(x) )[idx] },
                         ineqlb.nonzero()[0] )
                )

            return out

        bounds = np.vstack( ( self.prob.lb,
                              self.prob.ub ) ).T
        extraopts = self.options.toDict()
        ## remove internal options
        if( "method" in extraopts ):
            del extraopts[ "method" ]
        if( "tol" in extraopts ):
            del extraopts[ "tol" ]

        res = scipy.optimize.minimize( x0 = self.init,
                                       fun = fun,
                                       jac = jac,
                                       bounds = bounds,
                                       constraints = cons_dict(),
                                       method = self.options[ "method" ].value,
                                       tol = self.options[ "tol" ].value,
                                       options = extraopts )

        self.prob.soln = Soln()
        self.prob.soln.final = np.copy( res.x )
        self.prob.soln.retval = res.status
        self.prob.soln.message = res.message
        self.prob.soln.value = res.fun
