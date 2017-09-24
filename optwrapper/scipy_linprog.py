import numpy as np
import scipy.optimize

from optwrapper import base, utils, lp
import ipdb

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
        self.options[ "method" ] = "simplex"

        if( prob ):
            self.setupProblem( prob )


    def setupProblem( self, prob ):
        if( not isinstance( prob, lp.Problem ) ):
            raise TypeError( "Argument prob must be an instance of lp.Problem" )

        if( not prob.checkSetup() ):
            raise ValueError( "Argument 'prob' has not been properly configured" )

        self.prob = prob


    def solve( self ):
        consvar = ( self.prob.lb == self.prob.ub )
        c = self.prob.objL[ ~consvar ]
        bounds = np.vstack( ( self.prob.lb[ ~consvar ],
                              self.prob.ub[ ~consvar ] ) ).T

        A_ub = None
        b_ub = None
        A_eq = None
        b_eq = None
        if( self.prob.Nconslin > 0 ):
            conseq = ( self.prob.conslinlb == self.prob.conslinub )
            ineqlb = np.logical_and( ~conseq, np.isfinite( self.prob.conslinlb ) )
            inequb = np.logical_and( ~conseq, np.isfinite( self.prob.conslinub ) )
            if( any( ineqlb ) or any( inequb ) ):
                A_ub = np.vstack( ( self.prob.conslinA[ np.ix_( inequb, ~consvar ) ],
                                    - self.prob.conslinA[ np.ix_( ineqlb, ~consvar ) ] ) )
                b_ub = np.hstack( ( self.prob.conslinub[ inequb ],
                                    - self.prob.conslinlb[ ineqlb ] ) )
            if( any( conseq ) ):
                A_eq = self.prob.conslinA[ np.ix_( conseq, ~consvar ) ]
                b_eq = self.prob.conslinub[ conseq ]

        options = self.options.toDict()
        del options[ "method" ] ## remove internal options

        res = scipy.optimize.linprog( c = c,
                                      A_ub = A_ub,
                                      b_ub = b_ub,
                                      A_eq = A_eq,
                                      b_eq = b_eq,
                                      bounds = bounds,
                                      method = self.options["method"].value,
                                      options = options )

        self.prob.soln = Soln()
        self.prob.soln.final = np.empty( (self.prob.N,) )
        self.prob.soln.final[ consvar ] = self.prob.ub[ consvar ]
        self.prob.soln.final[ ~consvar ] = res.x
        self.prob.soln.retval = res.status
        self.prob.soln.message = res.message
        self.prob.soln.value = res.fun
