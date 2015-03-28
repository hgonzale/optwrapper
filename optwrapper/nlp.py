import types
import numpy as np

class Problem:
    """
    General nonlinear programming optimization problem.
    Requires a nonlinear objective function and its gradient.
    Accepts box, linear, and nonlinear constraints.
    """

    def __init__( self, N, Ncons=0, Nconslin=0 ):
        """
        Arguments:
        N         number of optimization variables (required).
        Nconslin  number of linear constraints (default: 0).
        Ncons     number of constraints (default: 0).

        prob = optProblem( N=2, Ncons=2, Nconslin=3 )
        """

        try:
            self.N = int( N )
        except:
            raise ValueError( "N must be an integer" )

        if( self.N <= 0 ):
            raise ValueError( "N must be strictly positive" )

        try:
            self.Nconslin = int( Nconslin )
        except:
            raise ValueError( "Nconslin was not provided or was not an integer" )

        if( self.Nconslin < 0 ):
            raise ValueError( "Nconslin must be positive" )

        try:
            self.Ncons = int( Ncons )
        except:
            raise ValueError( "Ncons was not provided or was not an integer" )

        if( self.Ncons < 0 ):
            raise ValueError( "Ncons must be positive" )

        self.init = np.zeros( self.N )
        self.lb = None
        self.ub = None
        self.objf = None
        self.objmixedA = None
        self.objg = None
        self.consf = None
        self.consg = None
        self.conslb = None
        self.consub = None
        self.consmixedA = None
        self.conslinA = None
        self.conslinlb = None
        self.conslinub = None
        self.soln = None


    def initPoint( self, init ):
        """
        Sets initial value for optimization variable.

        Arguments:
        init  initial condition, must be a one-dimensional array of size N
              (default: vector of zeros).

        prob.initCond( [ 1.0, 1.0 ] )
        """
        self.init = np.asfortranarray( init )

        if( self.init.shape != ( self.N, ) ):
            raise ValueError( "Argument must have size (" + str(self.N) + ",)." )


    def consBox( self, lb, ub ):
        """
        Defines box constraints.

        Arguments:
        lb  lower bounds, one-dimensional array of size N.
        ub  upper bounds, one-dimensional array of size N.

        prob.consBox( [-1,-2], [1,2] )
        """
        self.lb = np.asfortranarray( lb )
        self.ub = np.asfortranarray( ub )

        if( self.lb.shape != ( self.N, ) or
            self.ub.shape != ( self.N, ) ):
            raise ValueError( "Bound must have size (" + str(self.N) + ",)." )


    def consLinear( self, A, lb=None, ub=None ):
        """
        Defines linear constraints.

        Arguments:
        A   linear constraint matrix, two-dimensional array of size (Nconslin,N).
        lb  lower bounds, one-dimensional array of size Nconslin.
        ub  upper bounds, one-dimensional array of size Nconslin.

        prob.consLinear( [[1,-1],[1,1]], [-1,-2], [1,2] )
        """
        self.conslinA = np.asfortranarray( A )

        if( self.conslinA.shape != ( self.Nconslin, self.N ) ):
            raise ValueError( "Argument 'A' must have size (" + str(self.Nconslin)
                              + "," + str(self.N) + ")." )

        if( lb is None ):
            lb = -np.inf * np.ones( self.Nconslin )

        if( ub is None ):
            ub = np.zeros( self.Nconslin )

        self.conslinlb = np.asfortranarray( lb )
        self.conslinub = np.asfortranarray( ub )

        if( self.conslinlb.shape != ( self.Nconslin, ) or
            self.conslinub.shape != ( self.Nconslin, ) ):
            raise ValueError( "Bounds must have size (" + str(self.Nconslin) + ",)." )


    def objFctn( self, objf, A=None ):
        """
        Set objective function.

        Arguments:
        objf  objective function, must return a scalar.

        def objf(x):
            return x[1]
        prob.objFctn( objf )
        """
        if( type(objf) != types.FunctionType ):
            raise ValueError( "Argument must be a function" )

        self.objf = objf

        if( not A is None ):
            self.objmixedA = np.asfortranarray( A )

            if( self.objmixedA.shape != ( self.N, ) ):
                raise ValueError( "Argument 'A' must have size (" + str(self.N) + ",)." )


    def objGrad( self, objg ):
        """
        Set objective gradient.

        Arguments:
        objg  gradient function, must return a one-dimensional array of size N.

        def objg(x):
            return np.array( [2,-1] )
        prob.objGrad( objg )
        """
        if( type(objg) != types.FunctionType ):
            raise ValueError( "Argument must be a function" )

        self.objg = objg


    def consFctn( self, consf, lb=None, ub=None, A=None ):
        """
        Set nonlinear constraints function.

        Arguments:
        consf  constraint function, must return a one-dimensional array of
               size Ncons.
        lb     lower bounds, one-dimensional array of size Ncons (default: vector
               of -inf).
        ub     upper bounds, one-dimensional array of size Ncons (default: vector
               of zeros).

        def consf(x):
            return np.array( [ x[0] - x[1],
                               x[0] + x[1] ] )
        prob.consFctn( consf )
        """
        if( type(consf) != types.FunctionType ):
            raise ValueError( "Argument must be a function" )

        if( lb is None ):
            lb = -np.inf * np.ones( self.Ncons )

        if( ub is None ):
            ub = np.zeros( self.Ncons )

        self.consf = consf
        self.conslb = np.asfortranarray( lb )
        self.consub = np.asfortranarray( ub )

        if( self.conslb.shape != ( self.Ncons, ) or
            self.consub.shape != ( self.Ncons, ) ):
            raise ValueError( "Bound must have size (" + str(self.Ncons) + ",)." )

        if( not A is None ):
            self.consmixedA = np.asfortranarray( A )

            if( self.consmixedA.shape != ( self.Ncons, self.N ) ):
                raise ValueError( "Argument 'A' must have size ("
                                  + str(self.Ncons) + "," + str(self.N) + ")." )

    def consGrad( self, consg ):
        """
        Set nonlinear constraints gradient.

        Arguments:
        consg  constraint gradient, must return a two-dimensional array of
               size (Ncons,N), where entry [i,j] is the derivative of i-th
               constraint w.r.t. the j-th variables.

        def consg(x):
            return np.array( [ [ 2*x[0], 8*x[1] ],
                               [ 2*(x[0]-2), 2*x[1] ] ] )
        prob.consGrad( consg )
        """
        if( type(consg) != types.FunctionType ):
            raise ValueError( "Argument must be a function" )

        self.consg = consg


    def checkGrad( self, h=1e-5, etol=1e-4, point=None, debug=False ):
        """
        Checks if user-defined gradients are correct using finite
        differences.

        Arguments:
        h      optimization variable variation step size (default: 1e-5).
        etol   error tolerance (default: 1e-4).
        point  evaluation point one-dimensional array of size N (default:
               initial condition).
        debug  boolean to enable extra debug information (default: False).

        isCorrect = prob.checkGrad( h=1e-6, etol=1e-5, point, debug=False )
        """
        if( self.objf is None or
            self.objg is None ):
            raise StandardError( "Objective must be set before gradients are checked." )
        if( self.Ncons > 0 and
            ( self.consf is None or self.consg is None ) ):
            raise StandardError( "Constraints must be set before gradients are checked." )

        if( point is None ):
            point = self.init
        else:
            point = np.asfortranarray( point )
            if( point.shape != ( self.N, ) ):
                raise ValueError( "Argument 'point' must have size (" + str(self.N) + ",)." )

        usrgrad = np.zeros( [ self.Ncons + 1, self.N ] )
        numgrad = np.zeros( [ self.Ncons + 1, self.N ] )

        fph = np.zeros( self.Ncons + 1 )
        fmh = np.zeros( self.Ncons + 1 )
        for k in range( 0, self.N ):
            hvec = np.zeros( self.N )
            hvec[k] = h

            self.objf( fph[0:1], point + hvec )
            self.objf( fmh[0:1], point - hvec )
            self.consf( fph[1:], point + hvec )
            self.consf( fmh[1:], point - hvec )

            if( np.any( np.isnan( fph ) ) or np.any( np.isnan( fmh ) ) or
                np.any( np.isinf( fph ) ) or np.any( np.isinf( fmh ) ) ):
                raise ValueError( "Function returned NaN or inf at iteration " + str(k) )

            delta = ( fph - fmh ) / 2.0 / h
            numgrad[:,k] = delta

        self.objg( usrgrad[0,:], point )
        self.consg( usrgrad[1:,:], point )
        if( np.any( np.isnan( usrgrad ) ) or
            np.any( np.isinf( usrgrad ) ) ):
            raise ValueError( "Gradient returned NaN or inf." )

        errgrad = abs( usrgrad - numgrad )
        if( errgrad.max() < etol ):
            if( debug ):
                print( "Numerical gradient check passed. Max error was: " + str( errgrad.max() ) )
            return( True, errgrad.max(), errgrad )
        else:
            if( debug ):
                idx = np.unravel_index( np.argmax(err), err.shape )
                if( idx[0] == 0 ):
                    print( "Objective gradient incorrect in element=("
                           + str(idx[1]) + ")" )
                else:
                    print( "Constraint gradient incorrect in element=( "
                           + str(idx[0]-1) + "," + str(idx[1]) + ")" )
            return( False, errgrad.max(), errgrad )


    def check( self, debug=False ):
        """
        General checks required before solver is executed.

        Arguments:
        debug  boolean to enable extra debug information (default: False).

        isCorrect = prob.check()
        """

        if( self.lb is None or
            self.ub is None or
            np.any( self.lb > self.ub ) ):
            if( debug ):
                print( "Box constraints not set or lower bound larger than upper bound." )
            return False

        if( np.any( self.lb > self.init ) or
            np.any( self.init > self.ub ) ):
            if( debug ):
                print( "Initial condition not set or violates box constraints." )
            return False

        if( self.objf is None or
            np.any( np.isnan( self.objf( self.init ) ) ) or
            np.any( np.isinf( self.objf( self.init ) ) ) ):
            if( debug ):
                print( "Objective function not set or return NaN/inf for initial condition." )
            return False

        if( self.objf( self.init ).shape != (1,) ):
            if( debug ):
                print( "Objective function must return a scalar array." )
            return False

        if( self.objg is None or
            np.any( np.isnan( self.objg( self.init ) ) ) or
            np.any( np.isinf( self.objg( self.init ) ) ) ):
            if( debug ):
                print( "Objective gradient not set or return NaN/inf for initial condition." )
            return False

        if( self.objg( self.init ).shape != ( self.N, ) ):
            if( debug ):
                print( "Objective gradient must return array of size (" + str(self.N) + ",)." )
            return False

        if( Nconslin > 0 ):
            if( self.conslinlb is None or
                self.conslinub is None or
                np.any( self.conslinlb > self.conslinub ) ):
                if( debug ):
                    print( "Linear constraint bounds not set or lower bound larger than upper bound." )
                return False

        if( Ncons > 0 ):
            if( self.conslb is None or
                self.consub is None or
                np.any( self.conslb > self.consub ) ):
                if( debug ):
                    print( "Constraint bounds not set or lower bound larger than upper bound." )
                return False

            if( self.consf is None or
                np.any( np.isnan( self.consf( self.init ) ) ) or
                np.any( np.isinf( self.consf( self.init ) ) ) ):
                if( debug ):
                    print( "Constraint function not set or return NaN/inf for initial condition." )
                return False

            if( self.consf( self.init ).shape != ( self.Ncons, ) ):
                if( debug ):
                    print( "Constraint function must return array of size (" + str(self.Ncons) + ",)." )
                return False

            if( self.consg is None or
                np.any( np.isnan( self.consg( self.init ) ) ) or
                np.any( np.isinf( self.consg( self.init ) ) ) ):
                if( debug ):
                    print( "Constraint gradient not set or return NaN/inf for initial condition." )
                return False

            if( self.consg( self.init ).shape != ( self.Ncons, self.N ) ):
                if( debug ):
                    print( "Constraint gradient must return array of size ("
                           + str(self.Ncons) + "," + str(self.N) + ")." )
                return False

        return True



class SparseProblem( Problem ):
    """
    General nonlinear programming optimization problem.
    Requires a nonlinear objective function and its gradient.
    Accepts box, linear, and nonlinear constraints.
    """

    def __init__( self, N, Ncons=0, Nconslin=0 ):
        Problem.__init__( self, N, Ncons, Nconslin )

        self.objgpattern = None
        self.consgpattern = None


    def objGrad( self, objg, pattern=None ):
        """
        Set objective gradient.

        Arguments:
        objg  gradient function, must return a one-dimensional array of size N.

        def objg(x):
            return np.array( [2,-1] )
        prob.objGrad( objg )
        """
        Problem.objGrad( self, objg )

        if( not pattern is None ):
            self.objgpattern = np.asfortranarray( pattern, dtype=np.int )

            if( self.objgpattern.shape != ( self.N, ) ):
                raise ValueError( "Argument 'pattern' must have size (" + str(self.N) + ",)." )


    def consGrad( self, consg, pattern=None ):
        """
        Set nonlinear constraints gradient.

        Arguments:
        consg  constraint gradient, must return a two-dimensional array of
               size (Ncons,N), where entry [i,j] is the derivative of i-th
               constraint w.r.t. the j-th variables.

        def consg(x):
            return np.array( [ [ 2*x[0], 8*x[1] ],
                               [ 2*(x[0]-2), 2*x[1] ] ] )
        prob.consGrad( consg )
        """
        Problem.consGrad( self, consg )

        if( not pattern is None ):
            self.consgpattern = np.asfortranarray( pattern, dtype=np.int )

            if( self.consgpattern.shape != ( self.Ncons, self.N ) ):
                raise ValueError( "Argument 'pattern' must have size (" + str(self.Ncons)
                                  + "," + str(self.N) + ")." )


    def checkPatterns( self ):
        ## TODO
        pass
