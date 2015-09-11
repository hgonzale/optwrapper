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
        nonlinear programming optimization problem

        Arguments:
        N:        number of optimization variables.
        Nconslin: number of linear constraints (default: 0).
        Ncons:    number of constraints (default: 0).

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

        self.init = np.zeros( (self.N,) )
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
        sets initial value for optimization variable.

        Arguments:
        init: array of size (N,) as initial condition. (default: array of zeros).

        """

        self.init = np.asfortranarray( init )

        if( self.init.shape != ( self.N, ) ):
            raise ValueError( "Argument must have size (" + str(self.N) + ",)." )


    def consBox( self, lb, ub ):
        """
        sets box constraints.

        Arguments:
        lb: lower bounds, array of size (N,).
        ub: upper bounds, array of size (N,).

        """

        self.lb = np.asfortranarray( lb )
        self.ub = np.asfortranarray( ub )

        if( self.lb.shape != ( self.N, ) or
            self.ub.shape != ( self.N, ) ):
            raise ValueError( "Both arrays must have size (" + str(self.N) + ",)." )


    def consLinear( self, A, lb=None, ub=None ):
        """
        sets linear constraints.

        Arguments:
        A:  linear constraint matrix, array of size (Nconslin,N).
        lb: lower bounds, array of size (Nconslin,). (default: -inf).
        ub: upper bounds, array of size (Nconslin,). (default: zeros).

        """

        self.conslinA = np.asfortranarray( A )

        if( self.conslinA.shape != ( self.Nconslin, self.N ) ):
            raise ValueError( "Argument 'A' must have size (" + str(self.Nconslin)
                              + "," + str(self.N) + ")." )

        if( lb is None ):
            lb = -np.inf * np.ones( (self.Nconslin,) )

        if( ub is None ):
            ub = np.zeros( (self.Nconslin,) )

        self.conslinlb = np.asfortranarray( lb )
        self.conslinub = np.asfortranarray( ub )

        if( self.conslinlb.shape != ( self.Nconslin, ) or
            self.conslinub.shape != ( self.Nconslin, ) ):
            raise ValueError( "Bounds must have size (" + str(self.Nconslin) + ",)." )


    def objFctn( self, objf, A=None ):
        """
        sets objective function of the form: objf(x) + A.dot(x).

        Arguments:
        objf: objective function receiving an output array of size (1,) and an optimization vector
              of size (N,), e.g.:

              def objf( out, x ):
                  out[0] = x.dot(x)

        A:    array of size (N,) for linear terms in the objective.

        """
        if( type(objf) != types.FunctionType ):
            raise ValueError( "Argument must be a function" )

        self.objf = objf

        if( A is not None ):
            self.objmixedA = np.asfortranarray( A )

            if( self.objmixedA.shape != ( self.N, ) ):
                raise ValueError( "Argument 'A' must have size (" + str(self.N) + ",)." )


    def objGrad( self, objg ):
        """
        sets objective gradient of the form: objg(x) + A, where A is defined in objFctn.

        Arguments:
        objg: objective gradient receiving an output array of size (N,) and an optimization
              vector of size (N,), e.g.:

              def objg( out, x ):
                  out[:] = 2 * x

        """

        if( type(objg) != types.FunctionType ):
            raise ValueError( "Argument must be a function" )

        self.objg = objg


    def consFctn( self, consf, lb=None, ub=None, A=None ):
        """
        sets constraint function of the form: consf(x) + A.dot(x).

        Arguments:
        consf: constraint function receiving an output array of size (Ncons,) and an optimization
               vector of size (N,), e.g.:

               def objf( out, x ):
                   for k in range( Ncons ):
                       out[k] = k * x.dot(x)

        A:    array of size (Ncons,N) for linear terms in the objective.
        lb:   lower bounds, array of size (Ncons,). (default: -inf).
        ub:   upper bounds, array of size (Ncons,). (default: zeros).

        """

        if( type(consf) != types.FunctionType ):
            raise ValueError( "Argument must be a function" )

        if( lb is None ):
            lb = -np.inf * np.ones( (self.Ncons,) )

        if( ub is None ):
            ub = np.zeros( (self.Ncons,) )

        self.consf = consf
        self.conslb = np.asfortranarray( lb )
        self.consub = np.asfortranarray( ub )

        if( self.conslb.shape != ( self.Ncons, ) or
            self.consub.shape != ( self.Ncons, ) ):
            raise ValueError( "Bound must have size (" + str(self.Ncons) + ",)." )

        if( A is not None ):
            self.consmixedA = np.asfortranarray( A )

            if( self.consmixedA.shape != ( self.Ncons, self.N ) ):
                raise ValueError( "Argument 'A' must have size ("
                                  + str(self.Ncons) + "," + str(self.N) + ")." )


    def consGrad( self, consg ):
        """
        sets constraint gradient of the form: consg(x) + A, where A is defined in consFctn.

        Arguments:
        consg: constraint gradient receiving an output array of size (Ncons,N) and an optimization
               vector of size (N,), e.g.:

               def consg( out, x ):
                   for k in range( Ncons ):
                       out[k,:] = 2 * k * x

        """

        if( type(consg) != types.FunctionType ):
            raise ValueError( "Argument must be a function" )

        self.consg = consg


    def checkGrad( self, Ntries=10, h=1e-5, etol=1e-4, point=None, debug=False ):
        """
        checks if user-defined gradients are correct using central finite differences.

        Arguments:
        h:     optimization vector variation size. (default: 1e-5).
        etol:  error tolerance. (default: 1e-4).
        point: evaluation point, array of size (N,). (default: initial condition).
        debug: boolean to enable extra debug information. (default: False).

        Returns:
        isCorrect: boolean, True if numerical and user-defined gradients match within error margin.

        """

        if( self.objf is None or self.objg is None ):
            raise StandardError( "Objective must be set before gradients are checked." )
        if( self.Ncons > 0 and
            ( self.consf is None or self.consg is None ) ):
            raise StandardError( "Constraints must be set before gradients are checked." )

        if( point is None ):
            ub = self.ub
            ub[ np.isinf( ub ) ] = 1
            lb = self.lb
            lb[ np.isinf( lb ) ] = -1
            point = np.random.rand( self.N ) * ( ub - lb ) + lb
        else:
            Ntries = 1
            point = np.asfortranarray( point )
            if( point.shape != ( self.N, ) ):
                raise ValueError( "Argument 'point' must have size (" + str(self.N) + ",)." )

        usrgrad = np.zeros( ( self.Ncons + 1, self.N ) )
        numgrad = np.zeros( ( self.Ncons + 1, self.N ) )
        fph = np.zeros( (self.Ncons + 1,) )
        fmh = np.zeros( (self.Ncons + 1,) )

        for iter in range( Ntries ):
            if( Ntries > 1 ):
                point = np.random.rand( self.N ) * ( ub - lb ) + lb

            self.objg( usrgrad[0,:], point )
            if( self.Ncons > 0 ):
                self.consg( usrgrad[1:,:], point )
            if( not np.all( np.isfinite( usrgrad ) ) ):
                raise ValueError( "gradient returned non-finite value" )

            for k in range( self.N ):
                hvec = np.zeros( (self.N,) )
                hvec[k] = h

                self.objf( fph[0:1], point + hvec )
                self.objf( fmh[0:1], point - hvec )
                if( self.Ncons > 0 ):
                    self.consf( fph[1:], point + hvec )
                    self.consf( fmh[1:], point - hvec )

                if( not np.all( np.isfinite( fph ) ) or
                    not np.all( np.isfinite( fmh ) ) ):
                    raise ValueError( "non-finite value inf at iteration {0}".format( k ) )

                numgrad[:,k] = ( fph - fmh ) / 2.0 / h

            errgrad = abs( usrgrad - numgrad )
            if( errgrad.max() >= etol ):
                if( debug ):
                    print( ">>> Numerical gradient check failed. " +
                           "Max error: {0}".format( errgrad.max() ) )
                    idx = np.unravel_index( np.argmax(errgrad), errgrad.shape )
                    if( idx[0] == 0 ):
                        print( ">>> Max error achieved at element " +
                               "{0} of objg()".format( idx[1] ) )
                    else:
                        print( ">>> Max error achieved at element " +
                               "({0},{1}) of consg()".format( idx[0]-1,idx[1] ) )
                return False

        if( debug ):
            print( ">>> Numerical gradient check passed. " )
        return True


class SparseProblem( Problem ):
    """
    General nonlinear programming optimization problem with sparse gradient matrices.
    Requires a nonlinear objective function and its gradient.
    Accepts box, linear, and nonlinear constraints.

    """

    def __init__( self, N, Ncons=0, Nconslin=0 ):
        Problem.__init__( self, N, Ncons, Nconslin )

        self.objgpattern = None
        self.consgpattern = None


    def objGrad( self, objg, pattern=None ):
        """
        sets objective gradient of the form: objg(x) + A, where A is defined in objFctn.

        Arguments:
        objg: objective gradient receiving an output array of size (N,) and an optimization
              vector of size (N,), e.g.:

              def objg( out, x ):
                  out[:] = 2 * x

        pattern: array of size (N,) whose zero entries mark constant zero entries in the output
                 of objg()

        """

        Problem.objGrad( self, objg )

        if( pattern is not None ):
            self.objgpattern = np.asfortranarray( pattern, dtype=np.int )

            if( self.objgpattern.shape != ( self.N, ) ):
                raise ValueError( "Argument 'pattern' must have size (" + str(self.N) + ",)." )


    def consGrad( self, consg, pattern=None ):
        """
        sets constraint gradient of the form: consg(x) + A, where A is defined in consFctn.

        Arguments:
        consg: constraint gradient receiving an output array of size (Ncons,N) and an optimization
               vector of size (N,), e.g.:

               def consg( out, x ):
                   for k in range( Ncons ):
                       out[k,:] = 2 * k * x

        pattern: array of size (Ncons,N) whose zero entries mark constant zero entries in the output
                 of consg()

        """

        Problem.consGrad( self, consg )

        if( pattern is not None ):
            self.consgpattern = np.asfortranarray( pattern, dtype=np.int )

            if( self.consgpattern.shape != ( self.Ncons, self.N ) ):
                raise ValueError( "Argument 'pattern' must have size (" + str(self.Ncons)
                                  + "," + str(self.N) + ")." )


    def checkPattern( self, Ntries=100, debug=False ):
        """
        checks if sparse patterns cover all nonzero entries in user defined gradients by
        evaluating objg() and consg() at random points drawn from a Gaussian distribution.

        Arguments:
        Ntries: number of random tries. (default: 100).
        debug:  boolean to enable extra debug information. (default: False).

        Returns:
        isCorrect: boolean, True if pattern covers all nonzero entries in the gradients.

        """

        if( self.objg is None or self.objgpattern is None ):
            raise StandardError( "objective gradient and pattern must be set before check" )
        if( self.Ncons > 0 and
            ( self.consg is None or self.consgpattern is None ) ):
            raise StandardError( "constraint gradient and pattern must be set before check" )

        usrgrad = np.zeros( (self.Ncons + 1, self.N) )
        if( self.Ncons > 0 ):
            pattern = np.vstack( (self.objgpattern, self.consgpattern ) )
        else:
            pattern = self.objgpattern

        for k in range( Ntries ):
            point = np.random.randn( self.N )

            self.objg( usrgrad[0,:], point )
            if( self.Ncons > 0 ):
                self.consg( usrgrad[1:,:], point )

            err = np.zeros( (self.Ncons + 1, self.N) )
            err[ pattern == 0 ] = usrgrad[ pattern == 0 ]
            if( np.any( err ) ):
                if( debug ):
                    idx = np.unravel_index( np.argmax( np.abs(err) ), err.shape )
                    if( idx[0] == 0 ):
                        print( ">>> Pattern check failed. Found wrong nonzero value in " +
                               "objg() at element {0}".format( idx[1] ) )
                    else:
                        print( ">>> Pattern check failed. Found wrong nonzero value in " +
                               "consg() at element ({0},{1})".format( idx[0]-1, idx[1] ) )
                return False

        if( debug ):
            print( ">>> Pattern check passed" )

        return True
