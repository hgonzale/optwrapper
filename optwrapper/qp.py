import types
import numpy as np

class Problem:
    """
    General quadratic programming optimization problem.
    Requires a matrix and a vector to define the objective function.
    Accepts box and linear constraints.

    """

    def __init__( self, N, Nconslin=0 ):
        """
        quadratic programming optimization problem

        Arguments:
        N:        number of optimization variables.
        Nconslin: number of linear constraints (default: 0).

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

        self.init = np.zeros( (self.N,) )
        self.lb = None
        self.ub = None
        self.objQ = None
        self.objL = None
        self.conslinA = None
        self.conslinlb = None
        self.conslinub = None
        self.soln = None


    def initPoint( self, init ):
        """
        sets initial value for optimization variables.

        Arguments:
        init: initial condition, must be an array of size (N,). (default: zeros).

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

        if( self.Nconslin == 0 ):
            raise ValueError( "cannot set linear constraints when Nconslin=0" )

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


    def objFctn( self, quad=None, lin=None ):
        """
        sets objective function of the form: 0.5 * x.dot(Q).dot(x) + L.dot(x).

        Arguments:
        Q: array of size (N,N) for quadratic terms in the objective. (default: identity)
        L: array of size (N,) for linear terms in the objective. (default: zeros)

        """

        if( quad is not None ):
            self.objQ = np.asfortranarray( quad )

            if( self.objQ.shape != ( self.N, self.N ) ):
                raise ValueError( "Array Q must have size (" + str(self.N) + "," +
                                  str(self.N) + ")." )

        if( lin is not None ):
            self.objL = np.asfortranarray( lin )

            if( self.objL.shape != ( self.N, ) ):
                raise ValueError( "Array L must have size (" + str(self.N) + ",)." )
