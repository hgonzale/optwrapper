import numpy as np

class Problem:
    """
    General linear programming optimization problem.
    Requires a vector to define the objective function.
    Accepts box and linear constraints.

    """

    def __init__( self, N, Nconslin=0 ):
        """
        linear programming optimization problem

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
            raise ValueError( "Nconslin must be non-negative" )

        self.lb = None
        self.ub = None
        self.objL = None
        self.conslinA = None
        self.conslinlb = None
        self.conslinub = None
        self.soln = None


    def checkSetup( self ):
        out = ( self.lb is not None and
                self.ub is not None )

        if( self.Nconslin > 0 ):
            out = out and ( self.conslinA is not None and
                            self.conslinlb is not None and
                            self.conslinub is not None )

        return out


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


    def objFctn( self, lin=None ):
        """
        sets objective function of the form: L.dot(x).

        Arguments:
        L: array of size (N,) for linear terms in the objective.

        """

        if( lin is not None ):
            self.objL = np.asfortranarray( lin )

            if( self.objL.shape != ( self.N, ) ):
                raise ValueError( "Array L must have size (" + str(self.N) + ",)." )
