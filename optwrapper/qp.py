import numpy as np
from optwrapper import lp

class Problem( lp.Problem ):
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

        super().__init__( N, Nconslin )
        self.objQ = None


    def objFctn( self, quad=None, lin=None ):
        """
        sets objective function of the form: 0.5 * x.dot(Q).dot(x) + L.dot(x).

        Arguments:
        Q: array of size (N,N) for quadratic terms in the objective.
        L: array of size (N,) for linear terms in the objective.

        """

        super().objFctn( lin )

        if( quad is not None ):
            self.objQ = np.asfortranarray( quad )

            if( self.objQ.shape != ( self.N, self.N ) ):
                raise ValueError( "Array Q must have size (" + str(self.N) + "," +
                                  str(self.N) + ")." )
