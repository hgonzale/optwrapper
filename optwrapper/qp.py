import numpy as np
from optwrapper import lp
from enum import Enum

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
        self.objQtype = None


    def checkSetup( self ):
        out = super().checkSetup()

        out = out and ( self.objL is not None and
                        self.objQ is not None )

        return out


    def objFctn( self, quad=None, lin=None, quadtype=None ):
        """
        sets objective function of the form: 0.5 * x.dot(Q).dot(x) + L.dot(x).

        Arguments:
        Q: array of size (N,N) for quadratic terms in the objective.
        L: array of size (N,) for linear terms in the objective.

        """

        super().objFctn( lin )

        if( self.objL.shape != ( self.N, ) ):
            raise ValueError( "Array L must have size (" + str(self.N) + ",)." )

        self.objQ = np.asfortranarray( quad )

        if( self.objQ.shape != ( self.N, self.N ) ):
            raise ValueError( "Array Q must have size (" + str(self.N) + "," +
                              str(self.N) + ")." )

        if( quadtype ):
            if( not isinstance( quadtype, QuadType ) ):
                raise ValueError( "argument 'quadtype' must be of type 'qp.QuadType'" )
            self.quadtype = quadtype


class QuadType( Enum ):
    indef = 1
    posdef = 2
    possemidef = 3
    identity = 4
    zero = 5
