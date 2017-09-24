## define shell classes to fix basic interfaces

cdef class Soln:
    def getStatus( self ):
        return str()


cdef class Solver:
    def __init__( self ):
        self.debug = False

    def initPoint( self, init ):
        pass

    def solve( self ):
        pass

    def setupProblem( self, prob ):
        pass

    def warmStart( self ):
        pass
