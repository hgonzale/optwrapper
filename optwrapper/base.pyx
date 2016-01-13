## define shell classes to fix basic interfaces

cdef class Soln:
    pass

cdef class Solver:
    def __init__( self ):
        self.debug = False
