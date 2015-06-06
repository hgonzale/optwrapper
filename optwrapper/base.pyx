## define shell classes to fix basic interfaces

cdef class Soln:
    pass

cdef class Solver:
    def __init__( self ):
        self.printOpts = { "printFile":"" }
        self.solveOpts = { }
        self.debug = False
