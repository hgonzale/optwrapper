from optwSolver cimport *

cdef class optwSolver:
    def __init__( self ):
        self.stopOpts = { "stopval":None,
                          "maxeval":1e4,
                          "maxtime":1e4,
                          "xtol":1e-6,
                          "ftol":1e-6 }
        self.printOpts = { "printFile":None }
        self.solveOpts = { "warmStart":False,
                           "constraintViolation":1e-8 }

    def checkStopOpts( self ):
        """
        Check if dictionary self.stopOpts is valid.
        """
        return True


    def checkPrintOpts( self ):
        """
        Check if dictionary self.printOpts is valid.
        """
        return True


    def checkSolveOpts( self ):
        """
        Check if dictionary self.solveOpts is valid.
        """
        return True
