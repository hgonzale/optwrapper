from optwSolver cimport *

cdef class optwSolver:
    def __init__( self ):
        self.stopOpts = { "stopval":None,
                          "maxeval":1e4,
                          "maxtime":1e4,
                          "xtol":1e-6,
                          "ftol":1e-6 }
        self.printOpts = { "print_level":0,
                           "summary_level":0,
                           "options_file":None,
                           "print_file":None }
        self.solveOpts = { "warm_start":False,
                           "constraint_violation":1e-8 }

    def checkStopOpts( self ):
        """
        Check if dictionary self.stopOpts is valid.
        """
        return True


    def checkPrintOpts( self ):
        """
        Check if dictionary self.printOpts is valid.
        """
        # print_level: verbosity of file output (0-11 for SNOPT, 0-30 for NPSOL, 0-12 for IPOPT)
        # summary_level: verbosity of output to screen (0-1 for SNOPT, 0-12 for IPOPT)
        return True


    def checkSolveOpts( self ):
        """
        Check if dictionary self.solveOpts is valid.
        """
        return True
