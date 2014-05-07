cimport utils

cdef class Soln:
    pass

cdef class Solver:
    def __init__( self ):
        self.printOpts = { "printFile":None }
        self.solveOpts = { }


    def checkPrintOpts( self ):
        """
        Check if dictionary self.printOpts is valid.

        Optional entries:
        printFile        filename for debug information (default: None)
        """
        if( not utils.isString( self.printOpts[ "printFile" ] ) ):
            print( "printOpts['printFile'] must be a string." )
            return False

        return True


    def checkSolveOpts( self ):
        """
        Check if dictionary self.solveOpts is valid.
        """
        return True
