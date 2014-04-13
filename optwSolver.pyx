

cdef class optwSolver:
    def __init__():
        self.stopOpts = { "stopval":None,
                          "maxeval":1000,
                          "maxtime":1.0,
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
        if( not type(val) == types.DictType or
            not val.has_key("xtol") or
            not val.has_key("ftol") or
            not val.has_key("stopval") or
            not val.has_key("maxeval") or
            not val.has_key("maxtime") ):
            return False
        else:
            return True


    def checkPrintOpts( self ):
        """
        Check if dictionary self.printOpts is valid.
        """
        # print_level: verbosity of file output (0-11 for SNOPT, 0-30 for NPSOL, 0-12 for IPOPT)
        # summary_level: verbosity of output to screen (0-1 for SNOPT, 0-12 for IPOPT)
        if( not type(val) == types.DictType or
            not val.has_key("print_level") or
            not val.has_key("summary_level") or
            not val.has_key("options_file") or
            not val.has_key("print_file") ):
            return False
        else:
            return True


    def checkSolveOpts( self ):
        """
        Check if dictionary self.solveOpts is valid.
        """
        if( not type(val) == types.DictType or
            not val.has_key("constraint_violation") or
            not val.has_key("warm_start") ):
            return False
        else:
            return True
