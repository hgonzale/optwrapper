from libc.string cimport memcpy
from libc.math cimport sqrt
cimport cpython.mem as mem
cimport numpy as np
import numpy as np

from f2ch cimport *      ## tydefs from f2c.h
cimport npsolh as npsol  ## import every function exposed in npsol.h
cimport utils
cimport base
import nlp

## NPSOL's option strings
cdef char* STR_NOLIST = "Nolist"
cdef char* STR_PRINT_FILE = "Print file"
cdef char* STR_SUMMARY_FILE = "Summary file"
cdef char* STR_PRINT_LEVEL = "Print level"
cdef char* STR_MINOR_PRINT_LEVEL = "Minor print level"
cdef char* STR_INFINITE_BOUND_SIZE = "Infinite bound size"
cdef char* STR_ITERATION_LIMIT = "Iteration limit"
cdef char* STR_MINOR_ITERATION_LIMIT = "Minor iteration limit"
cdef char* STR_LINE_SEARCH_TOLERANCE = "Line search tolerance"
cdef char* STR_FEASIBILITY_TOLERANCE = "Feasibility tolerance"
cdef char* STR_OPTIMALITY_TOLERANCE = "Optimality tolerance"
cdef char* STR_FUNCTION_PRECISION = "Function precision"
cdef char* STR_VERIFY_LEVEL = "Verify level"
cdef char* STR_WARM_START = "Warm start"

cdef tuple statusInfo = ( "Optimality conditions satisfied", ## 0
                          "Optimality conditions satisfied, but sequence has not converged", ## 1
                          "Linear constraints could not be satisfied", ## 2
                          "Nonlinear constraints could not be satisfied", ## 3
                          "Iteration limit reached", ## 4
                          "N/A",
                          "Optimality conditions not satisfied, no improvement can be made", ## 6
                          "Derivatives appear to be incorrect", ## 7
                          "N/A",
                          "Invalid input parameter" ) ## 9


## The functions funobj and funcon should be static methods in npsol.Solver,
## but it appears that Cython doesn't support static cdef methods yet.
## Instead, this is a reasonable hack.
cdef object extprob

## pg. 17, Section 7.1
cdef int funobj( integer* mode, integer* n,
                 doublereal* x, doublereal* f, doublereal* g,
                 integer* nstate ):
    xarr = utils.wrap1dPtr( x, n[0], np.NPY_DOUBLE )

    if( mode[0] != 1 ):
        f[0] = extprob.objf( xarr )

    if( mode[0] > 0 ):
        tmpg = utils.convFortran( extprob.objg( xarr ) )
        memcpy( g, utils.getPtr( tmpg ),
                n[0] * sizeof( doublereal ) )


## pg. 18, Section 7.2
cdef int funcon( integer* mode, integer* ncnln,
                 integer* n, integer* ldJ, integer* needc,
                 doublereal* x, doublereal* c, doublereal* cJac,
                 integer* nstate ):
    xarr = utils.wrap1dPtr( x, n[0], np.NPY_DOUBLE )

    if( mode[0] != 1 ):
        tmpf = utils.convFortran( extprob.consf( xarr ) )
        memcpy( c, utils.getPtr( tmpf ),
                ncnln[0] * sizeof( doublereal ) )

    if( mode[0] > 0 ):
        tmpg = utils.convFortran( extprob.consg( xarr ) )
        memcpy( cJac, utils.getPtr( tmpg ),
                ncnln[0] * n[0] * sizeof( doublereal ) )


cdef class Soln( base.Soln ):
    cdef public np.ndarray istate
    cdef public np.ndarray clamda
    cdef public np.ndarray R
    cdef public int Niters
    cdef public int retval

    def __init__( self ):
        self.retval = 100

    def getStatus( self ):
        if( self.retval == 100 ):
            return "Return information is not defined yet"

        if( self.retval < 0 ):
            return "Execution terminated by user defined function (should not occur)"
        elif( self.retval >= 10 ):
            return "Invalid value"
        else:
            return statusInfo[ self.retval ]


cdef class Solver( base.Solver ):
    cdef integer ldA[1]
    cdef integer ldJ[1]
    cdef integer ldR[1]
    cdef integer leniw[1]
    cdef integer lenw[1]
    cdef doublereal *x
    cdef doublereal *bl
    cdef doublereal *bu
    cdef doublereal *objg_val
    cdef doublereal *consf_val
    cdef doublereal *consg_val
    cdef doublereal *clamda
    cdef integer *istate
    cdef integer *iw
    cdef doublereal *w
    cdef doublereal *A
    cdef doublereal *R

    cdef int nctotl
    cdef int default_iter_limit
    cdef float default_tol
    cdef float default_fctn_prec
    cdef int warm_start
    cdef int mem_alloc
    cdef int mem_size[3] ## { N, Nconslin, Ncons }


    def __init__( self, prob=None ):
        super().__init__()

        self.mem_alloc = False
        self.mem_size[0] = self.mem_size[1] = self.mem_size[2] = 0
        self.default_tol = sqrt( np.spacing(1) ) ## pg. 24
        self.default_fctn_prec = np.power( np.spacing(1), 0.9 ) ## pg. 24
        self.prob = None

        ## Set options
        self.printOpts[ "summaryFile" ] = ""
        self.printOpts[ "printLevel" ] = 0
        self.printOpts[ "minorPrintLevel" ] = 0
        self.solveOpts[ "infValue" ] = 1e20
        self.solveOpts[ "iterLimit" ] = self.default_iter_limit
        self.solveOpts[ "minorIterLimit" ] = self.default_iter_limit
        self.solveOpts[ "lineSearchTol" ] = 0.9
        self.solveOpts[ "fctnPrecision" ] = 0 ## Invalid value
        self.solveOpts[ "feasibilityTol" ] = 0 ## Invalid value
        self.solveOpts[ "optimalityTol" ] = 0 ## Invalid value
        self.solveOpts[ "verifyGrad" ] = False

        ## We are assuming np.float64 equals doublereal from now on
        ## At least we need to be sure that doublereal is 8 bytes in this architecture
        assert( sizeof( doublereal ) == 8 )

        if( prob ):
            self.setupProblem( prob )


    def setupProblem( self, prob ):
        global extprob

        if( not isinstance( prob, nlp.Problem ) ):
            raise TypeError( "Argument 'prob' must be of type 'nlp.Problem'" )

        self.prob = prob ## Save a copy of prob's pointer
        extprob = prob ## Save a global copy prob's pointer for funcon and funobj

        ## New problems cannot be warm started
        self.warm_start = False

        ## Set size-dependent constants
        self.nctotl = prob.N + prob.Nconslin + prob.Ncons
        if( prob.Nconslin == 0 ): ## pg. 5, ldA >= 1 even if nclin = 0
            self.ldA[0] = 1
        else:
            self.ldA[0] = prob.Nconslin
        if( prob.Ncons == 0 ): ## pg. 5, ldJ >= 1 even if ncnln = 0
            self.ldJ[0] = 1
        else:
            self.ldJ[0] = prob.Ncons
        self.ldR[0] = prob.N
        self.leniw[0] = 3 * prob.N + prob.Nconslin + 2 * prob.Ncons ## pg. 7
        self.lenw[0] = ( 2 * prob.N * prob.N + prob.N * prob.Nconslin
                         + 2 * prob.N * prob.Ncons + 20 * prob.N + 11 * prob.Nconslin
                         + 21 * prob.Ncons ) ## pg. 7
        self.default_iter_limit = max( 50, 3*( prob.N + prob.Nconslin ) + 10*prob.Ncons ) ## pg. 25

        ## Allocate if necessary
        if( not self.mem_alloc ):
            self.allocate()
        elif( self.mem_size[0] != prob.N or
              self.mem_size[1] != prob.Nconslin or
              self.mem_size[2] != prob.Ncons ):
            self.deallocate()
            self.allocate()

        ## Require all arrays we are going to copy to be:
        ## two-dimensional, float64, fortran contiguous, and type aligned
        tmplb = utils.convFortran( prob.lb )
        memcpy( &self.bl[0], utils.getPtr( tmplb ),
                prob.N * sizeof( doublereal ) )
        tmpub = utils.convFortran( prob.ub )
        memcpy( &self.bu[0], utils.getPtr( tmpub ),
                prob.N * sizeof( doublereal ) )
        if( prob.Nconslin > 0 ):
            tmpconslinlb = utils.convFortran( prob.conslinlb )
            memcpy( &self.bl[prob.N], utils.getPtr( tmpconslinlb ),
                    prob.Nconslin * sizeof( doublereal ) )
            tmpconslinub = utils.convFortran( prob.conslinub )
            memcpy( &self.bu[prob.N], utils.getPtr( tmpconslinub ),
                    prob.Nconslin * sizeof( doublereal ) )
            tmpconslinA = utils.convFortran( prob.conslinA )
            memcpy( self.A, utils.getPtr( tmpconslinA ),
                    self.ldA[0] * prob.N * sizeof( doublereal ) )
        if( prob.Ncons > 0 ):
            tmpconslb = utils.convFortran( prob.conslb )
            memcpy( &self.bl[prob.N+prob.Nconslin], utils.getPtr( tmpconslb ),
                    prob.Ncons * sizeof( doublereal ) )
            tmpconsub = utils.convFortran( prob.consub )
            memcpy( &self.bu[prob.N+prob.Nconslin], utils.getPtr( tmpconsub ),
                    prob.Ncons * sizeof( doublereal ) )


    cdef allocate( self ):
        if( self.mem_alloc ):
            return False

        self.x = <doublereal *> mem.PyMem_Malloc( self.prob.N * sizeof( doublereal ) )
        self.bl = <doublereal *> mem.PyMem_Malloc( self.nctotl * sizeof( doublereal ) )
        self.bu = <doublereal *> mem.PyMem_Malloc( self.nctotl * sizeof( doublereal ) )
        self.objg_val = <doublereal *> mem.PyMem_Malloc( self.prob.N * sizeof( doublereal ) )
        self.consf_val = <doublereal *> mem.PyMem_Malloc( self.prob.Ncons * sizeof( doublereal ) )
        self.consg_val = <doublereal *> mem.PyMem_Malloc( self.ldJ[0] * self.prob.N * sizeof( doublereal ) )
        self.clamda = <doublereal *> mem.PyMem_Malloc( self.nctotl * sizeof( doublereal ) )
        self.istate = <integer *> mem.PyMem_Malloc( self.nctotl * sizeof( integer ) )
        self.iw = <integer *> mem.PyMem_Malloc( self.leniw[0] * sizeof( integer ) )
        self.w = <doublereal *> mem.PyMem_Malloc( self.lenw[0] * sizeof( doublereal ) )
        self.A = <doublereal *> mem.PyMem_Malloc( self.ldA[0] * self.prob.N * sizeof( doublereal ) )
        self.R = <doublereal *> mem.PyMem_Malloc( self.prob.N * self.prob.N * sizeof( doublereal ) )

        if( self.x == NULL or
            self.bl == NULL or
            self.bu == NULL or
            self.objg_val == NULL or
            self.consf_val == NULL or
            self.consg_val == NULL or
            self.clamda == NULL or
            self.istate == NULL or
            self.iw == NULL or
            self.w == NULL or
            self.A == NULL or
            self.R == NULL ):
            raise MemoryError( "At least one memory allocation failed" )

        self.mem_alloc = True
        self.mem_size[0] = self.prob.N
        self.mem_size[1] = self.prob.Nconslin
        self.mem_size[2] = self.prob.Ncons
        return True


    cdef deallocate( self ):
        if( not self.mem_alloc ):
            return False

        mem.PyMem_Free( self.x )
        mem.PyMem_Free( self.bl )
        mem.PyMem_Free( self.bu )
        mem.PyMem_Free( self.objg_val )
        mem.PyMem_Free( self.consf_val )
        mem.PyMem_Free( self.consg_val )
        mem.PyMem_Free( self.clamda )
        mem.PyMem_Free( self.istate )
        mem.PyMem_Free( self.iw )
        mem.PyMem_Free( self.w )
        mem.PyMem_Free( self.A )
        mem.PyMem_Free( self.R )

        self.mem_alloc = False
        return True


    def warmStart( self ):
        if( not isinstance( self.prob.soln, Soln ) ):
            return False

        tmpistate = utils.convIntFortran( self.prob.soln.istate )
        memcpy( self.istate, utils.getPtr( tmpistate ), self.nctotl * sizeof( integer ) )

        tmpclamda = utils.convFortran( self.prob.soln.clamda )
        memcpy( self.clamda, utils.getPtr( tmpclamda ), self.nctotl * sizeof( doublereal ) )

        tmpR = utils.convFortran( self.prob.soln.R )
        memcpy( self.R, utils.getPtr( tmpR ), self.prob.N * self.prob.N * sizeof( doublereal ) )

        self.warm_start = True
        return True


    def __dealloc__( self ):
        self.deallocate()


    def checkPrintOpts( self ):
        """
        Check if dictionary self.printOpts is valid.

        Optional entries:
        printFile        filename for debug information (default: None)
        printLevel       verbosity level for major iterations (0-None, 1, 5, 10, 20, or 30-Full)
        minorPrintLevel  verbosity level for minor iterations (0-None, 1, 5, 10, 20, or 30-Full)
        """
        if( not super().checkPrintOpts() ):
            return False

        ## printLevel and minorPrintLevel
        if( not utils.isInt( self.printOpts[ "printLevel" ] ) or
            not utils.isInt( self.printOpts[ "minorPrintLevel" ] ) ):
            print( "printOpts['printLevel'] and printOpts['minorPrintLevel'] must be integers" )
            return False
        if( self.printOpts[ "printFile" ] == "" and
            self.printOpts[ "printLevel" ] > 0 ):
                print( "Must set printOpts['printFile'] whenever printOpts['printLevel'] > 0" )
                return False

        return True


    def checkSolveOpts( self ):
        """
        Check if dictionary self.solveOpts is valid.

        Optional entries:
        infValue        Value above which is considered infinity (default: 1e20)
        iterLimit       Maximum number of iterations (default: max{50,3*(N+Nconslin)+10*Ncons})
        minorIterLimit  Maximum number of minor iterations (default: max{50,3*(N+Nconslin)+10*Ncons})
        lineSearchTol   Line search tolerance parameter (default: 0.9)
        """
        if( not super().checkSolveOpts() ):
            return False

        ## infValue
        if( not utils.isFloat( self.solveOpts[ "infValue" ] ) ):
            print( "solveOpts['infValue'] must be float" )
            return False
        if( self.solveOpts[ "infValue" ] < 0 ):
            print( "solveOpts['infValue'] must be positive" )
            return False
        elif( self.solveOpts[ "infValue" ] > 1e20 ):
            print( "Values for solveOpts['infValue'] above 1e20 are ignored" )
            return False

        ## lineSearchTol
        if( not utils.isFloat( self.solveOpts[ "lineSearchTol" ] ) ):
            print( "solveOpts['lineSearchTol'] must be float" )
            return False
        if( self.solveOpts[ "lineSearchTol" ] < 0 or
            self.solveOpts[ "lineSearchTol" ] >= 1 ):
            print( "solveOpts['lineSearchTol'] must belong to the interval [0,1)" )
            return False

        ## iterLimit
        if( not utils.isInt( self.solveOpts[ "iterLimit" ] ) ):
            print( "solveOpts['iterLimit'] must be integer" )
            return False
        if( self.solveOpts[ "iterLimit" ] < self.default_iter_limit ):
            print( "Values for solveOpts['iterLimit'] below "
                   + str( self.default_iter_limit ) + " are ignored" )

        ## minorIterLimit
        if( not utils.isInt( self.solveOpts[ "minorIterLimit" ] ) ):
            print( "solveOpts['minorIterLimit'] must be integer" )
            return False
        if( self.solveOpts[ "minorIterLimit" ] < self.default_iter_limit ):
            print( "Values for solveOpts['minorIterLimit'] below "
                   + str( self.default_iter_limit ) + " are ignored" )

        ## fctnPrecision
        if( not utils.isFloat( self.solveOpts[ "fctnPrecision" ] ) ):
            print( "solveOpts['fctnPrecision'] must be float" )
            return False
        if( self.solveOpts[ "fctnPrecision" ] < self.default_fctn_prec ):
            print( "Values for solveOpts['fctnPrecision'] below "
                   + str( self.default_fctn_prec ) + " are ignored" )

        ## feasibilityTol
        if( not utils.isFloat( self.solveOpts[ "feasibilityTol" ] ) ):
            print( "solveOpts['feasibilityTol'] must be float" )
            return False
        if( self.solveOpts[ "feasibilityTol" ] < self.default_tol ):
            print( "Values for solveOpts['feasiblityTol'] below "
                   + str( self.default_tol ) + " are ignored" )

        ## optimalityTol
        if( not utils.isFloat( self.solveOpts[ "optimalityTol" ] ) ):
            print( "solveOpts['optimalityTol'] must be float" )
            return False
        if( self.solveOpts[ "optimalityTol" ] < np.power( self.default_fctn_prec, 0.8 ) ):
            print( "Values for solveOpts['optimalityTol'] below "
                   + str( np.power( self.default_fctn_prec, 0.8 ) ) + " are ignored" )

        return True


    def solve( self ):
        cdef bytes printFileTmp = self.printOpts[ "printFile" ].encode() ## temp container
        cdef char* printFile = printFileTmp
        cdef bytes summaryFileTmp = self.printOpts[ "summaryFile" ].encode() ## temp container
        cdef char* summaryFile = summaryFileTmp
        cdef integer* summaryFileUnit = [ 89 ] ## Hardcoded since nobody cares
        cdef integer* printFileUnit = [ 90 ] ## Hardcoded since nobody cares
        cdef integer* printLevel = [ self.printOpts[ "printLevel" ] ]
        cdef integer* minorPrintLevel = [ self.printOpts[ "minorPrintLevel" ] ]
        cdef doublereal* infValue = [ self.solveOpts["infValue"] ]
        cdef integer* iterLimit = [ self.solveOpts[ "iterLimit" ] ]
        cdef integer* minorIterLimit = [ self.solveOpts[ "minorIterLimit" ] ]
        cdef doublereal* lineSearchTol = [ self.solveOpts["lineSearchTol"] ]
        cdef doublereal* fctnPrecision = [ self.solveOpts["fctnPrecision"] ]
        cdef doublereal* feasiblityTol = [ self.solveOpts["feasibilityTol"] ]
        cdef doublereal* optimalityTol = [ self.solveOpts["optimalityTol"] ]
        cdef integer verifyLevel[1]
        cdef float tmpPrec

        cdef integer *n = [ self.prob.N ]
        cdef integer *nclin = [ self.prob.Nconslin ]
        cdef integer *ncnln = [ self.prob.Ncons ]
        cdef integer iter_out[1]
        cdef integer inform_out[1]
        cdef doublereal objf_val[1]

        ## Begin by setting up initial condition
        tmpinit = utils.convFortran( self.prob.init )
        memcpy( self.x, utils.getPtr( tmpinit ),
                self.prob.N * sizeof( doublereal ) )

        ## Set all options
        ## Supress echo options
        npsol.npoptn_( STR_NOLIST, len( STR_NOLIST ) )

        ## Open file if necessary
        if( self.printOpts[ "printFile" ] != "" and
            self.printOpts[ "printLevel" ] > 0 ):
            npsol.npopenappend_( printFileUnit, printFile, inform_out,
                                 len( self.printOpts[ "printFile" ] ) )
            if( inform_out[0] != 0 ):
                raise IOError( "Could not open file " + self.printOpts[ "printFile" ] )
            npsol.npopti_( STR_PRINT_FILE, printFileUnit, len( STR_PRINT_FILE ) )

        if( self.printOpts[ "summaryFile" ] != "" ):
            npsol.npopenappend_( summaryFileUnit, summaryFile, inform_out,
                                 len( self.printOpts[ "summaryFile" ] ) )
            if( inform_out[0] != 0 ):
                raise IOError( "Could not open file " + self.printOpts[ "summaryFile" ] )
            npsol.npopti_( STR_SUMMARY_FILE, summaryFileUnit, len( STR_SUMMARY_FILE ) )

        ## Set major and minor print levels
        npsol.npopti_( STR_PRINT_LEVEL, printLevel, len( STR_PRINT_LEVEL ) )
        npsol.npopti_( STR_MINOR_PRINT_LEVEL, minorPrintLevel, len( STR_MINOR_PRINT_LEVEL ) )

        ## Set infinite bound value if necessary
        if( self.solveOpts["infValue"] < 1e20 ):
            npsol.npoptr_( STR_INFINITE_BOUND_SIZE, infValue, len( STR_INFINITE_BOUND_SIZE ) )

        ## Set major and minor iteration limits if necessary
        if( self.solveOpts["iterLimit"] > self.default_iter_limit ):
            npsol.npopti_( STR_ITERATION_LIMIT, iterLimit, len( STR_ITERATION_LIMIT ) )
        if( self.solveOpts["minorIterLimit"] > self.default_iter_limit ):
            npsol.npopti_( STR_MINOR_ITERATION_LIMIT, minorIterLimit,
                           len( STR_MINOR_ITERATION_LIMIT ) )

        ## Set line search tolerance value
        npsol.npoptr_( STR_LINE_SEARCH_TOLERANCE, lineSearchTol, len( STR_LINE_SEARCH_TOLERANCE ) )

        ## Set fctn precision, and feasibility and optimality tolerances
        tmpPrec = self.default_fctn_prec
        if( self.solveOpts["fctnPrecision"] > self.default_fctn_prec ):
            tmpPrec = self.solveOpts["fctnPrecision"]
            npsol.npoptr_( STR_FUNCTION_PRECISION, fctnPrecision,
                           len( STR_FUNCTION_PRECISION ) )
        if( self.solveOpts["feasibilityTol"] > self.default_tol ):
            npsol.npoptr_( STR_FEASIBILITY_TOLERANCE, feasiblityTol,
                           len( STR_FEASIBILITY_TOLERANCE ) )
        if( self.solveOpts["optimalityTol"] > np.power( tmpPrec, 0.8 ) ):
            npsol.npoptr_( STR_OPTIMALITY_TOLERANCE, optimalityTol,
                           len( STR_OPTIMALITY_TOLERANCE ) )

        ## Set verify level if required, pg. 29
        if( self.solveOpts["verifyGrad"] ):
            verifyLevel[0] = 3 ## Check both obj and cons
        else:
            verifyLevel[0] = -1 ## Disabled
        npsol.npopti_( STR_VERIFY_LEVEL, verifyLevel, len( STR_VERIFY_LEVEL ) )

        if( self.warm_start ):
            npsol.npoptn_( STR_WARM_START, len( STR_WARM_START ) )

        ## Call NPSOL
        npsol.npsol_( n, nclin,
                      ncnln, self.ldA,
                      self.ldJ, self.ldR,
                      self.A, self.bl, self.bu,
                      <npsol.funcon_fp> funcon, <npsol.funobj_fp> funobj,
                      inform_out, iter_out, self.istate,
                      self.consf_val, self.consg_val, self.clamda,
                      objf_val, self.objg_val, self.R, self.x,
                      self.iw, self.leniw, self.w, self.lenw )

        ## Politely close files
        if( self.printOpts[ "printFile" ] != "" and
            self.printOpts[ "printLevel" ] > 0 ):
            npsol.npclose_( printFileUnit )

        if( self.printOpts[ "summaryFile" ] != "" ):
            npsol.npclose_( summaryFileUnit )

        ## Save result to prob
        self.prob.soln = Soln()
        self.prob.soln.value = float( objf_val[0] )
        self.prob.soln.final = np.copy( utils.wrap1dPtr( self.x, self.prob.N,
                                                         np.NPY_DOUBLE ) )
        self.prob.soln.istate = np.copy( utils.wrap1dPtr( self.istate, self.nctotl,
                                                          np.NPY_LONG ) )
        self.prob.soln.clamda = np.copy( utils.wrap1dPtr( self.clamda, self.nctotl,
                                                          np.NPY_DOUBLE ) )
        self.prob.soln.R = np.copy( utils.wrap2dPtr( self.R, self.prob.N, self.prob.N,
                                                     np.NPY_DOUBLE ) )
        self.prob.soln.Niters = int( iter_out[0] )
        self.prob.soln.retval = int( inform_out[0] )

        return( self.prob.soln.final,
                self.prob.soln.value,
                self.prob.soln.retval )
