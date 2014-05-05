from libc.stdlib cimport calloc,free
from libc.string cimport memcpy
import numpy as np
cimport numpy as np
cimport cpython.mem as mem
from libc.math cimport sqrt

from optwF2c cimport *
cimport optwNpsol as npsol
cimport arrayWrapper as arrwrap
from optwSolver cimport *
from optwrapper import *

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

cdef tuple statusInfo = ( "Optimality conditions satisfied",
                          "Optimality conditions satisfied, but sequence has not converged",
                          "Linear constraints could not be satisfied",
                          "Nonlinear constraints could not be satisfied",
                          "Iteration limit reached",
                          "N/A",
                          "Optimality conditions not satisfied, no improvement can be made",
                          "Derivatives appear to be incorrect",
                          "N/A",
                          "Invalid input parameter" )


## These functions should be static methods in optwNpsol, but it appears that
## Cython doesn't support static cdef methods yet.
## Instead, this is a reasonable hack.
cdef object extprob

## pg. 17, Section 7.1
cdef int funobj( integer* mode, integer* n,
                 doublereal* x, doublereal* f, doublereal* g,
                 integer* nstate ):
    xarr = arrwrap.wrap1dPtr( x, extprob.N, np.NPY_DOUBLE )

    if( mode[0] != 1 ):
        f[0] = extprob.objf( xarr )

    if( mode[0] > 0 ):
        tmpg = arrwrap.convFortran( extprob.objg( xarr ) )
        memcpy( g, arrwrap.getPtr( tmpg ),
                extprob.N * sizeof( doublereal ) )


## pg. 18, Section 7.2
cdef int funcon( integer* mode, integer* ncnln,
                 integer* n, integer* ldJ, integer* needc,
                 doublereal* x, doublereal* c, doublereal* cJac,
                 integer* nstate ):
    xarr = arrwrap.wrap1dPtr( x, extprob.N, np.NPY_DOUBLE )

    if( mode[0] != 1 ):
        tmpf = arrwrap.convFortran( extprob.consf( xarr ) )
        memcpy( c, arrwrap.getPtr( tmpf ),
                extprob.Ncons * sizeof( doublereal ) )

    if( mode[0] > 0 ):
        tmpg = arrwrap.convFortran( extprob.consg( xarr ) )
        memcpy( cJac, arrwrap.getPtr( tmpg ),
                extprob.Ncons * extprob.N * sizeof( doublereal ) )


cdef class optwNpsol( optwSolver ):
    cdef integer n[1]
    cdef integer nclin[1]
    cdef integer ncnln[1]
    cdef integer ldA[1]
    cdef integer ldJ[1]
    cdef integer ldR[1]
    cdef integer leniw[1]
    cdef integer lenw[1]
    cdef integer iter_out[1]
    cdef integer inform_out[1]
    cdef doublereal *x
    cdef doublereal *bl
    cdef doublereal *bu
    cdef doublereal objf_val[1]
    cdef doublereal *objg_val
    cdef doublereal *consf_val
    cdef doublereal *consg_val
    cdef doublereal *clamda
    cdef integer *istate
    cdef integer *iw
    cdef doublereal *w
    cdef doublereal *A
    cdef doublereal *R
    cdef object prob
    cdef integer nctotl
    cdef integer default_iter_limit
    cdef doublereal default_tol
    cdef doublereal default_fctn_prec
    cdef int warm_start

    def __init__( self, prob ):
        super().__init__()

        if( not isinstance( prob, optwProblem ) ):
            raise StandardError( "Argument 'prob' must be of type 'optwProblem'" )
        self.prob = prob ## Save a copy of the pointer
        self.setupProblem( self.prob )


    def setupProblem( self, prob ):
        global extprob
        extprob = prob ## Save static prob for funcon and funobj

        self.nctotl = prob.N + prob.Nconslin + prob.Ncons
        self.default_iter_limit = max( 50, 3*( prob.N + prob.Nconslin ) + 10*prob.Ncons ) ## pg. 25
        self.default_tol = sqrt( np.spacing(1) ) ## pg. 24
        self.default_fctn_prec = np.power( np.spacing(1), 0.9 ) ## pg. 24
        self.warm_start = False

        self.n[0] = prob.N
        self.nclin[0] = prob.Nconslin
        self.ncnln[0] = prob.Ncons
        if( prob.Nconslin == 0 ):
            self.ldA[0] = 1 ## pg. 5, ldA >= 1 even if nclin = 0
        else:
            self.ldA[0] = prob.Nconslin

        if( prob.Ncons == 0 ):
            self.ldJ[0] = 1 ## pg. 5, ldJ >= 1 even if ncnln = 0
        else:
            self.ldJ[0] = prob.Ncons

        self.ldR[0] = prob.N

        ## pg. 7
        self.leniw[0] = 3 * prob.N + prob.Nconslin + 2 * prob.Ncons
        self.lenw[0] = ( 2 * prob.N * prob.N + prob.N * prob.Nconslin
                         + 2 * prob.N * prob.Ncons + 20 * prob.N + 11 * prob.Nconslin
                         + 21 * prob.Ncons )

        self.iter_out[0] = 0
        self.inform_out[0] = 100

        self.x = <doublereal *> mem.PyMem_Malloc( prob.N * sizeof( doublereal ) )
        self.bl = <doublereal *> calloc( self.nctotl, sizeof( doublereal ) )
        self.bu = <doublereal *> calloc( self.nctotl, sizeof( doublereal ) )

        self.objf_val[0] = 0.0
        self.objg_val = <doublereal *> calloc( prob.N, sizeof( doublereal ) )
        self.consf_val = <doublereal *> calloc( prob.Ncons, sizeof( doublereal ) )
        self.consg_val = <doublereal *> calloc( self.ldJ[0] * prob.N, sizeof( doublereal ) )

        self.clamda = <doublereal *> calloc( self.nctotl, sizeof( doublereal ) )
        self.istate = <integer *> calloc( self.nctotl, sizeof( integer ) )

        self.iw = <integer *> calloc( self.leniw[0], sizeof( integer ) )
        self.w = <doublereal *> calloc( self.lenw[0], sizeof( doublereal ) )

        self.A = <doublereal *> calloc( self.ldA[0] * prob.N, sizeof( doublereal ) )
        self.R = <doublereal *> calloc( prob.N * prob.N, sizeof( doublereal ) )

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

        ## Require all arrays we are going to copy to be:
        ## two-dimensional, float64, fortran contiguous, and type aligned
        tmpinit = arrwrap.convFortran( prob.init )
        memcpy( self.x, arrwrap.getPtr( tmpinit ),
                prob.N * sizeof( doublereal ) )
        tmplb = arrwrap.convFortran( prob.lb )
        memcpy( &self.bl[0], arrwrap.getPtr( tmplb ),
                prob.N * sizeof( doublereal ) )
        tmpub = arrwrap.convFortran( prob.ub )
        memcpy( &self.bu[0], arrwrap.getPtr( tmpub ),
                prob.N * sizeof( doublereal ) )
        if( prob.Nconslin > 0 ):
            tmpconslinlb = arrwrap.convFortran( prob.conslinlb )
            memcpy( &self.bl[prob.N], arrwrap.getPtr( tmpconslinlb ),
                    prob.Nconslin * sizeof( doublereal ) )
            tmpconslinub = arrwrap.convFortran( prob.conslinub )
            memcpy( &self.bu[prob.N], arrwrap.getPtr( tmpconslinub ),
                    prob.Nconslin * sizeof( doublereal ) )
            tmpconslinA = arrwrap.convFortran( prob.conslinA )
            memcpy( self.A, arrwrap.getPtr( tmpconslinA ),
                    self.ldA[0] * prob.N * sizeof( doublereal ) )
        if( prob.Ncons > 0 ):
            tmpconslb = arrwrap.convFortran( prob.conslb )
            memcpy( &self.bl[prob.N+prob.Nconslin], arrwrap.getPtr( tmpconslb ),
                    prob.Ncons * sizeof( doublereal ) )
            tmpconsub = arrwrap.convFortran( prob.consub )
            memcpy( &self.bu[prob.N+prob.Nconslin], arrwrap.getPtr( tmpconsub ),
                    prob.Ncons * sizeof( doublereal ) )


    def warmStart( self, istate, clamda, R ):
        tmpistate = arrwrap.convIntFortran( istate )
        memcpy( self.istate, arrwrap.getPtr( tmpistate ), self.nctotl * sizeof( integer ) )

        tmpclamda = arrwrap.convFortran( clamda )
        memcpy( self.clamda, arrwrap.getPtr( tmpclamda ), self.nctotl * sizeof( doublereal ) )

        tmpR = arrwrap.convFortran( R )
        memcpy( self.R, arrwrap.getPtr( tmpR ), prob.N * prob.N * sizeof( doublereal ) )

        self.warm_start = True


    def __dealloc__( self ):
        mem.PyMem_Free( self.x )
        free( self.bl )
        free( self.bu )
        free( self.objg_val )
        free( self.consf_val )
        free( self.consg_val )
        free( self.clamda )
        free( self.istate )
        free( self.iw )
        free( self.w )
        free( self.A )
        free( self.R )


    def getStatus( self ):
        if( self.inform_out[0] == 100 ):
            return "Return information is not defined yet"
        elif( self.inform_out[0] < 0 ):
            return "Execution terminated by user defined function (should not occur)"
        elif( self.inform_out[0] >= 10 ):
            return "Invalid value"
        else:
            return statusInfo[ self.inform_out[0] ]


    def checkPrintOpts( self ):
        """
        Check if dictionary self.printOpts is valid.

        Optional entries:
        printFile        filename for debug information (default: None)
        printLevel       verbosity level for major iterations (0-None, 1, 5, 10, 20, or 30-Full)
        minorPrintLevel  verbosity level for minor iterations (0-None, 1, 5, 10, 20, or 30-Full)
        """
        if( super().checkPrintOpts() == False ):
            return False

        ## printLevel and minorPrintLevel
        if( not arrwrap.isInt( self.printOpts[ "printLevel" ] ) or
            not arrwrap.isInt( self.printOpts[ "minorPrintLevel" ] ) ):
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
        if( super().checkSolveOpts() == False ):
            return False

        ## infValue
        if( not arrwrap.isFloat( self.solveOpts[ "infValue" ] ) ):
            print( "solveOpts['infValue'] must be float" )
            return False
        if( self.solveOpts[ "infValue" ] < 0 ):
            print( "solveOpts['infValue'] must be positive" )
            return False
        elif( self.solveOpts[ "infValue" ] > 1e20 ):
            print( "Values for solveOpts['infValue'] above 1e20 are ignored" )
            return False

        ## lineSearchTol
        if( not arrwrap.isFloat( self.solveOpts[ "lineSearchTol" ] ) ):
            print( "solveOpts['lineSearchTol'] must be float" )
            return False
        if( self.solveOpts[ "lineSearchTol" ] < 0 or
            self.solveOpts[ "lineSearchTol" ] >= 1 ):
            print( "solveOpts['lineSearchTol'] must belong to the interval [0,1)" )
            return False

        ## iterLimit
        if( not arrwrap.isInt( self.solveOpts[ "iterLimit" ] ) ):
            print( "solveOpts['iterLimit'] must be integer" )
            return False
        if( self.solveOpts[ "iterLimit" ] < self.default_iter_limit ):
            print( "Values for solveOpts['iterLimit'] below "
                   + str( self.default_iter_limit ) + " are ignored" )

        ## minorIterLimit
        if( not arrwrap.isInt( self.solveOpts[ "minorIterLimit" ] ) ):
            print( "solveOpts['minorIterLimit'] must be integer" )
            return False
        if( self.solveOpts[ "minorIterLimit" ] < self.default_iter_limit ):
            print( "Values for solveOpts['minorIterLimit'] below "
                   + str( self.default_iter_limit ) + " are ignored" )

        ## fctnPrecision
        if( not arrwrap.isFloat( self.solveOpts[ "fctnPrecision" ] ) ):
            print( "solveOpts['fctnPrecision'] must be float" )
            return False
        if( self.solveOpts[ "fctnPrecision" ] < self.default_fctn_prec ):
            print( "Values for solveOpts['feasiblityTol'] below "
                   + str( self.default_fctn_prec ) + " are ignored" )

        ## feasibilityTol
        if( not arrwrap.isFloat( self.solveOpts[ "feasibilityTol" ] ) ):
            print( "solveOpts['feasibilityTol'] must be float" )
            return False
        if( self.solveOpts[ "feasibilityTol" ] < self.default_tol ):
            print( "Values for solveOpts['feasiblityTol'] below "
                   + str( self.default_tol ) + " are ignored" )

        ## optimalityTol
        if( not arrwrap.isFloat( self.solveOpts[ "optimalityTol" ] ) ):
            print( "solveOpts['optimalityTol'] must be float" )
            return False
        if( self.solveOpts[ "optimalityTol" ] < np.power( self.default_fctn_prec, 0.8 ) ):
            print( "Values for solveOpts['feasiblityTol'] below "
                   + str( np.power( self.default_fctn_prec, 0.8 ) ) + " are ignored" )

        return True


    def solve( self ):
        cdef integer inform[1]
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

        ## Supress echo options
        npsol.npoptn_( STR_NOLIST, len( STR_NOLIST ) )

        ## Open file if necessary
        if( self.printOpts[ "printFile" ] != "" and
            self.printOpts[ "printLevel" ] > 0 ):
            npsol.npopenappend_( printFileUnit, printFile, inform,
                                 len( self.printOpts[ "printFile" ] ) )
            if( inform[0] != 0 ):
                raise StandardError( "Could not open file " + self.printOpts[ "printFile" ] )
            npsol.npopti_( STR_PRINT_FILE, printFileUnit, len( STR_PRINT_FILE ) )

        if( self.printOpts[ "summaryFile" ] != "" ):
            npsol.npopenappend_( summaryFileUnit, summaryFile, inform,
                                 len( self.printOpts[ "summaryFile" ] ) )
            if( inform[0] != 0 ):
                raise StandardError( "Could not open file " + self.printOpts[ "summaryFile" ] )
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
        if( self.solveOpts["fctnPrecision"] > self.default_fctn_prec ):
            npsol.npoptr_( STR_FUNCTION_PRECISION, fctnPrecision,
                           len( STR_FUNCTION_PRECISION ) )
        if( self.solveOpts["feasibilityTol"] > self.default_tol ):
            npsol.npoptr_( STR_FEASIBILITY_TOLERANCE, feasiblityTol,
                           len( STR_FEASIBILITY_TOLERANCE ) )
        if( self.solveOpts["optimalityTol"] > np.power( self.default_fctn_prec, 0.8 ) ):
            npsol.npoptr_( STR_OPTIMALITY_TOLERANCE, optimalityTol,
                           len( STR_OPTIMALITY_TOLERANCE ) )

        ## Set verify level if required, pg. 29
        if( self.solveOpts["verifyGrad"] ):
            verifyLevel[0] = 3 ## Check both obj and cons
            print( "Verifying!" )
        else:
            verifyLevel[0] = -1 ## Disabled
        npsol.npopti_( STR_VERIFY_LEVEL, verifyLevel, len( STR_VERIFY_LEVEL ) )

        if( self.warm_start ):
            npsol.npoptn_( STR_WARM_START, len( STR_WARM_START ) )

        npsol.npsol_( self.n, self.nclin,
                      self.ncnln, self.ldA,
                      self.ldJ, self.ldR,
                      self.A, self.bl, self.bu,
                      <npsol.funcon_fp> funcon, <npsol.funobj_fp> funobj,
                      self.inform_out, self.iter_out, self.istate,
                      self.consf_val, self.consg_val, self.clamda,
                      self.objf_val, self.objg_val, self.R, self.x,
                      self.iw, self.leniw, self.w, self.lenw )

        self.prob.final = np.copy( arrwrap.wrap1dPtr( self.x, self.prob.N, np.NPY_DOUBLE ) )
        self.prob.value = float( self.objf_val[0] )
        self.prob.istate = np.copy( arrwrap.wrap1dPtr( self.istate, self.nctotl, np.NPY_LONG ) )
        self.prob.clamda = np.copy( arrwrap.wrap1dPtr( self.clamda, self.nctotl, np.NPY_DOUBLE ) )
        self.prob.R = np.copy( arrwrap.wrap2dPtr( self.R, self.prob.N, self.prob.N, np.NPY_DOUBLE ) )
        self.prob.Niters = int( self.iter_out[0] )
        self.prob.retval = int( self.inform_out[0] )

        return( self.prob.final, self.prob.value, self.prob.retval )
