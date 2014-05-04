from libc.stdlib cimport calloc,free
from libc.string cimport memcpy
import numpy as np
cimport numpy as np

from optwF2c cimport *
cimport optwNpsol as npsol
cimport arrayWrapper as arrwrap
from optwSolver cimport *
from optwrapper import *

## NPSOL's option strings
cdef char* STR_NOLIST = "Nolist"
cdef char* STR_WARM_START = "Warm start"
cdef char* STR_PRINT_FILE = "Print file"
cdef char* STR_PRINT_LEVEL = "Print level"
cdef char* STR_MINOR_PRINT_LEVEL = "Minor print level"
cdef char* STR_INFINITE_BOUND_SIZE = "Infinite bound size"
cdef char* STR_ITERATION_LIMIT = "Iteration limit"
cdef char* STR_MINOR_ITERATIONS_LIMIT = "Minor iterations limit"

cdef char* STR_FEASIBILITY_TOLERANCE = "Feasibility tolerance"
cdef char* STR_OPTIMALITY_TOLERANCE = "Optimality tolerance"
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
    xarr = arrwrap.wrapPtr( x, extprob.N, np.NPY_DOUBLE )

    if( mode[0] != 1 ):
        f[0] = extprob.objf( xarr )

    if( mode[0] > 0 ):
        tmpg = extprob.objg( xarr )
        tmpg = np.require( np.atleast_2d( tmpg ), dtype=np.float64, requirements=['F', 'A'] )
        memcpy( g, <doublereal *> arrwrap.getPtr( tmpg ),
                extprob.N * sizeof( doublereal ) )


## pg. 18, Section 7.2
cdef int funcon( integer* mode, integer* ncnln,
                 integer* n, integer* ldJ, integer* needc,
                 doublereal* x, doublereal* c, doublereal* cJac,
                 integer* nstate ):
    xarr = arrwrap.wrapPtr( x, extprob.N, np.NPY_DOUBLE )

    if( mode[0] != 1 ):
        tmpf = extprob.consf( xarr )
        tmpf = np.require( np.atleast_2d( tmpf ), dtype=np.float64, requirements=['F', 'A'] )
        memcpy( c, <doublereal *> arrwrap.getPtr( tmpf ),
                extprob.Ncons * sizeof( doublereal ) )

    if( mode[0] > 0 ):
        tmpg = extprob.consg( xarr )
        tmpg = np.require( np.atleast_2d( tmpg ), dtype=np.float64, requirements=['F', 'A'] )
        memcpy( cJac, <doublereal *> arrwrap.getPtr( tmpg ),
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
    cdef integer def_iter_limit

    def __init__( self, prob ):
        super().__init__()

        if( not isinstance( prob, optwProblem ) ):
            raise StandardError( "Argument 'prob' must be of type 'optwProblem'" )
        self.prob = prob ## Save a copy of the pointer
        self.setupProblem( self.prob )


    def setupProblem( self, prob ):
        global extprob
        extprob = prob ## Save static point for funcon and funobj

        self.nctotl = prob.N + prob.Nconslin + prob.Ncons
        self.def_iter_limit = max( 50, 3*( prob.N + prob.Nconslin ) + 10*prob.Ncons )

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

        self.x = <doublereal *> calloc( prob.N, sizeof( doublereal ) )
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

        if( self.x is NULL or
            self.bl is NULL or
            self.bu is NULL or
            self.objg_val is NULL or
            self.consf_val is NULL or
            self.consg_val is NULL or
            self.clamda is NULL or
            self.istate is NULL or
            self.iw is NULL or
            self.w is NULL or
            self.A is NULL or
            self.R is NULL ):
            raise MemoryError( "At least one memory allocation failed" )

        ## Set options
        self.printOpts[ "printLevel" ] = 0
        self.printOpts[ "minorPrintLevel" ] = 0
        self.solveOpts[ "infValue" ] = 1e20
        self.solveOpts[ "iterLimit" ] = self.def_iter_limit
        self.solveOpts[ "minorIterLimit" ] = self.def_iter_limit
        self.solveOpts[ "lineSearchTol" ] = 0.9
        self.solveOpts[ "feasibilityTol" ] = None
        self.solveOpts[ "optimalityTol" ] = None

        ## We are assuming np.float64 equals doublereal from now on
        ## At least we need to be sure that doublereal is 8 bytes in this architecture
        assert( sizeof( doublereal ) == 8 )

        ## Require all arrays we are going to copy to be:
        ## two-dimensional, float64, fortran contiguous, and type aligned
        tmpinit = np.require( np.atleast_2d( prob.init ), dtype=np.float64, requirements=['F', 'A'] )
        tmplb = np.require( np.atleast_2d( prob.lb ), dtype=np.float64, requirements=['F', 'A'] )
        tmpub = np.require( np.atleast_2d( prob.ub ), dtype=np.float64, requirements=['F', 'A'] )
        memcpy( self.x, <doublereal *> arrwrap.getPtr( tmpinit ),
                prob.N * sizeof( doublereal ) )
        memcpy( &self.bl[0], <doublereal *> arrwrap.getPtr( tmplb ),
                prob.N * sizeof( doublereal ) )
        memcpy( &self.bu[0], <doublereal *> arrwrap.getPtr( tmpub ),
                prob.N * sizeof( doublereal ) )
        if( prob.Nconslin > 0 ):
            tmpconslinA = np.require( np.atleast_2d( prob.conslinA ), dtype=np.float64, requirements=['F', 'A'] )
            tmpconslinlb = np.require( np.atleast_2d( prob.conslinlb ), dtype=np.float64, requirements=['F', 'A'] )
            tmpconslinub = np.require( np.atleast_2d( prob.conslinub ), dtype=np.float64, requirements=['F', 'A'] )
            memcpy( &self.bl[prob.N], <doublereal *> arrwrap.getPtr( tmpconslinlb ),
                    prob.Nconslin * sizeof( doublereal ) )
            memcpy( &self.bu[prob.N], <doublereal *> arrwrap.getPtr( tmpconslinub ),
                    prob.Nconslin * sizeof( doublereal ) )
            memcpy( self.A, <doublereal *> arrwrap.getPtr( tmpconslinA ),
                    self.ldA[0] * prob.N * sizeof( doublereal ) )
        if( prob.Ncons > 0 ):
            tmpconslb = np.require( np.atleast_2d( prob.conslb ), dtype=np.float64, requirements=['F', 'A'] )
            tmpconsub = np.require( np.atleast_2d( prob.consub ), dtype=np.float64, requirements=['F', 'A'] )
            memcpy( &self.bl[prob.N+prob.Nconslin], <doublereal *> arrwrap.getPtr( tmpconslb ),
                    prob.Ncons * sizeof( doublereal ) )
            memcpy( &self.bu[prob.N+prob.Nconslin], <doublereal *> arrwrap.getPtr( tmpconsub ),
                    prob.Ncons * sizeof( doublereal ) )



    def __dealloc__( self ):
        free( self.x )
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

        try:
            int( self.printOpts[ "printLevel" ] ) + 1
            int( self.printOpts[ "minorPrintLevel" ] ) + 1
        except:
            print( "printOpts['printLevel'] and printOpts['minorPrintLevel'] must be integers" )
            return False

        if( self.printOpts[ "printFile" ] == None and
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
        try:
            float( self.solveOpts[ "infValue" ] ) + 1.1
            float( self.solveOpts[ "lineSearchTol" ] ) + 1.1
        except:
            print( "solveOpts['infValue'] and solveOpts['lineSearchTol'] must be floats" )
            return False

        if( self.solveOpts[ "infValue" ] < 0 ):
            print( "solveOpts['infValue'] must be positive" )
            return False
        elif( self.solveOpts[ "infValue" ] > 1e20 ):
            print( "Values for solveOpts['infValue'] above 1e20 are ignored" )

        if( self.solveOpts[ "lineSearchTol" ] < 0 or
            self.solveOpts[ "lineSearchTol" ] >= 1 ):
            print( "solveOpts['lineSearchTol'] must belong to the interval [0,1)" )
            return False

        try:
            int( self.solveOpts[ "iterLimit" ] ) + 1
            int( self.solveOpts[ "minorIterLimit" ] ) + 1
        except:
            print( "solveOpts['iterLimit'] and solveOpts['minorIterLimit'] must be integers" )
            return False

        if( self.solveOpts[ "iterLimit" ] < self.def_iter_limit ):
            print( "Values for solveOpts['iterLimit'] below "
                   + str( self.def_iter_limit ) + " are ignored" )
        if( self.solveOpts[ "minorIterLimit" ] < self.def_iter_limit ):
            print( "Values for solveOpts['minorIterLimit'] below "
                   + str( self.def_iter_limit ) + " are ignored" )

        return True


    def solve( self ):
        cdef integer inform[1]
        cdef bytes printFileTmp = self.printOpts[ "printFile" ].encode() ## temp container
        cdef char* printFile = printFileTmp
        cdef integer* printFileUnit = [ 90 ] ## Hardcoded since nobody cares
        cdef integer* printLevel = [ self.printOpts[ "printLevel" ] ]
        cdef integer* minorPrintLevel = [ self.printOpts[ "minorPrintLevel" ] ]
        cdef doublereal* infValue = [ self.solveOpts["infValue"] ]
        cdef integer* iterLimit = [ self.solveOpts[ "iterLimit" ] ]
        cdef integer* minorIterLimit = [ self.solveOpts[ "minorIterLimit" ] ]
        cdef doublereal* lineSearchTol = [ self.solveOpts["lineSearchTol"] ]

        ## Supress echo options
        npsol.npoptn_( STR_NOLIST, len( STR_NOLIST ) )

        ## Open file if necessary
        if( self.printOpts[ "printFile" ] != None and
            self.printOpts[ "printLevel" ] > 0 ):
            npsol.npopenappend_( printFileUnit, printFile, inform,
                                 len( self.printOpts[ "printFile" ] ) )
            if( inform[0] != 0 ):
                raise StandardError( "Could not open file " + self.printOpts[ "printFile" ] )
            npsol.npopti_( STR_PRINT_FILE, printFileUnit, len( STR_PRINT_FILE ) )

        ## Set major and minor print levels
        npsol.npopti_( STR_PRINT_LEVEL, printLevel, len( STR_PRINT_LEVEL ) )
        npsol.npopti_( STR_MINOR_PRINT_LEVEL, minorPrintLevel, len( STR_MINOR_PRINT_LEVEL ) )

        ## Set infinite bound value if necessary
        if( self.solveOpts["infValue"] < 1e20 ):
            npsol.npoptr_( STR_INFINITE_BOUND_SIZE, infValue, len( STR_INFINITE_BOUND_SIZE ) )

        ## Set major and minor iteration limits if necessary
        if( self.solveOpts["iterLimit"] > self.def_iter_limit ):
            npsol.npopti_( STR_ITERATION_LIMIT, iterLimit, len( STR_ITERATION_LIMIT ) )
        if( self.solveOpts["minorIterLimit"] > self.def_iter_limit ):
            npsol.npopti_( STR_MINOR_ITERATION_LIMIT, minorIterLimit,
                           len( STR_MINOR_ITERATION_LIMIT ) )

        ## Set line search tolerance value
        npsol.npoptr_( STR_LINE_SEARCH_TOLERANCE, lineSearchTol, len( STR_LINE_SEARCH_TOLERANCE ) )

        # if( self.warmstart[0] == 1 ):
        #     cnpsol.npopti_( cnpsol.STR_WARM_START, self.warmstart, len(cnpsol.STR_WARM_START) )
        # npsol.npoptr_( npsol.STR_FEASIBILITY_TOLERANCE, self.constraint_violation,
        #                len(cnpsol.STR_FEASIBILITY_TOLERANCE) )
        # npsol.npoptr_( npsol.STR_OPTIMALITY_TOLERANCE, self.ftol,
        #                len(npsol.STR_OPTIMALITY_TOLERANCE) )
        # npsol.npopti_( npsol.STR_MINOR_ITERATIONS_LIMIT, self.maxeval,
        #                len(npsol.STR_MINOR_ITERATIONS_LIMIT) )

        npsol.npsol_( self.n, self.nclin,
                      self.ncnln, self.ldA,
                      self.ldJ, self.ldR,
                      self.A, self.bl, self.bu,
                      <npsol.funcon_fp> funcon, <npsol.funobj_fp> funobj,
                      self.inform_out, self.iter_out, self.istate,
                      self.consf_val, self.consg_val, self.clamda,
                      self.objf_val, self.objg_val, self.R, self.x,
                      self.iw, self.leniw, self.w, self.lenw )

        self.prob.final = np.copy( arrwrap.wrapPtr( self.x, self.prob.N, np.NPY_DOUBLE ) )
        self.prob.value = float( self.objf_val[0] )
        self.prob.istate = np.copy( arrwrap.wrapPtr( self.istate, self.nctotl, np.NPY_LONG ) )
        self.prob.clamda = np.copy( arrwrap.wrapPtr( self.clamda, self.nctotl, np.NPY_DOUBLE ) )
        self.prob.R = np.copy( arrwrap.wrapPtr( self.R, self.prob.N * self.prob.N, np.NPY_DOUBLE ) )
        self.prob.Niters = int( self.iter_out[0] )
        self.prob.retval = int( self.inform_out[0] )

        return( self.prob.final, self.prob.value, self.prob.retval )



## Crappy code repository

## 1. Create bl, bu, A, and x vectors by memcpy'ing contents from optwProblem.

        # if( not prob.lb.flags["F_CONTIGUOUS"] or
        #     not prob.lb.dtype == np.float64 or
        #     not prob.ub.flags["F_CONTIGUOUS"] or
        #     not prob.ub.dtype == np.float64 ):
        #     raise MemoryError( "At least one box bound array is not fortran-contiguous or not float64." )
        # memcpy( &bl[0], &prob.lb[0], prob.N * sizeof( doublereal ) )
        # memcpy( &bu[0], &prob.ub[0], prob.N * sizeof( doublereal ) )

        # if( prob.Nconslin > 0 ):
        #     if( not prob.conslinlb.flags["F_CONTIGUOUS"] or
        #         not prob.conslinlb.dtype == np.float64 or
        #         not prob.conslinub.flags["F_CONTIGUOUS"] or
        #         not prob.conslinub.dtype == np.float64 or
        #         not prob.conslinA.flags["F_CONTIGUOUS"] or
        #         not prob.conslinA.dtype == np.float64 ):
        #         raise MemoryError( "At least one linear constraint array is not fortran-contiguous." )
        #     memcpy( &self.bl[prob.N], &prob.conslinlb[0], prob.Nconslin * sizeof( doublereal ) )
        #     memcpy( &self.bu[prob.N], &prob.conslinub[0], prob.Nconslin * sizeof( doublereal ) )
        #     memcpy( &self.A[0], &prob.conslinA[0], prob.Nconslin * prob.N * sizeof( doublereal ) )

        # if( prob.Ncons > 0 ):
        #     if( not prob.conslb.flags["F_CONTIGUOUS"] or
        #         not prob.conslb.dtype == np.float64 or
        #         not prob.consub.flags["F_CONTIGUOUS"] or
        #         not prob.consub.dtype == np.float64 ):
        #         raise MemoryError( "At least one constraint array is not fortran-contiguous." )
        #     memcpy( &self.bl[prob.N+prob.Nconslin], &prob.conslb[0], prob.Ncons * sizeof( doublereal ) )
        #     memcpy( &self.bu[prob.N+prob.Nconslin], &prob.consub[0], prob.Ncons * sizeof( doublereal ) )

        # if( not prob.init.flags["F_CONTIGUOUS"] or
        #     not prob.init.dtype == np.float64 ):
        #     raise MemoryError( "Initial condition array is not fortran-contiguous." )
        # memcpy( &self.x[0], &prob.init[0], prob.N * sizeof( doublereal ) )
