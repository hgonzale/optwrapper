# cython: boundscheck=False
# cython: wraparound=False

from libc.string cimport memcpy, memset
from libc.stdlib cimport malloc, free
from libc.stdint cimport int32_t
cimport numpy as cnp
import numpy as np
import os

from ipstdcinterfaceh import Number, Index, Int, Bool
cimport ipstdcinterfaceh as ipopt ## import functions exposed in IpStdCInterface.h
cimport utils
cimport base
import nlp

## Match numpy's datatypes to those of the architecture
cdef int Number_type = cnp.NPY_FLOAT64
cdef type Number_dtype = np.float64
cdef int Index_type = cnp.NPY_INT64
cdef type Index_dtype = np.int64
if( sizeof( Index ) == 4 ):
    Index_type = cnp.NPY_INT32
    Index_dtype = np.int32

## helper static function usrfun that evaluate user-defined functions in Solver.prob
cdef object extprob
cdef utils.sMatrix consGsparse


## These evaluation functions are detailed under "C++ Interface" at:
## http://www.coin-or.org/Ipopt/documentation/
cdef Bool eval_objf( Index n, Number* x, Bool new_x,
                     Number* obj_value, UserDataPtr user_data ):
    xarr = utils.wrap1dPtr( x, n, Number_type )
    obj_value[0] = 0.0
    farr = utils.wrap1dPtr( obj_value, 1, Number_type )

    extprob.objf( farr, xarr )
    if( extprob.objmixedA is not None ):
        farr += extprob.objmixedA.dot( xarr )

    return True

cdef Bool eval_objg( Index n, Number* x, Bool new_x,
                     Number* grad_f, UserDataPtr user_data ):
    xarr = utils.wrap1dPtr( x, n, Number_type )
    memset( grad_f, 0, n * sizeof( Number ) )
    garr = utils.wrap1dPtr( grad_f, n, Number_type )

    extprob.objg( garr, xarr )
    if( extprob.objmixedA is not None ):
        garr += extprob.objmixedA

    return True

cdef Bool eval_consf( Index n, Number* x, Bool new_x,
                      Index m, Number* g, UserDataPtr user_data ):
    xarr = utils.wrap1dPtr( x, n, Number_type )
    memset( g, 0, m * sizeof( Number ) )
    garr = utils.wrap1dPtr( g, m, Number_type )

    extprob.consf( garr, xarr )
    if( extprob.consmixedA is not None ):
        garr += extprob.consmixedA.dot( xarr )

    return True

cdef Bool eval_consg( Index n, Number *x, Bool new_x,
                      Index m, Index nele_jac,
                      Index *iRow, Index *jCol, Number *values,
                      UserDataPtr user_data ):
    if( values == NULL ):
        if( sizeof( Index ) == 4 ):
            consGsparse.copyIdxs32( <int32_t *> self.iRow,
                                    <int32_t *> self.jCol )
        else:
            consGsparse.copyIdxs( <int64_t *> self.iRow,
                                  <int64_t *> self.jCol )
    else:
        xarr = utils.wrap1dPtr( x, n, Number_type )
        memset( values, 0, nele_jac * sizeof( Number ) )
        consGsparse.setDataPtr( values )

        extprob.consg( consGsparse, xarr )
        if( extprob.consmixedA is not None ):
            consGsparse += extprob.consmixedA

    return True


cdef Bool eval_lagrangianh( Index n, const Number* x, Bool new_x,
                            Number obj_factor, Index m, const Number* lambda_,
                            Bool new_lambda, Index nele_hess, Index* iRow,
                            Index* jCol, Number* values ):
    return False


cdef class Soln( base.Soln ):
    cdef public cnp.ndarray mult_x_L
    cdef public cnp.ndarray mult_x_U
    cdef public cnp.ndarray mult_g

    def __init__( self ):
        super().__init__()
        self.retval = -666

    def getStatus( self ):
        cdef dict statusInfo = { 0: "Solve Succeeded",
                                 1: "Solved To Acceptable Level",
                                 2: "Infeasible Problem Detected",
                                 3: "Search Direction Becomes Too Small",
                                 4: "Diverging Iterates",
                                 5: "User Requested Stop",
                                 6: "Feasible Point Found",
                                 -1: "Maximum Iterations Exceeded",
                                 -2: "Restoration Failed",
                                 -3: "Error In Step Computation",
                                 -4: "Maximum CpuTime Exceeded",
                                 -10: "Not Enough Degrees Of Freedom",
                                 -11: "Invalid Problem Definition",
                                 -12: "Invalid Option",
                                 -13: "Invalid Number Detected",
                                 -100: "Unrecoverable Exception",
                                 -101: "NonIpopt Exception Thrown",
                                 -102: "Insufficient Memory",
                                 -199: "Internal Error" }

        if( self.retval == -666 ):
            return "Return information is not defined yet"

        if( self.retval not in statusInfo ):
            return "Undefined return information"

        return statusInfo[ self.retval ]


cdef class Solver( base.Solver ):
    cdef Number *x
    cdef Number *x_L
    cdef Number *x_U
    cdef Number *mult_x_L
    cdef Number *mult_x_U
    cdef Number *g
    cdef Number *g_L
    cdef Number *g_U
    cdef Number *mult_g
    cdef Number obj
    cdef ipopt.IpoptProblem nlp
    cdef ipopt.ApplicationReturnStatus status
    cdef Index Ntotcons

    cdef int warm_start
    cdef int mem_alloc
    cdef int mem_size[2] ## { N, Ntotcons }


    def __init__( self, prob=None ):
        super().__init__()

        self.mem_alloc = False
        self.mem_size[0] = self.mem_size[1] = 0
        self.prob = None

        if( prob ):
            self.setupProblem( prob )

        ######################################
        ## Set print options
        self.printOpts[ "summaryFile" ] = None
        self.printOpts[ "printLevel" ] = None
        self.printOpts[ "minorPrintLevel" ] = None
        ## Set solve options
        self.solveOpts[ "centralDiffInterval" ] = None
        self.solveOpts[ "crashTol" ] = None
        self.solveOpts[ "diffInterval" ] = None
        self.solveOpts[ "feasibilityTol" ] = None
        self.solveOpts[ "fctnPrecision" ] = None
        self.solveOpts[ "infBoundSize" ] = None
        self.solveOpts[ "infStepSize" ] = None
        self.solveOpts[ "iterLimit" ] = None
        self.solveOpts[ "linesearchTol" ] = None
        self.solveOpts[ "minorIterLimit" ] = None
        self.solveOpts[ "optimalityTol" ] = None
        self.solveOpts[ "stepLimit" ] = None
        self.solveOpts[ "verifyLevel" ] = None


    def setupProblem( self, prob ):
        global extprob
        global consGsparse
        cdef cnp.ndarray tmparr

        if( not isinstance( prob, nlp.Problem ) ):
            raise TypeError( "Argument 'prob' must be of type 'nlp.Problem'" )

        self.prob = prob ## Save a copy of prob's pointer
        extprob = prob ## Save a global copy prob's pointer for funcon and funobj

        ## New problems cannot be warm started
        self.warm_start = False

        ## Set size-dependent constants
        self.N = prob.N
        self.Ntotcons = prob.Ncons + prob.Nconslin

        ## Allocate if necessary
        if( not self.mem_alloc ):
            self.allocate()
        elif( self.mem_size[0] < self.N or
              self.mem_size[1] < self.Ntotcons ):
            self.deallocate()
            self.allocate()

        ## Copy information from prob to NPSOL's working arrays
        tmparr = utils.arraySanitize( prob.lb, dtype=Number_dtype )
        memcpy( self.x_L, utils.getPtr( tmparr ), self.N * sizeof( Number ) )

        tmparr = utils.arraySanitize( prob.ub, dtype=Number_dtype )
        memcpy( self.x_U, utils.getPtr( tmparr ), self.N * sizeof( Number ) )

        if( prob.Nconslin > 0 ):
            tmparr = utils.arraySanitize( prob.conslinlb, dtype=Number_dtype )
            memcpy( &self.g_L[0], utils.getPtr( tmparr ),
                    prob.Nconslin * sizeof( Number ) )

            tmparr = utils.arraySanitize( prob.conslinub, dtype=Number_dtype )
            memcpy( &self.g_U[0], utils.getPtr( tmparr ),
                    prob.Nconslin * sizeof( Number ) )

        if( prob.Ncons > 0 ):
            tmparr = utils.arraySanitize( prob.conslb, dtype=Number_dtype )
            memcpy( &self.g_L[prob.Nconslin], utils.getPtr( tmparr ),
                    prob.Ncons * sizeof( doublereal ) )

            tmparr = utils.arraySanitize( prob.consub, dtype=Number_dtype )
            memcpy( &self.g_U[prob.Nconslin], utils.getPtr( tmparr ),
                    prob.Ncons * sizeof( doublereal ) )

        if( isinstance( prob, nlp.SparseProblem ) and prob.consgpattern is not None ):
            ###
        
        self.nlp = ipopt.CreateIpoptProblem( self.N, self.x_L, self.x_U, self.Ntotcons,
                                             self.g_L, self.g_U,


    cdef allocate( self ):
        if( self.mem_alloc ):
            return False

        self.x = <Number *> malloc( self.N * sizeof( Number ) )
        self.x_L = <Number *> malloc( self.N * sizeof( Number ) )
        self.x_U = <Number *> malloc( self.N * sizeof( Number ) )
        self.mult_x_L = <Number *> malloc( self.N * sizeof( Number ) )
        self.mult_x_U = <Number *> malloc( self.N * sizeof( Number ) )
        self.g = <Number *> malloc( self.Ntotcons * sizeof( Number ) )
        self.g_L = <Number *> malloc( self.Ntotcons * sizeof( Number ) )
        self.g_U = <Number *> malloc( self.Ntotcons * sizeof( Number ) )
        self.mult_g = <Number *> malloc( self.Ntotcons * sizeof( Number ) )

        if( self.x == NULL or
            self.x_L == NULL or
            self.x_U == NULL or
            self.mult_x_L == NULL or
            self.mult_x_U == NULL or
            self.g == NULL or
            self.g_L == NULL or
            self.g_U == NULL or
            self.mult_g == NULL ):
            raise MemoryError( "At least one memory allocation failed" )

        self.mem_alloc = True
        self.mem_size[0] = self.N
        self.mem_size[1] = self.Ntotcons
        return True


    cdef deallocate( self ):
        if( not self.mem_alloc ):
            return False

        free( self.x )
        free( self.x_L )
        free( self.x_U )
        free( self.mult_x_L )
        free( self.mult_x_U )
        free( self.g )
        free( self.g_L )
        free( self.g_U )
        free( self.mult_g )

        self.mem_alloc = False
        return True


    def __dealloc__( self ):
        self.deallocate()


    def warmStart( self ):
        if( not isinstance( self.prob.soln, Soln ) ):
            return False

        memcpy( self.istate,
                utils.getPtr( utils.convIntFortran( self.prob.soln.istate ) ),
                self.nctotl * sizeof( integer ) )
        memcpy( self.clamda,
                utils.getPtr( utils.convFortran( self.prob.soln.clamda ) ),
                self.nctotl * sizeof( doublereal ) )
        memcpy( self.R,
                utils.getPtr( utils.convFortran( self.prob.soln.R ) ),
                self.prob.N * self.prob.N * sizeof( doublereal ) )

        self.warm_start = True
        return True


    def solve( self ):
        ## Option strings
        cdef char* STR_NOLIST = "Nolist"
        cdef char* STR_DEFAULTS = "Defaults"
        cdef char* STR_WARM_START = "Warm Start"
        cdef char* STR_CENTRAL_DIFFERENCE_INTERVAL = "Central Difference Interval"
        cdef char* STR_CRASH_TOLERANCE = "Crash Tolerance"
        cdef char* STR_DIFFERENCE_INTERVAL = "Difference Interval"
        cdef char* STR_FEASIBILITY_TOLERANCE = "Feasibility Tolerance"
        cdef char* STR_FUNCTION_PRECISION = "Function Precision"
        cdef char* STR_HESSIAN_YES = "Hessian Yes"
        cdef char* STR_INFINITE_BOUND_SIZE = "Infinite Bound Size"
        cdef char* STR_INFINITE_STEP_SIZE = "Infinite Step Size"
        cdef char* STR_ITERATION_LIMIT = "Iteration Limit"
        cdef char* STR_LINE_SEARCH_TOLERANCE = "Line Search Tolerance"
        cdef char* STR_PRINT_LEVEL = "Print Level"
        cdef char* STR_MINOR_ITERATION_LIMIT = "Minor Iteration Limit"
        cdef char* STR_MINOR_PRINT_LEVEL = "Minor Print Level"
        cdef char* STR_OPTIMALITY_TOLERANCE = "Optimality Tolerance"
        cdef char* STR_PRINT_FILE = "Print File"
        cdef char* STR_STEP_LIMIT = "Step Limit"
        cdef char* STR_SUMMARY_FILE = "Summary File"
        cdef char* STR_VERIFY_LEVEL = "Verify Level"

        cdef bytes printFileTmp = self.printOpts[ "printFile" ].encode( "latin_1" )
        cdef char* printFile = printFileTmp
        cdef bytes summaryFileTmp = self.printOpts[ "summaryFile" ].encode( "latin_1" )
        cdef char* summaryFile = summaryFileTmp
        cdef integer summaryFileUnit[1]
        cdef integer printFileUnit[1]
        cdef doublereal centralDiffInterval[1]
        cdef doublereal crashTol[1]
        cdef doublereal diffInterval[1]
        cdef doublereal feasibilityTol[1]
        cdef doublereal fctnPrecision[1]
        cdef doublereal infBoundSize[1]
        cdef doublereal infStepSize[1]
        cdef integer iterLimit[1]
        cdef doublereal linesearchTol[1]
        cdef integer printLevel[1]
        cdef integer minorIterLimit[1]
        cdef integer minorPrintLevel[1]
        cdef doublereal optimalityTol[1]
        cdef doublereal stepLimit[1]
        cdef integer verifyLevel[1]

        cdef integer *n = [ self.prob.N ]
        cdef integer *nclin = [ self.prob.Nconslin ]
        cdef integer *ncnln = [ self.prob.Ncons ]
        cdef integer iter_out[1]
        cdef integer inform_out[1]
        cdef doublereal objf_val[1]

        ## Begin by setting up initial condition
        memcpy( self.x, utils.getPtr( utils.convFortran( self.prob.init ) ),
                self.prob.N * sizeof( doublereal ) )

        ## Supress echo options and reset optional values, pg. 21
        npsol.npoptn_( STR_NOLIST, len( STR_NOLIST ) )
        npsol.npoptn_( STR_DEFAULTS, len( STR_DEFAULTS ) )

        ## Handle debug files
        if( self.printOpts[ "printFile" ] is not None and
            self.printOpts[ "printFile" ] != "" ):
            printFileUnit[0] = 90 ## Hardcoded since nobody cares
        else:
            printFileUnit[0] = 0 ## disabled by default, pg. 27

        if( self.printOpts[ "summaryFile" ] is not None and
              self.printOpts[ "summaryFile" ] != "" ):
            if( self.printOpts[ "summaryFile" ].lower() == "stdout" ):
                summaryFileUnit[0] = 6 ## Fortran's magic value for stdout
            else:
                summaryFileUnit[0] = 89 ## Hardcoded since nobody cares
        else:
            summaryFileUnit[0] = 0 ## disabled by default, pg. 28

        npsol.npopti_( STR_PRINT_FILE, printFileUnit, len( STR_PRINT_FILE ) )
        npsol.npopti_( STR_SUMMARY_FILE, summaryFileUnit, len( STR_SUMMARY_FILE ) )

        ## Set optional parameters, pg. 22
        if( self.solveOpts[ "centralDiffInterval" ] is not None ):
            centralDiffInterval[0] = self.solveOpts[ "centralDiffInterval" ]
            npsol.npoptr_( STR_CENTRAL_DIFFERENCE_INTERVAL, centralDiffInterval,
                           len( STR_CENTRAL_DIFFERENCE_INTERVAL ) )

        if( self.warm_start ):
            npsol.npoptn_( STR_WARM_START, len( STR_WARM_START ) )
            ## follow recommendation in pg. 24
            npsol.npoptn_( STR_HESSIAN_YES, len( STR_HESSIAN_YES ) )
            self.warm_start = False ## Reset variable

        if( self.solveOpts[ "crashTol" ] is not None ):
            crashTol[0] = self.solveOpts[ "crashTol" ]
            npsol.npoptr_( STR_CRASH_TOLERANCE, crashTol, len( STR_CRASH_TOLERANCE ) )

        if( self.solveOpts[ "diffInterval" ] is not None ):
            diffInterval[0] = self.solveOpts[ "diffInterval" ]
            npsol.npoptr_( STR_DIFFERENCE_INTERVAL, diffInterval,
                           len( STR_DIFFERENCE_INTERVAL ) )

        if( self.solveOpts[ "feasibilityTol" ] is not None ):
            feasibilityTol[0] = self.solveOpts[ "feasibilityTol" ]
            npsol.npoptr_( STR_FEASIBILITY_TOLERANCE, feasibilityTol,
                           len( STR_FEASIBILITY_TOLERANCE ) )

        if( self.solveOpts[ "fctnPrecision" ] is not None ):
            fctnPrecision[0] = self.solveOpts[ "fctnPrecision" ]
            npsol.npoptr_( STR_FUNCTION_PRECISION, fctnPrecision,
                           len( STR_FUNCTION_PRECISION ) )

        if( self.solveOpts[ "infBoundSize" ] is not None ):
            infBoundSize[0] = self.solveOpts[ "infBoundSize" ]
            npsol.npoptr_( STR_INFINITE_BOUND_SIZE, infBoundSize, len( STR_INFINITE_BOUND_SIZE ) )

        if( self.solveOpts[ "infStepSize" ] is not None ):
            infStepSize[0] = self.solveOpts[ "infStepSize" ]
            npsol.npoptr_( STR_INFINITE_STEP_SIZE, infStepSize, len( STR_INFINITE_STEP_SIZE ) )

        if( self.solveOpts[ "iterLimit" ] is not None ):
            iterLimit[0] = self.solveOpts["iterLimit"]
            npsol.npopti_( STR_ITERATION_LIMIT, iterLimit, len( STR_ITERATION_LIMIT ) )

        if( self.printOpts[ "printLevel" ] is not None ):
            printLevel[0] = self.printOpts[ "printLevel" ]
            npsol.npopti_( STR_PRINT_LEVEL, printLevel, len( STR_PRINT_LEVEL ) )

        if( self.solveOpts[ "linesearchTol" ] is not None ):
            linesearchTol[0] = self.solveOpts[ "linesearchTol" ]
            npsol.npoptr_( STR_LINE_SEARCH_TOLERANCE, linesearchTol,
                           len( STR_LINE_SEARCH_TOLERANCE ) )

        if( self.solveOpts[ "minorIterLimit" ] is not None ):
            minorIterLimit[0] = self.solveOpts[ "minorIterLimit" ]
            npsol.npopti_( STR_MINOR_ITERATION_LIMIT, minorIterLimit,
                           len( STR_MINOR_ITERATION_LIMIT ) )

        if( self.printOpts[ "minorPrintLevel" ] is not None ):
            minorPrintLevel[0] = self.printOpts[ "minorPrintLevel" ]
            npsol.npopti_( STR_MINOR_PRINT_LEVEL, minorPrintLevel, len( STR_MINOR_PRINT_LEVEL ) )

        if( self.solveOpts[ "optimalityTol" ] is not None ):
            optimalityTol[0] = self.solveOpts[ "optimalityTol" ]
            npsol.npoptr_( STR_OPTIMALITY_TOLERANCE, optimalityTol,
                           len( STR_OPTIMALITY_TOLERANCE ) )

        if( self.solveOpts[ "stepLimit" ] is not None ):
            stepLimit[0] = self.solveOpts["stepLimit"]
            npsol.npoptr_( STR_STEP_LIMIT, stepLimit, len( STR_STEP_LIMIT ) )

        if( self.solveOpts[ "verifyLevel" ] is not None ):
            verifyLevel[0] = self.solveOpts[ "verifyLevel" ]
        else:
            verifyLevel[0] = -1 ## disabled by default, pg. 28
        npsol.npopti_( STR_VERIFY_LEVEL, verifyLevel, len( STR_VERIFY_LEVEL ) )

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

        ## Try to rename fortran print and summary files
        if( self.printOpts[ "printFile" ] is not None and
            self.printOpts[ "printFile" ] != "" ):
            try:
                os.rename( "fort.{0}".format( printFileUnit[0] ),
                           self.printOpts[ "printFile" ] )
            except:
                pass

        if( self.printOpts[ "summaryFile" ] is not None and
            self.printOpts[ "summaryFile" ] != "" and
            self.printOpts[ "summaryFile" ].lower() != "stdout" ):
            try:
                os.rename( "fort.{0}".format( summaryFileUnit[0] ),
                           self.printOpts[ "summaryFile" ] )
            except:
                pass

        ## Try to remove spurious print file fort.9 file left because NOLIST does not work
        try:
            os.remove( "fort.9" )
        except:
            pass

        ## Save result to prob
        self.prob.soln = Soln()
        self.prob.soln.value = float( objf_val[0] )
        self.prob.soln.final = np.copy( utils.wrap1dPtr( self.x, self.prob.N,
                                                         utils.doublereal_type ) )
        self.prob.soln.istate = np.copy( utils.wrap1dPtr( self.istate, self.nctotl,
                                                          utils.integer_type ) )
        self.prob.soln.clamda = np.copy( utils.wrap1dPtr( self.clamda, self.nctotl,
                                                          utils.doublereal_type ) )
        self.prob.soln.R = np.copy( utils.wrap2dPtr( self.R, self.prob.N, self.prob.N,
                                                     utils.doublereal_type ) )
        self.prob.soln.Niters = int( iter_out[0] )
        self.prob.soln.retval = int( inform_out[0] )

        return( self.prob.soln.final,
                self.prob.soln.value,
                self.prob.soln.retval )
