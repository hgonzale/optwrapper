# cython: boundscheck=False
# cython: wraparound=False

from __future__ import division
from libc.string cimport memcpy, memset
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libc.stdint cimport int64_t
cimport numpy as cnp
import numpy as np
import os

from .typedefs cimport *
cimport snopth as snopt
cimport utils
cimport base
import nlp

## we use these definitions to wrap C arrays in numpy arrays
## we can safely assume 64-bit values since we already checked using scons
cdef int doublereal_type = cnp.NPY_FLOAT64
cdef int integer_type = cnp.NPY_INT64
cdef type doublereal_dtype = np.float64
cdef type integer_dtype = np.int64


cdef class Soln( base.Soln ):
    cdef public cnp.ndarray xstate
    cdef public cnp.ndarray xmul
    cdef public cnp.ndarray Fstate
    cdef public cnp.ndarray Fmul
    cdef public int nS

    def __init__( self ):
        super().__init__()
        self.retval = 1000
        self.nS = 0

    def getStatus( self ):
        cdef dict statusInfo = { 0: "Finished successfully",
                                 1: "Optimality conditions satisfied",
                                 2: "Feasible point found",
                                 3: "Requested accuracy could not be achieved",
                                 10: "The problem appears to be infeasible",
                                 11: "Infeasible linear constraints",
                                 12: "Infeasible linear equalities",
                                 13: "Nonlinear infeasibilities minimized",
                                 14: "Infeasibilities minimized",
                                 20: "The problem appears to be unbounded",
                                 21: "Unbounded objective",
                                 22: "Constraint violation limit reached",
                                 30: "Resource limit error",
                                 31: "Iteration limit reached",
                                 32: "Major iteration limit reached",
                                 33: "The superbasics limit is too small",
                                 40: "Terminated after numerical difficulties",
                                 41: "Current point cannot be improved",
                                 42: "Singular basis",
                                 43: "Cannot satisfy the general constraints",
                                 44: "Ill-conditioned null-space basis",
                                 50: "Error in the user-supplied functions",
                                 51: "Incorrect objective derivatives",
                                 52: "Incorrect constraint derivatives",
                                 60: "Undefined user-supplied functions",
                                 61: "Undefined function at the first feasible point",
                                 62: "Undefined function at the initial point",
                                 63: "Unable to proceed into undefined region",
                                 70: "User requested termination",
                                 71: "Terminated during function evaluation",
                                 72: "Terminated during constraint evaluation",
                                 73: "Terminated during objective evaluation",
                                 74: "Terminated from monitor routine",
                                 80: "Insufficient storage allocated",
                                 81: "Work arrays must have at least 500 elements",
                                 82: "Not enough character storage",
                                 83: "Not enough integer storage",
                                 84: "Not enough real storage",
                                 90: "Input arguments out of range",
                                 91: "Invalud input argument",
                                 92: "Basis file dimensions do not match this problem",
                                 100: "SNOPT auxiliary routine finished successfully",
                                 110: "Errors while processing MPS data",
                                 120: "Errors while estimating Jacobian structure",
                                 130: "Errors while reading the Specs file",
                                 140: "System error",
                                 141: "Wrong number of basic variables",
                                 142: "Error in basis package",
                                 1000: "Return information undefined" }

        if( self.retval < 0 ):
            return "Execution terminated by user defined function (should not occur)"
        elif( self.retval not in statusInfo ):
            return "Invalid return value"
        else:
            return statusInfo[ self.retval ]


## helper static function usrfun that evaluate user-defined functions in Solver.prob
cdef object extprob
cdef utils.sMatrix objGsparse
cdef utils.sMatrix consGsparse

cdef int usrfun( integer *status, integer *n, doublereal *x,
                 integer *needF, integer *nF, doublereal *f,
                 integer *needG, integer *lenG, doublereal *G,
                 char *cu, integer *lencu,
                 integer *iu, integer *leniu,
                 doublereal *ru, integer *lenru ):
    ## FYI: Dual variables are at rw[ iw[329] ] onward, pg.25

    if( status[0] >= 2 ): ## Final call, do nothing
        return 0

    xarr = utils.wrap1dPtr( x, n[0], doublereal_type )

    if( needF[0] > 0 ):
        ## we zero out all arrays in case the user does not modify all the values,
        ## e.g., in sparse problems.
        memset( f, 0, nF[0] * sizeof( doublereal ) )
        farr = utils.wrap1dPtr( f, nF[0], doublereal_type )
        extprob.objf( farr[0:1], xarr )
        if( extprob.Ncons > 0 ):
            extprob.consf( farr[1+extprob.Nconslin:], xarr )

    if( needG[0] > 0 ):
        ## we zero out all arrays in case the user does not modify all the values,
        ## e.g., in sparse problems.
        memset( G, 0, lenG[0] * sizeof( doublereal ) )
        if( objGsparse.nnz > 0 ):
            objGsparse.setDataPtr( &G[0] )
            extprob.objg( objGsparse, xarr )
            # objGsparse.print_debug()
        if( extprob.Ncons > 0 and consGsparse.nnz > 0 ):
            consGsparse.setDataPtr( &G[objGsparse.nnz] )
            extprob.consg( consGsparse, xarr )
            # consGsparse.print_debug()
        # print( "G: [" ),
        # for k in range( lenG[0] ):
        #     printf( "%f ", G[k] )
        # print( "]" )


cdef class Solver( base.Solver ):
    cdef integer nF[1]
    cdef integer n[1]
    cdef integer lenA[1]
    cdef integer neA[1]
    cdef integer lenG[1]
    cdef integer neG[1]
    cdef doublereal *x
    cdef doublereal *xlow
    cdef doublereal *xupp
    cdef doublereal *xmul
    cdef doublereal *F
    cdef doublereal *Flow
    cdef doublereal *Fupp
    cdef doublereal *Fmul
    cdef doublereal *A
    cdef integer *iAfun
    cdef integer *jAvar
    cdef integer *iGfun
    cdef integer *jGvar
    cdef integer *xstate
    cdef integer *Fstate
    cdef char *cw
    cdef integer *iw
    cdef doublereal *rw
    cdef integer lencw[1]
    cdef integer leniw[1]
    cdef integer lenrw[1]
    cdef integer Start[1]
    cdef integer summaryFileUnit[1]
    cdef integer printFileUnit[1]
    cdef utils.sMatrix Asparse

    cdef int mem_alloc
    cdef integer mem_size[4]
    cdef int mem_alloc_ws
    cdef integer mem_size_ws[3]

    def __init__( self, prob=None ):
        cdef dict legacy

        super().__init__()

        self.mem_alloc = False
        self.mem_alloc_ws = False
        memset( self.mem_size, 0, 4 * sizeof( integer ) ) ## Set mem_size to zero
        memset( self.mem_size_ws, 0, 3 * sizeof( integer ) ) ## Set mem_size_ws to zero
        self.prob = None

        ## New problems cannot be warm started
        self.Start[0] = 0 ## Cold start
        self.printFileUnit[0] = 0
        self.summaryFileUnit[0] = 0

        ## legacy_label: real_label
        legacy = { "printLevel": "Major print level",
                   "minorPrintLevel": "Minor print level",
                   "printFreq": "Print frequency",
                   "solutionPrint": "Solution",
                   "systemInfo": "System information Yes",
                   "timingLevel": "Timing level",
                   "centralDiffInterval": "Central difference interval",
                   "checkFreq": "Check frequency",
                   "crashOpt": "Crash option",
                   "crashTol": "Crash tolerance",
                   "nonderivLinesearch": "Nonderivative linesearch",
                   "diffInterval": "Difference interval",
                   "elasticWeight": "Elastic weight",
                   "expandFreq": "Expand frequency",
                   "factorizationFreq": "Factorization frequency",
                   "feasiblePoint": "Feasible point",
                   "fctnPrecision": "Function precision",
                   "hessianFreq": "Hessian frequency",
                   "hessianUpdates": "Hessian updates",
                   "infBound": "Infinite bound",
                   "iterLimit": "Iterations limit",
                   "linesearchTol": "Linesearch tolerance",
                   "luFactorTol": "LU factor tolerance",
                   "luUpdateTol": "LU update tolerance",
                   "luDensityTol": "LU density tolerance",
                   "luSingularityTol": "LU singularity tolerance",
                   "majorFeasibilityTol": "Major feasibility tolerance",
                   "majorIterLimit": "Major iterations limit",
                   "majorOptimalityTol": "Major optimality tolerance",
                   "majorStepLimit": "Major step limit",
                   "minorIterLimit": "Minor iterations limit",
                   "minorFeasibilityTol": "Minor feasibility tolerance",
                   "newSuperbasicsLimit": "New superbasics limit",
                   "partialPrice": "Partial price",
                   "pivotTol": "Pivot tolerance",
                   "proximalPointMethod": "Proximal point method",
                   "reducedHessianDim": "Hessian dimension",
                   "Reduced hessian dimension": "Hessian dimension",
                   "scaleOption": "Scale option",
                   "scaleTol": "Scale tolerance",
                   "scalePrint": "Scale print",
                   "superbasicsLimit": "Superbasics limit",
                   "unboundedObjValue": "Unbounded objective value",
                   "unboundedStepSize": "Unbounded step size",
                   "verifyLevel": "Verify level",
                   "violationLimit": "Violation limit" }

        self.options = utils.Options( legacy )
        self.options[ "Verify level" ] = -1
        self.options[ "Solution" ] = "no"

        if( prob ):
            self.setupProblem( prob )


    def setupProblem( self, prob ):
        cdef cnp.ndarray tmparr

        global extprob
        global objGsparse
        global consGsparse

        if( not isinstance( prob, nlp.Problem ) ):
            raise TypeError( "Argument prob must be of type nlp.Problem" )

        if( not prob.checkSetup() ):
            raise ValueError( "Argument 'prob' has not been properly configured" )

        self.prob = prob ## Save a copy of prob's pointer
        extprob = prob ## Save another (global) copy of prob's pointer to use in usrfun

        ## Set n, nF
        self.n[0] = prob.N
        self.nF[0] = 1 + prob.Nconslin + prob.Ncons

        ## Create Asparse, set lenA
        if( prob.objmixedA is not None ):
            tmplist = ( prob.objmixedA, )
        else:
            tmplist = ( np.zeros( ( 1, self.n[0] ) ), )
        if( prob.Nconslin > 0 ):
            tmplist += ( prob.conslinA, )
        if( prob.consmixedA is not None ):
            tmplist += ( prob.consmixedA, )
        else:
            tmplist += ( np.zeros( ( prob.Ncons, self.n[0] ) ), )
        self.Asparse = utils.sMatrix( np.vstack( tmplist ), copy_data=True )
        # print( Asparse[:] )

        if( self.Asparse.nnz > 0 ):
            self.lenA[0] = self.Asparse.nnz
            self.neA[0] = self.lenA[0]
        else: ## Minimum allowed values, pg. 16
            self.lenA[0] = 1
            self.neA[0] = 0

        ## Create objGsparse and consGsparse, set lenG
        if( isinstance( prob, nlp.SparseProblem ) and prob.objgpattern is not None ):
            objGsparse = utils.sMatrix( prob.objgpattern )
        else:
            objGsparse = utils.sMatrix( np.ones( ( 1, self.n[0] ) ) )

        if( isinstance( prob, nlp.SparseProblem ) and prob.consgpattern is not None ):
            consGsparse = utils.sMatrix( prob.consgpattern )
        else:
            consGsparse = utils.sMatrix( np.ones( ( prob.Ncons, self.n[0] ) ) )
        if( objGsparse.nnz + consGsparse.nnz > 0 ):
            self.lenG[0] = objGsparse.nnz + consGsparse.nnz
            self.neG[0] = self.lenG[0]
        else:
            self.lenG[0] = 1
            self.neG[0] = 0

        ## Allocate if necessary
        if( self.mustAllocate( self.n[0], self.nF[0], self.lenA[0], self.lenG[0] ) ):
            self.deallocate()
            self.allocate()

        ## copy box constraints limits
        tmparr = utils.arraySanitize( prob.lb, dtype=doublereal_dtype, fortran=True )
        memcpy( self.xlow, utils.getPtr( tmparr ), self.n[0] * sizeof( doublereal ) )

        tmparr = utils.arraySanitize( prob.ub, dtype=doublereal_dtype, fortran=True )
        memcpy( self.xupp, utils.getPtr( tmparr ), self.n[0] * sizeof( doublereal ) )

        ## copy index data of G
        ## row 0 of G belongs to objg
        objGsparse.copyFortranIdxs( <int64_t *> &self.iGfun[0],
                                    <int64_t *> &self.jGvar[0] )
        ## rows 1:(1 + prob.Nconslin) of G are empty, these are pure linear constraints
        ## rows (1 + prob.Nconslin):(1 + prob.Nconslin + prob.Ncons) of G belong to consg
        consGsparse.copyFortranIdxs( <int64_t *> &self.iGfun[objGsparse.nnz],
                                     <int64_t *> &self.jGvar[objGsparse.nnz],
                                     roffset = 1 + prob.Nconslin )
        ## copy index data of A
        self.Asparse.copyFortranIdxs( <int64_t *> &self.iAfun[0],
                                      <int64_t *> &self.jAvar[0] )
        ## copy matrix data of A
        self.Asparse.copyData( &self.A[0] )

        ## copy general constraints limits
        ## objective function knows no limits (https://i.imgur.com/UuQbJ.gif)
        self.Flow[0] = -np.inf
        self.Fupp[0] = np.inf
        ## linear constraints limits
        if( prob.Nconslin > 0 ):
            tmparr = utils.arraySanitize( prob.conslinlb, dtype=doublereal_dtype, fortran=True )
            memcpy( &self.Flow[1], utils.getPtr( tmparr ),
                    prob.Nconslin * sizeof( doublereal ) )

            tmparr = utils.arraySanitize( prob.conslinub, dtype=doublereal_dtype, fortran=True )
            memcpy( &self.Fupp[1], utils.getPtr( tmparr ),
                    prob.Nconslin * sizeof( doublereal ) )
        ## nonlinear constraints limits
        if( prob.Ncons > 0 ):
            tmparr = utils.arraySanitize( prob.conslb, dtype=doublereal_dtype, fortran=True )
            memcpy( &self.Flow[1 + prob.Nconslin], utils.getPtr( tmparr ),
                    prob.Ncons * sizeof( doublereal ) )

            tmparr = utils.arraySanitize( prob.consub, dtype=doublereal_dtype, fortran=True )
            memcpy( &self.Fupp[1 + prob.Nconslin], utils.getPtr( tmparr ),
                    prob.Ncons * sizeof( doublereal ) )

        ## initialize other vectors with zeros
        memset( self.xstate, 0, self.n[0] * sizeof( integer ) )
        memset( self.Fstate, 0, self.nF[0] * sizeof( integer ) )
        memset( self.Fmul, 0, self.nF[0] * sizeof( doublereal ) )


    def initPoint( self, init ):
        if( not self.mem_alloc ):
            raise ValueError( "Internal memory has not been allocated" )

        tmparr = utils.arraySanitize( init, dtype=doublereal_dtype, fortran=True )
        memcpy( self.x, utils.getPtr( tmparr ),
                self.n[0] * sizeof( doublereal ) )

        return True


    cdef int allocate( self ):
        if( self.mem_alloc ):
            return False

        self.x = <doublereal *> malloc( self.n[0] * sizeof( doublereal ) )
        self.xlow = <doublereal *> malloc( self.n[0] * sizeof( doublereal ) )
        self.xupp = <doublereal *> malloc( self.n[0] * sizeof( doublereal ) )
        self.xmul = <doublereal *> malloc( self.n[0] * sizeof( doublereal ) )
        self.xstate = <integer *> malloc( self.n[0] * sizeof( integer ) )
        self.F = <doublereal *> malloc( self.nF[0] * sizeof( doublereal ) )
        self.Flow = <doublereal *> malloc( self.nF[0] * sizeof( doublereal ) )
        self.Fupp = <doublereal *> malloc( self.nF[0] * sizeof( doublereal ) )
        self.Fmul = <doublereal *> malloc( self.nF[0] * sizeof( doublereal ) )
        self.Fstate = <integer *> malloc( self.nF[0] * sizeof( integer ) )
        self.A = <doublereal *> malloc( self.lenA[0] * sizeof( doublereal ) )
        self.iAfun = <integer *> malloc( self.lenA[0] * sizeof( integer ) )
        self.jAvar = <integer *> malloc( self.lenA[0] * sizeof( integer ) )
        self.iGfun = <integer *> malloc( self.lenG[0] * sizeof( integer ) )
        self.jGvar = <integer *> malloc( self.lenG[0] * sizeof( integer ) )

        if( not ( self.x and
                  self.xlow and
                  self.xupp and
                  self.xmul and
                  self.xstate and
                  self.F and
                  self.Flow and
                  self.Fupp and
                  self.Fmul and
                  self.Fstate and
                  self.A and
                  self.iAfun and
                  self.jAvar and
                  self.iGfun and
                  self.jGvar ) ):
            raise MemoryError( "At least one memory allocation failed" )

        self.mem_alloc = True
        self.setMemSize( self.n[0], self.nF[0], self.lenA[0], self.lenG[0] )

        return True


    cdef int mustAllocate( self, integer N, integer nF, integer lenA, integer lenG ):
        if( not self.mem_alloc ):
            return True

        if( self.mem_size[0] < N or
            self.mem_size[1] < nF or
            self.mem_size[2] < lenA or
            self.mem_size[3] < lenG ):
            return True

        return False


    cdef void setMemSize( self, integer N, integer nF, integer lenA, integer lenG ):
        self.mem_size[0] = N
        self.mem_size[1] = nF
        self.mem_size[2] = lenA
        self.mem_size[3] = lenG


    cdef int deallocate( self ):
        if( not self.mem_alloc ):
            return False

        free( self.x )
        free( self.xlow )
        free( self.xupp )
        free( self.xstate )
        free( self.xmul )
        free( self.F )
        free( self.Flow )
        free( self.Fupp )
        free( self.Fstate )
        free( self.Fmul )
        free( self.A )
        free( self.iAfun )
        free( self.jAvar )
        free( self.iGfun )
        free( self.jGvar )

        self.mem_alloc = False
        self.setMemSize( 0, 0, 0, 0 )
        return True


    cdef int allocateWS( self ):
        if( self.mem_alloc_ws ):
            return False

        ## Allocate workspace memory
        self.cw = <char *> malloc( self.lencw[0] * 8 * sizeof( char ) )
        self.iw = <integer *> malloc( self.leniw[0] * sizeof( integer ) )
        self.rw = <doublereal *> malloc( self.lenrw[0] * sizeof( doublereal ) )

        if( not ( self.iw and
                  self.rw and
                  self.cw ) ):
            raise MemoryError( "At least one memory allocation failed" )

        self.mem_alloc_ws = True
        self.setMemSizeWS( self.lencw[0], self.leniw[0], self.lenrw[0] )
        return True


    cdef int mustAllocateWS( self, integer lencw, integer leniw, integer lenrw ):
        if( not self.mem_alloc_ws ):
            return True

        if( self.mem_size_ws[0] < lencw or
            self.mem_size_ws[1] < leniw or
            self.mem_size_ws[2] < lenrw ):
            return True

        return False


    cdef void setMemSizeWS( self, integer lencw, integer leniw, integer lenrw ):
        self.mem_size_ws[0] = lencw
        self.mem_size_ws[1] = leniw
        self.mem_size_ws[2] = lenrw


    cdef int deallocateWS( self ):
        if( not self.mem_alloc_ws ):
            return False

        free( self.cw )
        free( self.iw )
        free( self.rw )

        self.mem_alloc_ws = False
        self.setMemSizeWS( 0, 0, 0 )
        return True


    def __dealloc__( self ):
        self.deallocateWS()
        self.deallocate()


    def warmStart( self ):
        cdef cnp.ndarray tmparr

        if( not isinstance( self.prob.soln, Soln ) ):
            return False

        self.initPoint( self.prob.soln.final )

        tmparr = utils.arraySanitize( self.prob.soln.xstate, dtype=integer_dtype, fortran=True )
        memcpy( self.xstate, utils.getPtr( tmparr ),
                self.n[0] * sizeof( integer ) )

        tmparr = utils.arraySanitize( self.prob.soln.Fstate, dtype=integer_dtype, fortran=True )
        memcpy( self.Fstate, utils.getPtr( tmparr ),
                self.nF[0] * sizeof( integer ) )

        self.Start[0] = 2
        return True


    cdef int setOption( self, str option,
                        char* cw, integer* lencw,
                        integer* iw, integer* leniw,
                        doublereal* rw, integer* lenrw ):
        cdef integer* inform_out = [ 0 ]
        cdef bytes myopt

        if( len( option ) > 72 ):
            raise ValueError( "option string is too long: {0}".format( option ) )

        myopt = option.encode( "ascii" )
        snopt.snset_( myopt,
                      self.printFileUnit, self.summaryFileUnit, inform_out,
                      cw, lencw, iw, leniw, rw, lenrw,
                      len( myopt ), lencw[0]*8 )

        if( inform_out[0] != 0 ):
            raise ValueError( "Could not process option: {0}".format( option ) )

        return inform_out[0]


    cdef int processOptions( self ):
        cdef str mystr, key
        cdef int out = True

        for key in self.options:
            if( self.options[key].dtype == utils.NONE or
                ( self.options[key].dtype == utils.BOOL and not self.options[key].value ) ):
                continue

            mystr = key
            if( self.options[key].dtype != utils.BOOL ):
                mystr += " {0}".format( self.options[key].value )

            if( self.debug ):
                print( "processing option: '{0}'".format( mystr ) )

            out = out and ( self.setOption( mystr,
                                            self.cw, self.lencw,
                                            self.iw, self.leniw,
                                            self.rw, self.lenrw ) == 0 )

        return out


    def solve( self ):
        global objGsparse
        global consGsparse

        cdef integer mincw[1]
        cdef integer miniw[1]
        cdef integer minrw[1]
        cdef integer inform_out[1]
        cdef integer *ltmpcw = [ 500 ]
        cdef integer *ltmpiw = [ 500 ]
        cdef integer *ltmprw = [ 500 ]
        cdef char tmpcw[500 * 8]
        cdef integer tmpiw[500]
        cdef doublereal tmprw[500]
        cdef integer *nxname = [ 1 ] ## Do not provide vars names
        cdef integer *nFname = [ 1 ] ## Do not provide cons names
        cdef integer nS[1]
        cdef integer nInf[1]
        cdef doublereal sInf[1]
        cdef doublereal *ObjAdd = [ 0.0 ]
        cdef integer *ObjRow = [ 1 ]
        cdef char* probname = "optwrapp" ## Must have 8 characters
        cdef char* xnames = "dummy"
        cdef char* Fnames = "dummy"
        cdef cnp.ndarray tmparr

        ## Handle debug files
        if( "printFile" in self.options ):
            self.printFileUnit[0] = 90 ## Hardcoded since nobody cares
            if( self.debug ):
                print( ">>> Sending print file to {0}".format( self.options[ "printFile" ] ) )
        else:
            self.printFileUnit[0] = 0 ## disabled by default, pg. 27
            if( self.debug ):
                print( ">>> Print file is disabled" )

        if( "summaryFile" in self.options ):
            if( self.options[ "summaryFile" ] == "stdout" ):
                self.summaryFileUnit[0] = 6 ## Fortran's magic value for stdout
                if( self.debug ):
                    print( ">>> Sending summary to stdout" )
            else:
                self.summaryFileUnit[0] = 89 ## Hardcoded since nobody cares
                if( self.debug ):
                    print( ">>> Sending summary to {0}".format( self.options[ "summaryFile" ] ) )
        else:
            self.summaryFileUnit[0] = 0 ## disabled by default, pg. 28
            if( self.debug ):
                print( ">>> Summary is disabled" )

        ## Initialize
        snopt.sninit_( self.printFileUnit, self.summaryFileUnit,
                       tmpcw, ltmpcw, tmpiw, ltmpiw, tmprw, ltmprw,
                       ltmpcw[0]*8 )

        inform_out[0] = 0 ## Reset inform_out before running snset* functions

        ## Suppress parameter verbosity
        if( not self.debug ):
            self.setOption( "Suppress parameters", tmpcw, ltmpcw, tmpiw, ltmpiw, tmprw, ltmprw )

        ## The following settings change the outcome of snmema, pg. 29
        if( "Hessian full memory" in self.options ):
            self.setOption( "Hessian full memory", tmpcw, ltmpcw, tmpiw, ltmpiw, tmprw, ltmprw )
        elif( "Hessian limited memory" in self.options ):
            self.setOption( "Hessian limited memory", tmpcw, ltmpcw, tmpiw, ltmpiw, tmprw, ltmprw )

        if( "Hessian updates" in self.options ):
            self.setOption( "Hessian updtes {0}".format( self.options[ "Hessian updates" ] ),
                            tmpcw, ltmpcw, tmpiw, ltmpiw, tmprw, ltmprw )

        if( "Hessian dimension" in self.options ):
            self.setOption( "Hessian dimension {0}".format( self.options[ "Hessian dimension" ] ),
                            tmpcw, ltmpcw, tmpiw, ltmpiw, tmprw, ltmprw )

        if( "Superbasics limit" in self.options ):
            self.setOption( "Superbasics limit {0}".format( self.options[ "Superbasics limit" ] ),
                            tmpcw, ltmpcw, tmpiw, ltmpiw, tmprw, ltmprw )

        ## Now we get to know how much memory we need
        snopt.snmema_( inform_out, self.nF, self.n, nxname, nFname, self.lenA, self.lenG,
                       self.lencw, self.leniw, self.lenrw,
                       tmpcw, ltmpcw, tmpiw, ltmpiw, tmprw, ltmprw,
                       ltmpcw[0]*8 )
        if( inform_out[0] != 104 ):
            raise Exception( "snMemA failed to estimate workspace memory requirements" )

        ## Recallocate with new memory requirements
        if( self.mustAllocateWS( self.lencw[0], self.leniw[0], self.lenrw[0] ) ):
            self.deallocateWS()
            self.allocateWS()

        ## zero out workspace
        # memset( self.cw, 0, self.lencw[0] * 8 * sizeof( char ) )
        # memset( self.iw, 0, self.leniw[0] * sizeof( integer ) )
        # memset( self.rw, 0, self.lenrw[0] * sizeof( doublereal ) )

        ## Copy content of temp workspace arrays to malloc'ed workspace arrays
        memcpy( self.cw, tmpcw, ltmpcw[0] * 8 * sizeof( char ) )
        memcpy( self.iw, tmpiw, ltmpiw[0] * sizeof( integer ) )
        memcpy( self.rw, tmprw, ltmprw[0] * sizeof( doublereal ) )

        ## Set new workspace lengths
        self.setOption( "Total character workspace {0}".format( self.lencw[0] ),
                       self.cw, ltmpcw, self.iw, ltmpiw, self.rw, ltmprw )
        self.setOption( "Total integer workspace {0}".format( self.leniw[0] ),
                       self.cw, ltmpcw, self.iw, ltmpiw, self.rw, ltmprw )
        self.setOption( "Total real workspace {0}".format( self.lenrw[0] ),
                       self.cw, ltmpcw, self.iw, ltmpiw, self.rw, ltmprw )

        self.processOptions()

        if( self.debug ):
            print( ">>> All options successfully set up" )

        if( self.debug ):
            print( ">>> Memory allocated for data: {0:,.3} MB".format(
                        ( self.mem_size[0] * ( 4 * sizeof(doublereal) + sizeof(integer) ) +
                          self.mem_size[1] * ( 4 * sizeof(doublereal) + sizeof(integer) ) +
                          self.mem_size[2] * ( sizeof(doublereal) + 2 * sizeof(integer) ) +
                          self.mem_size[3] * 2 * sizeof(integer) ) / 1024 / 1024 ) )

            print( ">>> Memory allocated for workspace: {0:,.3} MB".format(
                        ( self.mem_size_ws[0] * 8 * sizeof(char) +
                          self.mem_size_ws[1] * sizeof(integer) +
                          self.mem_size_ws[2] * sizeof(doublereal) ) / 1024 / 1024 ) )

            if( isinstance( self.prob, nlp.SparseProblem ) ):
                if( self.prob.Nconslin > 0 ):
                    print( ">>> Sparse A: {0:r}".format( self.Asparse ) )
                print( ">>> Sparse gradient obj: {0:r}".format( objGsparse ) )
                if( self.prob.Ncons > 0 ):
                    print( ">>> Sparse gradient cons: {0:r}".format( consGsparse ) )

        ## Execute SNOPT
        snopt.snopta_( self.Start, self.nF,
                       self.n, nxname, nFname,
                       ObjAdd, ObjRow, probname,
                       <snopt.usrfun_fp> usrfun,
                       self.iAfun, self.jAvar, self.lenA, self.neA, self.A,
                       self.iGfun, self.jGvar, self.lenG, self.neG,
                       self.xlow, self.xupp, xnames,
                       self.Flow, self.Fupp, Fnames,
                       self.x, self.xstate, self.xmul,
                       self.F, self.Fstate, self.Fmul,
                       inform_out,
                       mincw, miniw, minrw,
                       nS, nInf, sInf,
                       self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                       self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                       len( probname ), len( xnames ), len( Fnames ),
                       self.lencw[0]*8, self.lencw[0]*8 )

        ## Try to rename fortran print and summary files
        if( "printFile" in self.options ):
            try:
                os.rename( "fort.{0}".format( self.printFileUnit[0] ),
                           str( self.options[ "printFile" ] ) )
            except:
                pass

        if( "summaryFile" in self.options and
            self.options[ "summaryFile" ] != "stdout" ):
            try:
                os.rename( "fort.{0}".format( self.summaryFileUnit[0] ),
                           str( self.options[ "summaryFile" ] ) )
            except:
                pass

        ## Reset warm start
        self.Start[0] = 0

        ## Save result to prob
        self.prob.soln = Soln()
        self.prob.soln.value = float( self.F[0] )
        self.prob.soln.final = np.copy( utils.wrap1dPtr( self.x, self.n[0],
                                                         doublereal_type ) )
        self.prob.soln.xstate = np.copy( utils.wrap1dPtr( self.xstate, self.n[0],
                                                          integer_type ) )
        self.prob.soln.xmul = np.copy( utils.wrap1dPtr( self.xmul, self.n[0],
                                                        doublereal_type ) )
        self.prob.soln.Fstate = np.copy( utils.wrap1dPtr( self.Fstate, self.nF[0],
                                                          integer_type ) )
        self.prob.soln.Fmul = np.copy( utils.wrap1dPtr( self.Fmul, self.nF[0],
                                                        doublereal_type ) )
        self.prob.soln.nS = int( nS[0] )
        self.prob.soln.retval = int( inform_out[0] )

        return( self.prob.soln.final,
                self.prob.soln.value,
                self.prob.soln.retval )
