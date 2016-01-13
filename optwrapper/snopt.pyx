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
        cdef tuple statusInfo = ( "Finished successfully", ## 0
                                  "The problem appears to be infeasible", ## 10
                                  "The problem appears to be unbounded", ## 20
                                  "Resource limit error", ## 30
                                  "Terminated after numerical difficulties", ## 40
                                  "Error in the user-supplied functions", ## 50
                                  "Undefined user-supplied functions", ## 60
                                  "User requested termination", ## 70
                                  "Insufficient storage allocated", ## 80
                                  "Input arguments out of range", ## 90
                                  "SNOPT auxiliary routine finished successfully", ## 100
                                  "Errors while processing MPS data", ## 110
                                  "Errors while estimating Jacobian structure", ## 120
                                  "Errors while reading the Specs file", ## 130
                                  "System error" ) ## 140

        if( self.retval == 1000 ):
            return "Return information is not defined yet"

        if( self.retval < 0 ):
            return "Execution terminated by user defined function (should not occur)"
        elif( self.retval >= 143 ):
            return "Invalid value"
        else:
            return statusInfo[ int( self.retval / 10 ) ]


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

        if( prob ):
            self.setupProblem( prob )

        ## legacy_label: real_label
        legacy = { "printLevel": "Major print level",
                   "minorPrintLevel": "Minor print level",
                   "printFreq": "Print frequency",
                   "scalePrint": "Scale print",
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
                   "reducedHessianDim": "Reduced Hessian dimension",
                   "scaleOption": "Scale option",
                   "scaleTol": "Scale tolerance",
                   "superbasicsLimit": "Superbasics limit",
                   "unboundedObjValue": "Unbounded objective value",
                   "unboundedStepSize": "Unbounded step size",
                   "verifyLevel": "Verify level",
                   "violationLimit": "Violation limit" }
        self.options = Options( legacy )


    def setupProblem( self, prob ):
        cdef utils.sMatrix Asparse
        cdef cnp.ndarray tmparr

        global extprob
        global objGsparse
        global consGsparse

        if( not isinstance( prob, nlp.Problem ) ):
            raise TypeError( "Argument prob must be of type nlp.Problem" )

        self.prob = prob ## Save a copy of prob's pointer
        extprob = prob ## Save another (global) copy of prob's pointer to use in usrfun

        ## New problems cannot be warm started
        self.Start[0] = 0 ## Cold start

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
        Asparse = utils.sMatrix( np.vstack( tmplist ), copy_data=True )
        print( Asparse[:] )
        if( Asparse.nnz > 0 ):
            self.lenA[0] = Asparse.nnz
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
        Asparse.copyFortranIdxs( <int64_t *> &self.iAfun[0],
                                 <int64_t *> &self.jAvar[0] )
        ## copy matrix data of A
        Asparse.copyData( &self.A[0] )

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

        tmparr = utils.arraySanitize( self.prob.soln.xstate, dtype=integer_dtype, fortran=True )
        memcpy( self.xstate, utils.getPtr( tmparr ),
                self.n[0] * sizeof( integer ) )

        tmparr = utils.arraySanitize( self.prob.soln.Fstate, dtype=integer_dtype, fortran=True )
        memcpy( self.Fstate, utils.getPtr( tmparr ),
                self.nF[0] * sizeof( integer ) )

        self.Start[0] = 2
        return True


    cdef int processOptions( self, integer* printFileUnit, integer* summaryFileUnit,
                             char* cw, integer* iw, doublereal* rw,
                             integer* lencw, integer* leniw, integer* lenrw ):
        cdef str mystr
        cdef integer inform_out[1]

        if( not self.nlp_alloc ):
            return False

        for key in self.options:
            mystr = key
            if( self.options[key].dtype != utils.NONE ):
                mystr += " {0}".format( self.options[key].value )

            if( debug ):
                print( "processing option: '{0}'".format( mystr ) )

            snopt.snset_( mystr,
                          printFileUnit, summaryFileUnit, inform_out,
                          cw, lencw, iw, leniw, rw, lenrw,
                          len( mystr ), lencw[0]*8 )

            if( inform_out[0] != 0 ):
                raise TypeError( "Could not process option " +
                                 "'{0}: {1}' with type {2}".format( key,
                                                                    self.options[key].value,
                                                                    self.options[key].dtype ) )

        return True


    def solve( self ):
        def ezset( char* s, bint temp=False ):
            if( temp ):
                snopt.snset_( s,
                              printFileUnit, summaryFileUnit, inform_out,
                              tmpcw, ltmpcw, tmpiw, ltmpiw, tmprw, ltmprw,
                              len( s ), ltmpcw[0]*8 )
            else:
                snopt.snset_( s,
                              printFileUnit, summaryFileUnit, inform_out,
                              self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                              len( s ), self.lencw[0]*8 )

        def ezseti( char* s, integer i, bint temp=False ):
            cdef integer* tmpInt = [ i ]
            if( temp ):
                snopt.snseti_( s, tmpInt,
                               printFileUnit, summaryFileUnit, inform_out,
                               tmpcw, ltmpcw, tmpiw, ltmpiw, tmprw, ltmprw,
                               len( s ), ltmpcw[0]*8 )
            else:
                snopt.snseti_( s, tmpInt,
                               printFileUnit, summaryFileUnit, inform_out,
                               self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                               len( s ), self.lencw[0]*8 )

        def ezsetr( char* s, doublereal r ):
            cdef doublereal* tmpReal = [ r ]
            snopt.snsetr_( s, tmpReal,
                           printFileUnit, summaryFileUnit, inform_out,
                           self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                           len( s ), self.lencw[0]*8 )

        ## option strings
        cdef char* STR_SUPPRESS_PARAMETERS = "Suppress parameters"
        cdef char* STR_TOTAL_CHARACTER_WORKSPACE = "Total character workspace"
        cdef char* STR_TOTAL_INTEGER_WORKSPACE = "Total integer workspace"
        cdef char* STR_TOTAL_REAL_WORKSPACE = "Total real workspace"

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
        cdef integer summaryFileUnit[1]
        cdef integer printFileUnit[1]
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

        ## Begin by setting up initial condition
        tmparr = utils.arraySanitize( self.prob.init, dtype=doublereal_dtype, fortran=True )
        memcpy( self.x, utils.getPtr( tmparr ),
                self.n[0] * sizeof( doublereal ) )

        ## Handle debug files
        if( self.options[ "printFile" ].value ):
            printFileUnit[0] = 90 ## Hardcoded since nobody cares
            if( self.debug ):
                print( ">>> Sending print file to " + self.options[ "printFile" ] )
        else:
            printFileUnit[0] = 0 ## disabled by default, pg. 27
            if( self.debug ):
                print( ">>> Print file is disabled" )

        if( self.options[ "summaryFile" ].value ):
            if( self.options[ "summaryFile" ].value == "stdout" ):
                summaryFileUnit[0] = 6 ## Fortran's magic value for stdout
                if( self.debug ):
                    print( ">>> Sending summary to stdout" )
            else:
                summaryFileUnit[0] = 89 ## Hardcoded since nobody cares
                if( self.debug ):
                    print( ">>> Sending summary to " + self.options[ "summaryFile" ] )
        else:
            summaryFileUnit[0] = 0 ## disabled by default, pg. 28
            if( self.debug ):
                print( ">>> Summary is disabled" )

        ## Initialize
        snopt.sninit_( printFileUnit, summaryFileUnit,
                       tmpcw, ltmpcw, tmpiw, ltmpiw, tmprw, ltmprw,
                       ltmpcw[0]*8 )

        inform_out[0] = 0 ## Reset inform_out before running snset* functions

        #####################################################
        #####################################################
        #####################################################
        ## Suppress parameter verbosity
        if( not self.debug ):
            ezset( STR_SUPPRESS_PARAMETERS, True )

        ## The following settings change the outcome of snmema, pg. 29
        if( self.options[ "hessianMemory" ] is not None ):
            if( self.options[ "hessianMemory" ].lower() == "full" ):
                ezset( STR_HESSIAN_FULL_MEMORY, True )
            elif( self.options[ "hessianMemory" ].lower() == "limited" ):
                ezset( STR_HESSIAN_LIMITED_MEMORY, True )

        if( self.options[ "hessianUpdates" ] is not None ):
            ezseti( STR_HESSIAN_UPDATES, self.options[ "hessianUpdates" ], True )

        if( self.options[ "reducedHessianDim" ] is not None ):
            ezseti( STR_REDUCED_HESSIAN_DIMENSION, self.options[ "reducedHessianDim" ], True )

        if( self.options[ "superbasicsLimit" ] is not None ):
            ezseti( STR_SUPERBASICS_LIMIT, self.options[ "superbasicsLimit" ], True )

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

        inform_out[0] = 0 ## Reset inform_out before running snset* functions

        ## Set new workspace lengths
        snopt.snseti_( STR_TOTAL_CHARACTER_WORKSPACE, self.lencw,
                       printFileUnit, summaryFileUnit, inform_out,
                       self.cw, ltmpcw, self.iw, ltmpiw, self.rw, ltmprw,
                       len( STR_TOTAL_CHARACTER_WORKSPACE ), self.lencw[0]*8 )
        snopt.snseti_( STR_TOTAL_INTEGER_WORKSPACE, self.leniw,
                       printFileUnit, summaryFileUnit, inform_out,
                       self.cw, ltmpcw, self.iw, ltmpiw, self.rw, ltmprw,
                       len( STR_TOTAL_INTEGER_WORKSPACE ), self.lencw[0]*8 )
        snopt.snseti_( STR_TOTAL_REAL_WORKSPACE, self.lenrw,
                       printFileUnit, summaryFileUnit, inform_out,
                       self.cw, ltmpcw, self.iw, ltmpiw, self.rw, ltmprw,
                       len( STR_TOTAL_REAL_WORKSPACE ), self.lencw[0]*8 )
        if( inform_out[0] != 0 ):
            raise Exception( "Could not set workspace lengths" )

        ## Set rest of the parameters
        if( self.options[ "centralDiffInterval" ] is not None ):
            ezsetr( STR_CENTRAL_DIFFERENCE_INTERVAL, self.options[ "centralDiffInterval" ] )

        if( self.options[ "checkFreq" ] is not None ):
            ezseti( STR_CHECK_FREQUENCY, self.options[ "checkFreq" ] )

        if( self.options[ "crashOpt" ] is not None ):
            ezseti( STR_CRASH_OPTION, self.options[ "crashOpt" ] )

        if( self.options[ "crashTol" ] is not None ):
            ezseti( STR_CRASH_TOLERANCE, self.options[ "crashTol" ] )

        if( self.options[ "nonderivLinesearch"] is not None ):
            ezset( STR_NONDERIVATIVE_LINESEARCH )

        if( self.options[ "diffInterval" ] is not None ):
            ezsetr( STR_DIFFERENCE_INTERVAL, self.options[ "diffInterval" ] )

        if( self.options[ "elasticWeight" ] is not None ):
            ezsetr( STR_ELASTIC_WEIGHT, self.options[ "elasticWeight" ] )

        if( self.options[ "expandFreq" ] is not None ):
            ezseti( STR_EXPAND_FREQUENCY, self.options[ "expandFreq" ] )

        if( self.options[ "factorizationFreq" ] is not None ):
            ezseti( STR_FACTORIZATION_FREQUENCY, self.options[ "factorizationFreq" ] )

        if( self.options[ "feasiblePoint" ] is not None ):
            ezset( STR_FEASIBLE_POINT )

        if( self.options[ "fctnPrecision" ] is not None ):
            ezsetr( STR_FUNCTION_PRECISION, self.options[ "fctnPrecision" ] )

        if( self.options[ "hessianFreq" ] is not None ):
            ezseti( STR_HESSIAN_FREQUENCY, self.options[ "hessianFreq" ] )

        if( self.options[ "infBound" ] is not None ):
            ezsetr( STR_INFINITE_BOUND, self.options[ "infBound" ] )

        if( self.options[ "iterLimit" ] is not None ):
            ezseti( STR_ITERATIONS_LIMIT, self.options[ "iterLimit" ] )

        if( self.options[ "linesearchTol" ] is not None ):
            ezsetr( STR_LINESEARCH_TOLERANCE, self.options[ "linesearchTol" ] )

        if( self.options[ "luFactorTol" ] is not None ):
            ezsetr( STR_LU_FACTOR_TOLERANCE, self.options[ "luFactorTol" ] )

        if( self.options[ "luUpdateTol" ] is not None ):
            ezsetr( STR_LU_UPDATE_TOLERANCE, self.options[ "luUpdateTol" ] )

        if( self.options[ "luPivoting" ] is not None ):
            if( self.options[ "luPivoting" ].lower() == "rook" ):
                ezset( STR_LU_ROOK_PIVOTING )
            elif( self.options[ "luPivoting" ].lower() == "complete" ):
                ezset( STR_LU_COMPLETE_PIVOTING )

        if( self.options[ "luDensityTol" ] is not None ):
            ezsetr( STR_LU_DENSITY_TOLERANCE, self.options[ "luDensityTol" ] )

        if( self.options[ "luSingularityTol" ] is not None ):
            ezsetr( STR_LU_SINGULARITY_TOLERANCE, self.options[ "luSingularityTol" ] )

        if( self.options[ "majorFeasibilityTol" ] is not None ):
            ezsetr( STR_MAJOR_FEASIBILITY_TOLERANCE, self.options[ "majorFeasibilityTol" ] )

        if( self.options[ "majorIterLimit" ] is not None ):
            ezseti( STR_MAJOR_ITERATIONS_LIMIT, self.options[ "majorIterLimit" ] )

        if( self.options[ "majorOptimalityTol" ] is not None ):
            ezsetr( STR_MAJOR_OPTIMALITY_TOLERANCE, self.options[ "majorOptimalityTol" ] )

        if( self.options[ "printLevel" ] is not None ):
            ezseti( STR_MAJOR_PRINT_LEVEL, self.options[ "printLevel" ] )

        if( self.options[ "majorStepLimit" ] is not None ):
            ezsetr( STR_MAJOR_STEP_LIMIT, self.options[ "majorStepLimit" ] )

        if( self.options[ "minorIterLimit" ] is not None ):
            ezseti( STR_MINOR_ITERATIONS_LIMIT, self.options[ "minorIterLimit" ] )

        if( self.options[ "minorFeasibilityTol" ] is not None ):
            ezsetr( STR_MINOR_FEASIBILITY_TOLERANCE, self.options[ "minorFeasibilityTol" ] )

        if( self.options[ "minorPrintLevel" ] is not None ):
            ezseti( STR_MINOR_PRINT_LEVEL, self.options[ "minorPrintLevel" ] )

        if( self.options[ "newSuperbasicsLimit" ] is not None ):
            ezseti( STR_NEW_SUPERBASICS_LIMIT, self.options[ "newSuperbasicsLimit" ] )

        if( self.options[ "partialPrice" ] is not None ):
            ezseti( STR_PARTIAL_PRICE, self.options[ "partialPrice" ] )

        if( self.options[ "pivotTol" ] is not None ):
            ezsetr( STR_PIVOT_TOLERANCE, self.options[ "pivotTol" ] )

        if( self.options[ "printFreq" ] is not None ):
            ezseti( STR_PRINT_FREQUENCY, self.options[ "printFreq" ] )

        if( self.options[ "proximalPointMethod" ] is not None ):
            ezseti( STR_PROXIMAL_POINT_METHOD, self.options[ "proximalPointMethod" ] )

        if( self.options[ "qpSolver" ] is not None ):
            if( self.options[ "qpSolver" ].lower() == "cg" ):
                ezset( STR_QPSOLVER_CG )
            elif( self.options[ "qpSolver" ].lower() == "qn" ):
                ezset( STR_QPSOLVER_QN )

        if( self.options[ "scaleOption" ] is not None ):
            ezseti( STR_SCALE_OPTION, self.options[ "scaleOption" ] )

        if( self.options[ "scaleTol" ] is not None ):
            ezsetr( STR_SCALE_TOLERANCE, self.options[ "scaleTol" ] )

        if( self.options[ "scalePrint" ] is not None ):
            ezset( STR_SCALE_PRINT )

        if( self.options[ "solutionPrint" ] == True ):
            ezset( STR_SOLUTION_YES )
        else:
            ezset( STR_SOLUTION_NO ) ## changed default value! we don't print soln unless requested

        if( self.options[ "systemInfo" ] is not None ):
            ezset( STR_SYSTEM_INFORMATION_YES )

        if( self.options[ "timingLevel" ] is not None ):
            ezseti( STR_TIMING_LEVEL, self.options[ "timingLevel" ] )

        if( self.options[ "unboundedObjValue" ] is not None ):
            ezsetr( STR_UNBOUNDED_OBJECTIVE_VALUE, self.options[ "unboundedObjValue" ] )

        if( self.options[ "unboundedStepSize" ] is not None ):
            ezsetr( STR_UNBOUNDED_STEP_SIZE, self.options[ "unboundedStepSize" ] )

        if( self.options[ "verifyLevel" ] is not None ):
            ezseti( STR_VERIFY_LEVEL, self.options[ "verifyLevel" ] )
        else:
            ezseti( STR_VERIFY_LEVEL, -1 ) ## changed default value! disabled by default, pg. 84

        if( self.options[ "violationLimit" ] is not None ):
            ezsetr( STR_VIOLATION_LIMIT, self.options[ "violationLimit" ] )

        ## Checkout if we had any errors before we run SNOPT
        if( inform_out[0] != 0 ):
            raise Exception( "At least one option setting failed" )
        elif( self.debug ):
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
                    print( ">>> Sparsity of A: {0:.1%}".format(
                           self.lenA[0] / ( self.n[0] * self.prob.Nconslin ) ) )
                if( self.prob.Ncons > 0 ):
                    print( ">>> Sparsity of gradient: {0:.1%}".format(
                           self.lenG[0] / ( self.n[0] * ( 1 + self.prob.Ncons ) ) ) )

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
        if( self.options[ "printFile" ] is not None and
            self.options[ "printFile" ] != "" ):
            try:
                os.rename( "fort.{0}".format( printFileUnit[0] ),
                           self.options[ "printFile" ] )
            except:
                pass

        if( self.options[ "summaryFile" ] is not None and
            self.options[ "summaryFile" ] != "" and
            self.options[ "summaryFile" ].lower() != "stdout" ):
            try:
                os.rename( "fort.{0}".format( summaryFileUnit[0] ),
                           self.options[ "summaryFile" ] )
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
