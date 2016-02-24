# cython: boundscheck=False
# cython: wraparound=False

from libc.string cimport memcpy, memset
from libc.stdlib cimport malloc, free
cimport numpy as cnp
import numpy as np
import os

from .typedefs cimport *
cimport npsolh as npsol   ## import every function exposed in npsol.h
cimport utils
cimport base
import nlp

## we use these definitions to wrap C arrays in numpy arrays
## we can safely assume 64-bit values since we already checked using scons
cdef int doublereal_type = cnp.NPY_FLOAT64
cdef int integer_type = cnp.NPY_INT64
cdef type doublereal_dtype = np.float64
cdef type integer_dtype = np.int64

## The functions funobj and funcon should be static methods in npsol.Solver,
## but it appears that Cython doesn't support static cdef methods yet.
## Instead, this is a reasonable hack.
cdef object extprob

## pg. 17, Section 7.1
cdef int funobj( integer* mode, integer* n,
                 doublereal* x, doublereal* f, doublereal* g,
                 integer* nstate ):
    xarr = utils.wrap1dPtr( x, n[0], doublereal_type )

    ## we zero out all arrays in case the user does not modify all the values,
    ## e.g., in sparse problems.
    if( mode[0] != 1 ):
        f[0] = 0.0
        farr = utils.wrap1dPtr( f, 1, doublereal_type )
        extprob.objf( farr, xarr )
        if( extprob.objmixedA is not None ):
            farr += extprob.objmixedA.dot( xarr )

    if( mode[0] != 0 ):
        memset( g, 0, n[0] * sizeof( doublereal ) )
        garr = utils.wrap1dPtr( g, n[0], doublereal_type )
        extprob.objg( garr, xarr )
        if( extprob.objmixedA is not None ):
            garr += extprob.objmixedA


## pg. 18, Section 7.2
cdef int funcon( integer* mode, integer* ncnln,
                 integer* n, integer* ldJ, integer* needc,
                 doublereal* x, doublereal* c, doublereal* cJac,
                 integer* nstate ):
    xarr = utils.wrap1dPtr( x, n[0], doublereal_type )

    ## we zero out all arrays in case the user does not modify all the values,
    ## e.g., in sparse problems.
    if( mode[0] != 1 ):
        memset( c, 0, ncnln[0] * sizeof( doublereal ) )
        carr = utils.wrap1dPtr( c, ncnln[0], doublereal_type )
        extprob.consf( carr, xarr )
        if( extprob.consmixedA is not None ):
            carr += extprob.consmixedA.dot( xarr )

    if( mode[0] > 0 ):
        memset( cJac, 0, ncnln[0] * n[0] * sizeof( doublereal ) )
        cJacarr = utils.wrap2dPtr( cJac, ncnln[0], n[0], doublereal_type, fortran=True )
        extprob.consg( cJacarr, xarr )
        if( extprob.consmixedA is not None ):
            cJacarr += extprob.consmixedA


cdef class Soln( base.Soln ):
    cdef public cnp.ndarray istate
    cdef public cnp.ndarray clamda
    cdef public cnp.ndarray R
    cdef public int Niters

    def __init__( self ):
        super().__init__()
        self.retval = 100

    def getStatus( self ):
        cdef dict status = { 0: "Optimality conditions satisfied",
                             1: "Optimality conditions satisfied, but sequence has not converged",
                             2: "Linear constraints could not be satisfied",
                             3: "Nonlinear constraints could not be satisfied",
                             4: "Iteration limit reached",
                             6: "Optimality conditions not satisfied, no improvement can be made",
                             7: "Derivatives appear to be incorrect",
                             9: "Invalid input parameter",
                             100: "Return information undefined" }

        if( self.retval < 0 ):
            return "Execution terminated by user defined function (should not occur)"
        elif( self.retval not in status ):
            return "Invalid return value"
        else:
            return status[ self.retval ]


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
    cdef int warm_start
    cdef int mem_alloc
    cdef int mem_size[3] ## { N, Nconslin, Ncons }


    def __init__( self, prob=None ):
        cdef dict legacy

        super().__init__()

        self.mem_alloc = False
        self.mem_size[0] = self.mem_size[1] = self.mem_size[2] = 0
        self.prob = None

        legacy = { "printLevel": "Print Level",
                   "minorPrintLevel": "Minor Print Level",
                   "centralDiffInterval": "Central Difference Interval",
                   "crashTol": "Crash Tolerance",
                   "diffInterval": "Difference Interval",
                   "feasibilityTol": "Feasibility Tolerance",
                   "fctnPrecision": "Function Precision",
                   "infBoundSize": "Infinite Bound Size",
                   "infStepSize": "Infinite Step Size",
                   "iterLimit": "Iteration Limit",
                   "linesearchTol": "Line Search Tolerance",
                   "minorIterLimit": "Minor Iteration Limit",
                   "optimalityTol": "Optimality Tolerance",
                   "stepLimit": "Step Limit",
                   "verifyLevel": "Verify Level",
                   "printFile": "Print File",
                   "summaryFile": "Summary File" }
        self.options = utils.Options( legacy )
        self.options[ "Hessian" ] = "yes"
        self.options[ "Verify level" ] = -1

        if( prob ):
            self.setupProblem( prob )


    def setupProblem( self, prob ):
        global extprob
        cdef cnp.ndarray tmparr

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

        ## Allocate if necessary
        if( not self.mem_alloc ):
            self.allocate()
        elif( self.mem_size[0] < prob.N or
              self.mem_size[1] < prob.Nconslin or
              self.mem_size[2] < prob.Ncons ):
            self.deallocate()
            self.allocate()

        ## Copy information from prob to NPSOL's working arrays
        tmparr = utils.arraySanitize( prob.lb, dtype=doublereal_dtype, fortran=True )
        memcpy( &self.bl[0], utils.getPtr( tmparr ), prob.N * sizeof( doublereal ) )

        tmparr = utils.arraySanitize( prob.ub, dtype=doublereal_dtype, fortran=True )
        memcpy( &self.bu[0], utils.getPtr( tmparr ), prob.N * sizeof( doublereal ) )

        if( prob.Nconslin > 0 ):
            tmparr = utils.arraySanitize( prob.conslinlb, dtype=doublereal_dtype, fortran=True )
            memcpy( &self.bl[prob.N], utils.getPtr( tmparr ), prob.Nconslin * sizeof( doublereal ) )

            tmparr = utils.arraySanitize( prob.conslinub, dtype=doublereal_dtype, fortran=True )
            memcpy( &self.bu[prob.N], utils.getPtr( tmparr ), prob.Nconslin * sizeof( doublereal ) )

            tmparr = utils.arraySanitize( prob.conslinA, dtype=doublereal_dtype, fortran=True )
            memcpy( &self.A[0], utils.getPtr( tmparr ),
                    self.ldA[0] * prob.N * sizeof( doublereal ) )

        if( prob.Ncons > 0 ):
            tmparr = utils.arraySanitize( prob.conslb, dtype=doublereal_dtype, fortran=True )
            memcpy( &self.bl[prob.N+prob.Nconslin], utils.getPtr( tmparr ),
                    prob.Ncons * sizeof( doublereal ) )

            tmparr = utils.arraySanitize( prob.consub, dtype=doublereal_dtype, fortran=True )
            memcpy( &self.bu[prob.N+prob.Nconslin], utils.getPtr( tmparr ),
                    prob.Ncons * sizeof( doublereal ) )


    cdef int allocate( self ):
        if( self.mem_alloc ):
            return False

        self.x = <doublereal *> malloc( self.prob.N * sizeof( doublereal ) )
        self.bl = <doublereal *> malloc( self.nctotl * sizeof( doublereal ) )
        self.bu = <doublereal *> malloc( self.nctotl * sizeof( doublereal ) )
        self.objg_val = <doublereal *> malloc( self.prob.N * sizeof( doublereal ) )
        self.consf_val = <doublereal *> malloc( self.prob.Ncons * sizeof( doublereal ) )
        self.consg_val = <doublereal *> malloc( self.ldJ[0] * self.prob.N * sizeof( doublereal ) )
        self.clamda = <doublereal *> malloc( self.nctotl * sizeof( doublereal ) )
        self.istate = <integer *> malloc( self.nctotl * sizeof( integer ) )
        self.iw = <integer *> malloc( self.leniw[0] * sizeof( integer ) )
        self.w = <doublereal *> malloc( self.lenw[0] * sizeof( doublereal ) )
        self.A = <doublereal *> malloc( self.ldA[0] * self.prob.N * sizeof( doublereal ) )
        self.R = <doublereal *> malloc( self.prob.N * self.prob.N * sizeof( doublereal ) )
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


    cdef int deallocate( self ):
        if( not self.mem_alloc ):
            return False

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

        self.mem_alloc = False
        return True


    def __dealloc__( self ):
        self.deallocate()


    def warmStart( self ):
        cdef cnp.ndarray tmparr

        if( not isinstance( self.prob.soln, Soln ) ):
            return False

        tmparr = utils.arraySanitize( self.prob.soln.istate, dtype=integer_dtype, fortran=True )
        memcpy( self.istate, utils.getPtr( tmparr ), self.nctotl * sizeof( integer ) )

        tmparr = utils.arraySanitize( self.prob.soln.clamda, dtype=doublereal_dtype, fortran=True )
        memcpy( self.clamda, utils.getPtr( tmparr ), self.nctotl * sizeof( doublereal ) )

        tmparr = utils.arraySanitize( self.prob.soln.R, dtype=doublereal_dtype, fortran=True )
        memcpy( self.R, utils.getPtr( tmparr ), self.prob.N * self.prob.N * sizeof( doublereal ) )

        self.warm_start = True
        self.options[ "Warm start" ] = True
        self.options[ "Hessian" ] = "yes"

        return True


    cdef void setOption( self, str option ):
        cdef bytes myopt

        if( len( option ) > 72 ):
            raise ValueError( "option string is too long: {0}".format( option ) )

        myopt = option.encode( "ascii" )
        npsol.npoptn_( myopt, len( myopt ) )


    cdef void processOptions( self ):
        cdef str mystr, key

        for key in self.options:
            if( self.options[key].dtype == utils.NONE or
                ( self.options[key].dtype == utils.BOOL and not self.options[key].value ) ):
                continue

            if( key == "print file" or
                key == "summary file" ): ## these options are set in self.solve()
                continue

            mystr = key
            if( self.options[key].dtype != utils.BOOL ):
                mystr += " {0}".format( self.options[key].value )

            if( self.debug ):
                print( "processing option: '{0}'".format( mystr ) )

            self.setOption( mystr )


    def solve( self ):
        cdef integer summaryFileUnit
        cdef integer printFileUnit

        cdef integer *n = [ self.prob.N ]
        cdef integer *nclin = [ self.prob.Nconslin ]
        cdef integer *ncnln = [ self.prob.Ncons ]
        cdef integer iter_out[1]
        cdef integer inform_out[1]
        cdef doublereal objf_val[1]
        cdef cnp.ndarray tmparr

        ## Begin by setting up initial condition
        tmparr = utils.arraySanitize( self.prob.init, dtype=doublereal_dtype, fortran=True )
        memcpy( self.x, utils.getPtr( tmparr ), self.prob.N * sizeof( doublereal ) )

        ## Supress echo options and reset optional values, pg. 21
        self.setOption( "Nolist" )
        self.setOption( "Defaults" )

        ## Handle debug files
        if( "Print File" in self.options ):
            printFileUnit = 90 ## Hardcoded since nobody cares
        else:
            printFileUnit = 0 ## disabled by default, pg. 27

        if( "Summary File" in self.options ):
            if( self.options[ "Summary File" ] == "stdout" ):
                summaryFileUnit = 6 ## Fortran's magic value for stdout
            else:
                summaryFileUnit = 89 ## Hardcoded since nobody cares
        else:
            summaryFileUnit = 0 ## disabled by default, pg. 28

        self.setOption( "Print file {0}".format( printFileUnit ) )
        self.setOption( "Summary file {0}".format( summaryFileUnit ) )

        self.processOptions()

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

        ## Reset warm start state
        self.warm_start = False
        del self.options[ "Warm start" ]

        ## Try to rename fortran print and summary files
        if( "Print File" in self.options ):
            try:
                os.rename( "fort.{0}".format( printFileUnit ),
                           str( self.options[ "Print File" ] ) )
            except:
                pass

        if( "Summary File" in self.options and
            self.options[ "Summary File" ] != "stdout" ):
            try:
                os.rename( "fort.{0}".format( summaryFileUnit ),
                           str( self.options[ "Summary File" ] ) )
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
                                                         doublereal_type ) )
        self.prob.soln.istate = np.copy( utils.wrap1dPtr( self.istate, self.nctotl,
                                                          integer_type ) )
        self.prob.soln.clamda = np.copy( utils.wrap1dPtr( self.clamda, self.nctotl,
                                                          doublereal_type ) )
        self.prob.soln.R = np.copy( utils.wrap2dPtr( self.R, self.prob.N, self.prob.N,
                                                     doublereal_type, fortran=True ) )
        self.prob.soln.Niters = int( iter_out[0] )
        self.prob.soln.retval = int( inform_out[0] )

        return( self.prob.soln.final,
                self.prob.soln.value,
                self.prob.soln.retval )
