# cython: boundscheck=False
# cython: wraparound=False

from libc.string cimport memcpy, memset
from libc.stdlib cimport malloc, free
cimport numpy as cnp
import numpy as np
import os

from .typedefs cimport *
cimport lssolh as lssol   ## import every function exposed in lssol.h
cimport utils
cimport base
import qp

## we use these definitions to wrap C arrays in numpy arrays
## we can safely assume 64-bit values since we already checked using scons
cdef int doublereal_type = cnp.NPY_FLOAT64
cdef int integer_type = cnp.NPY_INT64
cdef type doublereal_dtype = np.float64
cdef type integer_dtype = np.int64


cdef class Soln( base.Soln ):
    cdef public cnp.ndarray istate
    cdef public cnp.ndarray clamda
    cdef public cnp.ndarray R
    cdef public int Niters

    def __init__( self ):
        super().__init__()
        self.retval = 100

    def getStatus( self ):
        ## pg. 10
        cdef dict statusInfo = { 0: "Strong local minimum (likely unique)",
                                 1: "Weak local minimum (likely not unique)",
                                 2: "Solution appears to be unbounded",
                                 3: "No feasible point found",
                                 4: "Iteration limit reached",
                                 5: "Algorithm likely cycling, point has not changed in 50 iters.",
                                 6: "Invalid input parameter",
                                 100: "Return information undefined" }

        if( self.retval not in statusInfo ):
            return "Invalid return value"
        else:
            return statusInfo[ self.retval ]


cdef class Solver( base.Solver ):
    cdef integer nrowC[1]
    cdef integer ldR[1]
    cdef integer leniw[1]
    cdef integer lenw[1]
    cdef doublereal *x
    cdef doublereal *bl
    cdef doublereal *bu
    cdef doublereal *clamda
    cdef integer *istate
    cdef integer *kx
    cdef integer *iw
    cdef doublereal *w
    cdef doublereal *A
    cdef doublereal *C
    cdef doublereal B[1]
    cdef doublereal *cVec

    cdef int warm_start
    cdef int mem_alloc
    cdef int mem_size[2] ## { N, Nconslin }


    def __init__( self, prob=None ):
        cdef dict legacy

        super().__init__()

        self.mem_alloc = False
        self.mem_size[0] = self.mem_size[1] = 0
        self.prob = None

        legacy = { "crashTol": "Crash Tolerance",
                   "iterLimit": "Optimality Phase Iteration Limit",
                   "feasibilityTol": "Feasibility Tolerance",
                   "infBoundSize": "Infinite Bound Size",
                   "infStepSize": "Infinite Step Size",
                   "printLevel": "Print level",
                   "rankTol": "Rank Tolerance" }
        self.options = utils.Options( legacy )

        if( prob ):
            self.setupProblem( prob )


    def setupProblem( self, prob ):
        cdef cnp.ndarray tmparr

        if( not isinstance( prob, qp.Problem ) ):
            raise TypeError( "prob must be of type qp.Problem" )

        self.prob = prob

        ## New problems cannot be warm started
        self.warm_start = False

        ## Set problem type
        if( prob.objQ is None and prob.objL is None ):
            self.options[ "Problem type" ] = "FP"
        elif( prob.objQ is None ):
            self.options[ "Problem type" ] = "LP"
        else:
            self.options[ "Problem type" ] = "QP2"

        ## Set size-dependent constants
        if( prob.Nconslin == 0 ): ## pg. 7, nrowC >= 1 even if nclin = 0
            self.nrowC[0] = 1
        else:
            self.nrowC[0] = prob.Nconslin
        self.leniw[0] = prob.N ## pg. 11
        self.lenw[0] = ( 2 * prob.N * prob.N + 10 * prob.N + 6 * prob.Nconslin ) ## pg. 11, case QP2

        ## Allocate if necessary
        if( not self.mem_alloc ):
            self.allocate()
        elif( self.mem_size[0] < prob.N or
              self.mem_size[1] < prob.Nconslin ):
            self.deallocate()
            self.allocate()

        ## Copy information from prob to NPSOL's working arrays
        tmparr = utils.arraySanitize( prob.lb, dtype=doublereal_dtype, fortran=True )
        memcpy( &self.bl[0], utils.getPtr( tmparr ), prob.N * sizeof( doublereal ) )

        tmparr = utils.arraySanitize( prob.ub, dtype=doublereal_dtype, fortran=True )
        memcpy( &self.bu[0], utils.getPtr( tmparr ), prob.N * sizeof( doublereal ) )

        if( prob.objL is not None ):
            tmparr = utils.arraySanitize( prob.objL, dtype=doublereal_dtype, fortran=True )
            memcpy( &self.cVec[0], utils.getPtr( tmparr ), prob.N * sizeof( doublereal ) )

        if( prob.Nconslin > 0 ):
            tmparr = utils.arraySanitize( prob.conslinlb, dtype=doublereal_dtype, fortran=True )
            memcpy( &self.bl[prob.N], utils.getPtr( tmparr ),
                    prob.Nconslin * sizeof( doublereal ) )

            tmparr = utils.arraySanitize( prob.conslinub, dtype=doublereal_dtype, fortran=True )
            memcpy( &self.bu[prob.N], utils.getPtr( tmparr ),
                    prob.Nconslin * sizeof( doublereal ) )

            tmparr = utils.arraySanitize( prob.conslinA, dtype=doublereal_dtype, fortran=True )
            memcpy( &self.C[0], utils.getPtr( tmparr ),
                    prob.Nconslin * prob.N * sizeof( doublereal ) )


    cdef allocate( self ):
        if( self.mem_alloc ):
            return False

        cdef integer Ntot = self.prob.N + self.prob.Nconslin

        self.x = <doublereal *> malloc( self.prob.N * sizeof( doublereal ) )
        self.bl = <doublereal *> malloc( Ntot * sizeof( doublereal ) )
        self.bu = <doublereal *> malloc( Ntot * sizeof( doublereal ) )
        self.clamda = <doublereal *> malloc( Ntot * sizeof( doublereal ) )
        self.istate = <integer *> malloc( Ntot * sizeof( integer ) )
        self.iw = <integer *> malloc( self.leniw[0] * sizeof( integer ) )
        self.w = <doublereal *> malloc( self.lenw[0] * sizeof( doublereal ) )
        self.A = <doublereal *> malloc( self.prob.N * self.prob.N * sizeof( doublereal ) )
        self.C = <doublereal *> malloc( self.nrowC[0] * self.prob.N * sizeof( doublereal ) )
        self.cVec = <doublereal *> malloc( self.prob.N * sizeof( doublereal ) )
        self.kx = <integer *> malloc( self.prob.N * sizeof( integer ) )
        if( self.x == NULL or
            self.bl == NULL or
            self.bu == NULL or
            self.clamda == NULL or
            self.istate == NULL or
            self.iw == NULL or
            self.w == NULL or
            self.A == NULL or
            self.C == NULL or
            self.cVec == NULL or
            self.kx == NULL ):
            raise MemoryError( "At least one memory allocation failed" )

        self.mem_alloc = True
        self.mem_size[0] = self.prob.N
        self.mem_size[1] = self.prob.Nconslin
        return True


    cdef deallocate( self ):
        if( not self.mem_alloc ):
            return False

        free( self.x )
        free( self.bl )
        free( self.bu )
        free( self.clamda )
        free( self.istate )
        free( self.iw )
        free( self.w )
        free( self.A )
        free( self.C )
        free( self.cVec )
        free( self.kx )

        self.mem_alloc = False
        return True


    def __dealloc__( self ):
        self.deallocate()


    def warmStart( self ):
        cdef cnp.ndarray tmparr

        if( not isinstance( self.prob.soln, Soln ) ):
            return False

        tmparr = utils.arraySanitize( self.prob.soln.istate, dtype=integer_dtype, fortran=True )
        memcpy( self.istate, utils.getPtr( tmparr ),
                ( self.prob.N + self.prob.Nconslin ) * sizeof( integer ) )

        self.warm_start = True
        self.options[ "Warm start" ] = True

        return True


    cdef void setOption( self, str option ):
        cdef bytes myopt

        if( len( option ) > 72 ):
            raise ValueError( "option string is too long: {0}".format( option ) )

        myopt = option.encode( "ascii" )
        lssol.lsoptn_( myopt, len( myopt ) )


    cdef void processOptions( self ):
        cdef str mystr, key

        for key in self.options:
            if( self.options[key].dtype == utils.NONE or
                ( self.options[key].dtype == utils.BOOL and not self.options[key].value ) ):
                continue

            mystr = key
            if( self.options[key].dtype != utils.BOOL ):
                mystr += " {0}".format( self.options[key].value )

            if( self.debug ):
                print( "processing option: '{0}'".format( mystr ) )

            self.setOption( mystr )


    def solve( self ):
        cdef integer printFileUnit
        cdef integer summaryFileUnit

        cdef integer *n = [ self.prob.N ]
        cdef integer *nclin = [ self.prob.Nconslin ]
        cdef integer iter_out[1]
        cdef integer inform_out[1]
        cdef doublereal objf_val[1]
        cdef cnp.ndarray tmparr

        ## Begin by setting up initial condition
        tmparr = utils.arraySanitize( self.prob.init, dtype=doublereal_dtype, fortran=True )
        memcpy( self.x, utils.getPtr( tmparr ), self.prob.N * sizeof( doublereal ) )

        ## Set quadratic obj term if available, it is overwritten after each run, pg. 9
        if( self.prob.objQ is not None ):
            tmparr = utils.arraySanitize( self.prob.objQ, dtype=doublereal_dtype, fortran=True )
            memcpy( &self.A[0], utils.getPtr( tmparr ),
                    self.prob.N * self.prob.N * sizeof( doublereal ) )

        ## Supress echo options and reset optional values, pg. 13
        self.setOptions( "Nolist" )
        self.setOptions( "Defaults" )

        ## Handle debug files
        if( "printFile" in self.options ):
            printFileUnit = 90 ## Hardcoded since nobody cares
        else:
            printFileUnit = 0 ## disabled by default, pg. 27

        if( "summaryFile" in self.options ):
            if( self.options[ "summaryFile" ] == "stdout" ):
                summaryFileUnit = 6 ## Fortran's magic value for stdout
            else:
                summaryFileUnit = 89 ## Hardcoded since nobody cares
        else:
            summaryFileUnit = 0 ## disabled by default, pg. 28

        self.setOption( "Print file {0}".format( printFileUnit ) )
        self.setOption( "Summary file {0}".format( summaryFileUnit ) )

        self.processOptions()

        ## Call NPSOL
        lssol.lssol_( n, n,
                      nclin, self.nrowC, n,
                      self.C, self.bl, self.bu, self.cVec,
                      self.istate, self.kx, self.x, self.A, self.B,
                      inform_out, iter_out, objf_val, self.clamda,
                      self.iw, self.leniw, self.w, self.lenw )

        ## Reset warm start state
        self.warm_start = False
        del self.options[ "Warm start" ]

        ## Try to rename fortran print and summary files
        if( "printFile" in self.options ):
            try:
                os.rename( "fort.{0}".format( printFileUnit ),
                           str( self.options[ "printFile" ] ) )
            except:
                pass

        if( "summaryFile" in self.options and
            self.options[ "summaryFile" ] != "stdout" ):
            try:
                os.rename( "fort.{0}".format( summaryFileUnit ),
                           str( self.options[ "summaryFile" ] ) )
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
        self.prob.soln.istate = np.copy( utils.wrap1dPtr( self.istate,
                                                          self.prob.N + self.prob.Nconslin,
                                                          integer_type ) )
        self.prob.soln.clamda = np.copy( utils.wrap1dPtr( self.clamda,
                                                          self.prob.N + self.prob.Nconslin,
                                                          doublereal_type ) )
        self.prob.soln.R = np.copy( utils.wrap2dPtr( self.A,
                                                     self.prob.N, self.prob.N,
                                                     doublereal_type, fortran=True ) )
        self.prob.soln.Niters = int( iter_out[0] )
        self.prob.soln.retval = int( inform_out[0] )

        return( self.prob.soln.final,
                self.prob.soln.value,
                self.prob.soln.retval )
