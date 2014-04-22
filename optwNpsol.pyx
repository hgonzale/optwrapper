from libc.stdlib cimport calloc,free
from libc.string cimport memcpy
import numpy as np
cimport numpy as np

from optwF2c cimport *
cimport optwNpsol as npsol
cimport arrayWrapper as arrwrap
from optwSolver cimport *
from optwrapper import *


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
        valf = extprob.objf( xarr )
        f[0] = valf[0]

    if( mode[0] > 0 ):
        memcpy( g, <doublereal *> arrwrap.getPtr( extprob.objg( xarr ) ),
                extprob.N * sizeof( doublereal ) )

## pg. 18, Section 7.2
cdef int funcon( integer* mode, integer* ncnln,
                 integer* n, integer* ldJ, integer* needc,
                 doublereal* x, doublereal* c, doublereal* cJac,
                 integer* nstate ):
    xarr = arrwrap.wrapPtr( x, extprob.N, np.NPY_DOUBLE )

    if( mode[0] != 1 ):
        valf = np.asfortranarray( extprob.consf( xarr ), dtype=np.float64 )
        memcpy( c, <doublereal *> arrwrap.getPtr( extprob.consf( xarr ) ),
                extprob.Ncons * sizeof( doublereal ) )

    if( mode[0] > 0 ):
        memcpy( cJac, <doublereal *> arrwrap.getPtr( extprob.consg( xarr ) ),
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

    def __cinit__( self, prob ):
        if( not prob is optwProblem ):
            raise StandardError( "Argument 'prob' must be of type 'optwProblem'." )
        self.prob = prob ## Save a copy of the pointer
        setupProblem( self.prob )


    def setupProblem( self, prob ):
        extprob = prob ## Save static point for funcon and funobj

        self.nctotl = prob.N + prob.Nconslin + prob.Ncons

        self.inform[0] = 0
        self.printLevel[0] = 0

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

        # self.x = <doublereal *> calloc( prob.N, sizeof( doublereal ) )
        # self.bl = <doublereal *> calloc( self.nctotl, sizeof( doublereal ) )
        # self.bu = <doublereal *> calloc( self.nctotl, sizeof( doublereal ) )

        self.objf_val[0] = 0.0
        self.objg_val = <doublereal *> calloc( prob.N, sizeof( doublereal ) )
        self.consf_val = <doublereal *> calloc( prob.Ncons, sizeof( doublereal ) )
        self.consg_val = <doublereal *> calloc( self.ldJ[0] * prob.N, sizeof( doublereal ) )

        self.clamda = <doublereal *> calloc( self.nctotl, sizeof( doublereal ) )
        self.istate = <integer *> calloc( self.nctotl, sizeof( integer ) )

        self.iw = <integer *> calloc( self.leniw[0], sizeof( integer ) )
        self.w = <doublereal *> calloc( self.lenw[0], sizeof( doublereal ) )

        # self.A = <doublereal *> calloc( self.ldA[0] * prob.N, sizeof( doublereal ) )
        self.R = <doublereal *> calloc( prob.N * prob.N, sizeof( doublereal ) )

        if( # self.x is NULL or
            # self.bl is NULL or
            # self.bu is NULL or
            self.objg_val is NULL or
            self.consf_val is NULL or
            self.consg_val is NULL or
            self.clamda is NULL or
            self.istate is NULL or
            self.iw is NULL or
            self.w is NULL or
            # self.A is NULL or
            self.R is NULL ):
            raise MemoryError( "At least one memory allocation failed." )

        ## We are assuming np.float64 equals doublereal from now on
        ## At least we need to be sure that doublereal is 8 bytes in this architecture
        assert( sizeof( doublereal ) == 8 )

        tmpbl = prob.lb
        tmpbu = prob.ub
        if( prob.Nconslin > 0 ):
            tmpbl = np.vstack( tmpbl, prob.conslinlb )
            tmpbu = np.vstack( tmpbu, prob.conslinub )
        if( prob.Ncons > 0 ):
            tmpbl = np.vstack( tmpbl, prob.conslb )
            tmpbu = np.vstack( tmpbu, prob.consub )
        ## Make sure arrays are contiguous, fortran-ordered, and float64
        self.bl = <doublereal *> arrwrap.getPtr( tmpbl )
        self.bu = <doublereal *> arrwrap.getPtr( tmpbu )
        self.A = <doublereal *> arrwrap.getPtr( prob.conslinA )
        self.x = <doublereal *> arrwrap.getPtr( prob.init )

        ## Set options
        self.printOpts.printLevel = 0
        self.printOpts.minorPrintLevel = 0


    def __dealloc__( self ):
        # free( self.x )
        # free( self.bl )
        # free( self.bu )
        free( self.objg_val )
        free( self.consf_val )
        free( self.consg_val )
        free( self.clamda )
        free( self.istate )
        free( self.iw )
        free( self.w )
        # free( self.A )
        free( self.R )


    def getStatus( self ):
        if( self.inform_out[0] == None ):
            return "Return information is not defined"
        elif( self.inform_out[0] == 0 ):
            return "Optimality conditions satisfied"
        elif( self.inform_out[0] == 1 ):
            return "Feasible point found but no further improvement can be made"
        elif( self.inform_out[0] == 2 ):
            return "The problem appears to have infeasible linear constraints"
        elif( self.inform_out[0] == 3 ):
            return "The problem appears to have infeasible nonlinear constraints"
        elif( self.inform_out[0] == 4 ):
            return "Iteration limit reached"
        elif( self.inform_out[0] == 6 ):
            return "Point does not satisfy first-order optimality conditions"
        elif( self.inform_out[0] == 7 ):
            return "Derivatives appear to be incorrect"
        else:
            return "Undefined return information value"

    # def set_options( self, warmstart, maxeval, constraint_violation, ftol ):
    #     if( not warmstart is None ):
    #         self.warmstart[0] = <integer> warmstart
    #     if( not maxeval is None ):
    #         self.maxeval[0] = <integer> maxeval
    #     if( not constraint_violation is None ):
    #         self.constraint_violation[0] = <doublereal> constraint_violation
    #     if( not ftol is None ):
    #         self.ftol[0] = <doublereal> ftol


    def checkPrintOpts( self ):
        """
        Check if dictionary self.printOpts is valid.

        Optional entries:
        printFile        filename for debug information (default: None)
        printLevel       verbosity level for major iterations (0-None, 1, 5, 10, 20, or 30-Full)
        minorPrintLevel  verbosity level for minor iterations (0-None, 1, 5, 10, 20, or 30-Full)
        """
        try:
            int( self.printOpts.printLevel ) + 1
            int( self.printOpts.minorPrintLevel ) + 1
        except:
            print( "printOpts.printLevel and printOpts.minorPrintLevel must be integers." )
            return False

        if( self.printOpts.printFile != None ):
            try:
                str( self.printOpts.printFile ) + "x"
            except:
                print( "printOpts.printFile must be a string." )
                return False
        else:
            if( self.printOpts.printLevel > 0 ):
                print( "Must set printOpts.printFile to get debug information." )
                return False

        return True


    def solve( self ):
        cdef integer inform[1]
        cdef char* printFile = self.printOpts.printFile
        cdef integer* printFileUnit = [ 90 ] ## Hardcoded since nobody cares
        cdef integer* printLevel = [ self.printOpts.printLevel ]
        cdef integer* minorPrintLevel = [ self.printOpts.minorPrintLevel ]

        if( self.printFile != None and
            self.printLevel > 0 ):
            npsol.npopenappend_( printFileUnit, printFile, inform,
                                 len( self.printOpts.printFile ) )

            if( inform[0] != 0 ):
                raise StandardError( "Could not open file " + self.printOpts.printFile )

            npsol.npopti_( npsol.STR_PRINT_FILE, printFileUnit,
                           len( npsol.STR_PRINT_FILE ) )

        npsol.npopti_( npsol.STR_PRINT_LEVEL, printLevel,
                       len( npsol.STR_PRINT_LEVEL ) )
        npsol.npopti_( npsol.STR_MINOR_PRINT_LEVEL, minorPrintLevel,
                       len( npsol.STR_MINOR_PRINT_LEVEL ) )

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

        self.prob.final = arrwrap.wrapPtr( self.x, prob.N, np.NPY_DOUBLE )
        self.prob.value = float( self.objf_val[0] )
        self.prob.istate = arrwrap.wrapPtr( self.istate, self.nctotl, np.NPY_LONG )
        self.prob.clamda = arrwrap.wrapPtr( self.clamda, self.nctotl, np.NPY_DOUBLE )
        self.prob.R = arrwrap.wrapPtr( self.R, prob.N * prob.N, np.NPY_DOUBLE )
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
