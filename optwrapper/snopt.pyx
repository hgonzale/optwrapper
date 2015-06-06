# cython: boundscheck=False
# cython: wraparound=False

from libc.string cimport memcpy, memset
from libc.stdlib cimport malloc, free
cimport numpy as cnp
import numpy as np
import os

from .f2ch cimport *
cimport snopth as snopt
cimport utils
cimport base
import nlp


## sMatrix helper to create arrays with valid indices out of keys with heterogeneous data types
cdef cnp.ndarray key_to_array( object key, integer limit ):
    if( isinstance( key, slice ) ):
        return np.arange( *key.indices( limit ), dtype=np.int_ )
    else:
        try:
            key = np.asarray( key, dtype=np.int_ )
        except:
            raise TypeError( "key cannot be converted to an integer array" )
        ## http://docs.scipy.org/doc/numpy/reference/arrays.nditer.html#modifying-array-values
        for val in np.nditer( key, op_flags=[ "readwrite" ] ):
            if( val >= limit or val < -limit ):
                raise IndexError( "index {0} is out of bounds, limit is {1}".format( val, limit ) )
            if( val < 0 ):
                val[...] = limit + val
        return key


cdef class sMatrix:
    cdef doublereal *data
    cdef integer *rptr
    cdef integer *ridx
    cdef integer *cidx
    cdef readonly integer nnz
    cdef readonly integer nrows
    cdef readonly integer ncols
    cdef readonly tuple shape
    cdef int data_alloc

    def __cinit__( self, arr, int copy_data=False ):
        self.data_alloc = True

        try:
            arr = np.atleast_2d( np.asarray( arr, dtype=np.float64 ) )
        except:
            raise TypeError( "argument must be an array" )

        if( arr.ndim > 2 ):
            raise ValueError( "argument can have at most two dimensions" )

        ( self.nrows, self.ncols ) = arr.shape
        ( rowarr, colarr ) = np.nonzero( arr )
        self.nnz = colarr.size

        self.data = <doublereal *> malloc( self.nnz * sizeof( doublereal ) )
        self.rptr = <integer *> malloc( ( self.nrows + 1 ) * sizeof( integer ) )
        self.ridx = <integer *> malloc( self.nnz * sizeof( integer ) )
        self.cidx = <integer *> malloc( self.nnz * sizeof( integer ) )

        self.shape = ( self.nrows, self.ncols )

        ## copy ridx and cidx
        memcpy( self.ridx,
                utils.getPtr( utils.convIntFortran( rowarr ) ),
                self.nnz * sizeof( integer ) )
        memcpy( self.cidx,
                utils.getPtr( utils.convIntFortran( colarr ) ),
                self.nnz * sizeof( integer ) )
        ## write rptr
        self.rptr[0] = 0
        for k in range( self.nrows ):
            self.rptr[k+1] = self.rptr[k] + np.sum( rowarr == k )
        ## zero data
        if( copy_data ):
            memcpy( self.data,
                    utils.getPtr( utils.convFortran( arr[rowarr,colarr].flatten() ) ),
                    self.nnz * sizeof( doublereal ) )
        else:
            memset( self.data, 0, self.nnz * sizeof( doublereal ) )


    def print_debug( self ):
        """
        print internal C arrays containing representation data of this sparse matrix, which
        cannot be accessed using Python

        """

        print( "nrows: {0} - ncols: {1} - nnz: {2} - data_alloc: {3}".format( self.nrows,
                                                                              self.ncols,
                                                                              self.nnz,
                                                                              self.data_alloc ) )

        print( "rptr: [" ),
        for k in range( self.nrows+1 ):
            print( self.rptr[k] ),
        print( "]" )

        print( "ridx: [" ),
        for k in range( self.nnz ):
            print( self.ridx[k] ),
        print( "]" )

        print( "cidx: [" ),
        for k in range( self.nnz ):
            print( self.cidx[k] ),
        print( "]" )

        print( "data: [" ),
        for k in range( self.nnz ):
            print( self.data[k] ),
        print( "]" )


    def __dealloc__( self ):
        if( self.data_alloc ):
            free( self.data )
        free( self.rptr )
        free( self.ridx )
        free( self.cidx )


    cdef void setDataPtr( self, void *ptr ):
        if( self.data_alloc ):
            free( self.data )
            self.data_alloc = False

        self.data = <doublereal *> ptr


    cdef void copyFortranIdxs( self, integer* ridx, integer* cidx,
                               integer roffset=0, integer coffset=0 ):
        memcpy( ridx, self.ridx, self.nnz * sizeof( integer ) )
        memcpy( cidx, self.cidx, self.nnz * sizeof( integer ) )

        for k in range( self.nnz ): ## have to add one because Fortran
            ridx[k] += roffset + 1
            cidx[k] += coffset + 1


    cdef void copyData( self, doublereal* data ):
        memcpy( data, self.data, self.nnz * sizeof( doublereal ) )


    cdef doublereal get_elem_at( self, integer row, integer col ):
        for k in range( self.rptr[row], self.rptr[row+1] ):
            if( self.cidx[k] == col ):
                return self.data[k]
        return 0


    cdef int set_elem_at( self, integer row, integer col, doublereal val ):
        for k in range( self.rptr[row], self.rptr[row+1] ):
            if( self.cidx[k] == col ):
                self.data[k] = val
                return True
        return False


    cdef cnp.broadcast key_to_bcast( self, object key ):
        if( self.nrows > 1 ):
            if( isinstance( key, tuple ) and len(key) == 2 ):
                rowiter = key_to_array( key[0], self.nrows )
                coliter = key_to_array( key[1], self.ncols )
                if( ( isinstance( key[0], slice ) and isinstance( key[1], slice ) ) or
                    ( isinstance( key[0], slice ) and coliter.size > 1 ) or
                    ( isinstance( key[1], slice ) and rowiter.size > 1 ) ): ## slices form meshes
                    ( rowiter, coliter ) = np.ix_( rowiter, coliter )
            else:
                rowiter = key_to_array( key, self.nrows )
                coliter = key_to_array( slice( None, None, None ), self.ncols )
                if( rowiter.size > 1 ):  ## slices form meshes
                    ( rowiter, coliter ) = np.ix_( rowiter, coliter )

        elif( self.nrows == 1 ):
            rowiter = np.array( 0 )
            coliter = key_to_array( key, self.ncols )
        else:
            raise TypeError( "key cannot be applied to this sMatrix" )

        ## here is where all the magic happens to figure out new dimensions
        return np.broadcast( rowiter, coliter )


    def __setitem__( self, key, value ):
        try:
            value = np.asarray( value, dtype=np.float64 )
        except:
            raise TypeError( "value cannot be converted into a float array" )

        bcast = self.key_to_bcast( key )
        origshape = value.shape

        ## this algorithm follow the rules listed here:
        ## http://docs.scipy.org/doc/numpy/reference/ufuncs.html#broadcasting

        ## shave extra dimensions we can't deal with
        while( len( bcast.shape ) < len( value.shape ) ):
            if( value.shape[0] > 1 ):
                raise ValueError( "could not broadcast value array from shape " +
                                  "{0} into shape {1}".format( origshape, bcast.shape ) )
            value = np.squeeze( value, axis=0 )

        ## add size 1 dimensions to the left
        while( len( bcast.shape ) > len( value.shape ) ):
            value = value[np.newaxis]

        ## now try to match different dimensions
        for k in range( len( value.shape ) ):
            if( bcast.shape[-k-1] != value.shape[-k-1] ):
                if( value.shape[-k-1] != 1 ):
                    raise ValueError( "could not broadcast value array from shape " +
                                      "{0} into shape {1}".format( origshape, bcast.shape ) )
                value = np.tile( value, (bcast.shape[-k-1],) + (1,) * k )

        ## finally set values, assuming (!) bcast is C-ordered
        for ( (row,col), val ) in zip( bcast, np.nditer( value, order='C' ) ):
            self.set_elem_at( row, col, val )


    def __getitem__( self, key ):
        bcast = self.key_to_bcast( key )

        ## cool trick copied from
        ## http://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast.html
        out = np.empty( bcast.shape )
        out.flat = [ self.get_elem_at( row, col ) for (row,col) in bcast ]

        return out


    def __str__( self ):
        return str( self[:] )


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
        if( self.retval == 1000 ):
            return "Return information is not defined yet"

        if( self.retval < 0 ):
            return "Execution terminated by user defined function (should not occur)"
        elif( self.retval >= 143 ):
            return "Invalid value"
        else:
            return statusInfo[ int( self.retval / 10 ) ]


## SNOPT's option strings
cdef char* STR_NONDERIVATIVE_LINESEARCH = "Nonderivative linesearch"
cdef char* STR_DIFFERENCE_INTERVAL = "Difference interval"
cdef char* STR_FUNCTION_PRECISION = "Function precision"
cdef char* STR_MAJOR_FEASIBILITY_TOLERANCE = "Major feasibility tolerance"
cdef char* STR_MINOR_FEASIBILITY_TOLERANCE = "Minor feasibility tolerance"
cdef char* STR_HESSIAN_FULL_MEMORY = "Hessian full memory"
cdef char* STR_HESSIAN_UPDATES = "Hessian updates"
cdef char* STR_INFINITE_BOUND = "Infinite bound"
cdef char* STR_ITERATIONS_LIMIT = "Iterations limit"
cdef char* STR_MAJOR_ITERATIONS_LIMIT = "Major iterations limit"
cdef char* STR_MINOR_ITERATIONS_LIMIT = "Minor iterations limit"
cdef char* STR_LINESEARCH_TOLERANCE = "Linesearch tolerance"
cdef char* STR_MAJOR_OPTIMALITY_TOLERANCE = "Major optimality tolerance"
cdef char* STR_MAJOR_PRINT_LEVEL = "Major print level"
cdef char* STR_MINOR_PRINT_LEVEL = "Minor print level"
cdef char* STR_PIVOT_TOLERANCE = "Pivot tolerance"
cdef char* STR_QPSOLVER_CHOLESKY = "QPSolver Cholesky"
cdef char* STR_QPSOLVER_CG = "QPSolver CG"
cdef char* STR_QPSOLVER_QN = "QPSolver QN"
cdef char* STR_SCALE_OPTION = "Scale option"
cdef char* STR_SCALE_PRINT = "Scale print"
cdef char* STR_SOLUTION_NO = "Solution No"
cdef char* STR_SUPPRESS_PARAMETERS = "Suppress parameters"
cdef char* STR_TOTAL_CHARACTER_WORKSPACE = "Total character workspace"
cdef char* STR_TOTAL_INTEGER_WORKSPACE = "Total integer workspace"
cdef char* STR_TOTAL_REAL_WORKSPACE = "Total real workspace"
cdef char* STR_VERIFY_LEVEL = "Verify level"
cdef char* STR_VIOLATION_LIMIT = "Violation limit"

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


## helper static function usrfun used to evaluate user defined functions in Solver.prob
cdef object extprob
cdef sMatrix objGsparse
cdef sMatrix consGsparse

cdef int usrfun( integer *status, integer *n, doublereal *x,
                 integer *needF, integer *nF, doublereal *f,
                 integer *needG, integer *lenG, doublereal *G,
                 char *cu, integer *lencu,
                 integer *iu, integer *leniu,
                 doublereal *ru, integer *lenru ):
    ## FYI: Dual variables are at rw[ iw[329] ] onward, pg.25

    if( status[0] >= 2 ): ## Final call, do nothing
        return 0

    xarr = utils.wrap1dPtr( x, n[0], utils.doublereal_type )

    if( needF[0] > 0 ):
        ## we zero out all arrays in case the user does not modify all the values,
        ## e.g., in sparse problems.
        memset( f, 0, nF[0] * sizeof( doublereal ) )
        farr = utils.wrap1dPtr( f, nF[0], utils.doublereal_type )
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
        if( extprob.Ncons > 0 and consGsparse.nnz > 0 ):
            consGsparse.setDataPtr( &G[objGsparse.nnz] )
            extprob.consg( consGsparse, xarr )


cdef class Solver( base.Solver ):
    cdef integer nF[1]
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

    cdef int warm_start
    cdef int mem_alloc
    cdef integer mem_size[4]
    cdef int mem_alloc_ws
    cdef integer mem_size_ws[3]

    cdef float default_tol
    cdef float default_fctn_prec
    cdef float default_feas_tol
    cdef int default_iter_limit
    cdef int default_maj_iter_limit
    cdef int default_min_iter_limit
    cdef float default_violation_limit

    def __init__( self, prob=None ):
        super().__init__()

        self.mem_alloc = False
        self.mem_alloc_ws = False
        memset( self.mem_size, 0, 4 * sizeof( integer ) ) ## Set mem_size to zero
        memset( self.mem_size_ws, 0, 3 * sizeof( integer ) ) ## Set mem_size_ws to zero
        self.default_tol = np.sqrt( np.spacing(1) ) ## "Difference interval", pg. 71
        self.default_fctn_prec = np.power( np.spacing(1), 2.0/3.0 ) ## pg. 72, there is a typo there
        self.default_feas_tol = 1.0e-6 ## pg. 76
        self.default_min_iter_limit = 500 ## pg. 78
        self.default_violation_limit = 10 ## pg. 85
        self.prob = None

        if( prob ):
            self.setupProblem( prob )

        ## Set options
        self.printOpts[ "summaryFile" ] = "stdout"
        self.printOpts[ "printLevel" ] = 0
        self.printOpts[ "minorPrintLevel" ] = 0
        self.solveOpts[ "derivLinesearch" ] = True
        self.solveOpts[ "diffInterval" ] = self.default_tol
        self.solveOpts[ "fctnPrecision" ] = self.default_fctn_prec
        self.solveOpts[ "majorFeasibilityTol" ] = self.default_feas_tol
        self.solveOpts[ "minorFeasibilityTol" ] = self.default_feas_tol
        self.solveOpts[ "forceFullHessian" ] = False
        self.solveOpts[ "bfgsResetFreq" ] = 10 ## pg. 74
        self.solveOpts[ "infValue" ] = 1.0e20 ## pg. 74
        self.solveOpts[ "iterLimit" ] = self.default_iter_limit ## defined in setupProblem
        self.solveOpts[ "majorIterLimit" ] = self.default_maj_iter_limit ## defined in setupProblem
        self.solveOpts[ "minorIterLimit" ] = self.default_min_iter_limit
        self.solveOpts[ "lineSearchTol" ] = 0.9
        self.solveOpts[ "majorOptimalityTol" ] = self.default_feas_tol
        self.solveOpts[ "pivotTol" ] = self.default_fctn_prec
        self.solveOpts[ "qpSolver" ] = "Cholesky"
        self.solveOpts[ "disableScaling" ] = False
        self.solveOpts[ "printScaling" ] = False
        self.solveOpts[ "verifyGrad" ] = False
        self.solveOpts[ "violationLimit" ] = self.default_violation_limit


    def setupProblem( self, prob ):
        cdef sMatrix Asparse

        global extprob
        global objGsparse
        global consGsparse

        if( not isinstance( prob, nlp.Problem ) ):
            raise TypeError( "Argument prob must be of type nlp.Problem" )

        self.prob = prob ## Save a copy of prob's pointer
        extprob = prob ## Save another (global) copy of prob's pointer to use in usrfun

        ## New problems cannot be warm started
        self.Start[0] = 0 ## Cold start

        ## Set nF
        self.nF[0] = 1 + prob.Nconslin + prob.Ncons

        ## Create Asparse, set lenA
        if( prob.objmixedA is not None ):
            tmplist = ( prob.objmixedA, )
        else:
            tmplist = ( np.zeros( ( 1, prob.N ) ), )
        if( prob.Nconslin > 0 ):
            tmplist += ( prob.conslinA, )
        if( prob.consmixedA is not None ):
            tmplist += ( prob.consmixedA, )
        else:
            tmplist += ( np.zeros( ( prob.Ncons, prob.N ) ), )
        Asparse = sMatrix( np.vstack( tmplist ), copy_data=True )
        self.lenA[0] = Asparse.nnz
        if( self.lenA[0] > 0 ):
            self.neA[0] = self.lenA[0]
        else: ## Minimum allowed values, pg. 16
            self.lenA[0] = 1
            self.neA[0] = 0

        ## Create objGsparse and consGsparse, set lenG
        if( isinstance( prob, nlp.SparseProblem ) and prob.objgpattern is not None ):
            objGsparse = sMatrix( prob.objgpattern )
        else:
            objGsparse = sMatrix( np.ones( ( 1, prob.N ) ) )

        if( isinstance( prob, nlp.SparseProblem ) and prob.consgpattern is not None ):
            consGsparse = sMatrix( prob.consgpattern )
        else:
            consGsparse = sMatrix( np.ones( ( prob.Ncons, prob.N ) ) )
        self.lenG[0] = objGsparse.nnz + consGsparse.nnz
        self.neG[0] = self.lenG[0]

        ## I'm guessing the definition of m in pgs. 74,76
        self.default_iter_limit = max( 1000, 20*( prob.Ncons + prob.Nconslin ) )
        self.default_maj_iter_limit = max( 1000, prob.Ncons + prob.Nconslin )

        ## Allocate if necessary
        if( self.mustAllocate( prob.N, self.nF[0], self.lenA[0], self.lenG[0] ) ):
            self.deallocate()
            self.allocate()

        ## copy box constraints limits
        memcpy( self.xlow, utils.getPtr( utils.convFortran( prob.lb ) ),
                prob.N * sizeof( doublereal ) )
        memcpy( self.xupp, utils.getPtr( utils.convFortran( prob.ub ) ),
                prob.N * sizeof( doublereal ) )

        ## copy index data of G
        ## row 0 of G belongs to objg
        objGsparse.copyFortranIdxs( &self.iGfun[0], &self.jGvar[0] )
        ## rows 1:(1 + prob.Nconslin) of G are empty, these are pure linear constraints
        ## rows (1 + prob.Nconslin):(1 + prob.Nconslin + prob.Ncons) of G belong to consg
        consGsparse.copyFortranIdxs( &self.iGfun[objGsparse.nnz], &self.jGvar[objGsparse.nnz],
                                     roffset = 1 + prob.Nconslin )
        ## copy index data of A
        Asparse.copyFortranIdxs( &self.iAfun[0], &self.jAvar[0] )
        ## copy matrix data of A
        Asparse.copyData( &self.A[0] )

        ## copy general constraints limits
        ## objective function knows no limits (https://i.imgur.com/UuQbJ.gif)
        self.Flow[0] = -np.inf
        self.Fupp[0] = np.inf
        ## linear constraints limits
        if( prob.Nconslin > 0 ):
            memcpy( &self.Flow[1],
                    utils.getPtr( utils.convFortran( prob.conslinlb ) ),
                    prob.Nconslin * sizeof( doublereal ) )
            memcpy( &self.Fupp[1],
                    utils.getPtr( utils.convFortran( prob.conslinub ) ),
                    prob.Nconslin * sizeof( doublereal ) )
        ## nonlinear constraints limits
        if( prob.Ncons > 0 ):
            memcpy( &self.Flow[1 + prob.Nconslin],
                    utils.getPtr( utils.convFortran( prob.conslb ) ),
                    prob.Ncons * sizeof( doublereal ) )
            memcpy( &self.Fupp[1 + prob.Nconslin],
                    utils.getPtr( utils.convFortran( prob.consub ) ),
                    prob.Ncons * sizeof( doublereal ) )

        ## initialize other vectors with zeros
        memset( self.xstate, 0, self.prob.N * sizeof( integer ) )
        memset( self.Fstate, 0, self.nF[0] * sizeof( integer ) )
        memset( self.Fmul, 0, self.nF[0] * sizeof( doublereal ) )


    cdef int allocate( self ):
        if( self.mem_alloc ):
            return False

        self.x = <doublereal *> malloc( self.prob.N * sizeof( doublereal ) )
        self.xlow = <doublereal *> malloc( self.prob.N * sizeof( doublereal ) )
        self.xupp = <doublereal *> malloc( self.prob.N * sizeof( doublereal ) )
        self.xmul = <doublereal *> malloc( self.prob.N * sizeof( doublereal ) )
        self.xstate = <integer *> malloc( self.prob.N * sizeof( integer ) )
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

        if( self.x is NULL or
            self.xlow is NULL or
            self.xupp is NULL or
            self.xmul is NULL or
            self.xstate is NULL or
            self.F is NULL or
            self.Flow is NULL or
            self.Fupp is NULL or
            self.Fmul is NULL or
            self.Fstate is NULL or
            self.A is NULL or
            self.iAfun is NULL or
            self.jAvar is NULL or
            self.iGfun is NULL or
            self.jGvar is NULL ):
            raise MemoryError( "At least one memory allocation failed" )

        self.mem_alloc = True
        self.setMemSize( self.prob.N, self.nF[0], self.lenA[0], self.lenG[0] )

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

        if( self.iw is NULL or
            self.rw is NULL or
            self.cw is NULL ):
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


    cdef void debugMem( self ):
        print( ">>> Memory allocated for data: " +
               str( self.mem_size[0] * ( 4 * sizeof(doublereal) + sizeof(integer) ) +
                    self.mem_size[1] * ( 4 * sizeof(doublereal) + sizeof(integer) ) +
                    self.mem_size[2] * ( sizeof(doublereal) + 2*sizeof(integer) ) +
                    self.mem_size[3] * 2 * sizeof(integer) ) +
               " bytes." )

        print( ">>> Memory allocated for workspace: " +
               str( self.mem_size_ws[0] * 8 * sizeof(char) +
                    self.mem_size_ws[1] * sizeof(integer) +
                    self.mem_size_ws[2] * sizeof(doublereal) ) +
               " bytes." )


    def __dealloc__( self ):
        self.deallocateWS()
        self.deallocate()


    def warmStart( self ):
        if( not isinstance( self.prob.soln, Soln ) ):
            return False

        memcpy( self.xstate, utils.getPtr( utils.convIntFortran( self.prob.soln.xstate ) ),
                self.prob.N * sizeof( integer ) )
        memcpy( self.Fstate, utils.getPtr( utils.convIntFortran( self.prob.soln.Fstate ) ),
                self.nF[0] * sizeof( integer ) )

        self.Start[0] = 2
        return True


    def solve( self ):
        cdef integer nS[1]
        cdef integer nInf[1]
        cdef doublereal sInf[1]
        cdef integer mincw[1]
        cdef integer miniw[1]
        cdef integer minrw[1]
        cdef integer inform_out[1]
        cdef integer *ltmpcw = [ 500 ]
        cdef integer *ltmpiw = [ 500 ]
        cdef integer *ltmprw = [ 500 ]
        cdef char tmpcw[500*8]
        cdef integer tmpiw[500]
        cdef doublereal tmprw[500]

        cdef integer *n = [ self.prob.N ]
        cdef integer *nxname = [ 1 ] ## Do not provide vars names
        cdef integer *nFname = [ 1 ] ## Do not provide cons names
        cdef doublereal *ObjAdd = [ 0.0 ]
        cdef integer *ObjRow = [ 1 ]
        cdef char *probname = "optwrapp" ## Must have 8 characters
        cdef char *xnames = "dummy"
        cdef char *Fnames = "dummy"
        cdef bytes printFileTmp = self.printOpts[ "printFile" ].encode( "latin_1" )
        cdef char* printFile = printFileTmp
        cdef bytes summaryFileTmp = self.printOpts[ "summaryFile" ].encode( "latin_1" )
        cdef char* summaryFile = summaryFileTmp
        cdef integer* summaryFileUnit = [ 89 ] ## Hardcoded since nobody cares
        cdef integer* printFileUnit = [ 90 ] ## Hardcoded since nobody cares
        cdef integer* printLevel = [ self.printOpts[ "printLevel" ] ]
        cdef integer* minorPrintLevel = [ self.printOpts[ "minorPrintLevel" ] ]
        cdef doublereal* diffInterval = [ self.solveOpts[ "diffInterval" ] ]
        cdef doublereal* fctnPrecision = [ self.solveOpts[ "fctnPrecision" ] ]
        cdef doublereal* majorFeasibilityTol = [ self.solveOpts[ "majorFeasibilityTol" ] ]
        cdef doublereal* minorFeasibilityTol = [ self.solveOpts[ "minorFeasibilityTol" ] ]
        cdef integer* bfgsResetFreq = [ self.solveOpts[ "bfgsResetFreq" ] ]
        cdef doublereal* infValue = [ self.solveOpts[ "infValue" ] ]
        cdef integer* iterLimit = [ self.solveOpts[ "iterLimit" ] ]
        cdef integer* majorIterLimit = [ self.solveOpts[ "majorIterLimit" ] ]
        cdef integer* minorIterLimit = [ self.solveOpts[ "minorIterLimit" ] ]
        cdef doublereal* lineSearchTol = [ self.solveOpts[ "lineSearchTol" ] ]
        cdef doublereal* majorOptimalityTol = [ self.solveOpts[ "majorOptimalityTol" ] ]
        cdef doublereal* pivotTol = [ self.solveOpts[ "pivotTol" ] ]
        cdef doublereal* violationLimit = [ self.solveOpts[ "violationLimit" ] ]
        cdef integer* zero = [ 0 ]
        cdef integer verifyLevel[1]

        ## Begin by setting up initial condition
        tmpinit = utils.convFortran( self.prob.init )
        memcpy( self.x, utils.getPtr( tmpinit ),
                self.prob.N * sizeof( doublereal ) )

        ## Handle debug files
        if( self.printOpts[ "printFile" ] == "" ):
            printFileUnit[0] = 0
        # else:
        #     fh.openfile_( printFileUnit, printFile, inform_out,
        #                   len( self.printOpts[ "printFile" ] ) )
        #     if( inform_out[0] != 0 ):
        #         raise IOError( "Could not open file " + self.printOpts[ "printFile" ] )

        if( self.printOpts[ "summaryFile" ] == "stdout" ):
            summaryFileUnit[0] = 6 ## Fortran's magic value for stdout
        elif( self.printOpts[ "summaryFile" ] == "" ):
            summaryFileUnit[0] = 0 ## Disable, pg. 6
        # else:
        #     fh.openfile_( summaryFileUnit, summaryFile, inform_out,
        #                   len( self.printOpts[ "summaryFile" ] ) )
        #     if( inform_out[0] != 0 ):
        #         raise IOError( "Could not open file " + self.printOpts[ "summaryFile" ] )

        ## Initialize
        snopt.sninit_( printFileUnit, summaryFileUnit,
                       tmpcw, ltmpcw, tmpiw, ltmpiw, tmprw, ltmprw,
                       ltmpcw[0]*8 )

        inform_out[0] = 0 ## Reset inform_out before running snset* functions
        ## The following two settings change the outcome of snmema, pg. 29
        ## Force full hessian
        if( self.solveOpts[ "forceFullHessian" ] ):
            snopt.snset_( STR_HESSIAN_FULL_MEMORY,
                          printFileUnit, summaryFileUnit, inform_out,
                          tmpcw, ltmpcw, tmpiw, ltmpiw, tmprw, ltmprw,
                          len( STR_HESSIAN_FULL_MEMORY ), ltmpcw[0]*8 )

        ## Set BFGS reset frequency
        snopt.snseti_( STR_HESSIAN_UPDATES, bfgsResetFreq,
                       printFileUnit, summaryFileUnit, inform_out,
                       tmpcw, ltmpcw, tmpiw, ltmpiw, tmprw, ltmprw,
                       len( STR_HESSIAN_UPDATES ), ltmpcw[0]*8 )

        ## Estimate workspace memory requirements
        snopt.snmema_( inform_out, self.nF, n, nxname, nFname, self.lenA, self.lenG,
                       self.lencw, self.leniw, self.lenrw,
                       tmpcw, ltmpcw, tmpiw, ltmpiw, tmprw, ltmprw,
                       ltmpcw[0]*8 )
        if( inform_out[0] != 104 ):
            raise Exception( "snopt.snMemA failed to estimate workspace memory requirements" )

        if( self.mustAllocateWS( self.lencw[0], self.leniw[0], self.lenrw[0] ) ):
            self.deallocateWS()
            self.allocateWS()

        ## Copy content of temp workspace arrays to malloc'ed workspace arrays
        memcpy( self.cw, tmpcw, ltmpcw[0] * sizeof( char ) )
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

        ## Suppress parameters
        snopt.snset_( STR_SUPPRESS_PARAMETERS,
                      printFileUnit, summaryFileUnit, inform_out,
                      self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                      len( STR_SUPPRESS_PARAMETERS ), self.lencw[0]*8 )


        ## Set major print level
        snopt.snseti_( STR_MAJOR_PRINT_LEVEL, printLevel,
                       printFileUnit, summaryFileUnit, inform_out,
                       self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                       len( STR_MAJOR_PRINT_LEVEL ), self.lencw[0]*8 )

        ## Set minor print level
        snopt.snseti_( STR_MINOR_PRINT_LEVEL, minorPrintLevel,
                       printFileUnit, summaryFileUnit, inform_out,
                       self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                       len( STR_MINOR_PRINT_LEVEL ), self.lencw[0]*8 )

        ## Disable derivative linesearch is necessary
        if( not self.solveOpts[ "derivLinesearch" ] ):
            snopt.snset_( STR_NONDERIVATIVE_LINESEARCH,
                          printFileUnit, summaryFileUnit, inform_out,
                          self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                          len( STR_NONDERIVATIVE_LINESEARCH ), self.lencw[0]*8 )

        ## Set difference interval if necessary
        if( self.solveOpts[ "diffInterval" ] > self.default_tol ):
            snopt.snsetr_( STR_DIFFERENCE_INTERVAL, diffInterval,
                           printFileUnit, summaryFileUnit, inform_out,
                           self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                           len( STR_DIFFERENCE_INTERVAL ), self.lencw[0]*8 )

        ## Set functino precision if necessary
        if( self.solveOpts[ "fctnPrecision" ] > self.default_fctn_prec ):
            snopt.snsetr_( STR_FUNCTION_PRECISION, fctnPrecision,
                           printFileUnit, summaryFileUnit, inform_out,
                           self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                           len( STR_FUNCTION_PRECISION ), self.lencw[0]*8 )

        ## Set major feasibility tolerance if necessary
        if( self.solveOpts[ "majorFeasibilityTol" ] > self.default_feas_tol ):
            snopt.snsetr_( STR_MAJOR_FEASIBILITY_TOLERANCE, majorFeasibilityTol,
                           printFileUnit, summaryFileUnit, inform_out,
                           self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                           len( STR_MAJOR_FEASIBILITY_TOLERANCE ), self.lencw[0]*8 )

        ## Set minor feasibility tolerance if necessary
        if( self.solveOpts[ "minorFeasibilityTol" ] > self.default_feas_tol ):
            snopt.snsetr_( STR_MINOR_FEASIBILITY_TOLERANCE, minorFeasibilityTol,
                           printFileUnit, summaryFileUnit, inform_out,
                           self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                           len( STR_MINOR_FEASIBILITY_TOLERANCE ), self.lencw[0]*8 )

        ## Set infinity value
        snopt.snsetr_( STR_INFINITE_BOUND, infValue,
                       printFileUnit, summaryFileUnit, inform_out,
                       self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                       len( STR_INFINITE_BOUND ), self.lencw[0]*8 )

        ## Set iterations limit if necessary
        if( self.solveOpts[ "iterLimit" ] > self.default_iter_limit ):
            snopt.snseti_( STR_ITERATIONS_LIMIT, iterLimit,
                           printFileUnit, summaryFileUnit, inform_out,
                           self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                           len( STR_ITERATIONS_LIMIT ), self.lencw[0]*8 )

        ## Set major iterations limit if necessary
        if( self.solveOpts[ "majorIterLimit" ] > self.default_maj_iter_limit ):
            snopt.snseti_( STR_MAJOR_ITERATIONS_LIMIT, majorIterLimit,
                           printFileUnit, summaryFileUnit, inform_out,
                           self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                           len( STR_MAJOR_ITERATIONS_LIMIT ), self.lencw[0]*8 )

        ## Set minor iterations limit if necessary
        if( self.solveOpts[ "minorIterLimit" ] > self.default_min_iter_limit ):
            snopt.snseti_( STR_MINOR_ITERATIONS_LIMIT, minorIterLimit,
                           printFileUnit, summaryFileUnit, inform_out,
                           self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                           len( STR_MINOR_ITERATIONS_LIMIT ), self.lencw[0]*8 )

        ## Set line search tolerance
        snopt.snsetr_( STR_LINESEARCH_TOLERANCE, lineSearchTol,
                       printFileUnit, summaryFileUnit, inform_out,
                       self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                       len( STR_LINESEARCH_TOLERANCE ), self.lencw[0]*8 )

        ## Set major optimality tolerance if necessary
        if( self.solveOpts[ "majorOptimalityTol" ] > self.default_feas_tol ):
            snopt.snsetr_( STR_MAJOR_OPTIMALITY_TOLERANCE, majorOptimalityTol,
                           printFileUnit, summaryFileUnit, inform_out,
                           self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                           len( STR_MAJOR_OPTIMALITY_TOLERANCE ), self.lencw[0]*8 )

        ## Set pivot tolerance
        if( self.solveOpts[ "pivotTol" ] > self.default_fctn_prec ):
            snopt.snsetr_( STR_PIVOT_TOLERANCE, pivotTol,
                           printFileUnit, summaryFileUnit, inform_out,
                           self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                           len( STR_PIVOT_TOLERANCE ), self.lencw[0]*8 )

        ## Set QP solver
        if( self.solveOpts[ "qpSolver" ].lower() == "cholesky" ):
            snopt.snset_( STR_QPSOLVER_CHOLESKY,
                          printFileUnit, summaryFileUnit, inform_out,
                          self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                          len( STR_QPSOLVER_CHOLESKY ), self.lencw[0]*8 )
        elif( self.solveOpts[ "qpSolver" ].lower() == "cg" ):
            snopt.snset_( STR_QPSOLVER_CG,
                          printFileUnit, summaryFileUnit, inform_out,
                          self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                          len( STR_QPSOLVER_CG ), self.lencw[0]*8 )
        elif( self.solveOpts[ "qpSolver" ].lower() == "qn" ):
            snopt.snset_( STR_QPSOLVER_QN,
                          printFileUnit, summaryFileUnit, inform_out,
                          self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                          len( STR_QPSOLVER_QN ), self.lencw[0]*8 )

        ## Scaling option and print
        if( self.solveOpts[ "disableScaling" ] ):
            snopt.snseti_( STR_SCALE_OPTION, zero,
                          printFileUnit, summaryFileUnit, inform_out,
                          self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                          len( STR_SCALE_OPTION ), self.lencw[0]*8 )
        if( self.solveOpts[ "printScaling" ] ):
            snopt.snset_( STR_SCALE_PRINT,
                          printFileUnit, summaryFileUnit, inform_out,
                          self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                          len( STR_SCALE_PRINT ), self.lencw[0]*8 )

        ## Do not print solution to print file
        snopt.snset_( STR_SOLUTION_NO,
                      printFileUnit, summaryFileUnit, inform_out,
                      self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                      len( STR_SOLUTION_NO ), self.lencw[0]*8 )

        ## Set verify level
        if( self.solveOpts[ "verifyGrad" ] ):
            verifyLevel[0] = 3 ## Check gradients with two algorithms
        else:
            verifyLevel[0] = -1 ## Disabled
        snopt.snseti_( STR_VERIFY_LEVEL, verifyLevel,
                       printFileUnit, summaryFileUnit, inform_out,
                       self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                       len( STR_VERIFY_LEVEL ), self.lencw[0]*8 )

        ## Set constraint violation limit
        if( self.solveOpts[ "violationLimit" ] > self.default_violation_limit ):
            snopt.snsetr_( STR_VIOLATION_LIMIT, violationLimit,
                           printFileUnit, summaryFileUnit, inform_out,
                           self.cw, self.lencw, self.iw, self.leniw, self.rw, self.lenrw,
                           len( STR_VIOLATION_LIMIT ), self.lencw[0]*8 )

        if( inform_out[0] != 0 ):
            raise Exception( "At least one option setting failed" )

        if( self.debug ):
            self.debugMem()

            if( isinstance( self.prob, nlp.SparseProblem ) ):
                if( self.prob.Nconslin > 0 ):
                    print( ">>> Sparsity of A: %.1f" %
                           ( self.lenA[0] * 100 / ( self.prob.N * self.prob.Nconslin ) ) + "%" )
                if( self.prob.Ncons > 0 ):
                    print( ">>> Sparsity of gradient: %.1f" %
                           ( self.lenG[0] * 100 / ( self.prob.N * (1 + self.prob.Ncons ) ) ) + "%" )

        ## Execute SNOPT
        snopt.snopta_( self.Start, self.nF,
                       n, nxname, nFname,
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
        if( self.printOpts[ "printFile" ] != "" ):
            try:
                os.rename( "fort.{0}".format( printFileUnit[0] ),
                           self.printOpts[ "printFile" ] )
            except FileNotFoundError:
                pass

        if( self.printOpts[ "summaryFile" ] != "" and
            self.printOpts[ "summaryFile" ] != "stdout" ):
            try:
                os.rename( "fort.{0}".format( summaryFileUnit[0] ),
                           self.printOpts[ "summaryFile" ] )
            except FileNotFoundError:
                pass

        ## Reset warm start
        self.Start[0] = 0

        ## Save result to prob
        self.prob.soln = Soln()
        self.prob.soln.value = float( self.F[0] )
        self.prob.soln.final = np.copy( utils.wrap1dPtr( self.x, self.prob.N,
                                                         utils.doublereal_type ) )
        self.prob.soln.xstate = np.copy( utils.wrap1dPtr( self.xstate, self.prob.N,
                                                          utils.integer_type ) )
        self.prob.soln.xmul = np.copy( utils.wrap1dPtr( self.xmul, self.prob.N,
                                                        utils.doublereal_type ) )
        self.prob.soln.Fstate = np.copy( utils.wrap1dPtr( self.Fstate, self.nF[0],
                                                          utils.integer_type ) )
        self.prob.soln.Fmul = np.copy( utils.wrap1dPtr( self.Fmul, self.nF[0],
                                                        utils.doublereal_type ) )
        self.prob.soln.nS = int( nS[0] )
        self.prob.soln.retval = int( inform_out[0] )

        return( self.prob.soln.final,
                self.prob.soln.value,
                self.prob.soln.retval )
