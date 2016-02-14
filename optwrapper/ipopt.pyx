# cython: boundscheck=False
# cython: wraparound=False

from libc.string cimport memcpy, memset
from libc.stdlib cimport malloc, free
from libc.stdint cimport int32_t, int64_t
cimport numpy as cnp
import numpy as np
import os

from .ipstdcinterfaceh cimport Number, Index, Int, Bool, UserDataPtr
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
cdef utils.sMatrix Asparse
cdef utils.sMatrix consGsparse
cdef utils.sMatrix consmixedAsparse

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

    if( extprob.Nconslin > 0 ):
        garrlin = utils.wrap1dPtr( &g[0], extprob.Nconslin, Number_type )
        garrlin[:] = Asparse.dot( xarr )

    if( extprob.Ncons > 0 ):
        garr = utils.wrap1dPtr( &g[extprob.Nconslin], extprob.Ncons, Number_type )
        extprob.consf( garr, xarr )
        if( extprob.consmixedA is not None ):
            garr += consmixedAsparse.dot( xarr )

    return True

cdef Bool eval_consg( Index n, Number *x, Bool new_x,
                      Index m, Index nele_jac,
                      Index *iRow, Index *jCol, Number *values,
                      UserDataPtr user_data ):
    if( values == NULL ):
        if( sizeof( Index ) == 4 ):
            if( Asparse.nnz > 0 ):
                Asparse.copyIdxs32( <int32_t *> &iRow[0],
                                    <int32_t *> &jCol[0] )
            if( consGsparse.nnz > 0 ):
                consGsparse.copyIdxs32( <int32_t *> &iRow[Asparse.nnz],
                                        <int32_t *> &jCol[Asparse.nnz],
                                        roffset = extprob.Nconslin )
        else:
            if( Asparse.nnz > 0 ):
                Asparse.copyIdxs( <int64_t *> &iRow[0],
                                  <int64_t *> &jCol[0] )
            if( consGsparse.nnz > 0 ):
                consGsparse.copyIdxs( <int64_t *> &iRow[Asparse.nnz],
                                      <int64_t *> &jCol[Asparse.nnz],
                                      roffset = extprob.Nconslin )
    else:
        xarr = utils.wrap1dPtr( x, n, Number_type )
        if( Asparse.nnz > 0 ):
            memcpy( &values[0], Asparse.data, Asparse.nnz * sizeof( Number ) )

        if( consGsparse.nnz > 0 ):
            memset( &values[Asparse.nnz], 0, consGsparse.nnz * sizeof( Number ) )
            consGsparse.setDataPtr( &values[Asparse.nnz] )
            extprob.consg( consGsparse, xarr )
            if( consmixedAsparse.nnz > 0 ):
                consGsparse.add_sparse( consmixedAsparse )

    return True

cdef Bool eval_lagrangianh( Index n, const Number* x, Bool new_x,
                            Number obj_factor, Index m, const Number* lambda_,
                            Bool new_lambda, Index nele_hess, Index* iRow,
                            Index* jCol, Number* values ):
    return False


cdef class Soln( base.Soln ):
    cdef public cnp.ndarray mult_x_L
    cdef public cnp.ndarray mult_x_U
    cdef public cnp.ndarray g
    cdef public cnp.ndarray mult_g

    def __init__( self ):
        super().__init__()
        self.retval = 100

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
                                 -199: "Internal Error",
                                 100: "Return information undefined" }

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
    cdef ipopt.IpoptProblem nlp
    cdef ipopt.ApplicationReturnStatus status

    cdef int N
    cdef int Ntotcons
    cdef int warm_start
    cdef int mem_alloc
    cdef int nlp_alloc
    cdef int mem_size[2] ## { N, Ntotcons }


    def __init__( self, prob=None ):
        super().__init__()

        self.mem_alloc = False
        self.nlp_alloc = False
        self.mem_size[0] = self.mem_size[1] = 0
        self.prob = None

        self.options = utils.Options( { "printFile": "output_file" } ) ## legacy_label: real_label
        self.options[ "hessian_approximation" ] = "limited-memory"

        if( prob ):
            self.setupProblem( prob )


    def setupProblem( self, prob ):
        global extprob
        global Asparse
        global consGsparse
        global consmixedAsparse
        cdef cnp.ndarray tmparr

        if( not isinstance( prob, nlp.Problem ) ):
            raise TypeError( "Argument 'prob' must be of type 'nlp.Problem'" )

        self.prob = prob ## Save a copy of prob's pointer
        extprob = prob

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
                    prob.Ncons * sizeof( Number ) )

            tmparr = utils.arraySanitize( prob.consub, dtype=Number_dtype )
            memcpy( &self.g_U[prob.Nconslin], utils.getPtr( tmparr ),
                    prob.Ncons * sizeof( Number ) )

        Asparse = utils.sMatrix( prob.conslinA, copy_data=True )

        consmixedAsparse = utils.sMatrix( prob.consmixedA, copy_data=True )

        if( isinstance( prob, nlp.SparseProblem ) and prob.consgpattern is not None ):
            if( prob.consmixedA is None ):
                consGsparse = utils.sMatrix( prob.consgpattern )
            else:
                consGsparse = utils.sMatrix( np.logical_or( prob.consgpattern,
                                                            prob.consmixedA ) )
        else:
            consGsparse = utils.sMatrix( np.ones( ( prob.Ncons, self.N ) ) )

        if( self.nlp_alloc ):
            ipopt.FreeIpoptProblem( self.nlp )

        self.nlp = ipopt.CreateIpoptProblem( self.N, self.x_L, self.x_U,
                                             self.Ntotcons, self.g_L, self.g_U,
                                             Asparse.nnz + consGsparse.nnz, 0, 0,
                                             <ipopt.Eval_F_CB> eval_objf,
                                             <ipopt.Eval_G_CB> eval_consf,
                                             <ipopt.Eval_Grad_F_CB> eval_objg,
                                             <ipopt.Eval_Jac_G_CB> eval_consg,
                                             <ipopt.Eval_H_CB> eval_lagrangianh )
        self.nlp_alloc = True


    def initPoint( self, init ):
        if( not self.mem_alloc ):
            raise ValueError( "Internal memory has not been allocated" )

        tmparr = utils.arraySanitize( init, dtype=Number_dtype )
        memcpy( self.x, utils.getPtr( tmparr ), self.N * sizeof( Number ) )

        return True


    cdef int allocate( self ):
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


    cdef int deallocate( self ):
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
        if( self.nlp_alloc ):
            ipopt.FreeIpoptProblem( self.nlp )

        self.deallocate()


    def warmStart( self ):
        if( not isinstance( self.prob.soln, Soln ) ):
            return False

        self.initPoint( self.prob.soln.final )

        tmparr = utils.arraySanitize( self.prob.soln.mult_g, dtype=Number_dtype )
        memcpy( self.mult_g, utils.getPtr( tmparr ), self.Ntotcons * sizeof( Number ) )

        tmparr = utils.arraySanitize( self.prob.soln.mult_x_L, dtype=Number_dtype )
        memcpy( self.mult_x_L, utils.getPtr( tmparr ), self.N * sizeof( Number ) )

        tmparr = utils.arraySanitize( self.prob.soln.mult_x_U, dtype=Number_dtype )
        memcpy( self.mult_x_U, utils.getPtr( tmparr ), self.N * sizeof( Number ) )

        self.warm_start = True
        self.options[ "warm_start_init_point" ] = "yes"

        return True


    cdef int processOptions( self ):
        cdef bytes tmpkey
        cdef bytes tmpval
        cdef Bool ret, out
        if( not self.nlp_alloc ):
            return False

        out = True
        for key in self.options:
            ret = False
            tmpkey = key.encode( "ascii" )
            if( self.options[key].dtype == utils.INT ):
                ret = ipopt.AddIpoptIntOption( self.nlp, tmpkey, self.options[key].value )
            elif( self.options[key].dtype == utils.DOUBLE ):
                ret = ipopt.AddIpoptNumOption( self.nlp, tmpkey, self.options[key].value )
            elif( self.options[key].dtype == utils.STR ):
                tmpval = self.options[key].value.encode( "ascii" )
                ret = ipopt.AddIpoptStrOption( self.nlp, tmpkey, tmpval )

            if( not ret ):
                raise TypeError( "Could not process option " +
                                 "'{0}: {1:r}'".format( key, self.options[key] ) )
            out = out and ret

        return out


    def solve( self ):
        cdef cnp.ndarray tmparr
        cdef Number obj_val[1]

        self.processOptions()

        ## unless output_file is stdout, we redirect stdout to /dev/null
        if( self.options[ "output_file" ] != "stdout" ):
            old_stdout = os.dup(1)
            os.close(1)
            os.open( os.devnull, os.O_WRONLY )

        ## Call Ipopt
        status = ipopt.IpoptSolve( self.nlp, self.x, self.g, obj_val, self.mult_g,
                                   self.mult_x_L, self.mult_x_U, NULL )

        ## undo redirection of stdout to /dev/null
        if( self.options[ "output_file" ] != "stdout" ):
            os.close(1)
            os.dup( old_stdout ) # should dup to 1
            os.close( old_stdout ) # get rid of left overs

        ## disable warm start after execution
        self.warm_start = False
        del self.options[ "warm_start_init_point" ]

        ## Save result to prob
        self.prob.soln = Soln()
        self.prob.soln.value = float( obj_val[0] )
        self.prob.soln.final = np.copy( utils.wrap1dPtr( self.x, self.N, Number_type ) )
        self.prob.soln.g = np.copy( utils.wrap1dPtr( self.g, self.Ntotcons, Number_type ) )
        self.prob.soln.mult_g = np.copy( utils.wrap1dPtr( self.mult_g, self.Ntotcons,
                                                          Number_type ) )
        self.prob.soln.mult_x_L = np.copy( utils.wrap1dPtr( self.mult_x_L, self.Ntotcons,
                                                            Number_type ) )
        self.prob.soln.mult_x_U = np.copy( utils.wrap1dPtr( self.mult_x_U, self.Ntotcons,
                                                            Number_type ) )
        self.prob.soln.retval = int( status )

        return( self.prob.soln.final,
                self.prob.soln.value,
                self.prob.soln.retval )
