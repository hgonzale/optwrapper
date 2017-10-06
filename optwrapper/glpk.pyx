import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
# from cython.operator cimport dereference as deref

cimport glpkh as glpk
from utils cimport getPtr, arraySanitize, Options, sMatrix
cimport base
import lp

cdef type double_dtype = np.float64

cdef class Soln( base.Soln ):
    cdef public cnp.ndarray dual

    def __init__( self ):
        super().__init__()
        self.retval = -100

    def getStatus( self ):
        cdef dict statusInfo = {
            glpk.GLP_EBADB: "Invalid basis",
            glpk.GLP_ESING: "Singular matrix",
            glpk.GLP_ECOND: "Ill-conditioned matrix",
            glpk.GLP_EBOUND: "Invalid bounds",
            glpk.GLP_EFAIL: "Solver failed",
            glpk.GLP_EOBJLL: "Objective lower limit reached",
            glpk.GLP_EOBJUL: "Objective upper limit reached",
            glpk.GLP_EITLIM: "Iteration limit exceeded",
            glpk.GLP_ETMLIM: "Time limit exceeded",
            glpk.GLP_ENOPFS: "No primal feasible solution",
            glpk.GLP_ENODFS: "No dual feasible solution",
            glpk.GLP_EROOT: "Root LP optimum not provided",
            glpk.GLP_ESTOP: "Search terminated by application",
            glpk.GLP_EMIPGAP: "Relative MIP gap tolerance reached",
            glpk.GLP_ENOFEAS: "No primal/dual feasible solution",
            glpk.GLP_ENOCVG: "No convergence",
            glpk.GLP_EINSTAB: "Numerical instability",
            glpk.GLP_EDATA: "Invalid data",
            glpk.GLP_ERANGE: "Result out of range"
        }

        if( self.retval not in statusInfo ):
            return "Undefined return information"

        return statusInfo[ self.retval ]



cdef class Solver( base.Solver ):
    cdef glpk.glp_prob *ptr
    cdef int warm_start
    cdef int mem_alloc
    cdef int mem_size
    cdef int nlp_alloc
    cdef int *Arows
    cdef int *Acols
    cdef double *Adata
    cdef int Annz


    def __init__( self, prob=None ):
        super().__init__()

        self.prob = None
        self.warm_start = False
        self.mem_alloc = False
        self.mem_size = 0
        self.nlp_alloc = False
        self.Annz = 0

        self.options = Options( case_sensitive = True )

        if( prob ):
            self.setupProblem( prob )


    def setupProblem( self, prob ):
        if( not isinstance( prob, lp.Problem ) ):
            raise TypeError( "Argument 'prob' must be an instance of lp.Problem" )

        if( not prob.checkSetup() ):
            raise ValueError( "Argument 'prob' has not been properly configured" )

        self.prob = prob

        ## free ptr/bptr if recycling solver
        self.__dealloc__()

        self.ptr = glpk.glp_create_prob()
        self.nlp_alloc = True

        ## set problem as minimization
        glpk.glp_set_obj_dir( self.ptr, GLP_MIN )

        glpk.glp_add_cols( ptr, self.prob.N )
        glpk.glp_add_rows( ptr, self.prob.Nconslin )

        for k in range( self.prob.N ):
            if( np.isinf( self.prob.lb[k] ) and np.isinf( self.prob.ub[k] ) ):
                glpk.glp_set_col_bnds( self.ptr, k+1, glpk.GLP_FR, 0.0, 0.0 )
            elif( np.isinf( self.prob.lb[k] ) ):
                glpk.glp_set_col_bnds( self.ptr, k+1, glpk.GLP_UP, 0.0, self.prob.ub[k] )
            elif( np.isinf( self.prob.ub[k] ) ):
                glpk.glp_set_col_bnds( self.ptr, k+1, glpk.GLP_LO, self.prob.lb[k], 0.0 )
            elif( self.prob.lb[k] == self.prob.ub[k] ):
                glpk.glp_set_col_bnds( self.ptr, k+1, glpk.GLP_FX,
                                       self.prob.lb[k], self.prob.ub[k] )
            else:
                glpk.glp_set_col_bnds( self.ptr, k+1, glpk.GLP_DB,
                                       self.prob.lb[k], self.prob.ub[k] )

            glpk.glp_set_obj_coef( self.ptr, k+1, self.prob.objL[k] );

        for k in range( self.prob.Nconslin ):
            if( np.isinf( self.prob.conslinlb[k] ) and np.isinf( self.prob.conslinub[k] ) ):
                glpk.glp_set_row_bnds( self.ptr, k+1, glpk.GLP_FR, 0.0, 0.0 )
            elif( np.isinf( self.prob.conslinlb[k] ) ):
                glpk.glp_set_row_bnds( self.ptr, k+1, glpk.GLP_UP, 0.0, self.prob.conslinub[k] )
            elif( np.isinf( self.prob.conslinub[k] ) ):
                glpk.glp_set_row_bnds( self.ptr, k+1, glpk.GLP_LO, self.prob.conslinlb[k], 0.0 )
            elif( self.prob.conslinlb[k] == self.prob.conslinub[k] ):
                glpk.glp_set_row_bnds( self.ptr, k+1, glpk.GLP_FX,
                                       self.prob.conslinlb[k], self.prob.conslinub[k] )
            else:
                glpk.glp_set_row_bnds( self.ptr, k+1, glpk.GLP_DB,
                                       self.prob.conslinlb[k], self.prob.conslinub[k] )

        Asparse = sMatrix( self.prob.conslinA, copy_data=True )
        self.Annz = Asparse.nnz

        ## Allocate if necessary
        if( not self.mem_alloc ):
            self.allocate()
        elif( self.mem_size < self.Annz )
            self.deallocate()
            self.allocate()

        ## first element of every array is reserved by GLPK
        if( Asparse.nnz > 0 ):
            if( sizeof( int ) == 8 ):
                Asparse.copyIdxs( &self.Arows[1], &self.Acols[1], roffset=1, coffset=1 )
            else:
                Asparse.copyIdxs32( &self.Arows[1], &self.Acols[1], roffset=1, coffset=1 )
            Asparse.copyData( &self.Adata[1] )
            glpk.glp_load_matrix( self.ptr, self.Annz, self.Arows, self.Acols, self.Adata )



    cdef int allocate( self ):
        if( self.mem_alloc ):
            return False

        self.Arows = <int *> malloc( ( self.Annz + 1 ) * sizeof( int ) )
        self.Acols = <int *> malloc( ( self.Annz + 1 ) * sizeof( int ) )
        self.Adata = <double *> malloc( ( self.Annz + 1 ) * sizeof( double ) )

        if( self.Arows == NULL or
            self.Acols == NULL or
            self.Adata == NULL ):
            raise MemoryError( "At least one memory allocation failed" )

        self.mem_alloc = True
        self.mem_size = self.Annz

        return True


    cdef int deallocate( self ):
        if( not self.mem_alloc ):
            return False

        free( self.Arows )
        free( self.Acols )
        free( self.Adata )

        self.mem_alloc = False
        return True


    def __dealloc__( self ):
        if( self.nlp_alloc ):
            del self.ptr
            self.nlp_alloc = False

        self.deallocate()


    cdef void processOptions( self ):
        


    def solve( self ):

        if( "solver" in self.options and
            self.options[ "solver" ] == "exact" ):
            retval = glpk.glp_exact( self.ptr, self.params )
        else:
            retval = glpk.glp_simplex( self.ptr, self.params )





        self.prob.soln.final = arraySanitize( np.empty( ( self.prob.N, ), dtype=real_dtype ) )
        self.prob.soln.dual = arraySanitize( np.empty( ( self.prob.N + self.prob.Nconslin,),
                                                       dtype=real_dtype ) )

        self.prob.soln.value = self.bptr.getObjVal()
        self.bptr.getPrimalSolution( <real_t*> getPtr( self.prob.soln.final ) )
        self.bptr.getDualSolution( <real_t*> getPtr( self.prob.soln.dual ) )

        ## endif
