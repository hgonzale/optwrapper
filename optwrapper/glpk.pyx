import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free

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
            0: "LP successfully solved",
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
    cdef glpk.glp_smcp opt_smcp
    cdef glpk.glp_iptcp opt_iptcp
    cdef glpk.glp_iocp opt_iocp
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

        self.options = Options()
        self.options[ "solver" ] = "interior"

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
                glpk.glp_set_col_bnds( self.ptr, k+1, glpk.GLP_FR,
                                       0.0, 0.0 )
            elif( np.isinf( self.prob.lb[k] ) ):
                glpk.glp_set_col_bnds( self.ptr, k+1, glpk.GLP_UP,
                                       0.0, self.prob.ub[k] )
            elif( np.isinf( self.prob.ub[k] ) ):
                glpk.glp_set_col_bnds( self.ptr, k+1, glpk.GLP_LO,
                                       self.prob.lb[k], 0.0 )
            elif( self.prob.lb[k] == self.prob.ub[k] ):
                glpk.glp_set_col_bnds( self.ptr, k+1, glpk.GLP_FX,
                                       self.prob.lb[k], self.prob.ub[k] )
            else:
                glpk.glp_set_col_bnds( self.ptr, k+1, glpk.GLP_DB,
                                       self.prob.lb[k], self.prob.ub[k] )

            glpk.glp_set_obj_coef( self.ptr, k+1, self.prob.objL[k] );

        for k in range( self.prob.Nconslin ):
            if( np.isinf( self.prob.conslinlb[k] ) and np.isinf( self.prob.conslinub[k] ) ):
                glpk.glp_set_row_bnds( self.ptr, k+1, glpk.GLP_FR,
                                       0.0, 0.0 )
            elif( np.isinf( self.prob.conslinlb[k] ) ):
                glpk.glp_set_row_bnds( self.ptr, k+1, glpk.GLP_UP,
                                       0.0, self.prob.conslinub[k] )
            elif( np.isinf( self.prob.conslinub[k] ) ):
                glpk.glp_set_row_bnds( self.ptr, k+1, glpk.GLP_LO,
                                       self.prob.conslinlb[k], 0.0 )
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
        elif( self.mem_size < self.Annz ):
            self.deallocate()
            self.allocate()

        ## first element of every array is reserved by GLPK
        if( Asparse.nnz > 0 ):
            if( sizeof(int) == 8 ):
                Asparse.copyIdxs( &self.Arows[1], &self.Acols[1], roffset=1, coffset=1 )
            else:
                Asparse.copyIdxs32( &self.Arows[1], &self.Acols[1], roffset=1, coffset=1 )
            Asparse.copyData( &self.Adata[1] )
            glpk.glp_load_matrix( self.ptr, self.Annz, self.Arows, self.Acols, self.Adata )



    cdef int allocate( self ):
        if( self.mem_alloc ):
            return False

        ## first element of every array is reserved by GLPK
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
        self.mem_size = 0
        return True


    def __dealloc__( self ):
        if( self.nlp_alloc ):
            del self.ptr
            self.nlp_alloc = False

        self.deallocate()


    cdef void processOptions( self ):
        if( "solver" not in self.options ):
            raise ValueError( "option 'solver' is not defined" )

        if( self.options[ "solver" ].value == "exact" or
            self.options[ "solver" ].value == "simplex" ):
            glpk.glp_init_smcp( &self.opt_smcp )

            if( "msg_lev" in self.options ):
                self.opt_smcp.msg_lev = self.options[ "msg_lev" ].value
            if( "meth" in self.options ):
                self.opt_smcp.meth = self.options[ "meth" ].value
            if( "pricing" in self.options ):
                self.opt_smcp.pricing = self.options[ "pricing" ].value
            if( "r_test" in self.options ):
                self.opt_smcp.r_test = self.options[ "r_test" ].value
            if( "tol_bnd" in self.options ):
                self.opt_smcp.tol_bnd = self.options[ "tol_bnd" ].value
            if( "tol_dj" in self.options ):
                self.opt_smcp.tol_dj = self.options[ "tol_dj" ].value
            if( "tol_piv" in self.options ):
                self.opt_smcp.tol_piv = self.options[ "tol_piv" ].value
            if( "obj_ll" in self.options ):
                self.opt_smcp.obj_ll = self.options[ "obj_ll" ].value
            if( "obj_ul" in self.options ):
                self.opt_smcp.obj_ul = self.options[ "obj_ul" ].value
            if( "it_lim" in self.options ):
                self.opt_smcp.it_lim = self.options[ "it_lim" ].value
            if( "tm_lim" in self.options ):
                self.opt_smcp.tm_lim = self.options[ "tm_lim" ].value
            if( "out_frq" in self.options ):
                self.opt_smcp.out_frq = self.options[ "out_frq" ].value
            if( "out_dly" in self.options ):
                self.opt_smcp.out_dly = self.options[ "out_dly" ].value
            if( "presolve" in self.options ):
                self.opt_smcp.presolve = self.options[ "presolve" ].value
            if( "excl" in self.options ):
                self.opt_smcp.excl = self.options[ "excl" ].value
            if( "shift" in self.options ):
                self.opt_smcp.shift = self.options[ "shift" ].value
            if( "aorn" in self.options ):
                self.opt_smcp.aorn = self.options[ "aorn" ].value
        # elif( self.options[ "solver" ].value == "intopt" ):
        #     glpk.glp_init_iocp( &self.opt_iocp )

        #     if( "msg_lev" in self.options ):
        #         self.opt_iocp.msg_lev = self.options[ "msg_lev" ].value
        #     if( "br_tech" in self.options ):
        #         self.opt_iocp.br_tech = self.options[ "br_tech" ].value
        #     if( "bt_tech" in self.options ):
        #         self.opt_iocp.bt_tech = self.options[ "bt_tech" ].value
        #     if( "tol_int" in self.options ):
        #         self.opt_iocp.tol_int = self.options[ "tol_int" ].value
        #     if( "tol_obj" in self.options ):
        #         self.opt_iocp.tol_obj = self.options[ "tol_obj" ].value
        #     if( "tm_lim" in self.options ):
        #         self.opt_iocp.tm_lim = self.options[ "tm_lim" ].value
        #     if( "out_frq" in self.options ):
        #         self.opt_iocp.out_frq = self.options[ "out_frq" ].value
        #     if( "out_dly" in self.options ):
        #         self.opt_iocp.out_dly = self.options[ "out_dly" ].value
        #     if( "pp_tech" in self.options ):
        #         self.opt_iocp.pp_tech = self.options[ "pp_tech" ].value
        #     if( "mip_gap" in self.options ):
        #         self.opt_iocp.mip_gap = self.options[ "mip_gap" ].value
        #     if( "mir_cuts" in self.options ):
        #         self.opt_iocp.mir_cuts = self.options[ "mir_cuts" ].value
        #     if( "gmi_cuts" in self.options ):
        #         self.opt_iocp.gmi_cuts = self.options[ "gmi_cuts" ].value
        #     if( "cov_cuts" in self.options ):
        #         self.opt_iocp.cov_cuts = self.options[ "cov_cuts" ].value
        #     if( "clq_cuts" in self.options ):
        #         self.opt_iocp.clq_cuts = self.options[ "clq_cuts" ].value
        #     if( "presolve" in self.options ):
        #         self.opt_iocp.presolve = self.options[ "presolve" ].value
        #     if( "binarize" in self.options ):
        #         self.opt_iocp.binarize = self.options[ "binarize" ].value
        #     if( "fp_heur" in self.options ):
        #         self.opt_iocp.fp_heur = self.options[ "fp_heur" ].value
        #     if( "ps_heur" in self.options ):
        #         self.opt_iocp.ps_heur = self.options[ "ps_heur" ].value
        #     if( "ps_tm_lim" in self.options ):
        #         self.opt_iocp.ps_tm_lim = self.options[ "ps_tm_lim" ].value
        #     if( "sr_heur" in self.options ):
        #         self.opt_iocp.sr_heur = self.options[ "sr_heur" ].value
        #     if( "use_sol" in self.options ):
        #         self.opt_iocp.use_sol = self.options[ "use_sol" ].value
        #     if( "alien" in self.options ):
        #         self.opt_iocp.alien = self.options[ "alien" ].value
        #     if( "flip" in self.options ):
        #         self.opt_iocp.flip = self.options[ "flip" ].value
        elif( self.options[ "solver" ].value == "interior" ):
            glpk.glp_init_iptcp( &self.opt_iptcp )

            if( "msg_lev" in self.options ):
                self.opt_iptcp.msg_lev = self.options[ "msg_lev" ].value
            if( "ord_alg" in self.options ):
                self.opt_iptcp.ord_alg = self.options[ "ord_alg" ].value
        else:
            raise ValueError( "unknown solver {}".format( self.options[ "solver" ].value ) )


    def solve( self ):
        self.processOptions()
        self.prob.soln = Soln()

        self.prob.soln.final = np.zeros( ( self.prob.N,) )
        self.prob.soln.dual = np.zeros( ( self.prob.N + self.prob.Nconslin,) )

        if( self.options[ "solver" ].value == "simplex" ):
            retval = glpk.glp_simplex( self.ptr, self.opt_smcp )

            self.prob.soln.value = glpk.glp_get_obj_val( self.ptr )
            for k in range( self.prob.N ):
                self.prob.soln.final[k] = glpk.glp_get_col_prim( self.ptr, k+1 )
                self.prob.soln.dual[k] = glpk.glp_get_col_dual( self.ptr, k+1 )
            for k in range( self.prob.Nconslin ):
                self.prob.soln.dual[ k + self.prob.N ] = glpk.glp_get_row_dual( self.ptr, k+1 )

        elif( self.options[ "solver" ].value == "exact" ):
            retval = glpk.glp_exact( self.ptr, self.opt_smcp )

            self.prob.soln.value = glpk.glp_get_obj_val( self.ptr )
            for k in range( self.prob.N ):
                self.prob.soln.final[k] = glpk.glp_get_col_prim( self.ptr, k+1 )
                self.prob.soln.dual[k] = glpk.glp_get_col_dual( self.ptr, k+1 )
            for k in range( self.prob.Nconslin ):
                self.prob.soln.dual[ k + self.prob.N ] = glpk.glp_get_row_dual( self.ptr, k+1 )

        elif( self.options[ "solver" ].value == "interior" ):
            retval = glpk.glp_interior( self.ptr, self.opt_iptcp )

            self.prob.soln.value = glpk.glp_ipt_obj_val( self.ptr )
            for k in range( self.prob.N ):
                self.prob.soln.final[k] = glpk.glp_ipt_col_prim( self.ptr, k+1 )
                self.prob.soln.dual[k] = glpk.glp_ipt_col_dual( self.ptr, k+1 )
            for k in range( self.prob.Nconslin ):
                self.prob.soln.dual[ k + self.prob.N ] = glpk.glp_ipt_row_dual( self.ptr, k+1 )

        # elif( self.options[ "solver" ].value == "intopt" ):
        #     retval = glpk.glp_intopt( self.ptr, self.opt_iocp )

        #     self.prob.soln.value = glpk.glp_mip_obj_val( self.ptr )

        else:
            raise ValueError( "unknown solver {}".format( self.options[ "solver" ].value ) )






        ## endif
