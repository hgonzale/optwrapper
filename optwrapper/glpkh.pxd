cdef extern from "glpk.h":
    enum:
        ## optimization direction flag:
        GLP_MIN  ## minimization
        GLP_MAX  ## maximization

        ## kind of structural variable:
        GLP_CV  ## continuous variable
        GLP_IV  ## integer variable
        GLP_BV  ## binary variable

        ## type of auxiliary/structural variable:
        GLP_FR  ## free (unbounded) variable
        GLP_LO  ## variable with lower bound
        GLP_UP  ## variable with upper bound
        GLP_DB  ## double-bounded variable
        GLP_FX  ## fixed variable

        ## status of auxiliary/structural variable:
        GLP_BS  ## basic variable
        GLP_NL  ## non-basic variable on lower bound
        GLP_NU  ## non-basic variable on upper bound
        GLP_NF  ## non-basic free (unbounded) variable
        GLP_NS  ## non-basic fixed variable

        ## scaling options:
        GLP_SF_GM  ## perform geometric mean scaling
        GLP_SF_EQ  ## perform equilibration scaling
        GLP_SF_2N  ## round scale factors to power of two
        GLP_SF_SKIP  ## skip if problem is well scaled
        GLP_SF_AUTO  ## choose scaling options automatically

        ## solution indicator:
        GLP_SOL  ## basic solution
        GLP_IPT  ## interior-point solution
        GLP_MIP  ## mixed integer solution

        ## solution status:
        GLP_UNDEF  ## solution is undefined
        GLP_FEAS  ## solution is feasible
        GLP_INFEAS  ## solution is infeasible
        GLP_NOFEAS  ## no feasible solution exists
        GLP_OPT  ## solution is optimal
        GLP_UNBND  ## solution is unbounded

        ## simplex method control parameters:
        GLP_MSG_OFF  ## no output
        GLP_MSG_ERR  ## warning and error messages only
        GLP_MSG_ON  ## normal output
        GLP_MSG_ALL  ## full output
        GLP_MSG_DBG  ## debug output
        GLP_PRIMAL  ## use primal simplex
        GLP_DUALP  ## use dual; if it fails, use primal
        GLP_DUAL  ## use dual simplex
        GLP_PT_STD  ## standard (Dantzig's rule)
        GLP_PT_PSE  ## projected steepest edge
        GLP_RT_STD  ## standard (textbook)
        GLP_RT_HAR  ## Harris' two-pass ratio test
        GLP_RT_FLIP  ## long-step (flip-flop) ratio test
        GLP_USE_AT  ## use A matrix in row-wise format
        GLP_USE_NT  ## use N matrix in row-wise format

        ## interior-point solver control parameters:
        GLP_ORD_NONE  ## natural (original) ordering
        GLP_ORD_QMD  ## quotient minimum degree (QMD)
        GLP_ORD_AMD  ## approx. minimum degree (AMD)
        GLP_ORD_SYMAMD  ## approx. minimum degree (SYMAMD)

        ## enable/disable flag:
        GLP_ON  ## enable something
        GLP_OFF  ## disable something

        ## reason codes:
        GLP_IROWGEN  ## request for row generation
        GLP_IBINGO  ## better integer solution found
        GLP_IHEUR  ## request for heuristic solution
        GLP_ICUTGEN  ## request for cut generation
        GLP_IBRANCH  ## request for branching
        GLP_ISELECT  ## request for subproblem selection
        GLP_IPREPRO  ## request for preprocessing

        ## return codes:
        GLP_EBADB  ## invalid basis
        GLP_ESING  ## singular matrix
        GLP_ECOND  ## ill-conditioned matrix
        GLP_EBOUND  ## invalid bounds
        GLP_EFAIL  ## solver failed
        GLP_EOBJLL  ## objective lower limit reached
        GLP_EOBJUL  ## objective upper limit reached
        GLP_EITLIM  ## iteration limit exceeded
        GLP_ETMLIM  ## time limit exceeded
        GLP_ENOPFS  ## no primal feasible solution
        GLP_ENODFS  ## no dual feasible solution
        GLP_EROOT  ## root LP optimum not provided
        GLP_ESTOP  ## search terminated by application
        GLP_EMIPGAP  ## relative mip gap tolerance reached
        GLP_ENOFEAS  ## no primal/dual feasible solution
        GLP_ENOCVG  ## no convergence
        GLP_EINSTAB  ## numerical instability
        GLP_EDATA  ## invalid data
        GLP_ERANGE  ## result out of range

        ## condition indicator:
        GLP_KKT_PE  ## primal equalities
        GLP_KKT_PB  ## primal bounds
        GLP_KKT_DE  ## dual equalities
        GLP_KKT_DB  ## dual bounds
        GLP_KKT_CS  ## complementary slackness


    ## problem object (obscure struct)
    ctypedef struct glp_prob


    ## simplex method control parameters
    ctypedef struct glp_smcp:
        cdef int msg_lev  ## message level
        cdef int meth  ## simplex method option
        cdef int pricing            ## pricing technique
        cdef int r_test             ## ratio test technique
        cdef double tol_bnd         ## primal feasibility tolerance
        cdef double tol_dj          ## dual feasibility tolerance
        cdef double tol_piv         ## pivot tolerance
        cdef double obj_ll          ## lower objective limit
        cdef double obj_ul          ## upper objective limit
        cdef int it_lim             ## simplex iteration limit
        cdef int tm_lim             ## time limit, ms
        cdef int out_frq            ## display output frequency, ms
        cdef int out_dly            ## display output delay, ms
        cdef int presolve           ## enable/disable using LP presolver
        cdef int excl               ## exclude fixed non-basic variables
        cdef int shift              ## shift bounds of variables to zero
        cdef int aorn               ## option to use A or N
        cdef double foo_bar[33]     ## (reserved)


    ## interior-point solver control parameters
    ctypedef struct glp_iptcp:
        cdef int msg_lev  ## message level (see glp_smcp)
        cdef int ord_alg  ## ordering algorithm
        cdef double foo_bar[48]  ## (reserved)


    ## create problem object
    glp_prob *glp_create_prob()

    ## assign (change) problem name
    void glp_set_prob_name( glp_prob *P, const char *name )

    ## assign (change) objective function name
    void glp_set_obj_name( glp_prob *P, const char *name )

    ## set (change) optimization direction flag
    void glp_set_obj_dir( glp_prob *P, int dir )

    ## add new rows to problem object
    int glp_add_rows( glp_prob *P, int nrs )

    ## add new columns to problem object
    int glp_add_cols( glp_prob *P, int ncs )

    ## assign (change) row name
    void glp_set_row_name( glp_prob *P, int i, const char *name )

    ## assign (change) column name
    void glp_set_col_name( glp_prob *P, int j, const char *name )

    ## set (change) row bounds
    void glp_set_row_bnds( glp_prob *P, int i, int type, double lb, double ub )

    ## set (change) column bounds
    void glp_set_col_bnds( glp_prob *P, int j, int type, double lb, double ub )

    ## set (change) obj. coefficient or constant term
    void glp_set_obj_coef( glp_prob *P, int j, double coef )

    ## set (replace) row of the constraint matrix
    void glp_set_mat_row( glp_prob *P, int i, int len, const int ind[], const double val[] )

    ## set (replace) column of the constraint matrix
    void glp_set_mat_col( glp_prob *P, int j, int len, const int ind[], const double val[] )

    ## load (replace) the whole constraint matrix
    void glp_load_matrix( glp_prob *P, int ne, const int ia[], const int ja[], const double ar[] )

    ## check for duplicate elements in sparse matrix
    int glp_check_dup( int m, int n, int ne, const int ia[], const int ja[] )

    ## sort elements of the constraint matrix
    void glp_sort_matrix( glp_prob *P )

    ## delete specified rows from problem object
    void glp_del_rows( glp_prob *P, int nrs, const int num[] )

    ## delete specified columns from problem object
    void glp_del_cols( glp_prob *P, int ncs, const int num[] )

    ## copy problem object content
    void glp_copy_prob( glp_prob *dest, glp_prob *prob, int names )

    ## erase problem object content
    void glp_erase_prob( glp_prob *P )

    ## delete problem object
    void glp_delete_prob( glp_prob *P )

    ## retrieve problem name
    const char *glp_get_prob_name( glp_prob *P)

    ## retrieve objective function name
    const char *glp_get_obj_name( glp_prob *P )

    ## retrieve optimization direction flag
    int glp_get_obj_dir( glp_prob *P )

    ## retrieve number of rows
    int glp_get_num_rows( glp_prob *P )

    ## retrieve number of columns
    int glp_get_num_cols( glp_prob *P )

    ## retrieve row name
    const char *glp_get_row_name( glp_prob *P, int i )

    ## retrieve column name
    const char *glp_get_col_name( glp_prob *P, int j )

    ## retrieve row type
    int glp_get_row_type( glp_prob *P, int i )

    ## retrieve row lower bound
    double glp_get_row_lb( glp_prob *P, int i )

    ## retrieve row upper bound
    double glp_get_row_ub( glp_prob *P, int i )

    ## retrieve column type
    int glp_get_col_type( glp_prob *P, int j )

    ## retrieve column lower bound
    double glp_get_col_lb( glp_prob *P, int j )

    ## retrieve column upper bound
    double glp_get_col_ub( glp_prob *P, int j )

    ## retrieve obj. coefficient or constant term
    double glp_get_obj_coef(glp_prob *P, int j)

    ## retrieve number of constraint coefficients
    int glp_get_num_nz( glp_prob *P )

    ## retrieve row of the constraint matrix
    int glp_get_mat_row( glp_prob *P, int i, int ind[], double val[] )

    ## retrieve column of the constraint matrix
    int glp_get_mat_col( glp_prob *P, int j, int ind[], double val[] )

    ## create the name index
    void glp_create_index( glp_prob *P )

    ## find row by its name
    int glp_find_row( glp_prob *P, const char *name )

    ## find column by its name
    int glp_find_col( glp_prob *P, const char *name )

    ## delete the name index
    void glp_delete_index( glp_prob *P )

    ## set (change) row scale factor
    void glp_set_rii( glp_prob *P, int i, double rii )

    ## set (change) column scale factor
    void glp_set_sjj( glp_prob *P, int j, double sjj )

    ## retrieve row scale factor
    double glp_get_rii( glp_prob *P, int i )

    ## retrieve column scale factor
    double glp_get_sjj( glp_prob *P, int j )

    ## scale problem data
    void glp_scale_prob( glp_prob *P, int flags )

    ## unscale problem data
    void glp_unscale_prob( glp_prob *P )

    ## set (change) row status
    void glp_set_row_stat( glp_prob *P, int i, int stat )

    ## set (change) column status
    void glp_set_col_stat( glp_prob *P, int j, int stat )

    ## construct standard initial LP basis
    void glp_std_basis( glp_prob *P )

    ## construct advanced initial LP basis
    void glp_adv_basis( glp_prob *P, int flags )

    ## construct Bixby's initial LP basis
    void glp_cpx_basis( glp_prob *P )

    ## solve LP problem with the simplex method
    int glp_simplex( glp_prob *P, const glp_smcp *parm )

    ## solve LP problem in exact arithmetic
    int glp_exact( glp_prob *P, const glp_smcp *parm )

    ## initialize simplex method control parameters
    void glp_init_smcp( glp_smcp *parm )

    ## retrieve generic status of basic solution
    int glp_get_status( glp_prob *P )

    ## retrieve status of primal basic solution
    int glp_get_prim_stat( glp_prob *P )

    ## retrieve status of dual basic solution
    int glp_get_dual_stat( glp_prob *P )

    ## retrieve objective value (basic solution)
    double glp_get_obj_val( glp_prob *P )

    ## retrieve row status
    int glp_get_row_stat( glp_prob *P, int i )

    ## retrieve row primal value (basic solution)
    double glp_get_row_prim( glp_prob *P, int i )

    ## retrieve row dual value (basic solution)
    double glp_get_row_dual( glp_prob *P, int i )

    ## retrieve column status
    int glp_get_col_stat( glp_prob *P, int j )

    ## retrieve column primal value (basic solution)
    double glp_get_col_prim( glp_prob *P, int j )

    ## retrieve column dual value (basic solution)
    double glp_get_col_dual( glp_prob *P, int j )

    ## determine variable causing unboundedness
    int glp_get_unbnd_ray( glp_prob *P )

    ## get simplex solver iteration count
    int glp_get_it_cnt( glp_prob *P )

    ## set simplex solver iteration count
    void glp_set_it_cnt( glp_prob *P, int it_cnt )

    ## solve LP problem with the interior-point method
    int glp_interior( glp_prob *P, const glp_iptcp *parm )

    ## initialize interior-point solver control parameters
    void glp_init_iptcp( glp_iptcp *parm )

    ## retrieve status of interior-point solution
    int glp_ipt_status( glp_prob *P )

    ## retrieve objective value (interior point)
    double glp_ipt_obj_val( glp_prob *P )

    ## retrieve row primal value (interior point)
    double glp_ipt_row_prim( glp_prob *P, int i )

    ## retrieve row dual value (interior point)
    double glp_ipt_row_dual( glp_prob *P, int i )

    ## retrieve column primal value (interior point)
    double glp_ipt_col_prim( glp_prob *P, int j )

    ## retrieve column dual value (interior point)
    double glp_ipt_col_dual( glp_prob *P, int j )

    ## set (change) column kind
    void glp_set_col_kind( glp_prob *P, int j, int kind )

    ## retrieve column kind
    int glp_get_col_kind( glp_prob *P, int j )

    ## retrieve number of integer columns
    int glp_get_num_int( glp_prob *P )

    ## retrieve number of binary columns
    int glp_get_num_bin( glp_prob *P )

    ## solve MIP problem with the branch-and-bound method
    int glp_intopt( glp_prob *P, const glp_iocp *parm )

    ## initialize integer optimizer control parameters
    void glp_init_iocp( glp_iocp *parm )

    ## retrieve status of MIP solution
    int glp_mip_status( glp_prob *P )

    ## retrieve objective value (MIP solution)
    double glp_mip_obj_val( glp_prob *P )

    ## retrieve row value (MIP solution)
    double glp_mip_row_val( glp_prob *P, int i )

    ## retrieve column value (MIP solution)
    double glp_mip_col_val( glp_prob *P, int j )

    ## check feasibility/optimality conditions
    void glp_check_kkt( glp_prob *P, int sol, int cond, double *ae_max, int *ae_ind, double *re_max,
                        int *re_ind )

    ## write basic solution in printable format
    int glp_print_sol( glp_prob *P, const char *fname )

    ## read basic solution from text file
    int glp_read_sol( glp_prob *P, const char *fname )

    ## write basic solution to text file
    int glp_write_sol( glp_prob *P, const char *fname )

    ## print sensitivity analysis report
    int glp_print_ranges( glp_prob *P, int len, const int list[], int flags, const char *fname )

    ## write interior-point solution in printable format
    int glp_print_ipt( glp_prob *P, const char *fname )

    ## read interior-point solution from text file
    int glp_read_ipt( glp_prob *P, const char *fname )

    ## write interior-point solution to text file
    int glp_write_ipt( glp_prob *P, const char *fname )

    ## write MIP solution in printable format
    int glp_print_mip( glp_prob *P, const char *fname )

    ## read MIP solution from text file
    int glp_read_mip( glp_prob *P, const char *fname )

    ## write MIP solution to text file
    int glp_write_mip( glp_prob *P, const char *fname )

    ## check if LP basis factorization exists
    int glp_bf_exists( glp_prob *P )

    ## compute LP basis factorization
    int glp_factorize( glp_prob *P )

    ## check if LP basis factorization has been updated
    int glp_bf_updated( glp_prob *P )

    ## retrieve LP basis factorization control parameters
    void glp_get_bfcp( glp_prob *P, glp_bfcp *parm )

    ## change LP basis factorization control parameters
    void glp_set_bfcp( glp_prob *P, const glp_bfcp *parm )

    ## retrieve LP basis header information
    int glp_get_bhead( glp_prob *P, int k )

    ## retrieve row index in the basis header
    int glp_get_row_bind( glp_prob *P, int i )

    ## retrieve column index in the basis header
    int glp_get_col_bind( glp_prob *P, int j )

    ## perform forward transformation (solve system B*x = b)
    void glp_ftran( glp_prob *P, double x[] )

    ## perform backward transformation (solve system B'*x = b)
    void glp_btran( glp_prob *P, double x[] )

    ## "warm up" LP basis
    int glp_warm_up( glp_prob *P )

    ## compute row of the simplex tableau
    int glp_eval_tab_row( glp_prob *P, int k, int ind[], double val[] )

    ## compute column of the simplex tableau
    int glp_eval_tab_col( glp_prob *P, int k, int ind[], double val[] )

    ## transform explicitly specified row
    int glp_transform_row( glp_prob *P, int len, int ind[], double val[] )

    ## transform explicitly specified column
    int glp_transform_col( glp_prob *P, int len, int ind[], double val[] )

    ## perform primal ratio test
    int glp_prim_rtest( glp_prob *P, int len, const int ind[], const double val[], int dir,
                        double eps )

    ## perform dual ratio test
    int glp_dual_rtest( glp_prob *P, int len, const int ind[], const double val[], int dir,
                        double eps )

    ## analyze active bound of non-basic variable
    void glp_analyze_bound( glp_prob *P, int k, double *value1, int *var1, double *value2,
                            int *var2 )

    ## analyze objective coefficient at basic variable
    void glp_analyze_coef( glp_prob *P, int k, double *coef1, int *var1, double *value1,
                           double *coef2, int *var2, double *value2 )

    ## initialize GLPK environment
    int glp_init_env()

    ## determine library version
    const char *glp_version()

    ## determine library configuration
    const char *glp_config( const char *option )

    ## free GLPK environment
    int glp_free_env()

    ## write string on terminal
    void glp_puts( const char *s )

    ## write formatted output on terminal
    void glp_printf( const char *fmt, ... )

    ## write formatted output on terminal
    void glp_vprintf( const char *fmt, va_list arg )

    ## enable/disable terminal output
    int glp_term_out( int flag )

    ## install hook to intercept terminal output
    void glp_term_hook( int (*func)( void *info, const char *s ), void *info )

    ## start copying terminal output to text file
    int glp_open_tee( const char *name )

    ## stop copying terminal output to text file
    int glp_close_tee()

    ## install hook to intercept abnormal termination
    void glp_error_hook( void (*func)( void *info ), void *info )

    ## allocate memory block
    void *glp_alloc( int n, int size )

    ## reallocate memory block
    void *glp_realloc( void *ptr, int n, int size )

    ## free (deallocate) memory block
    void glp_free( void *ptr )

    ## set memory usage limit
    void glp_mem_limit( int limit )

    ## get memory usage information
    void glp_mem_usage( int *count, int *cpeak, size_t *total, size_t *tpeak )

    ## determine current universal time
    double glp_time()

    ## compute difference between two time values
    double glp_difftime( double t1, double t0 )
