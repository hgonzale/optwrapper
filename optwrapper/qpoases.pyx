import numpy as np
cimport numpy as cnp
from cython.operator cimport dereference as deref

cimport qpoaseshpp as qpoases
from utils cimport getPtr, arraySanitize
cimport base
import qp


cdef int real_type = cnp.NPY_FLOAT64
cdef type real_dtype = np.float64
cdef int int_type = cnp.NPY_INT64
cdef type int_dtype = np.int64
if( sizeof( int_t ) == 4 ):
    int_type = cnp.NPY_INT32
    int_dtype = np.int32



cdef class Soln( base.Soln ):
    cdef public returnValue retval
    cdef public cnp.ndarray dual

    def __init__( self ):
        super().__init__()
        self.retval = TERMINAL_LIST_ELEMENT

    def getStatus( self ):
        cdef dict statusInfo = {
            TERMINAL_LIST_ELEMENT: "Terminal List Element",
            SUCCESSFUL_RETURN: "Successful Return",
            RET_DIV_BY_ZERO: "Division By Zero",
            RET_INDEX_OUT_OF_BOUNDS: "Index Out Of Bounds",
            RET_INVALID_ARGUMENTS: "Invalid Arguments",
            RET_ERROR_UNDEFINED: "Error Undefined",
            RET_WARNING_UNDEFINED: "Warning Undefined",
            RET_INFO_UNDEFINED: "Info Undefined",
            RET_EWI_UNDEFINED: "EWI Undefined",
            RET_AVAILABLE_WITH_LINUX_ONLY: "Available With Linux Only",
            RET_UNKNOWN_BUG: "Unknown Bug",
            RET_PRINTLEVEL_CHANGED: "PrintLevel Changed",
            RET_NOT_YET_IMPLEMENTED: "Not Yet Implemented",
            RET_INDEXLIST_MUST_BE_REORDERD: "IndexList Must Be Reorderd",
            RET_INDEXLIST_EXCEEDS_MAX_LENGTH: "IndexList Exceeds Max Length",
            RET_INDEXLIST_CORRUPTED: "IndexList Corrupted",
            RET_INDEXLIST_OUTOFBOUNDS: "IndexList Outofbounds",
            RET_INDEXLIST_ADD_FAILED: "IndexList Add Failed",
            RET_INDEXLIST_INTERSECT_FAILED: "IndexList Intersect Failed",
            RET_INDEX_ALREADY_OF_DESIRED_STATUS: "Index Already Of Desired Status",
            RET_ADDINDEX_FAILED: "AddIndex Failed",
            RET_REMOVEINDEX_FAILED: "RemoveIndex Failed",
            RET_SWAPINDEX_FAILED: "SwapIndex Failed",
            RET_NOTHING_TO_DO: "Nothing To Do",
            RET_SETUP_BOUND_FAILED: "Setup Bound Failed",
            RET_SETUP_CONSTRAINT_FAILED: "Setup Constraint Failed",
            RET_MOVING_BOUND_FAILED: "Moving Bound Failed",
            RET_MOVING_CONSTRAINT_FAILED: "Moving Constraint Failed",
            RET_SHIFTING_FAILED: "Shifting Failed",
            RET_ROTATING_FAILED: "Rotating Failed",
            RET_QPOBJECT_NOT_SETUP: "QPobject Not Setup",
            RET_QP_ALREADY_INITIALISED: "QP Already Initialised",
            RET_NO_INIT_WITH_STANDARD_SOLVER: "No Init with Standard Solver",
            RET_RESET_FAILED: "Reset Failed",
            RET_INIT_FAILED: "Init Failed",
            RET_INIT_FAILED_TQ: "Init Failed TQ",
            RET_INIT_FAILED_CHOLESKY: "Init Failed Cholesky",
            RET_INIT_FAILED_HOTSTART: "Init Failed Hotstart",
            RET_INIT_FAILED_INFEASIBILITY: "Init Failed Infeasibility",
            RET_INIT_FAILED_UNBOUNDEDNESS: "Init Failed Unboundedness",
            RET_INIT_FAILED_REGULARISATION: "Init Failed Regularisation",
            RET_INIT_SUCCESSFUL: "Init Successful",
            RET_OBTAINING_WORKINGSET_FAILED: "Obtaining Workingset Failed",
            RET_SETUP_WORKINGSET_FAILED: "Setup Workingset Failed",
            RET_SETUP_AUXILIARYQP_FAILED: "Setup AuxiliaryQP Failed",
            RET_NO_CHOLESKY_WITH_INITIAL_GUESS: "No Cholesky with Initial Guess",
            RET_NO_EXTERN_SOLVER: "No Extern Solver",
            RET_QP_UNBOUNDED: "QP Unbounded",
            RET_QP_INFEASIBLE: "QP Infeasible",
            RET_QP_NOT_SOLVED: "QP Not Solved",
            RET_QP_SOLVED: "QP Solved",
            RET_UNABLE_TO_SOLVE_QP: "Unable To Solve QP",
            RET_INITIALISATION_STARTED: "Initialisation Started",
            RET_HOTSTART_FAILED: "Hotstart Failed",
            RET_HOTSTART_FAILED_TO_INIT: "Hotstart Failed To Init",
            RET_HOTSTART_FAILED_AS_QP_NOT_INITIALISED: "Hotstart Failed As QP Not Initialised",
            RET_ITERATION_STARTED: "Iteration Started",
            RET_SHIFT_DETERMINATION_FAILED: "Shift Determination Failed",
            RET_STEPDIRECTION_DETERMINATION_FAILED: "StepDirection Determination Failed",
            RET_STEPLENGTH_DETERMINATION_FAILED: "StepLength Determination Failed",
            RET_OPTIMAL_SOLUTION_FOUND: "Optimal Solution Found",
            RET_HOMOTOPY_STEP_FAILED: "Homotopy Step Failed",
            RET_HOTSTART_STOPPED_INFEASIBILITY: "Hotstart Stopped Infeasibility",
            RET_HOTSTART_STOPPED_UNBOUNDEDNESS: "Hotstart Stopped Unboundedness",
            RET_WORKINGSET_UPDATE_FAILED: "Workingset Update Failed",
            RET_MAX_NWSR_REACHED: "Max nWSR Reached",
            RET_CONSTRAINTS_NOT_SPECIFIED: "Constraints Not Specified",
            RET_INVALID_FACTORISATION_FLAG: "Invalid Factorisation Flag",
            RET_UNABLE_TO_SAVE_QPDATA: "Unable To Save QPdata",
            RET_STEPDIRECTION_FAILED_TQ: "StepDirection Failed TQ",
            RET_STEPDIRECTION_FAILED_CHOLESKY: "StepDirection Failed Cholesky",
            RET_CYCLING_DETECTED: "Cycling Detected",
            RET_CYCLING_NOT_RESOLVED: "Cycling Not Resolved",
            RET_CYCLING_RESOLVED: "Cycling Resolved",
            RET_STEPSIZE: "StepSize",
            RET_STEPSIZE_NONPOSITIVE: "StepSize Nonpositive",
            RET_SETUPSUBJECTTOTYPE_FAILED: "SetupSubjectToType Failed",
            RET_ADDCONSTRAINT_FAILED: "AddConstraint Failed",
            RET_ADDCONSTRAINT_FAILED_INFEASIBILITY: "AddConstraint Failed Infeasibility",
            RET_ADDBOUND_FAILED: "AddBound Failed",
            RET_ADDBOUND_FAILED_INFEASIBILITY: "AddBound Failed Infeasibility",
            RET_REMOVECONSTRAINT_FAILED: "RemoveConstraint Failed",
            RET_REMOVEBOUND_FAILED: "RemoveBound Failed",
            RET_REMOVE_FROM_ACTIVESET: "Remove from Activeset",
            RET_ADD_TO_ACTIVESET: "Add to Activeset",
            RET_REMOVE_FROM_ACTIVESET_FAILED: "Remove from Activeset Failed",
            RET_ADD_TO_ACTIVESET_FAILED: "Add to Activeset Failed",
            RET_CONSTRAINT_ALREADY_ACTIVE: "Constraint Already Active",
            RET_ALL_CONSTRAINTS_ACTIVE: "All Constraints Active",
            RET_LINEARLY_DEPENDENT: "Linearly Dependent",
            RET_LINEARLY_INDEPENDENT: "Linearly Independent",
            RET_LI_RESOLVED: "LI Resolved",
            RET_ENSURELI_FAILED: "EnsureLI Failed",
            RET_ENSURELI_FAILED_TQ: "EnsureLI Failed TQ",
            RET_ENSURELI_FAILED_NOINDEX: "EnsureLI Failed Noindex",
            RET_ENSURELI_FAILED_CYCLING: "EnsureLI Failed Cycling",
            RET_BOUND_ALREADY_ACTIVE: "Bound Already Active",
            RET_ALL_BOUNDS_ACTIVE: "All Bounds Active",
            RET_CONSTRAINT_NOT_ACTIVE: "Constraint Not Active",
            RET_BOUND_NOT_ACTIVE: "Bound Not Active",
            RET_HESSIAN_NOT_SPD: "Hessian Not SPD",
            RET_HESSIAN_INDEFINITE: "Hessian Indefinite",
            RET_MATRIX_SHIFT_FAILED: "Matrix Shift Failed",
            RET_MATRIX_FACTORISATION_FAILED: "Matrix Factorisation Failed",
            RET_PRINT_ITERATION_FAILED: "Print Iteration Failed",
            RET_NO_GLOBAL_MESSAGE_OUTPUTFILE: "No Global Message Outputfile",
            RET_DISABLECONSTRAINTS_FAILED: "DisableConstraints Failed",
            RET_ENABLECONSTRAINTS_FAILED: "EnableConstraints Failed",
            RET_ALREADY_ENABLED: "Already Enabled",
            RET_ALREADY_DISABLED: "Already Disabled",
            RET_NO_HESSIAN_SPECIFIED: "No Hessian Specified",
            RET_USING_REGULARISATION: "Using Regularisation",
            RET_EPS_MUST_BE_POSITVE: "EPS Must Be Positve",
            RET_REGSTEPS_MUST_BE_POSITVE: "RegSteps Must Be Positve",
            RET_HESSIAN_ALREADY_REGULARISED: "Hessian Already Regularised",
            RET_CANNOT_REGULARISE_IDENTITY: "Cannot Regularise Identity",
            RET_CANNOT_REGULARISE_SPARSE: "Cannot Regularise Sparse",
            RET_NO_REGSTEP_NWSR: "No RegStep nWSR",
            RET_FEWER_REGSTEPS_NWSR: "Fewer RegSteps nWSR",
            RET_CHOLESKY_OF_ZERO_HESSIAN: "Cholesky Of Zero Hessian",
            RET_ZERO_HESSIAN_ASSUMED: "Zero Hessian Assumed",
            RET_CONSTRAINTS_ARE_NOT_SCALED: "Constraints Are Not Scaled",
            RET_INITIAL_BOUNDS_STATUS_NYI: "Initial Bounds Status NYI",
            RET_ERROR_IN_CONSTRAINTPRODUCT: "Error In ConstraintProduct",
            RET_FIX_BOUNDS_FOR_LP: "Fix Bounds For LP",
            RET_USE_REGULARISATION_FOR_LP: "Use Regularisation For LP",
            RET_UPDATEMATRICES_FAILED: "UpdateMatrices Failed",
            RET_UPDATEMATRICES_FAILED_AS_QP_NOT_SOLVED: "UpdateMatrices Failed As QP Not Solved",
            RET_UNABLE_TO_OPEN_FILE: "Unable To Open File",
            RET_UNABLE_TO_WRITE_FILE: "Unable To Write File",
            RET_UNABLE_TO_READ_FILE: "Unable To Read File",
            RET_FILEDATA_INCONSISTENT: "Filedata Inconsistent",
            RET_UNABLE_TO_ANALYSE_QPROBLEM: "Unable To Analyse QProblem",
            RET_OPTIONS_ADJUSTED: "Options Adjusted",
            RET_NWSR_SET_TO_ONE: "nWSR Set To One",
            RET_UNABLE_TO_READ_BENCHMARK: "Unable To Read Benchmark",
            RET_BENCHMARK_ABORTED: "Benchmark Aborted",
            RET_INITIAL_QP_SOLVED: "Initial QP Solved",
            RET_QP_SOLUTION_STARTED: "QP Solution Started",
            RET_BENCHMARK_SUCCESSFUL: "Benchmark Successful",
            RET_NO_DIAGONAL_AVAILABLE: "No Diagonal Available",
            RET_DIAGONAL_NOT_INITIALISED: "Diagonal Not Initialised",
            RET_ENSURELI_DROPPED: "EnsureLI Dropped",
            RET_KKT_MATRIX_SINGULAR: "KKT Matrix Singular",
            RET_QR_FACTORISATION_FAILED: "QR Factorisation Failed",
            RET_INERTIA_CORRECTION_FAILED: "Inertia Correction Failed",
            RET_NO_SPARSE_SOLVER: "No Sparse Solver",
            RET_SIMPLE_STATUS_P1: "Simple Status P1",
            RET_SIMPLE_STATUS_P0: "Simple Status P0",
            RET_SIMPLE_STATUS_M1: "Simple Status M1",
            RET_SIMPLE_STATUS_M2: "Simple Status M2",
            RET_SIMPLE_STATUS_M3: "Simple Status M3"
        }

        if( self.retval not in statusInfo ):
            return "Undefined return information"

        return statusInfo[ self.retval ]



cdef class Solver( base.Solver ):
    cdef QProblem *ptr
    cdef QProblemB *bptr
    cdef Options *optptr
    cdef int warm_start

    def __dealloc__(self):

    def __init__( self, prob=None ):
        super().__init__()

        self.prob = None

        self.options = utils.Options( case_sensitive = True )
        self.options[ "nWSR" ] = 10
        self.options[ "setTo" ] = "default"

        if( prob ):
            self.setupProblem( prob )


    def setupProblem( self, prob ):
        if( not prob.checkSetup() ):
            raise ValueError( "Argument 'prob' has not been properly configured" )

        self.prob = prob
        self.warm_start = False

        self.optptr = new Options()

        if( self.prob.Nconslin > 0 ):
            self.ptr = new QProblem( self.prob.N, self.prob.Nconslin, HST_UNKNOWN )
        else:
            self.bptr = new QProblemB( self.prob.N, HST_UNKNOWN )


    def __dealloc__( self ):
        del self.optptr

        if( self.prob.Nconslin > 0 ):
            del self.ptr
        else:
            del self.bptr


    def warmStart( self ):
        self.warm_start = True


    ## FIXME
    cdef processOptions( self ):
        if( self.options[ "setTo" ] == "default" ):
            self.optptr.setToDefault()
        elif( self.options[ "setTo" ] == "reliable" ):
            self.optptr.setToReliable()
        elif( self.options[ "setTo" ] == "mpc" ):
            self.optptr.setToMPC()
        elif( self.options[ "setTo" ] == "fast" ):
            self.optptr.setToFast()
        else:
            raise ValueError( "options['setTo'] has an invalid value" )

        for item in self.options:
            if( item == "nWSR" or item == "setTo" or item == "cputime" ):
                continue
            setattr( self.optptr, item, self.options.value )

        if( self.optptr.ensureConsistency() != SUCCESSFUL_RETURN ):
            raise ValueError( "invalid option value" )

        if( self.prob.Nconslin > 0 ):
            self.ptr.setOptions( self.optptr ) )
        else:
            self.bptr.setOptions( self.optptr ) )


    def solve( self ):
        cdef int_t tmpnWSR
        cdef real_t tmpcputime

        self.prob.soln = Soln()
        self.processOptions()

        tmpnWSR = int( self.options[ "nWSR" ] )

        if( self.prob.Nconslin > 0 ):
            if( "cputime" in self.options ):
                tmpcputime = float( self.options[ "cputime" ] )
                if( not self.warm_start ):
                    self.prob.soln.retval = self.ptr.init(
                        <double*> getPtr( arraySanitize( self.prob.objQ, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.objL, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.conslinA, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.lb, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.ub, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.conslinlb, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.conslinub, dtype=real_dtype ) ),
                        <int&>    tmpnWSR,
                        <double*> &tmpcputime )
                else:
                    self.warm_start = False
                    self.prob.soln.retval = self.ptr.hotstart(
                        <double*> getPtr( arraySanitize( self.prob.objL, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.lb, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.ub, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.conslinlb, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.conslinub, dtype=real_dtype ) ),
                        <int&>    tmpnWSR,
                        <double*> &tmpcputime )
            else:
                if( not self.warm_start ):
                    self.prob.soln.retval = self.ptr.init(
                        <double*> getPtr( arraySanitize( self.prob.objQ, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.objL, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.conslinA, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.lb, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.ub, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.conslinlb, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.conslinub, dtype=real_dtype ) ),
                        <int&>    tmpnWSR )
                else:
                    self.warm_start = False
                    self.prob.soln.retval = self.ptr.hotstart(
                        <double*> getPtr( arraySanitize( self.prob.objL, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.lb, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.ub, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.conslinlb, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.conslinub, dtype=real_dtype ) ),
                        <int&>    tmpnWSR )

            self.prob.soln.final = arraySanitize( np.empty( ( self.prob.N, ), dtype=real_dtype ) )
            self.prob.soln.dual = arraySanitize( np.empty( ( self.prob.N + self.prob.Nconslin,),
                                                           dtype=real_dtype ) )

            self.prob.soln.value = self.ptr.getObjVal()
            self.ptr.getPrimalSolution( <double*> getPtr( self.prob.soln.final ) )
            self.ptr.getDualSolution( <double*> getPtr( self.prob.soln.dual ) )

        else: ## self.prob.Nconslin == 0
            if( "cputime" in self.options ):
                tmpcputime = float( self.options[ "cputime" ] )
                if( not self.warm_start ):
                    self.prob.soln.retval = self.bptr.init(
                        <double*> getPtr( arraySanitize( self.prob.objQ, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.objL, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.lb, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.ub, dtype=real_dtype ) ),
                        <int&>    tmpnWSR,
                        <double*> &tmpcputime )
                else:
                    self.warm_start = False
                    self.prob.soln.retval = self.bptr.hotstart(
                        <double*> getPtr( arraySanitize( self.prob.objL, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.lb, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.ub, dtype=real_dtype ) ),
                        <int&>    tmpnWSR,
                        <double*> &tmpcputime )
            else:
                if( not self.warm_start ):
                    self.prob.soln.retval = self.bptr.init(
                        <double*> getPtr( arraySanitize( self.prob.objQ, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.objL, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.lb, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.ub, dtype=real_dtype ) ),
                        <int&>    tmpnWSR )
                else:
                    self.warm_start = False
                    self.prob.soln.retval = self.bptr.hotstart(
                        <double*> getPtr( arraySanitize( self.prob.objL, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.lb, dtype=real_dtype ) ),
                        <double*> getPtr( arraySanitize( self.prob.ub, dtype=real_dtype ) ),
                        <int&>    tmpnWSR )

            self.prob.soln.final = arraySanitize( np.empty( ( self.prob.N, ), dtype=real_dtype ) )
            self.prob.soln.dual = arraySanitize( np.empty( ( self.prob.N + self.prob.Nconslin,),
                                                           dtype=real_dtype ) )

            self.prob.soln.value = self.bptr.getObjVal()
            self.bptr.getPrimalSolution( <double*> getPtr( self.prob.soln.final ) )
            self.bptr.getDualSolution( <double*> getPtr( self.prob.soln.dual ) )

        ## endif
