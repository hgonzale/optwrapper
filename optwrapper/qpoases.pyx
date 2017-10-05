import numpy as np
cimport numpy as cnp
from cython.operator cimport dereference as deref

cimport qpoaseshpp as qpoases
from qpoaseshpp cimport *
from utils cimport getPtr, arraySanitize, Options as uOptions
cimport base
import qp, lp

cdef type real_dtype = np.float64

cdef class Soln( base.Soln ):
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

class Params:
    BT_FALSE              = qpoases.BT_FALSE
    BT_TRUE               = qpoases.BT_TRUE

    PL_DEBUG_ITER         = qpoases.PL_DEBUG_ITER
    PL_TABULAR            = qpoases.PL_TABULAR
    PL_NONE               = qpoases.PL_NONE
    PL_LOW                = qpoases.PL_LOW
    PL_MEDIUM             = qpoases.PL_MEDIUM
    PL_HIGH               = qpoases.PL_HIGH

    HST_ZERO              = qpoases.HST_ZERO
    HST_IDENTITY          = qpoases.HST_IDENTITY
    HST_POSDEF            = qpoases.HST_POSDEF
    HST_POSDEF_NULLSPACE  = qpoases.HST_POSDEF_NULLSPACE
    HST_SEMIDEF           = qpoases.HST_SEMIDEF
    HST_INDEF             = qpoases.HST_INDEF
    HST_UNKNOWN           = qpoases.HST_UNKNOWN

    ST_LOWER              = qpoases.ST_LOWER
    ST_INACTIVE           = qpoases.ST_INACTIVE
    ST_UPPER              = qpoases.ST_UPPER
    ST_INFEASIBLE_LOWER   = qpoases.ST_INFEASIBLE_LOWER
    ST_INFEASIBLE_UPPER   = qpoases.ST_INFEASIBLE_UPPER
    ST_UNDEFINED          = qpoases.ST_UNDEFINED


cdef class OptionsWrap:
    cdef qpoases.Options *ptr

    def __cinit__( self ):
        self.ptr = new qpoases.Options()

    def __dealloc__( self ):
        del self.ptr

    @property
    def printLevel( self ): return self.ptr.printLevel
    @printLevel.setter
    def printLevel( self, val ): self.ptr.printLevel = val

    @property
    def enableRamping( self ): return self.ptr.enableRamping
    @enableRamping.setter
    def enableRamping( self, val ): self.ptr.enableRamping = val

    @property
    def enableFarBounds( self ): return self.ptr.enableFarBounds
    @enableFarBounds.setter
    def enableFarBounds( self, val ): self.ptr.enableFarBounds = val

    @property
    def enableFlippingBounds( self ): return self.ptr.enableFlippingBounds
    @enableFlippingBounds.setter
    def enableFlippingBounds( self, val ): self.ptr.enableFlippingBounds = val

    @property
    def enableRegularisation( self ): return self.ptr.enableRegularisation
    @enableRegularisation.setter
    def enableRegularisation( self, val ): self.ptr.enableRegularisation = val

    @property
    def enableFullLITests( self ): return self.ptr.enableFullLITests
    @enableFullLITests.setter
    def enableFullLITests( self, val ): self.ptr.enableFullLITests = val

    @property
    def enableNZCTests( self ): return self.ptr.enableNZCTests
    @enableNZCTests.setter
    def enableNZCTests( self, val ): self.ptr.enableNZCTests = val

    @property
    def enableDriftCorrection( self ): return self.ptr.enableDriftCorrection
    @enableDriftCorrection.setter
    def enableDriftCorrection( self, val ): self.ptr.enableDriftCorrection = val

    @property
    def enableCholeskyRefactorisation( self ): return self.ptr.enableCholeskyRefactorisation
    @enableCholeskyRefactorisation.setter
    def enableCholeskyRefactorisation( self, val ): self.ptr.enableCholeskyRefactorisation = val

    @property
    def enableEqualities( self ): return self.ptr.enableEqualities
    @enableEqualities.setter
    def enableEqualities( self, val ): self.ptr.enableEqualities = val

    @property
    def terminationTolerance( self ): return self.ptr.terminationTolerance
    @terminationTolerance.setter
    def terminationTolerance( self, val ): self.ptr.terminationTolerance = val

    @property
    def boundTolerance( self ): return self.ptr.boundTolerance
    @boundTolerance.setter
    def boundTolerance( self, val ): self.ptr.boundTolerance = val

    @property
    def boundRelaxation( self ): return self.ptr.boundRelaxation
    @boundRelaxation.setter
    def boundRelaxation( self, val ): self.ptr.boundRelaxation = val

    @property
    def epsNum( self ): return self.ptr.epsNum
    @epsNum.setter
    def epsNum( self, val ): self.ptr.epsNum = val

    @property
    def epsDen( self ): return self.ptr.epsDen
    @epsDen.setter
    def epsDen( self, val ): self.ptr.epsDen = val

    @property
    def maxPrimalJump( self ): return self.ptr.maxPrimalJump
    @maxPrimalJump.setter
    def maxPrimalJump( self, val ): self.ptr.maxPrimalJump = val

    @property
    def maxDualJump( self ): return self.ptr.maxDualJump
    @maxDualJump.setter
    def maxDualJump( self, val ): self.ptr.maxDualJump = val

    @property
    def initialRamping( self ): return self.ptr.initialRamping
    @initialRamping.setter
    def initialRamping( self, val ): self.ptr.initialRamping = val

    @property
    def finalRamping( self ): return self.ptr.finalRamping
    @finalRamping.setter
    def finalRamping( self, val ): self.ptr.finalRamping = val

    @property
    def initialFarBounds( self ): return self.ptr.initialFarBounds
    @initialFarBounds.setter
    def initialFarBounds( self, val ): self.ptr.initialFarBounds = val

    @property
    def growFarBounds( self ): return self.ptr.growFarBounds
    @growFarBounds.setter
    def growFarBounds( self, val ): self.ptr.growFarBounds = val

    @property
    def initialStatusBounds( self ): return self.ptr.initialStatusBounds
    @initialStatusBounds.setter
    def initialStatusBounds( self, val ): self.ptr.initialStatusBounds = val

    @property
    def epsFlipping( self ): return self.ptr.epsFlipping
    @epsFlipping.setter
    def epsFlipping( self, val ): self.ptr.epsFlipping = val

    @property
    def numRegularisationSteps( self ): return self.ptr.numRegularisationSteps
    @numRegularisationSteps.setter
    def numRegularisationSteps( self, val ): self.ptr.numRegularisationSteps = val

    @property
    def epsRegularisation( self ): return self.ptr.epsRegularisation
    @epsRegularisation.setter
    def epsRegularisation( self, val ): self.ptr.epsRegularisation = val

    @property
    def numRefinementSteps( self ): return self.ptr.numRefinementSteps
    @numRefinementSteps.setter
    def numRefinementSteps( self, val ): self.ptr.numRefinementSteps = val

    @property
    def epsIterRef( self ): return self.ptr.epsIterRef
    @epsIterRef.setter
    def epsIterRef( self, val ): self.ptr.epsIterRef = val

    @property
    def epsLITests( self ): return self.ptr.epsLITests
    @epsLITests.setter
    def epsLITests( self, val ): self.ptr.epsLITests = val

    @property
    def epsNZCTests( self ): return self.ptr.epsNZCTests
    @epsNZCTests.setter
    def epsNZCTests( self, val ): self.ptr.epsNZCTests = val

    @property
    def rcondSMin( self ): return self.ptr.rcondSMin
    @rcondSMin.setter
    def rcondSMin( self, val ): self.ptr.rcondSMin = val

    @property
    def enableInertiaCorrection( self ): return self.ptr.enableInertiaCorrection
    @enableInertiaCorrection.setter
    def enableInertiaCorrection( self, val ): self.ptr.enableInertiaCorrection = val

    @property
    def enableDropInfeasibles( self ): return self.ptr.enableDropInfeasibles
    @enableDropInfeasibles.setter
    def enableDropInfeasibles( self, val ): self.ptr.enableDropInfeasibles = val

    @property
    def dropBoundPriority( self ): return self.ptr.dropBoundPriority
    @dropBoundPriority.setter
    def dropBoundPriority( self, val ): self.ptr.dropBoundPriority = val

    @property
    def dropEqConPriority( self ): return self.ptr.dropEqConPriority
    @dropEqConPriority.setter
    def dropEqConPriority( self, val ): self.ptr.dropEqConPriority = val

    @property
    def dropIneqConPriority( self ): return self.ptr.dropIneqConPriority
    @dropIneqConPriority.setter
    def dropIneqConPriority( self, val ): self.ptr.dropIneqConPriority = val



cdef class Solver( base.Solver ):
    cdef qpoases.QProblem *ptr
    cdef qpoases.QProblemB *bptr
    cdef OptionsWrap optptrw
    cdef int warm_start
    cdef int cons_prob
    cdef int nlp_alloc
    cdef qpoases.HessianType quadtype

    def __init__( self, prob=None ):
        super().__init__()

        self.prob = None
        self.warm_start = False
        self.cons_prob = False
        self.nlp_alloc = False
        self.quadtype = qpoases.HST_UNKNOWN

        self.options = uOptions( case_sensitive = True )
        self.options[ "nWSR" ] = 100
        self.options[ "setTo" ] = "default"
        self.options[ "printLevel" ] = qpoases.PL_NONE

        if( prob ):
            self.setupProblem( prob )


    def setupProblem( self, prob ):
        if( not isinstance( prob, lp.Problem ) ):
            raise TypeError( "Argument prob must be an instance of lp.Problem" )

        if( not prob.checkSetup() ):
            raise ValueError( "Argument 'prob' has not been properly configured" )

        self.prob = prob
        self.optptrw = OptionsWrap()
        self.warm_start = False
        self.cons_prob = ( self.prob.Nconslin > 0 )

        if( isinstance( prob, qp.Problem ) ):
            self.quadtype = qpoases.HST_UNKNOWN
            if( self.prob.objQtype is qp.QuadType.indef ):
                self.quadtype = qpoases.HST_INDEF
            elif( self.prob.objQtype is qp.QuadType.posdef ):
                self.quadtype = qpoases.HST_POSDEF
            elif( self.prob.objQtype is qp.QuadType.possemidef ):
                self.quadtype = qpoases.HST_SEMIDEF
            elif( self.prob.objQtype is qp.QuadType.identity ):
                self.quadtype = qpoases.HST_IDENTITY
            elif( self.prob.objQtype is qp.QuadType.zero ):
                self.quadtype = qpoases.HST_ZERO
        else: ## isinstance( prob, lp.Problem )
            self.quadtype = qpoases.HST_ZERO

        ## free ptr/bptr if recycling solver
        self.__dealloc__()

        if( self.cons_prob ):
            self.ptr = new QProblem( self.prob.N, self.prob.Nconslin, self.quadtype )
        else:
            self.bptr = new QProblemB( self.prob.N, self.quadtype )

        self.nlp_alloc = True


    def __dealloc__( self ):
        if( self.nlp_alloc ):
            if( self.cons_prob ):
                del self.ptr
            else:
                del self.bptr

            self.nlp_alloc = False


    def warmStart( self ):
        self.warm_start = True


    cdef processOptions( self ):
        if( self.options[ "setTo" ] == "default" ):
            self.optptrw.ptr.setToDefault()
        elif( self.options[ "setTo" ] == "reliable" ):
            if( self.debug ):
                print( "initializing parameters as 'reliable'" )
            self.optptrw.ptr.setToReliable()
        elif( self.options[ "setTo" ] == "mpc" ):
            if( self.debug ):
                print( "initializing parameters as 'MPC'" )
            self.optptrw.ptr.setToMPC()
        elif( self.options[ "setTo" ] == "fast" ):
            if( self.debug ):
                print( "initializing parameters as 'fast'" )
            self.optptrw.ptr.setToFast()
        else:
            raise ValueError( "options['setTo'] has an invalid value" )

        for item in self.options:
            if( item == "nWSR" or item == "setTo" or item == "cputime" ):
                continue

            if( self.debug ):
                print( "processing option {} = {}".format( item, self.options[item].value ) )

            setattr( self.optptrw, item, self.options[ item ].value )

        if( self.optptrw.ptr.ensureConsistency() != SUCCESSFUL_RETURN ):
            raise ValueError( "invalid option value" )

        if( self.cons_prob ):
            self.ptr.setOptions( deref( self.optptrw.ptr ) )
        else:
            self.bptr.setOptions( deref( self.optptrw.ptr ) )


    def solve( self ):
        cdef int_t tmpnWSR
        cdef real_t tmpcputime
        cdef real_t *objQptr
        cdef real_t *objLptr
        cdef real_t *conslinAptr
        cdef real_t *lbptr
        cdef real_t *ubptr
        cdef real_t *conslinlbptr
        cdef real_t *conslinubptr

        if( self.quadtype != qpoases.HST_IDENTITY and self.quadtype != qpoases.HST_ZERO ):
            objQptr = <real_t*> getPtr( arraySanitize( self.prob.objQ, dtype=real_dtype ) )
        else:
            objQptr = NULL

        objLptr = <real_t*> getPtr( arraySanitize( self.prob.objL, dtype=real_dtype ) )
        lbptr = <real_t*> getPtr( arraySanitize( self.prob.lb, dtype=real_dtype ) )
        ubptr = <real_t*> getPtr( arraySanitize( self.prob.ub, dtype=real_dtype ) )

        tmpnWSR = <int_t> self.options[ "nWSR" ].value

        self.prob.soln = Soln()
        self.processOptions()

        if( self.cons_prob ):
            conslinAptr = <real_t*> getPtr( arraySanitize( self.prob.conslinA, dtype=real_dtype ) )
            conslinlb = <real_t*> getPtr( arraySanitize( self.prob.conslinlb, dtype=real_dtype ) )
            conslinub = <real_t*> getPtr( arraySanitize( self.prob.conslinub, dtype=real_dtype ) )

            if( "cputime" in self.options ):
                tmpcputime = <real_t> self.options[ "cputime" ].value
                if( not self.warm_start ):
                    self.prob.soln.retval = self.ptr.init( objQptr, objLptr,
                                                           conslinAptr,
                                                           lbptr, ubptr,
                                                           conslinlb, conslinub,
                                                           <int_t&> tmpnWSR,
                                                           <real_t*> &tmpcputime )
                else:
                    self.warm_start = False
                    self.prob.soln.retval = self.ptr.hotstart( objLptr,
                                                               lbptr, ubptr,
                                                               conslinlb, conslinub,
                                                               <int_t&> tmpnWSR,
                                                               <real_t*> &tmpcputime )
            else:
                if( not self.warm_start ):
                    self.prob.soln.retval = self.ptr.init( objQptr, objLptr,
                                                           conslinAptr,
                                                           lbptr, ubptr,
                                                           conslinlb, conslinub,
                                                           <int_t&> tmpnWSR )
                else:
                    self.warm_start = False
                    self.prob.soln.retval = self.ptr.hotstart( objLptr,
                                                               lbptr, ubptr,
                                                               conslinlb, conslinub,
                                                               <int_t&> tmpnWSR )

            self.prob.soln.final = arraySanitize( np.empty( ( self.prob.N, ), dtype=real_dtype ) )
            self.prob.soln.dual = arraySanitize( np.empty( ( self.prob.N + self.prob.Nconslin,),
                                                           dtype=real_dtype ) )

            self.prob.soln.value = self.ptr.getObjVal()
            self.ptr.getPrimalSolution( <real_t*> getPtr( self.prob.soln.final ) )
            self.ptr.getDualSolution( <real_t*> getPtr( self.prob.soln.dual ) )

        else: ## self.prob.Nconslin == 0
            if( self.debug ):
                print( "using QProblemB class since Nconslin is zero" )

            if( "cputime" in self.options ):
                tmpcputime = <real_t> self.options[ "cputime" ].value
                if( not self.warm_start ):
                    self.prob.soln.retval = self.bptr.init( objQptr, objLptr,
                                                            lbptr, ubptr,
                                                            <int_t&> tmpnWSR,
                                                            <real_t*> &tmpcputime )
                else:
                    self.warm_start = False
                    self.prob.soln.retval = self.bptr.hotstart( objLptr,
                                                                lbptr, ubptr,
                                                                <int_t&> tmpnWSR,
                                                                <real_t*> &tmpcputime )
            else:
                if( not self.warm_start ):
                    self.prob.soln.retval = self.bptr.init( objQptr, objLptr,
                                                            lbptr, ubptr,
                                                            <int_t&> tmpnWSR )
                else:
                    self.warm_start = False
                    self.prob.soln.retval = self.bptr.hotstart( objLptr,
                                                                lbptr, ubptr,
                                                                <int_t&> tmpnWSR )

            self.prob.soln.final = arraySanitize( np.empty( ( self.prob.N, ), dtype=real_dtype ) )
            self.prob.soln.dual = arraySanitize( np.empty( ( self.prob.N + self.prob.Nconslin,),
                                                           dtype=real_dtype ) )

            self.prob.soln.value = self.bptr.getObjVal()
            self.bptr.getPrimalSolution( <real_t*> getPtr( self.prob.soln.final ) )
            self.bptr.getDualSolution( <real_t*> getPtr( self.prob.soln.dual ) )

        ## endif
