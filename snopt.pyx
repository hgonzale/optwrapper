from libc.string cimport memcpy, memset
from libc.stdlib cimport malloc, free
cimport cpython.mem as mem
cimport numpy as np
import numpy as np
from scipy.sparse import coo_matrix

from f2ch cimport *
cimport filehandler as fh
cimport snopth as snopt
cimport utils
cimport base
import nlp

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

## The function usrfun should be a static method in snopt.Solver,
## but it appears that Cython doesn't support static cdef methods yet.
## Instead, this is a reasonable hack.
cdef object extprob

cdef int usrfun( integer *status, integer *n, doublereal *x,
                 integer *needF, integer *nF, doublereal *f,
                 integer *needG, integer *lenG, doublereal *G,
                 char *cu, integer *lencu,
                 integer *iu, integer *leniu,
                 doublereal *ru, integer *lenru ):
    ## Dual variables are at rw[ iw[329] ] onward, pg.25
    cdef int lenobjg
    cdef int lenconsg

    if( status[0] >= 2 ): ## Final call, do nothing
        return 0

    xarr = utils.wrap1dPtr( x, n[0], utils.doublereal_type )

    if( extprob.mixedCons ):
        tmpidx = 1
    else:
        tmpidx = 1 + extprob.Nconslin

    if( needF[0] > 0 ):
        f[0] = extprob.objf( xarr )
        if( extprob.Ncons > 0 ):
            tmpconsf = utils.convFortran( extprob.consf( xarr ) )
            memcpy( &f[tmpidx], utils.getPtr( tmpconsf ),
                    extprob.Ncons * sizeof( doublereal ) )

    if( needG[0] > 0 ):
        objg = extprob.objg( xarr )
        if( isinstance( extprob, nlp.SparseProblem ) and extprob.objgpattern != None ):
            tmpobjg = utils.convFortran( objg.ravel()[ np.flatnonzero( extprob.objgpattern ) ] )
            lenobjg = extprob.objgpattern.sum()
        else:
            tmpobjg = utils.convFortran( objg )
            lenobjg = extprob.N

        memcpy( G, utils.getPtr( tmpobjg ),
                lenobjg * sizeof( doublereal ) )

        consg = extprob.consg( xarr )
        if( isinstance( extprob, nlp.SparseProblem ) and extprob.consgpattern != None ):
            tmpconsg = utils.convFortran( consg.ravel()[ np.flatnonzero( extprob.consgpattern ) ] )
            lenconsg = extprob.consgpattern.sum()
        else:
            tmpconsg = utils.convFortran( consg )
            lenconsg = extprob.N * extprob.Ncons

        memcpy( &G[lenobjg], utils.getPtr( tmpconsg ),
                lenconsg * sizeof( doublereal ) )


cdef class Soln( base.Soln ):
    cdef public np.ndarray xstate
    cdef public np.ndarray xmul
    cdef public np.ndarray Fstate
    cdef public np.ndarray Fmul
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
    cdef int mem_size[4]
    cdef int mem_alloc_ws
    cdef int mem_size_ws[3]

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
        memset( self.mem_size, 0, 4 ) ## Set mem_size to zero
        memset( self.mem_size_ws, 0, 3 ) ## Set mem_size_ws to zero
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
        cdef int lenobjg
        cdef int lenconsg
        cdef int tmpidx
        global extprob

        if( not isinstance( prob, nlp.Problem ) ):
            raise TypeError( "Argument 'prob' must be of type 'nlp.Problem'" )

        self.prob = prob ## Save a copy of prob's pointer
        extprob = prob ## Save a global copy of prob's pointer usrfun

        ## New problems cannot be warm started
        self.Start[0] = 0 ## Cold start

        ## Set nF
        if( prob.mixedCons ):
            self.nF[0] = 1 + prob.Ncons
        else:
            self.nF[0] = 1 + prob.Nconslin + prob.Ncons

        ## Set lenA
        if( prob.Nconslin > 0 ):
            self.lenA[0] = ( prob.conslinA != 0 ).sum()
            self.neA[0] = self.lenA[0]
        else: ## Minimum allowed values, pg. 16
            self.lenA[0] = 1
            self.neA[0] = 0

        ## Set lenG
        if( isinstance( prob, nlp.SparseProblem ) and prob.objgpattern != None ):
            lenobjg = ( prob.objgpattern != 0 ).sum()
        else:
            lenobjg = prob.N

        if( isinstance( prob, nlp.SparseProblem ) and prob.consgpattern != None ):
            lenconsg = ( prob.consgpattern != 0 ).sum()
        else:
            lenconsg = prob.Ncons * prob.N

        self.lenG[0] = lenobjg + lenconsg
        self.neG[0] = self.lenG[0]

        ## I'm guessing the definition of m in pgs. 74,76
        self.default_iter_limit = max( 1000, 20*( prob.Ncons + prob.Nconslin ) )
        self.default_maj_iter_limit = max( 1000, prob.Ncons + prob.Nconslin )

        ## Allocate if necessary
        if( self.mustAllocate( prob.N, self.nF[0], self.lenA[0], self.lenG[0] ) ):
            self.deallocate()
            self.allocate()

        tmplb = utils.convFortran( prob.lb )
        memcpy( self.xlow, utils.getPtr( tmplb ),
                prob.N * sizeof( doublereal ) )

        tmpub = utils.convFortran( prob.ub )
        memcpy( self.xupp, utils.getPtr( tmpub ),
                prob.N * sizeof( doublereal ) )

        self.Flow[0] = -np.inf
        self.Fupp[0] = np.inf

        ## First row belongs to objg
        ## These are Fortran indices, must start from 1!
        if( isinstance( prob, nlp.SparseProblem ) and prob.objgpattern != None ):
            objgpatsparse = coo_matrix( prob.objgpattern )

            tmpiGfun = utils.convIntFortran( objgpatsparse.row + 1 )
            memcpy( &self.iGfun[0], utils.getPtr( tmpiGfun ),
                    lenobjg * sizeof( integer ) )
            tmpjGvar = utils.convIntFortran( objgpatsparse.col + 1 )
            memcpy( &self.jGvar[0], utils.getPtr( tmpjGvar ),
                    lenobjg * sizeof( integer ) )
        else:
            for idx in range( prob.N ):
                self.iGfun[idx] = 1
                self.jGvar[idx] = 1 + idx

        if( prob.Nconslin > 0 ):
            if( not prob.mixedCons ):
                tmpconslinlb = utils.convFortran( prob.conslinlb )
                memcpy( &self.Flow[1], utils.getPtr( tmpconslinlb ),
                        prob.Nconslin * sizeof( doublereal ) )

                tmpconslinub = utils.convFortran( prob.conslinub )
                memcpy( &self.Fupp[1], utils.getPtr( tmpconslinub ),
                        prob.Nconslin * sizeof( doublereal ) )

            Asparse = coo_matrix( prob.conslinA )
            assert( self.lenA[0] == len( Asparse.data ) )
            ## Linear cons come below objective, in rows 2 to prob.Nconslin + 1
            tmpiAfun = utils.convIntFortran( Asparse.row + 2 )
            memcpy( self.iAfun, utils.getPtr( tmpiAfun ),
                    self.lenA[0] * sizeof( integer ) )

            tmpjAvar = utils.convIntFortran( Asparse.col + 1 )
            memcpy( self.jAvar, utils.getPtr( tmpjAvar ),
                    self.lenA[0] * sizeof( integer ) )

            tmpA = utils.convFortran( Asparse.data )
            memcpy( self.A, utils.getPtr( tmpA ),
                    self.lenA[0] * sizeof( doublereal ) )

        if( prob.Ncons > 0 ):
            tmpidx = 1
            if( not prob.mixedCons ):
                tmpidx += prob.Nconslin

            tmpconslb = utils.convFortran( prob.conslb )
            memcpy( &self.Flow[tmpidx], utils.getPtr( tmpconslb ),
                    prob.Ncons * sizeof( doublereal ) )

            tmpconsub = utils.convFortran( prob.consub )
            memcpy( &self.Fupp[tmpidx], utils.getPtr( tmpconsub ),
                    prob.Ncons * sizeof( doublereal ) )

            if( isinstance( prob, nlp.SparseProblem ) and prob.consgpattern != None ):
                consgpatsparse = coo_matrix( prob.consgpattern )

                tmpiGfun = utils.convIntFortran( consgpatsparse.row + 1 + tmpidx )
                memcpy( &self.iGfun[lenobjg], utils.getPtr( tmpiGfun ),
                        lenconsg * sizeof( integer ) )
                tmpjGvar = utils.convIntFortran( consgpatsparse.col + 1 )
                memcpy( &self.jGvar[lenobjg], utils.getPtr( tmpjGvar ),
                        lenconsg * sizeof( integer ) )
            else:
                for jdx in range( prob.N ):
                    for idx in range( prob.Ncons ):
                        self.iGfun[lenobjg + idx + prob.Ncons * jdx] = 1 + idx + tmpidx
                        self.jGvar[lenobjg + idx + prob.Ncons * jdx] = 1 + jdx

        memset( self.xstate, 0, self.prob.N * sizeof( integer ) )
        memset( self.Fstate, 0, self.nF[0] * sizeof( integer ) )
        memset( self.Fmul, 0, self.nF[0] * sizeof( doublereal ) )


    cdef allocate( self ):
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


    cdef mustAllocate( self, N, nF, lenA, lenG ):
        if( not self.mem_alloc ):
            return True

        if( self.mem_size[0] < N or
            self.mem_size[1] < nF or
            self.mem_size[2] < lenA or
            self.mem_size[3] < lenG ):
            return True

        return False


    cdef setMemSize( self, N, nF, lenA, lenG ):
        self.mem_size[0] = N
        self.mem_size[1] = nF
        self.mem_size[2] = lenA
        self.mem_size[3] = lenG


    cdef deallocate( self ):
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


    cdef allocateWS( self ):
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


    cdef mustAllocateWS( self, lencw, leniw, lenrw ):
        if( not self.mem_alloc_ws ):
            return True

        if( self.mem_size_ws[0] < lencw or
            self.mem_size_ws[1] < leniw or
            self.mem_size_ws[2] < lenrw ):
            return True

        return False


    cdef setMemSizeWS( self, lencw, leniw, lenrw ):
        self.mem_size_ws[0] = lencw
        self.mem_size_ws[1] = leniw
        self.mem_size_ws[2] = lenrw


    cdef deallocateWS( self ):
        if( not self.mem_alloc_ws ):
            return False

        free( self.cw )
        free( self.iw )
        free( self.rw )

        self.mem_alloc_ws = False
        self.setMemSizeWS( 0, 0, 0 )
        return True


    def __dealloc__( self ):
        self.deallocateWS()
        self.deallocate()


    def warmStart( self ):
        if( not isinstance( self.prob.soln, Soln ) ):
            return False

        tmpxstate = utils.convIntFortran( self.prob.soln.xstate )
        memcpy( self.xstate, utils.getPtr( tmpxstate ), self.prob.N * sizeof( integer ) )

        tmpFstate = utils.convIntFortran( self.prob.soln.Fstate )
        memcpy( self.Fstate, utils.getPtr( tmpFstate ), self.nF[0] * sizeof( integer ) )

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
        cdef bytes printFileTmp = self.printOpts[ "printFile" ].encode() ## temp container
        cdef char* printFile = printFileTmp
        cdef bytes summaryFileTmp = self.printOpts[ "summaryFile" ].encode() ## temp container
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
        else:
            fh.openfile_( printFileUnit, printFile, inform_out,
                          len( self.printOpts[ "printFile" ] ) )
            if( inform_out[0] != 0 ):
                raise IOError( "Could not open file " + self.printOpts[ "printFile" ] )

        if( self.printOpts[ "summaryFile" ] == "stdout" ):
            summaryFileUnit[0] = 6 ## Fortran's magic value for stdout
        elif( self.printOpts[ "summaryFile" ] == "" ):
            summaryFileUnit[0] = 0 ## Disable, pg. 6
        else:
            fh.openfile_( summaryFileUnit, summaryFile, inform_out,
                          len( self.printOpts[ "summaryFile" ] ) )
            if( inform_out[0] != 0 ):
                raise IOError( "Could not open file " + self.printOpts[ "summaryFile" ] )

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

        ## Reset warm start
        self.Start[0] = 0

        ## Politely close files
        if( self.printOpts[ "printFile" ] != "" ):
            fh.closefile_( printFileUnit )

        if( self.printOpts[ "summaryFile" ] != "" and
            self.printOpts[ "summaryFile" ] != "stdout" ):
            fh.closefile_( summaryFileUnit )

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
