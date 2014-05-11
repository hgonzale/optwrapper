from libc.string cimport memcpy, memset
from libc.math cimport sqrt
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
cdef char* STR_NOLIST = "Nolist"
cdef char* STR_PRINT_FILE = "Print file"
cdef char* STR_SUMMARY_FILE = "Summary file"
cdef char* STR_PRINT_LEVEL = "Print level"
cdef char* STR_MINOR_PRINT_LEVEL = "Minor print level"
cdef char* STR_INFINITE_BOUND_SIZE = "Infinite bound size"
cdef char* STR_ITERATION_LIMIT = "Iteration limit"
cdef char* STR_MINOR_ITERATION_LIMIT = "Minor iteration limit"
cdef char* STR_LINE_SEARCH_TOLERANCE = "Line search tolerance"
cdef char* STR_FEASIBILITY_TOLERANCE = "Feasibility tolerance"
cdef char* STR_OPTIMALITY_TOLERANCE = "Optimality tolerance"
cdef char* STR_FUNCTION_PRECISION = "Function precision"
cdef char* STR_VERIFY_LEVEL = "Verify level"
cdef char* STR_WARM_START = "Warm start"
cdef char* STR_TOTAL_CHARACTER_WORKSPACE = "Total character workspace"
cdef char* STR_TOTAL_INTEGER_WORKSPACE = "Total integer workspace"
cdef char* STR_TOTAL_REAL_WORKSPACE = "Total real workspace"

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
                 integer *needF, integer *neF, doublereal *f,
                 integer *needG, integer *neG, doublereal *G,
                 char *cu, integer *lencu,
                 integer *iu, integer *leniu,
                 doublereal *ru, integer *lenru ):
    if( status[0] == 2 ): ## Final call, do nothing
        return 0

    xarr = utils.wrap1dPtr( x, n[0], np.NPY_DOUBLE )

    if( needF[0] > 0 ):
        f[0] = extprob.objf( xarr )
        tmpconsf = utils.convFortran( extprob.consf( xarr ) )
        memcpy( &f[extprob.Nconslin+1], utils.getPtr( tmpconsf ),
                extprob.Ncons * sizeof( doublereal ) )

    ## Saving every entry in gradients
    ## This behavior should change in the near future
    if( needG[0] > 0 ):
        tmpobjg = utils.convFortran( extprob.objg( xarr ) )
        memcpy( G, utils.getPtr( tmpobjg ),
                extprob.N * sizeof( doublereal ) )
        tmpconsg = utils.convFortran( extprob.consg( xarr ) )
        memcpy( &G[extprob.N], utils.getPtr( tmpconsg ),
                extprob.N * extprob.Ncons * sizeof( doublereal ) )


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
    cdef int mem_size[3] ## N, Nconslin, Ncons


    def __init__( self, prob=None ):
        super().__init__()

        self.mem_alloc = False
        self.mem_size[0] = self.mem_size[1] = self.mem_size[2] = 0
        # self.default_tol = sqrt( np.spacing(1) ) ## pg. 24
        # self.default_fctn_prec = np.power( np.spacing(1), 0.9 ) ## pg. 24
        self.prob = None

        ## Set options
        self.printOpts[ "summaryFile" ] = ""
        self.printOpts[ "printLevel" ] = 0
        self.printOpts[ "minorPrintLevel" ] = 0
        # self.solveOpts[ "infValue" ] = 1e20
        # self.solveOpts[ "iterLimit" ] = self.default_iter_limit
        # self.solveOpts[ "minorIterLimit" ] = self.default_iter_limit
        # self.solveOpts[ "lineSearchTol" ] = 0.9
        # self.solveOpts[ "fctnPrecision" ] = 0 ## Invalid value
        # self.solveOpts[ "feasibilityTol" ] = 0 ## Invalid value
        # self.solveOpts[ "optimalityTol" ] = 0 ## Invalid value
        # self.solveOpts[ "verifyGrad" ] = False

        ## We are assuming np.float64 equals doublereal from now on
        ## At least we need to be sure that doublereal is 8 bytes in this architecture
        assert( sizeof( doublereal ) == 8 )

        if( prob ):
            self.setupProblem( prob )



    def setupProblem( self, prob ):
        global extprob

        if( not isinstance( prob, nlp.Problem ) ):
            raise TypeError( "Argument 'prob' must be of type 'nlp.Problem'" )

        self.prob = prob ## Save a copy of prob's pointer
        extprob = prob ## Save a global copy prob's pointer for funcon and funobj

        ## New problems cannot be warm started
        self.warm_start = False
        self.Start[0] = 0

        ## Set size-dependent constants
        self.nF[0] = 1 + prob.Nconslin + prob.Ncons
        self.lenA[0] = prob.Nconslin * prob.N
        self.neA[0] = self.lenA[0]
        self.lenG[0] = ( 1 + prob.Ncons ) * prob.N
        self.neG[0] = self.lenG[0]

        ## Allocate if necessary
        if( not self.mem_alloc ):
            self.allocate()
        elif( self.mem_size[0] != prob.N or
              self.mem_size[1] != prob.Nconslin or
              self.mem_size[2] != prob.Ncons ):
            self.deallocate()
            self.allocate()

        ## Require all arrays we are going to copy to be:
        ## two-dimensional, float64, fortran contiguous, and type aligned
        tmplb = utils.convFortran( prob.lb )
        memcpy( self.xlow, utils.getPtr( tmplb ),
                prob.N * sizeof( doublereal ) )

        tmpub = utils.convFortran( prob.ub )
        memcpy( self.xupp, utils.getPtr( tmpub ),
                prob.N * sizeof( doublereal ) )

        self.Flow[0] = -np.inf
        self.Fupp[0] = np.inf

        ## We fill iGfun and jGvar as we write them in usrfun
        ## First row belongs to objg
        for idx in range( self.prob.N ):
            self.iGfun[idx] = 1 ## These are Fortran indices, must start from 1!
            self.jGvar[idx] = 1 + idx

        if( prob.Nconslin > 0 ):
            tmpconslinlb = utils.convFortran( prob.conslinlb )
            memcpy( &self.Flow[1], utils.getPtr( tmpconslinlb ),
                    prob.Nconslin * sizeof( doublereal ) )

            tmpconslinub = utils.convFortran( prob.conslinub )
            memcpy( &self.Fupp[1], utils.getPtr( tmpconslinub ),
                    prob.Nconslin * sizeof( doublereal ) )

            Asparse = coo_matrix( prob.conslinA )
            tmpiAfun = utils.convIntFortran( Asparse.row )
            memcpy( self.iAfun, utils.getPtr( tmpiAfun ),
                    self.lenA[0] * sizeof( integer ) )

            tmpjAvar = utils.convIntFortran( Asparse.col )
            memcpy( self.jAvar, utils.getPtr( tmpjAvar ),
                    self.lenA[0] * sizeof( integer ) )

            tmpA = utils.convFortran( Asparse.data )
            memcpy( self.A, utils.getPtr( tmpA ),
                    self.lenA[0] * sizeof( doublereal ) )

        if( prob.Ncons > 0 ):
            tmpconslb = utils.convFortran( prob.conslb )
            memcpy( &self.Flow[1], utils.getPtr( tmpconslb ),
                    prob.Ncons * sizeof( doublereal ) )

            tmpconsub = utils.convFortran( prob.consub )
            memcpy( &self.Fupp[1], utils.getPtr( tmpconsub ),
                    prob.Ncons * sizeof( doublereal ) )

            ## Rest is filled in Fortran-order
            for jdx in range( self.prob.Ncons ):
                for idx in range( self.prob.N ):
                    self.iGfun[self.prob.N + idx + self.prob.N * jdx] = 2 + idx + self.prob.Nconslin
                    self.jGvar[self.prob.N + idx + self.prob.N * jdx] = 1 + jdx

        memset( self.xstate, 0, self.prob.N * sizeof( integer ) )
        memset( self.Fstate, 0, self.nF[0] * sizeof( integer ) )
        memset( self.Fmul, 0, self.nF[0] * sizeof( doublereal ) )


    cdef allocate( self ):
        if( self.mem_alloc ):
            return False

        self.x = <doublereal *> mem.PyMem_Malloc( self.prob.N * sizeof( doublereal ) )
        self.xlow = <doublereal *> mem.PyMem_Malloc( self.prob.N * sizeof( doublereal ) )
        self.xupp = <doublereal *> mem.PyMem_Malloc( self.prob.N * sizeof( doublereal ) )
        self.xmul = <doublereal *> mem.PyMem_Malloc( self.prob.N * sizeof( doublereal ) )
        self.xstate = <integer *> mem.PyMem_Malloc( self.prob.N * sizeof( integer ) )
        self.F = <doublereal *> mem.PyMem_Malloc( self.nF[0] * sizeof( doublereal ) )
        self.Flow = <doublereal *> mem.PyMem_Malloc( self.nF[0] * sizeof( doublereal ) )
        self.Fupp = <doublereal *> mem.PyMem_Malloc( self.nF[0] * sizeof( doublereal ) )
        self.Fmul = <doublereal *> mem.PyMem_Malloc( self.nF[0] * sizeof( doublereal ) )
        self.Fstate = <integer *> mem.PyMem_Malloc( self.nF[0] * sizeof( integer ) )
        self.A = <doublereal *> mem.PyMem_Malloc( self.lenA[0] * sizeof( doublereal ) )
        self.iAfun = <integer *> mem.PyMem_Malloc( self.lenA[0] * sizeof( integer ) )
        self.jAvar = <integer *> mem.PyMem_Malloc( self.lenA[0] * sizeof( integer ) )
        self.iGfun = <integer *> mem.PyMem_Malloc( self.lenG[0] * sizeof( integer ) )
        self.jGvar = <integer *> mem.PyMem_Malloc( self.lenG[0] * sizeof( integer ) )

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
        self.mem_size[0] = self.prob.N
        self.mem_size[1] = self.prob.Nconslin
        self.mem_size[2] = self.prob.Ncons
        return True


    cdef deallocate( self ):
        if( not self.mem_alloc ):
            return False

        mem.PyMem_Free( self.x )
        mem.PyMem_Free( self.xlow )
        mem.PyMem_Free( self.xupp )
        mem.PyMem_Free( self.xstate )
        mem.PyMem_Free( self.xmul )
        mem.PyMem_Free( self.F )
        mem.PyMem_Free( self.Flow )
        mem.PyMem_Free( self.Fupp )
        mem.PyMem_Free( self.Fstate )
        mem.PyMem_Free( self.Fmul )
        mem.PyMem_Free( self.A )
        mem.PyMem_Free( self.iAfun )
        mem.PyMem_Free( self.jAvar )
        mem.PyMem_Free( self.iGfun )
        mem.PyMem_Free( self.jGvar )

        self.mem_alloc = False
        return True


    def __dealloc__( self ):
        self.deallocate()


    def get_status(self):
        if( self.INFO[0] == 1 ):
            return 'optimality conditions satisfied'
        elif( self.INFO[0] == 2 ):
            return 'feasible point found'
        elif( self.INFO[0] == 3 ):
            return 'requested accuracy could not be achieved'
        elif( self.INFO[0] < 20 ):
            return 'the problem appears to be infeasible'
        elif( self.INFO[0] < 30 ):
            return 'the problem appears to be unbounded'
        elif( self.INFO[0] < 40 ):
            return 'resource limit error'
        elif( self.INFO[0] < 50 ):
            return 'terminated after numerical difficulties'
        else:
            return 'error in user supplied information'


    def solve( self ):
        cdef integer *n = [ self.prob.N ]
        cdef integer *nxname = [ 1 ] ## Do not provide vars names
        cdef integer *nFname = [ 1 ] ## Do not provide cons names
        cdef doublereal *ObjAdd = [ 0.0 ]
        cdef integer *ObjRow = [ 1 ]
        cdef char *probname = "optwrapp" ## Must have 8 characters
        cdef char *xnames = "dummy"
        cdef char *Fnames = "dummy"
        cdef integer nS[1]
        cdef integer nInf[1]
        cdef doublereal sInf[1]
        cdef integer mincw[1]
        cdef integer miniw[1]
        cdef integer minrw[1]

        cdef bytes printFileTmp = self.printOpts[ "printFile" ].encode() ## temp container
        cdef char* printFile = printFileTmp
        cdef bytes summaryFileTmp = self.printOpts[ "summaryFile" ].encode() ## temp container
        cdef char* summaryFile = summaryFileTmp
        cdef integer* summaryFileUnit = [ 89 ] ## Hardcoded since nobody cares
        cdef integer* printFileUnit = [ 90 ] ## Hardcoded since nobody cares

        cdef integer inform_out[1]
        cdef integer *ltmpcw = [ 500 ]
        cdef integer *ltmpiw = [ 500 ]
        cdef integer *ltmprw = [ 500 ]
        cdef char tmpcw[500*8]
        cdef integer tmpiw[500]
        cdef doublereal tmprw[500]

        ## Begin by setting up initial condition
        tmpinit = utils.convFortran( self.prob.init )
        memcpy( self.x, utils.getPtr( tmpinit ),
                self.prob.N * sizeof( doublereal ) )

                ## Handle debug files
        if( self.printOpts[ "printFile" ] != "" ):
            fh.openfile_( printFileUnit, printFile, inform_out,
                          len( self.printOpts[ "printFile" ] ) )
            if( inform_out[0] != 0 ):
                raise IOError( "Could not open file " + self.printOpts[ "printFile" ] )
        else:
            printFileUnit[0] = 0

        # if( self.printOpts[ "summaryFile" ] != "" ):
        #     fh.openfile_( summaryFileUnit, summaryFile, inform_out,
        #                   len( self.printOpts[ "summaryFile" ] ) )
        #     if( inform_out[0] != 0 ):
        #         raise IOError( "Could not open file " + self.printOpts[ "summaryFile" ] )
        # else:
        #     summaryFileUnit[0] = 0

        ## Initialize
        snopt.sninit_( printFileUnit, summaryFileUnit,
                       tmpcw, ltmpcw, tmpiw, ltmpiw, tmprw, ltmprw,
                       ltmpcw[0]*8 )

        ## Estimate workspace memory requirements
        snopt.snmema_( inform_out, self.nF, n, nxname, nFname, self.lenA, self.lenG,
                       self.lencw, self.leniw, self.lenrw,
                       tmpcw, ltmpcw, tmpiw, ltmpiw, tmprw, ltmprw,
                       ltmpcw[0]*8 )

        # print( "info: " + str( inform_out[0] ) + " "
        #        "cw: " + str( self.lencw[0] ) + " "
        #        "iw: " + str( self.leniw[0] ) + " "
        #        "rw: " + str( self.lenrw[0] ) )

        if( inform_out[0] != 104 ):
            raise Exception( "snopt.snMemA failed to estimate workspace memory requirements" )

        ## Allocate workspace memory
        self.cw = <char *> mem.PyMem_Malloc( self.lencw[0] * 8 * sizeof( char ) )
        self.iw = <integer *> mem.PyMem_Malloc( self.leniw[0] * sizeof( integer ) )
        self.rw = <doublereal *> mem.PyMem_Malloc( self.lenrw[0] * sizeof( doublereal ) )

        if( self.iw is NULL or
            self.rw is NULL or
            self.cw is NULL ):
            raise MemoryError( "At least one memory allocation failed" )

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

        mem.PyMem_Free( self.cw )
        mem.PyMem_Free( self.iw )
        mem.PyMem_Free( self.rw )

        ## Politely close files
        if( self.printOpts[ "printFile" ] != "" ):
            fh.closefile_( printFileUnit )

        if( self.printOpts[ "summaryFile" ] != "" ):
            fh.closefile_( summaryFileUnit )

        ## Save result to prob
        self.prob.soln = Soln()
        self.prob.soln.value = float( self.F[0] )
        self.prob.soln.final = np.copy( utils.wrap1dPtr( self.x, self.prob.N,
                                                         np.NPY_DOUBLE ) )
        self.prob.soln.xstate = np.copy( utils.wrap1dPtr( self.xstate, self.prob.N,
                                                          np.NPY_LONG ) )
        self.prob.soln.xmul = np.copy( utils.wrap1dPtr( self.xmul, self.prob.N,
                                                        np.NPY_DOUBLE ) )
        self.prob.soln.Fstate = np.copy( utils.wrap1dPtr( self.xstate, self.nF[0],
                                                          np.NPY_LONG ) )
        self.prob.soln.Fmul = np.copy( utils.wrap1dPtr( self.Fmul, self.nF[0],
                                                        np.NPY_DOUBLE ) )
        self.prob.soln.nS = int( nS[0] )
        self.prob.soln.retval = int( inform_out[0] )

        return( self.prob.soln.final,
                self.prob.soln.value,
                self.prob.soln.retval )
