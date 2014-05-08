from libc.string cimport memcpy
from libc.math cimport sqrt
cimport cpython.mem as mem
cimport numpy as np
import numpy as np

from f2ch cimport *
cimport snopth as snopt
cimport utils
cimport base
import nlp

## The functions funobj and funcon should be static methods in npsol.Solver,
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
        return

    xarr = utils.wrap1dPtr( x, n[0], np.NPY_DOUBLE )

    if( needF[0] > 0 ):
        f[0] = extprob.objf( xarr )
        tmpconsf = utils.convFortran( extprob.consf( xarr ) )
        memcpy( &f[extprob.Nconslin+1], utils.getPtr( tmpconsf ),
                extprob.Ncons * sizeof( doublereal ) )

    if( needG[0] > 0 ):
        tmpobjg = utils.convFortran( extprob.objg( xarr ) )
        memcpy( G, utils.getPtr( tmpobjg ),
                extprob.N * sizeof( doublereal ) )
        tmpconsg = utils.convFortran( extprob.consg( xarr ) )
        memcpy( &G[extprob.N], utils.getPtr( tmpconsg ),
                extprob.N * extprob.Ncons * sizeof( doublereal ) )


cdef class Solver:
    cdef integer nF[1]
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
    cdef integer *iw
    cdef doublereal *rw
    cdef char *cw

    def __init__( self, prob ):
        self.nF[0] = 1 + prob.Nconslin + prob.Ncons
        self.lenA[0] = prob.Nconslin * prob.N
        self.neA[0] = self.lenA[0]
        self.lenG[0] = ( 1 + prob.Ncons ) * prob.N
        self.neG[0] = self.lenG[0]
        ## I'm here

    cdef allocate( self )
        self.x = <doublereal *> mem.PyMem_Malloc( self.prob.N * sizeof( doublereal ) )
        self.xlow = <doublereal *> mem.PyMem_Malloc( self.prob.N * sizeof( doublereal ) )
        self.xupp = <doublereal *> mem.PyMem_Malloc( self.prob.N * sizeof( doublereal ) )
        self.xmul = <doublereal *> mem.PyMem_Malloc( self.prob.N * sizeof( doublereal ) )
        self.xstate = <integer *> mem.PyMem_Malloc( self.prob.N * sizeof( integer ) )
        self.F = <doublereal *> mem.PyMem_Malloc( self.neF[0] * sizeof( doublereal ) )
        self.Flow = <doublereal *> mem.PyMem_Malloc( self.neF[0] * sizeof( doublereal ) )
        self.Fupp = <doublereal *> mem.PyMem_Malloc( self.neF[0] * sizeof( doublereal ) )
        self.Fmul = <doublereal *> mem.PyMem_Malloc( self.neF[0] * sizeof( doublereal ) )
        self.Fstate = <integer *> mem.PyMem_Malloc( self.neF[0] * sizeof( integer ) )
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
            raise MemoryError()


    cdef deallocate( self ):
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
        mem.PyMem_Free( self.rw )
        mem.PyMem_Free( self.iw )
        mem.PyMem_Free( self.cw )


    def __dealloc__( self ):
        self.deallocate()

    def x_bounds( self, low, high ):
        for i in xrange( self.n[0] ):
            self.xlow[i] = <doublereal> low[i]
            self.xupp[i] = <doublereal> high[i]

    def F_bounds( self, low, high ):
        for i in xrange( self.neF[0] ):
            self.Flow[i] = <doublereal> low[i]
            self.Fupp[i] = <doublereal> high[i]
        ## first row (objective) is ignored by solver

    def A_indices( self, i, j ):
        for k in xrange( self.lenA[0] ):
            self.iAfun[k] = <integer> i[k]
            self.jAvar[k] = <integer> j[k]

    def G_indices( self, i, j ):
        ### TODO: maybe this is the problem ###
        for k in xrange( self.lenG[0] ):
            self.iGfun[k] = <integer> i[k]
            self.jGvar[k] = <integer> j[k]

    def set_A( self, val ):
        for i in xrange( self.lenA[0] ):
            self.A[i] = <doublereal> val[i]

    def set_x( self, val ):
        for i in xrange( self.n[0] ):
            self.x[i] = <doublereal> val[i]

    def set_funobj( self, usr_function ):
        global objfun
        objfun = usr_function

    def get_x(self):
        return arrwrap.wrapPtr( self.x, self.n[0], np.NPY_DOUBLE )

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

    def set_options( self, warmstart, maxeval, constraint_violation, ftol ):
        if( not warmstart is None ):
            self.status[0] = <integer> warmstart
        if( not maxeval is None ):
            self.maxeval[0] = <integer> maxeval
        if( not constraint_violation is None ):
            self.constraint_violation[0] = <doublereal> constraint_violation
        if( not ftol is None ):
            self.ftol[0] = <doublereal> ftol

    def solve( self ):
        ## Begin calling snMemA, as described in pg. 29 of SNOPT's manual.
        cdef integer mincw[1]
        cdef integer miniw[1]
        cdef integer minrw[1]
        cdef integer lencw[1]
        cdef integer leniw[1]
        cdef integer lenrw[1]
        cdef char tmpcw[1000*8]
        cdef integer tmpiw[1000]
        cdef doublereal tmprw[1000]
        lencw[0] = leniw[0] = lenrw[0] = 1000

        print( "1. sninit" )
        sninit_( self.iPrint, self.iSumm,
                        tmpcw, lencw, tmpiw, leniw, tmprw, lenrw,
                        lencw[0]*8 )

        print( "nF: " + str( self.neF[0] ) + " "
               "n: " + str( self.n[0] ) + " "
               "nxname: " + str( self.nxname[0] ) + " "
               "nFname: " + str( self.nFname[0] ) + " "
               "lenA: " + str( self.lenA[0] ) + " "
               "lenG: " + str( self.lenG[0] ) )

        print( "2. snmema" )
        snmema_( self.INFO, self.neF, self.n, self.nxname, self.nFname, self.lenA, self.lenG,
                        mincw, miniw, minrw,
                        tmpcw, lencw, tmpiw, leniw, tmprw, lenrw,
                        lencw[0]*8 )

        print( "info: " + str( self.INFO[0] ) + " "
               "cw: " + str( mincw[0] ) + " "
               "iw: " + str( miniw[0] ) + " "
               "rw: " + str( minrw[0] ) )
        lencw[0] = mincw[0]
        leniw[0] = miniw[0]
        lenrw[0] = minrw[0]

        self.cw = <char *> calloc( lencw[0]*2, 8 * sizeof( char ) )
        self.iw = <integer *> calloc( leniw[0]*2, sizeof( integer ) )
        self.rw = <doublereal *> calloc( lenrw[0]*2, sizeof( doublereal ) )

        if( self.iw is NULL or
            self.rw is NULL or
            self.cw is NULL ):
            raise MemoryError()

        print( "3. sninit" )
        sninit_( self.iPrint, self.iSumm,
                        self.cw, lencw, self.iw, leniw, self.rw, lenrw,
                        lencw[0]*8 )

        ## Done with the setup of cw, iw, and rw.
        ## Continuing
        self.option[0] = 1 # TODO: Hardcoding the derivative option is wrong.

        print( "4. snseti minimize/maximize" )
        if( self.maximize==0 ):
            snseti_( STR_MINIMIZE, self.option,
                            self.iPrint, self.iSumm, self.INFO,
                            self.cw, lencw, self.iw, leniw, self.rw, lenrw,
                            len( STR_MINIMIZE ), lencw[0]*8 )
        else:
            snseti_( STR_MAXIMIZE, self.option,
                            self.iPrint, self.iSumm, self.INFO,
                            self.cw, lencw, self.iw, leniw, self.rw, lenrw,
                            len( STR_MAXIMIZE ), lencw[0]*8 )

        print( "5. snseti der option" )
        snseti_( STR_DERIVATIVE_OPTION, self.option,
                        self.iPrint, self.iSumm, self.INFO,
                        self.cw, lencw, self.iw, leniw, self.rw, lenrw,
                        len( STR_DERIVATIVE_OPTION ), lencw[0]*8 )

        print( "6. snseti print level" )
        self.option[0] = self.printLevel[0] # TODO: Recycling 'option' is wrong.
        snseti_( STR_MAJOR_PRINT_LEVEL, self.option,
                        self.iPrint, self.iSumm, self.INFO,
                        self.cw, lencw, self.iw, leniw, self.rw, lenrw,
                        len( STR_MAJOR_PRINT_LEVEL ), lencw[0]*8 )

        print( "7. snseti iter limit" )
        snseti_( STR_ITERATIONS_LIMIT, self.maxeval,
                        self.iPrint, self.iSumm, self.INFO,
                        self.cw, lencw, self.iw, leniw, self.rw, lenrw,
                        len( STR_ITERATIONS_LIMIT ), lencw[0]*8 )

        print( "8. snseti warm start" )
        if( self.status[0] == 1 ):
            snseti_( STR_WARM_START, self.option,
                            self.iPrint, self.iSumm, self.INFO,
                            self.cw, lencw, self.iw, leniw, self.rw, lenrw,
                            len( STR_WARM_START ), lencw[0]*8 )

        print( "9. snsetr major feas tol" )
        snsetr_( STR_MAJOR_FEASIBILITY_TOLERANCE, self.constraint_violation,
                        self.iPrint, self.iSumm, self.INFO,
                        self.cw, lencw, self.iw, leniw, self.rw, lenrw,
                        len( STR_MAJOR_FEASIBILITY_TOLERANCE ), lencw[0]*8 )

        print( "10. snsetr major opt tol" )
        snsetr_( STR_MAJOR_OPTIMALITY_TOLERANCE, self.ftol,
                        self.iPrint, self.iSumm, self.INFO,
                        self.cw, lencw, self.iw, leniw, self.rw, lenrw,
                        len( STR_MAJOR_OPTIMALITY_TOLERANCE ), lencw[0]*8 )

        print( "11. snopta" )
        snopta_( self.status, self.neF,
                        self.n, self.nxname, self.nFname,
                        self.ObjAdd, self.ObjRow, self.prob,
                        <U_fp> callback,
                        self.iAfun, self.jAvar, self.lenA, self.lenA, self.A,
                        self.iGfun, self.jGvar, self.lenG, self.lenG,
                        self.xlow, self.xupp, self.xnames,
                        self.Flow, self.Fupp, self.Fnames,
                        self.x, self.xstate, self.xmul,
                        self.F, self.Fstate, self.Fmul,
                        self.INFO,
                        mincw, miniw, minrw,
                        self.nS, self.nInf, self.sInf,
                        self.cw, lencw, self.iw, leniw, self.rw, lenrw,
                        self.cw, lencw, self.iw, leniw, self.rw, lenrw,
                        len(self.prob), self.nxname[0], self.nFname[0],
                        lencw[0]*8, lencw[0]*8 )

        # int snopta_( integer *start, integer *nf, integer *n,
        #          integer *nxname, integer *nfname, doublereal *objadd, integer *objrow,
        #          char *prob, U_fp usrfun, integer *iafun, integer *javar,
        #          integer *lena, integer *nea, doublereal *a, integer *igfun,
        #          integer *jgvar, integer *leng, integer *neg,
        #          doublereal *xlow, doublereal *xupp,
        #          char *xnames, doublereal *flow, doublereal *fupp, char *fnames,
        #          doublereal *x, integer *xstate, doublereal *xmul, doublereal *f,
        #          integer *fstate, doublereal *fmul, integer *info, integer *mincw,
        #          integer *miniw, integer *minrw, integer *ns, integer *ninf,
        #          doublereal *sinf, char *cu, integer *lencu, integer *iu, integer *leniu,
        #          doublereal *ru, integer *lenru, char *cw, integer *lencw,
        #          integer *iw, integer *leniw, doublereal *rw, integer *lenrw,
        #          ftnlen prob_len, ftnlen xnames_len, ftnlen fnames_len, ftnlen cu_len,
        #          ftnlen cw_len )

        # print( "info: " + str( self.INFO[0] ) )
        return self.F[0]

    def solve_old( self ):
        if( self.jacobian_style == "sparse" ):
            lenG = len(self.iG) + self.n
        else:
            lenG = ( self.nconstraint + 1 ) * self.n
        prob = SnoptSolver( n = self.n, neF = ( self.nconstraint + 1 ),
                            lenA = 0, lenG = lenG,
                            summaryLevel = self.print_options["summary_level"],
                            printLevel = self.print_options["print_level"],
                            printFile = self.print_options["print_file"],
                            maximize = int(self.maximize) )
        prob.x_bounds( self.xlow, self.xupp )
        prob.F_bounds( np.append( [-1e20], self.Flow), np.append( [1e20], self.Fupp ) )
        prob.set_x( self.x )
        if( self.jacobian_style == "sparse" ):
            indGx = [1 for j in range( 1, self.n+1 ) ]
            indGy = range( 1, self.n + 1 )
            indGx.extend( [ x+2 for x in self.iG ] )
            indGy.extend( [ y+1 for y in self.jG ] )
        else:
            indGx = [ i for i in range( 1, self.nconstraint + 2 ) for j in range( 1, self.n + 1 ) ]
            indGy = [ j for i in range( 1, self.nconstraint + 2 ) for j in range( 1, self.n + 1 ) ]
        print( indGx )
        print( indGy )
        prob.G_indices( indGx, indGy )


        prob.set_funobj( snoptcallback )
        prob.set_options( int( self.solve_options["warm_start"] ),
                          self.stop_options["maxeval"],
                          self.solve_options["constraint_violation"],
                          self.stop_options["ftol"] )

        answer = prob.solve()

        finalX = prob.get_x()
        status = prob.get_status()
        finalXArray = [ finalX[i] for i in range( 0, len(finalX) ) ]
        return answer, finalXArray, status
