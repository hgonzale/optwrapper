cimport csnopt
from libc.stdlib cimport calloc, free
cimport numpy as np
cimport arrayWrapper as arrwrap

cdef object objfun

cdef int callback( csnopt.integer *status, csnopt.integer *n, csnopt.doublereal *x,
                   csnopt.integer *needF, csnopt.integer *neF, csnopt.doublereal *F,
                   csnopt.integer *needG, csnopt.integer *neG, csnopt.doublereal *G,
                   char *cu, csnopt.integer *lencu,
                   csnopt.integer *iu, csnopt.integer *leniu,
                   csnopt.doublereal *ru, csnopt.integer *lenru ):
    px = arrwrap.wrapPtr( x, n[0], np.NPY_DOUBLE )
    pF = arrwrap.wrapPtr( F, neF[0], np.NPY_DOUBLE )
    pG = arrwrap.wrapPtr( G, neG[0], np.NPY_DOUBLE )
    objfun( status[0], n[0], px, needF[0], neF[0], pF, needG[0], neG[0], pG )

cdef class SnoptSolver:
    cdef csnopt.integer status[1]
    cdef csnopt.integer INFO[1]
    cdef csnopt.integer neF[1]
    cdef csnopt.integer n[1]
    cdef csnopt.integer nxname[1]
    cdef csnopt.integer nFname[1]
    cdef csnopt.integer lenA[1]
    cdef csnopt.integer lenG[1]
    cdef csnopt.integer option[1]
    cdef csnopt.integer iPrint[1]
    cdef csnopt.integer iSumm[1]
    cdef csnopt.integer ObjRow[1]
    cdef csnopt.integer nS[1]
    cdef csnopt.integer nInf[1]
    cdef csnopt.integer npname[1]
    cdef csnopt.integer maxeval[1]
    cdef csnopt.doublereal ObjAdd[1]
    cdef csnopt.doublereal sInf[1]
    cdef csnopt.doublereal constraint_violation[1]
    cdef csnopt.doublereal ftol[1]
    cdef csnopt.doublereal *x
    cdef csnopt.doublereal *xlow
    cdef csnopt.doublereal *xupp
    cdef csnopt.doublereal *xmul
    cdef csnopt.doublereal *F
    cdef csnopt.doublereal *Flow
    cdef csnopt.doublereal *Fupp
    cdef csnopt.doublereal *Fmul
    cdef csnopt.doublereal *A
    cdef csnopt.integer *iAfun
    cdef csnopt.integer *jAvar
    cdef csnopt.integer *iGfun
    cdef csnopt.integer *jGvar
    cdef csnopt.integer *xstate
    cdef csnopt.integer *Fstate
    cdef csnopt.integer *iw
    cdef csnopt.doublereal *rw
    cdef char *cw
    cdef char prob[200]
    cdef char xnames[1]
    cdef char Fnames[1]
    cdef csnopt.integer printLevel[1]
    cdef csnopt.integer summaryLevel[1]
    cdef int maximize
    # cdef csnopt.olist ol[1]

    def __cinit__( self, **kwargs ):
        self.status[0] = 0
        self.INFO[0] = 0
        self.nxname[0] = 1
        self.nFname[0] = 1
        self.ObjAdd[0] = 0
        self.ObjRow[0] = 1
        self.npname[0] = 0
        self.n[0] = kwargs['n']
        self.neF[0] = kwargs['neF']
        self.printLevel[0] = kwargs['printLevel']
        self.summaryLevel[0] = kwargs['summaryLevel']
        if( self.summaryLevel[0] > 0 ):
            self.iSumm[0] = 6
        if( kwargs['printFile'] == None ):
            self.iPrint[0] = 0
        else:
            self.iPrint[0] = 9
            ## Sketchy code. I'm sure there are easier ways to open a file.
            # self.ol[0].oerr=1
            # self.ol[0].ounit=self.iPrint[0]
            # self.ol[0].ofnmlen=len(kwargs['printFile'])
            # #for i in range(0,self.ol[0].ofnmlen):
            #     #self.ol[0].ofnm[i]=kwargs['printFile'][i]
            # self.ol[0].ofnm=kwargs['printFile']
            # self.ol[0].orl = 0
            # self.ol[0].osta = "UNKNOWN"
            # self.ol[0].oacc = NULL
            # self.ol[0].ofm = NULL
            # self.ol[0].oblnk = NULL
            # self.INFO[0]=csnopt.f_open(self.ol)
            # if self.INFO[0]>0:
            #     raise ValueError('invalid file inform: '+str(self.INFO[0]))
        self.lenA[0] = kwargs['lenA']
        self.lenG[0] = kwargs['lenG']
        self.maximize = kwargs['maximize']
        self.maxeval[0] = 10000
        self.constraint_violation[0] = 1e-6
        self.ftol[0] = 1e-6
        self.status[0] = 0
        self.x = <csnopt.doublereal *> calloc( self.n[0], sizeof( csnopt.doublereal ) )
        self.xlow = <csnopt.doublereal *> calloc( self.n[0], sizeof( csnopt.doublereal ) )
        self.xupp = <csnopt.doublereal *> calloc( self.n[0], sizeof( csnopt.doublereal ) )
        self.xmul = <csnopt.doublereal *> calloc( self.n[0], sizeof( csnopt.doublereal ) )
        self.xstate = <csnopt.integer *> calloc( self.n[0], sizeof( csnopt.integer ) )
        self.F = <csnopt.doublereal *> calloc( self.neF[0], sizeof( csnopt.doublereal ) )
        self.Flow = <csnopt.doublereal *> calloc( self.neF[0], sizeof( csnopt.doublereal ) )
        self.Fupp = <csnopt.doublereal *> calloc( self.neF[0], sizeof( csnopt.doublereal ) )
        self.Fmul = <csnopt.doublereal *> calloc( self.neF[0], sizeof( csnopt.doublereal ) )
        self.Fstate = <csnopt.integer *> calloc( self.neF[0], sizeof( csnopt.integer ) )
        self.A = <csnopt.doublereal *> calloc( self.lenA[0], sizeof( csnopt.doublereal ) )
        self.iAfun = <csnopt.integer *> calloc( self.lenA[0], sizeof( csnopt.integer ) )
        self.jAvar = <csnopt.integer *> calloc( self.lenA[0], sizeof( csnopt.integer ) )
        self.iGfun = <csnopt.integer *> calloc( self.lenG[0], sizeof( csnopt.integer ) )
        self.jGvar = <csnopt.integer *> calloc( self.lenG[0], sizeof( csnopt.integer ) )

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

    def __dealloc__( self ):
        if( self.x is not NULL ):
            free( self.x )
        if( self.xlow is not NULL ):
            free( self.xlow )
        if( self.xupp is not NULL ):
            free( self.xupp )
        if( self.xstate is not NULL ):
            free( self.xstate )
        if( self.xmul is not NULL ):
            free( self.xmul )
        if( self.F is not NULL ):
            free( self.F )
        if( self.Flow is not NULL ):
            free( self.Flow )
        if( self.Fupp is not NULL ):
            free( self.Fupp )
        if( self.Fstate is not NULL ):
            free( self.Fstate )
        if( self.Fmul is not NULL ):
            free( self.Fmul )
        if( self.A is not NULL ):
            free( self.A )
        if( self.iAfun is not NULL ):
            free( self.iAfun )
        if( self.jAvar is not NULL ):
            free( self.jAvar )
        if( self.iGfun is not NULL ):
            free( self.iGfun )
        if( self.jGvar is not NULL ):
            free( self.jGvar )
        if( self.rw is not NULL ):
            free( self.rw )
        if( self.iw is not NULL ):
            free( self.iw )
        if( self.cw is not NULL ):
            free( self.cw )

    def x_bounds( self, low, high ):
        for i in xrange( self.n[0] ):
            self.xlow[i] = <csnopt.doublereal> low[i]
            self.xupp[i] = <csnopt.doublereal> high[i]

    def F_bounds( self, low, high ):
        for i in xrange( self.neF[0] ):
            self.Flow[i] = <csnopt.doublereal> low[i]
            self.Fupp[i] = <csnopt.doublereal> high[i]
        ## first row (objective) is ignored by solver

    def A_indices( self, i, j ):
        for k in xrange( self.lenA[0] ):
            self.iAfun[k] = <csnopt.integer> i[k]
            self.jAvar[k] = <csnopt.integer> j[k]

    def G_indices( self, i, j ):
        ### TODO: maybe this is the problem ###
        for k in xrange( self.lenG[0] ):
            self.iGfun[k] = <csnopt.integer> i[k]
            self.jGvar[k] = <csnopt.integer> j[k]

    def set_A( self, val ):
        for i in xrange( self.lenA[0] ):
            self.A[i] = <csnopt.doublereal> val[i]

    def set_x( self, val ):
        for i in xrange( self.n[0] ):
            self.x[i] = <csnopt.doublereal> val[i]

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
            self.status[0] = <csnopt.integer> warmstart
        if( not maxeval is None ):
            self.maxeval[0] = <csnopt.integer> maxeval
        if( not constraint_violation is None ):
            self.constraint_violation[0] = <csnopt.doublereal> constraint_violation
        if( not ftol is None ):
            self.ftol[0] = <csnopt.doublereal> ftol

    def solve( self ):
        ## Begin calling snMemA, as described in pg. 29 of SNOPT's manual.
        cdef csnopt.integer mincw[1]
        cdef csnopt.integer miniw[1]
        cdef csnopt.integer minrw[1]
        cdef csnopt.integer lencw[1]
        cdef csnopt.integer leniw[1]
        cdef csnopt.integer lenrw[1]
        cdef char tmpcw[1000*8]
        cdef csnopt.integer tmpiw[1000]
        cdef csnopt.doublereal tmprw[1000]
        lencw[0] = leniw[0] = lenrw[0] = 1000

        print( "1. sninit" )
        csnopt.sninit_( self.iPrint, self.iSumm,
                        tmpcw, lencw, tmpiw, leniw, tmprw, lenrw,
                        lencw[0]*8 )

        print( "nF: " + str( self.neF[0] ) + " "
               "n: " + str( self.n[0] ) + " "
               "nxname: " + str( self.nxname[0] ) + " "
               "nFname: " + str( self.nFname[0] ) + " "
               "lenA: " + str( self.lenA[0] ) + " "
               "lenG: " + str( self.lenG[0] ) )

        print( "2. snmema" )
        csnopt.snmema_( self.INFO, self.neF, self.n, self.nxname, self.nFname, self.lenA, self.lenG,
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
        self.iw = <csnopt.integer *> calloc( leniw[0]*2, sizeof( csnopt.integer ) )
        self.rw = <csnopt.doublereal *> calloc( lenrw[0]*2, sizeof( csnopt.doublereal ) )

        if( self.iw is NULL or
            self.rw is NULL or
            self.cw is NULL ):
            raise MemoryError()

        print( "3. sninit" )
        csnopt.sninit_( self.iPrint, self.iSumm,
                        self.cw, lencw, self.iw, leniw, self.rw, lenrw,
                        lencw[0]*8 )

        ## Done with the setup of cw, iw, and rw.
        ## Continuing
        self.option[0] = 1 # TODO: Hardcoding the derivative option is wrong.

        print( "4. snseti minimize/maximize" )
        if( self.maximize==0 ):
            csnopt.snseti_( csnopt.STR_MINIMIZE, self.option,
                            self.iPrint, self.iSumm, self.INFO,
                            self.cw, lencw, self.iw, leniw, self.rw, lenrw,
                            len( csnopt.STR_MINIMIZE ), lencw[0]*8 )
        else:
            csnopt.snseti_( csnopt.STR_MAXIMIZE, self.option,
                            self.iPrint, self.iSumm, self.INFO,
                            self.cw, lencw, self.iw, leniw, self.rw, lenrw,
                            len( csnopt.STR_MAXIMIZE ), lencw[0]*8 )

        print( "5. snseti der option" )
        csnopt.snseti_( csnopt.STR_DERIVATIVE_OPTION, self.option,
                        self.iPrint, self.iSumm, self.INFO,
                        self.cw, lencw, self.iw, leniw, self.rw, lenrw,
                        len( csnopt.STR_DERIVATIVE_OPTION ), lencw[0]*8 )

        print( "6. snseti print level" )
        self.option[0] = self.printLevel[0] # TODO: Recycling 'option' is wrong.
        csnopt.snseti_( csnopt.STR_MAJOR_PRINT_LEVEL, self.option,
                        self.iPrint, self.iSumm, self.INFO,
                        self.cw, lencw, self.iw, leniw, self.rw, lenrw,
                        len( csnopt.STR_MAJOR_PRINT_LEVEL ), lencw[0]*8 )

        print( "7. snseti iter limit" )
        csnopt.snseti_( csnopt.STR_ITERATIONS_LIMIT, self.maxeval,
                        self.iPrint, self.iSumm, self.INFO,
                        self.cw, lencw, self.iw, leniw, self.rw, lenrw,
                        len( csnopt.STR_ITERATIONS_LIMIT ), lencw[0]*8 )

        print( "8. snseti warm start" )
        if( self.status[0] == 1 ):
            csnopt.snseti_( csnopt.STR_WARM_START, self.option,
                            self.iPrint, self.iSumm, self.INFO,
                            self.cw, lencw, self.iw, leniw, self.rw, lenrw,
                            len( csnopt.STR_WARM_START ), lencw[0]*8 )

        print( "9. snsetr major feas tol" )
        csnopt.snsetr_( csnopt.STR_MAJOR_FEASIBILITY_TOLERANCE, self.constraint_violation,
                        self.iPrint, self.iSumm, self.INFO,
                        self.cw, lencw, self.iw, leniw, self.rw, lenrw,
                        len( csnopt.STR_MAJOR_FEASIBILITY_TOLERANCE ), lencw[0]*8 )

        print( "10. snsetr major opt tol" )
        csnopt.snsetr_( csnopt.STR_MAJOR_OPTIMALITY_TOLERANCE, self.ftol,
                        self.iPrint, self.iSumm, self.INFO,
                        self.cw, lencw, self.iw, leniw, self.rw, lenrw,
                        len( csnopt.STR_MAJOR_OPTIMALITY_TOLERANCE ), lencw[0]*8 )

        print( "11. snopta" )
        csnopt.snopta_( self.status, self.neF,
                        self.n, self.nxname, self.nFname,
                        self.ObjAdd, self.ObjRow, self.prob,
                        <csnopt.U_fp> callback,
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

        ## Page 23, Section 3.6, of SNOPT"s manual
        def snoptcallback( status, n, x, needF, neF, F, needG, neG, G ):
            if( needF > 0 ):
                F[0] = self.objf(x)
                con = self.con(x)
                for i in range( 0, self.nconstraint ):
                    F[i+1] = con[i]

            if( needG > 0 ):
                objgrad = self.objgrad(x)
                for i in range( 0, self.n ):
                    G[i] = objgrad[i]

                if( self.jacobian_style == "sparse" ):
                    congrad = self.congrad(x)
                else:
                    congrad = np.asarray( self.congrad(x) ).reshape(-1)

                for i in range( 0, len(congrad) ):
                    G[i + self.n] = congrad[i]

            print( "F: " + str(F) + " G: " + str(G) )
            assert( len(F) == neF and len(G) == neG ) # Just being a bit paranoid.

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
