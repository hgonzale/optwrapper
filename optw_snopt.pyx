cimport csnopt
cimport cpython
from libc.stdlib cimport calloc, free
cimport cpython.pycapsule as pycapsule

cdef object objfun

cdef int callback( csnopt.integer* status, csnopt.integer* n, csnopt.doublereal* x,
                   csnopt.integer* needF, csnopt.integer* neF, csnopt.doublereal* F,
                   csnopt.integer* needG, csnopt.integer* neG, csnopt.doublereal* G,
                   char* cu, csnopt.integer* lencu,
                   csnopt.integer* iu, csnopt.integer* leniu,
                   csnopt.doublereal* ru, csnopt.integer* lenru ):
    px = plist( pycapsule.PyCapsule_New( x, NULL, NULL ), n[0] )
    pF = plist( pycapsule.PyCapsule_New( F, NULL, NULL ), neF[0] )
    pG = plist( pycapsule.PyCapsule_New( G, NULL, NULL ), neG[0] )
    objfun( status[0], n[0], px, needF[0], neF[0], pF, needG[0], neG[0], pG )

cdef class plist:
    cdef csnopt.doublereal* val
    cdef int len

    def __cinit__( self, capsule, int size ):
        self.val = <csnopt.doublereal*> pycapsule.PyCapsule_GetPointer( capsule, NULL )
        self.len = size

    def __len__( self ):
        return self.len

    def __getitem__( self, i ):
        if( type(i) == int and i >= 0 and i < self.len ):
            return self.val[i]
        else:
            raise ValueError( 'invalid index on getitem: ' + str(i) )

    def __delitem__( self, i ):
        raise AttributeError( 'cannot delete item' )

    def __setitem__( self, i, v ):
        if( type(i) == int and i >= 0 and i < len ):
            self.val[i] = <csnopt.doublereal> v
        else:
            raise ValueError( 'invalid index on setitem: ' + str(i) )

    def __str__( self ):
        s = "["
        for i in range( 0, self.len ):
            s = s + str( self.val[i] ) + ","
        return s + "]"

    def insert( self, i, v ):
        raise AttributeError( 'cannot insert item' )

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
            self.iSumm[0]=6
        if( kwargs['printFile'] == None ):
            self.iPrint[0]=0
        else:
            self.iPrint[0]=9
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
        return plist( pycapsule.PyCapsule_New( self.x, NULL, NULL), self.n[0] )

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
        cdef csnopt.integer ltmpcw[1]
        cdef csnopt.integer ltmpiw[1]
        cdef csnopt.integer ltmprw[1]
        cdef char tmpcw[1000*8]
        cdef csnopt.integer tmpiw[1000]
        cdef csnopt.doublereal tmprw[1000]
        ltmpcw[0] = ltmpiw[0] = ltmprw[0] = 1000

        csnopt.sninit_( self.iPrint, self.iSumm,
                        tmpcw, ltmpcw, tmpiw, ltmpiw, tmprw, ltmprw,
                        ltmpcw[0]*8 )

        print( "nF: " + str( self.neF[0] ) + " "
               "n: " + str( self.n[0] ) + " "
               "nxname: " + str( self.nxname[0] ) + " "
               "nFname: " + str( self.nFname[0] ) + " "
               "lenA: " + str( self.lenA[0] ) + " "
               "lenG: " + str( self.lenG[0] ) )

        csnopt.snmema_( self.INFO, self.neF, self.n, self.nxname, self.nFname, self.lenA, self.lenG,
                        mincw, miniw, minrw,
                        tmpcw, ltmpcw, tmpiw, ltmpiw, tmprw, ltmprw,
                        ltmpcw[0]*8 )

        print( "info: " + str( self.INFO[0] ) + " "
               "cw: " + str( mincw[0] ) + " "
               "iw: " + str( miniw[0] ) + " "
               "rw: " + str( minrw[0] ) )

        self.cw = <char *> calloc( mincw[0], 8 * sizeof( char ) )
        self.iw = <csnopt.integer *> calloc( miniw[0], sizeof( csnopt.integer ) )
        self.rw = <csnopt.doublereal *> calloc( minrw[0], sizeof( csnopt.doublereal ) )

        if( self.iw is NULL or
            self.rw is NULL or
            self.cw is NULL ):
            raise MemoryError()

        csnopt.sninit_( self.iPrint, self.iSumm,
                        self.cw, mincw, self.iw, miniw, self.rw, minrw,
                        mincw[0]*8 )

        ## Done with the setup of cw, iw, and rw.
        ## Continuing
        self.option[0] = 1
        if( self.maximize==0 ):
            csnopt.snseti_( csnopt.STR_MINIMIZE, self.option,
                            self.iPrint, self.iSumm, self.INFO,
                            self.cw, mincw, self.iw, miniw, self.rw, minrw,
                            len( csnopt.STR_MINIMIZE ), mincw[0]*8 )
        else:
            csnopt.snseti_( csnopt.STR_MAXIMIZE, self.option,
                            self.iPrint, self.iSumm, self.INFO,
                            self.cw, mincw, self.iw, miniw, self.rw, minrw,
                            len( csnopt.STR_MAXIMIZE ), mincw[0]*8 )

        csnopt.snseti_( csnopt.STR_DERIVATIVE_OPTION, self.option,
                        self.iPrint, self.iSumm, self.INFO,
                        self.cw, mincw, self.iw, miniw, self.rw, minrw,
                        len( csnopt.STR_DERIVATIVE_OPTION ), mincw[0]*8 )

        self.option[0] = self.printLevel[0]
        csnopt.snseti_( csnopt.STR_MAJOR_PRINT_LEVEL, self.option,
                        self.iPrint, self.iSumm, self.INFO,
                        self.cw, mincw, self.iw, miniw, self.rw, minrw,
                        len( csnopt.STR_MAJOR_PRINT_LEVEL ), mincw[0]*8 )
        csnopt.snseti_( csnopt.STR_ITERATIONS_LIMIT, self.maxeval,
                        self.iPrint, self.iSumm, self.INFO,
                        self.cw, mincw, self.iw, miniw, self.rw, minrw,
                        len( csnopt.STR_ITERATIONS_LIMIT ), mincw[0]*8 )

        if( self.status[0] == 1 ):
            csnopt.snseti_( csnopt.STR_WARM_START, self.option,
                            self.iPrint, self.iSumm, self.INFO,
                            self.cw, mincw, self.iw, miniw, self.rw, minrw,
                            len( csnopt.STR_WARM_START ), mincw[0]*8 )
        csnopt.snsetr_( csnopt.STR_MAJOR_FEASIBILITY_TOLERANCE, self.constraint_violation,
                        self.iPrint, self.iSumm, self.INFO,
                        self.cw, mincw, self.iw, miniw, self.rw, minrw,
                        len( csnopt.STR_MAJOR_FEASIBILITY_TOLERANCE ), mincw[0]*8 )
        csnopt.snsetr_( csnopt.STR_MAJOR_OPTIMALITY_TOLERANCE, self.ftol,
                        self.iPrint, self.iSumm, self.INFO,
                        self.cw, mincw, self.iw, miniw, self.rw, minrw,
                        len( csnopt.STR_MAJOR_OPTIMALITY_TOLERANCE ), mincw[0]*8 )

        csnopt.snopta_( self.status, self.neF, self.n, self.nxname, self.nFname,
                        self.ObjAdd, self.ObjRow, self.prob, <csnopt.U_fp> callback,
                        self.iAfun, self.jAvar, self.lenA, self.lenA, self.A,
                        self.iGfun, self.jGvar, self.lenG, self.lenG,
                        self.xlow, self.xupp, self.xnames, self.Flow, self.Fupp, self.Fnames,
                        self.x, self.xstate, self.xmul, self.F, self.Fstate, self.Fmul,
                        self.INFO, mincw, miniw, minrw,
                        self.nS, self.nInf, self.sInf,
                        self.cw, mincw, self.iw, miniw, self.rw, minrw,
                        self.cw, mincw, self.iw, miniw, self.rw, minrw,
                        len( self.prob ), len( self.xnames ), len( self.Fnames ), mincw[0]*8, mincw[0]*8 )

        # print( "info: " + str( self.INFO[0] ) )
        return self.F[0]
