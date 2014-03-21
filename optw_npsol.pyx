cimport cnpsol
cimport cpython
from libc.stdlib cimport calloc,free
cimport numpy as np
cimport cpython.pycapsule as pycapsule
from arrayWrapper cimport wrapPtr

cdef object objfun,confun

cdef int objcallback( cnpsol.integer* mode, cnpsol.integer* n,
                      cnpsol.doublereal* x, cnpsol.doublereal* f, cnpsol.doublereal* g,
                      cnpsol.integer* nstate):
    px = wrapPtr( x, n[0], np.NPY_DOUBLE )
    pF = wrapPtr( f, 1, np.NPY_DOUBLE )
    pG = wrapPtr( g, n[0], np.NPY_DOUBLE )
    objfun( px, pF, pG )

## note that the gradient must be defined in Fortran-style indexing
cdef int concallback( cnpsol.integer* mode, cnpsol.integer* ncnln,
                      cnpsol.integer* n, cnpsol.integer* ldJ, cnpsol.integer* needc,
                      cnpsol.doublereal* x, cnpsol.doublereal* c, cnpsol.doublereal* cJac,
                      cnpsol.integer* nstate):
    px = wrapPtr( x, n[0], np.NPY_DOUBLE )
    pC = wrapPtr( c, ncnln[0], np.NPY_DOUBLE )
    pJ = wrapPtr( cJac, ldJ[0] * n[0], np.NPY_DOUBLE )
    confun( px, pC, pJ )

cdef class NpsolSolver:
    cdef cnpsol.integer iPrint[1]
    cdef cnpsol.integer iSumm[1]
    cdef cnpsol.integer iOptns[1]
    cdef cnpsol.integer printLevel[1]
    cdef cnpsol.integer maximize[1]
    cdef cnpsol.integer n[1]
    cdef cnpsol.integer nclin[1]
    cdef cnpsol.integer ncnln[1]
    cdef cnpsol.integer nrowa[1]
    cdef cnpsol.integer nrowuj[1]
    cdef cnpsol.integer nrowr[1]
    cdef cnpsol.integer maxbnd[1]
    cdef cnpsol.integer inform[1]
    cdef cnpsol.integer itern[1]
    cdef cnpsol.integer leniw[1]
    cdef cnpsol.integer lenw[1]
    cdef cnpsol.doublereal objf[1]
    cdef cnpsol.doublereal *bl
    cdef cnpsol.doublereal *bu
    cdef cnpsol.doublereal *c
    cdef cnpsol.doublereal *clamda
    cdef cnpsol.doublereal *ugrad
    cdef cnpsol.doublereal *x
    cdef cnpsol.doublereal *a,
    cdef cnpsol.doublereal *ujac
    cdef cnpsol.doublereal *r
    cdef cnpsol.integer *istate
    cdef cnpsol.integer *iw
    cdef cnpsol.doublereal *w
    # cdef cnpsol.olist ol[1]
    # cdef object confun
    # cdef object objfun
    cdef cnpsol.integer warmstart[1]
    cdef cnpsol.integer maxeval[1]
    cdef cnpsol.doublereal constraint_violation[1]
    cdef cnpsol.doublereal ftol[1]

    def __cinit__( self, **kwargs ):
        self.inform[0] = 0
        self.n[0] = kwargs['n']
        self.nclin[0] = kwargs['nclin']
        self.ncnln[0] = kwargs['ncnln']
        self.printLevel[0] = kwargs['printLevel']
        self.maximize[0] = kwargs['maximize']
        self.maxbnd[0] = self.n[0] + self.nclin[0] + self.ncnln[0]
        self.nrowa[0] = self.nclin[0]
        self.nrowuj[0] = self.ncnln[0]
        self.nrowr[0] = self.n[0]
        self.itern[0] = 0
        self.leniw[0] = 3*self.n[0] + self.nclin[0] + 2*self.ncnln[0]
        self.lenw[0] = ( 2*self.n[0]*self.n[0] + self.n[0]*self.nclin[0] +
                         2*self.n[0]*self.ncnln[0] + 20*self.n[0] + 11*self.nclin[0] +
                         21*self.ncnln[0] )
        self.objf[0] = 0.0
        self.warmstart[0] = 0
        self.maxeval[0] = 3*self.maxbnd[0]
        self.constraint_violation[0] = 1e-8
        self.ftol[0] = 1e-12
        if( kwargs['printFile'] == None ):
            self.iPrint[0]=0
        else:
            self.iPrint[0]=9
            ## Sketchy code. I'm sure there are easier ways to open a file.
            # self.ol[0].oerr=1
            # self.ol[0].ounit=self.iPrint[0]
            # self.ol[0].ofnmlen=len(kwargs['printFile'])
            # # for i in range(0,self.ol[0].ofnmlen):
            # # self.ol[0].ofnm[i]=kwargs['printFile'][i]
            # self.ol[0].ofnm=kwargs['printFile']
            # self.ol[0].orl = 0
            # self.ol[0].osta = "UNKNOWN"
            # self.ol[0].oacc = NULL
            # self.ol[0].ofm = NULL
            # self.ol[0].oblnk = NULL
            # self.inform[0]=cnpsol.f_open(self.ol)
            # if self.inform[0]>0:
            #   raise ValueError('invalid file inform: '+str(self.INFO[0]))
        self.bl = <cnpsol.doublereal *> calloc( self.maxbnd[0], sizeof( cnpsol.doublereal ) )
        self.bu = <cnpsol.doublereal *> calloc( self.maxbnd[0], sizeof( cnpsol.doublereal ) )
        self.c = <cnpsol.doublereal *> calloc( self.ncnln[0], sizeof( cnpsol.doublereal ) )
        self.clamda = <cnpsol.doublereal *> calloc( self.maxbnd[0], sizeof( cnpsol.doublereal ) )
        self.ugrad = <cnpsol.doublereal *> calloc( self.n[0], sizeof( cnpsol.doublereal ) )
        self.x = <cnpsol.doublereal *> calloc( self.n[0], sizeof( cnpsol.doublereal ) )
        self.istate = <cnpsol.integer *> calloc( self.maxbnd[0], sizeof( cnpsol.integer ) )
        self.iw = <cnpsol.integer *> calloc( self.leniw[0], sizeof( cnpsol.integer ) )
        self.w = <cnpsol.doublereal *> calloc( self.lenw[0], sizeof( cnpsol.doublereal ) )
        self.a = <cnpsol.doublereal *> calloc( self.nrowa[0]*self.n[0], sizeof( cnpsol.doublereal ) )
        self.ujac = <cnpsol.doublereal *> calloc( self.nrowuj[0]*self.n[0], sizeof( cnpsol.doublereal ) )
        self.r = <cnpsol.doublereal *> calloc( self.nrowr[0]*self.n[0], sizeof( cnpsol.doublereal ) )
        if( self.bl is NULL or
            self.bu is NULL or
            self.c is NULL or
            self.clamda is NULL or
            self.ugrad is NULL or
            self.x is NULL or
            self.istate is NULL or
            self.iw is NULL or
            self.w is NULL or
            self.a is NULL or
            self.ujac is NULL or
            self.r is NULL ):
            raise MemoryError()

    def __dealloc__(self):
        if self.bl is not NULL:
            free( self.bl )
        if self.bu is not NULL:
            free( self.bu )
        if self.c is not NULL:
            free( self.c )
        if self.clamda is not NULL:
            free( self.clamda )
        if self.ugrad is not NULL:
            free( self.ugrad )
        if self.x is not NULL:
            free( self.x )
        if self.istate is not NULL:
            free( self.istate )
        if self.iw is not NULL:
            free( self.iw )
        if self.w is not NULL:
            free( self.w )
        if self.a is not NULL:
            free( self.a )
        if self.ujac is not NULL:
            free( self.ujac )
        if self.r is not NULL:
            free( self.r )

    def set_bounds( self, xlow, xupp, lnlow, lnupp, nllow, nlupp ):
        for i in xrange( 0, self.n[0] ):
            self.bl[i] = xlow[i]
            self.bu[i] = xupp[i]
        for i in xrange( self.n[0], self.n[0] + self.nclin[0] ):
            self.bl[i] = lnlow[i - self.n[0]]
            self.bu[i] = lnupp[i - self.n[0]]
        for i in xrange( self.n[0] + self.nclin[0], self.n[0] + self.nclin[0] + self.ncnln[0] ):
            self.bl[i] = nllow[i - self.n[0] - self.nclin[0]]
            self.bu[i] = nlupp[i - self.n[0] - self.nclin[0]]

    def set_x( self, val ):
        for i in xrange( self.n[0] ):
            self.x[i] = <cnpsol.doublereal> val[i]

    def set_user_function( self, cf, of ):
        global confun, objfun
        confun = cf
        objfun = of

    def get_x( self ):
        return wrapPtr( self.x, self.n[0], np.NPY_DOUBLE )

    def get_status( self ):
        if( self.inform[0] == 0 ):
            return 'optimality conditions satisfied'
        elif( self.inform[0] == 1 ):
            return 'feasible point found but no further improvement can be made'
        elif( self.inform[0] == 2 ):
            return 'the problem appears to have infeasible linear constraints'
        elif( self.inform[0] == 3 ):
            return 'the problem appears to have infeasible nonlinear constraints'
        elif( self.inform[0] == 4 ):
            return 'iteration limit reached'
        elif( self.inform[0] == 6 ):
            return 'x does not satisfy first-order optimality conditions'
        elif( self.inform[0] == 7 ):
            return 'derivatives appear to be incorrect'
        else:
            return 'error in user supplied information'

    def set_options( self, warmstart, maxeval, constraint_violation, ftol ):
        if( not warmstart is None ):
            self.warmstart[0] = <cnpsol.integer> warmstart
        if( not maxeval is None ):
            self.maxeval[0] = <cnpsol.integer> maxeval
        if( not constraint_violation is None ):
            self.constraint_violation[0] = <cnpsol.doublereal> constraint_violation
        if( not ftol is None ):
            self.ftol[0] = <cnpsol.doublereal> ftol

    def solve(self):
        if( self.warmstart[0] == 1 ):
            cnpsol.npopti_( cnpsol.STR_WARM_START, self.warmstart, len(cnpsol.STR_WARM_START) )
            # cnpsol.npopti_( "Warm start", self.warmstart ) #, 10 )
        cnpsol.npopti_( cnpsol.STR_PRINT_FILE, self.iPrint, len(cnpsol.STR_PRINT_FILE) )
        cnpsol.npopti_( cnpsol.STR_MAJOR_PRINT_LEVEL, self.printLevel, len(cnpsol.STR_MAJOR_PRINT_LEVEL) )
        cnpsol.npoptr_( cnpsol.STR_FEASIBILITY_TOLERANCE, self.constraint_violation,
                        len(cnpsol.STR_FEASIBILITY_TOLERANCE) )
        cnpsol.npoptr_( cnpsol.STR_OPTIMALITY_TOLERANCE, self.ftol, len(cnpsol.STR_OPTIMALITY_TOLERANCE) )
        cnpsol.npopti_( cnpsol.STR_MINOR_ITERATIONS_LIMIT, self.maxeval, len(cnpsol.STR_MINOR_ITERATIONS_LIMIT) )

        cnpsol.npsol_( self.n, self.nclin,
                       self.ncnln, self.nrowa,
                       self.nrowuj, self.nrowr,
                       self.a, self.bl, self.bu,
                       <cnpsol.c_fp> concallback, <cnpsol.o_fp> objcallback,
                       self.inform, self.itern,
                       self.istate, self.c, self.ujac,
                       self.clamda, self.objf,
                       self.ugrad, self.r, self.x,
                       self.iw, self.leniw,
                       self.w, self.lenw )

        return self.objf[0]
