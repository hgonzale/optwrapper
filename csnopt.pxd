cdef extern from "snopt.h":

    cdef char* STR_MINIMIZE
    cdef char* STR_MAXIMIZE
    cdef char* STR_DERIVATIVE_OPTION
    cdef char* STR_MAJOR_PRINT_LEVEL
    cdef char* STR_ITERATIONS_LIMIT
    cdef char* STR_WARM_START
    cdef char* STR_MAJOR_FEASIBILITY_TOLERANCE
    cdef char* STR_MAJOR_OPTIMALITY_TOLERANCE

    ctypedef long int integer
    ctypedef double doublereal
    ctypedef long int ftnlen
    # ctypedef long int flag
    # ctypedef long int ftnint
    ctypedef int (*U_fp)( integer *Status, integer *n, doublereal *x,
                          integer *needf, integer *nF, doublereal *f,
                          integer *needG, integer *lenG, doublereal *G,
                          char *cu, integer *lencu, integer *iu, integer *leniu,
                          doublereal *ru, integer *lenru )

    # ctypedef struct olist:
    #   flag oerr
    #   ftnint ounit
    #   char *ofnm
    #   ftnlen ofnmlen
    #   char *osta
    #   char *oacc
    #   char *ofm
    #   ftnint orl
    #   char *oblnk
    # integer f_open(olist* a)

    int snopta_( integer *start, integer *nf, integer *n,
                 integer *nxname, integer *nfname, doublereal *objadd, integer *objrow,
                 char *prob, U_fp usrfun, integer *iafun, integer *javar,
                 integer *lena, integer *nea, doublereal *a, integer *igfun,
                 integer *jgvar, integer *leng, integer *neg,
                 doublereal *xlow, doublereal *xupp,
                 char *xnames, doublereal *flow, doublereal *fupp, char *fnames,
                 doublereal *x, integer *xstate, doublereal *xmul, doublereal *f,
                 integer *fstate, doublereal *fmul, integer *info, integer *mincw,
                 integer *miniw, integer *minrw, integer *ns, integer *ninf,
                 doublereal *sinf, char *cu, integer *lencu, integer *iu, integer *leniu,
                 doublereal *ru, integer *lenru, char *cw, integer *lencw,
                 integer *iw, integer *leniw, doublereal *rw, integer *lenrw,
                 ftnlen prob_len, ftnlen xnames_len, ftnlen fnames_len, ftnlen cu_len,
                 ftnlen cw_len )

    int sninit_( integer *iprint, integer *isumm, char *cw,
                 integer *lencw, integer *iw, integer *leniw,
                 doublereal *rw, integer *lenrw, ftnlen cw_len )

    int sngeti_( char *buffer, integer *ivalue, integer *errors,
                 char *cw, integer *lencw, integer *iw, integer *leniw, doublereal *rw,
                 integer *lenrw, ftnlen buffer_len, ftnlen cw_len )

    int snset_( char *buffer, integer *iprint, integer *isumm,
                integer *errors, char *cw, integer *lencw, integer *iw, integer *leniw,
                doublereal *rw, integer *lenrw, ftnlen buffer_len, ftnlen cw_len )

    int snseti_( char *buffer, integer *ivalue, integer *iprint,
                 integer *isumm, integer *errors, char *cw, integer *lencw,
                 integer *iw, integer *leniw, doublereal *rw, integer *lenrw,
                 ftnlen buffer_len, ftnlen cw_len )

    int snsetr_( char *buffer, doublereal *rvalue, integer *iprint, integer *isumm,
                 integer *errors, char *cw, integer *lencw, integer *iw,
                 integer *leniw, doublereal *rw, integer *lenrw, ftnlen buffer_len,
                 ftnlen cw_len )

    int snspec_( integer *ispecs, integer *iexit, char *cw,
                 integer *lencw, integer *iw, integer *leniw, doublereal *rw,
                 integer *lenrw, ftnlen cw_len )

    int snmema_( integer *iexit, integer *nf, integer *n, integer *nxname,
                 integer *nfname, integer *nea, integer *neg, integer *mincw,
                 integer *miniw, integer *minrw, char *cw, integer *lencw, integer *iw,
                 integer *leniw, doublereal *rw, integer *lenrw, ftnlen cw_len )

    int snjac_( integer *iexit, integer *nf, integer *n, U_fp userfg, integer *iafun,
                integer *javar, integer *lena, integer *nea, doublereal *a,
                integer *igfun, integer *jgvar, integer *leng, integer *neg,
                doublereal *x, doublereal *xlow, doublereal *xupp, integer *mincw,
                integer *miniw, integer *minrw, char *cu, integer *lencu, integer *iu,
                integer *leniu, doublereal *ru, integer *lenru, char *cw,
                integer *lencw, integer *iw, integer *leniw, doublereal *rw,
                integer *lenrw, ftnlen cu_len, ftnlen cw_len )
