# from .typedefs cimport *

## A few typedef declarations from f2c.h
cdef extern from "f2c.h":
    ctypedef long int integer
    ctypedef double doublereal
    ctypedef long int ftnlen

ctypedef int (*usrfun_fp)( integer *Status, integer *n, doublereal *x,
                           integer *needf, integer *nF, doublereal *f,
                           integer *needG, integer *lenG, doublereal *G,
                           char *cu, integer *lencu, integer *iu, integer *leniu,
                           doublereal *ru, integer *lenru )

cdef extern from "snopt.h":
    int snopta_( integer *start, integer *nef, integer *n,
                 integer *nxname, integer *nfname, doublereal *objadd, integer *objrow,
                 char *prob, usrfun_fp usrfun, integer *iafun, integer *javar,
                 integer *lena, integer *nea, doublereal *a, integer *igfun,
                 integer *jgvar, integer *leng, integer *neg,
                 doublereal *xlow, doublereal *xupp,
                 char *xnames, doublereal *flow, doublereal *fupp, char *fnames,
                 doublereal *x, integer *xstate, doublereal *xmul, doublereal *f,
                 integer *fstate, doublereal *fmul, integer *inform, integer *mincw,
                 integer *miniw, integer *minrw, integer *ns, integer *ninf,
                 doublereal *sinf, char *cu, integer *lencu, integer *iu, integer *leniu,
                 doublereal *ru, integer *lenru, char *cw, integer *lencw,
                 integer *iw, integer *leniw, doublereal *rw, integer *lenrw, ftnlen
                 prob_len, ftnlen xnames_len, ftnlen fnames_len, ftnlen cu_len, ftnlen cw_len )

    int sninit_( integer *iPrint, integer *iSumm, char *cw,
                 integer *lencw, integer *iw, integer *leniw,
                 doublereal *rw, integer *lenrw, ftnlen cw_len )

    int sngeti_( char *buffer, integer *ivalue, integer *inform,
                 char *cw, integer *lencw, integer *iw,
                 integer *leniw, doublereal *rw, integer *lenrw,
                 ftnlen buffer_len, ftnlen cw_len )

    int snset_( char *buffer, integer *iprint, integer *isumm,
                integer *inform, char *cw, integer *lencw,
                integer *iw, integer *leniw,
                doublereal *rw, integer *lenrw,
                ftnlen buffer_len, ftnlen cw_len )

    int snseti_( char *buffer, integer *ivalue, integer *iprint,
                 integer *isumm, integer *inform, char *cw,
                 integer *lencw, integer *iw, integer *leniw,
                 doublereal *rw, integer *lenrw, ftnlen buffer_len,
                 ftnlen cw_len )

    int snsetr_( char *buffer, doublereal *rvalue, integer * iprint,
                 integer *isumm, integer *inform, char *cw,
                 integer *lencw, integer *iw, integer *leniw,
                 doublereal *rw, integer *lenrw, ftnlen buffer_len,
                 ftnlen cw_len )

    int snspec_( integer *ispecs, integer *inform, char *cw,
                 integer *lencw, integer *iw, integer *leniw,
                 doublereal *rw, integer *lenrw, ftnlen cw_len)

    int snmema_( integer *iExit, integer *nef, integer *n, integer *nxname,
                 integer *nfname, integer *nea, integer *neg,
                 integer *mincw, integer *miniw, integer *minrw,
                 char *cw, integer *lencw, integer *iw,
                 integer *leniw, doublereal *rw, integer *lenrw,
                 ftnlen cw_len )

    int snjac_( integer *iExit, integer *nef, integer *n, usrfun_fp userfg,
                integer *iafun, integer *javar, integer *lena, integer *nea, doublereal *a,
                integer *igfun, integer *jgvar, integer *leng, integer *neg,
                doublereal *x, doublereal *xlow, doublereal *xupp,
                integer *mincw, integer *miniw, integer *minrw,
                char *cu, integer *lencu, integer *iu, integer *leniu, doublereal *ru, integer *lenru,
                char *cw, integer *lencw, integer *iw, integer *leniw, doublereal *rw, integer *lenrw,
                ftnlen cu_len, ftnlen cw_len )

    ## snfilewrapper.h
    # int snopenappend_( integer *iunit, char *name, integer *inform, ftnlen name_len )

    # int snfilewrapper_( char *name__, integer *ispec, integer *inform__,
    #                     char *cw, integer *lencw, integer *iw,
    #                     integer *leniw, doublereal *rw, integer *lenrw,
    #                     ftnlen name_len, ftnlen cw_len )

    # int snclose_( integer *iunit )

    # int snopen_( integer *iunit )
