cdef extern from "npsol.h":

    cdef char* STR_WARM_START
    cdef char* STR_PRINT_FILE
    cdef char* STR_MAJOR_PRINT_LEVEL
    cdef char* STR_FEASIBILITY_TOLERANCE
    cdef char* STR_OPTIMALITY_TOLERANCE
    cdef char* STR_MINOR_ITERATIONS_LIMIT

    ctypedef long int integer
    ctypedef double doublereal
    ctypedef long int ftnlen
    # ctypedef long int flag
    # ctypedef long int ftnint
    # ctypedef int (*U_fp)
    ctypedef int (*c_fp)( integer*, integer*, integer*,
                          integer*, integer*, doublereal*,
                          doublereal*, doublereal*, integer* )
    ctypedef int (*o_fp)( integer*, integer*, doublereal*,
                          doublereal*, doublereal*, integer* )
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

    int npopti_( char *name, integer *value, ftnlen string_len )
    int npoptr_( char *name, doublereal *value, ftnlen string_len )
    int npoptn_( char *name, ftnlen string_len )

    # int npfile_( integer* file, integer* inform )
    # int npfilewrapper_( char *name__, integer *inform__, ftnlen name_len )
    # integer f_open(olist* a)

    int npsol_( integer* n, integer* nclin,
                integer* ncnln, integer* nrowa,
                integer* nrowuj, integer* nrowr,
                doublereal* a, doublereal* bl, doublereal* bu,
                c_fp confun, o_fp objfun,
                integer* inform, integer* iter,
                integer* istate, doublereal* c, doublereal* ujac,
                doublereal* clamda, doublereal* objf,
                doublereal* ugrad, doublereal* r, doublereal* x,
                integer* iw, integer* leniw,
                doublereal* w, integer* lenw )
