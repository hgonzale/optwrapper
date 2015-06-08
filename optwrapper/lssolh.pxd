from .typedefs cimport *

cdef extern from "lssol.h":
    int lssol_( integer *mm, integer *n, integer *nclin, integer *lda,
                integer *ldr, doublereal *a, doublereal *bl, doublereal *bu,
                doublereal *cvec, integer *istate, integer *kx, doublereal *x,
                doublereal *r__, doublereal *b, integer *inform__, integer *iter,
                doublereal *obj, doublereal *clamda, integer *iw, integer *leniw,
                doublereal *w, integer *lenw )

    int lsfile_( integer *ioptns, integer *inform__ )

    int lsoptn_( char *string, ftnlen string_len )

    int lsopti_( char *string, integer *ivalue, ftnlen string_len )

    int lsoptr_( char *string, doublereal *rvalue, ftnlen string_len )

    # int lsfilewrapper_( char *name__, integer *inform__, ftnlen name_len )

    ## lsfilewrapper.h
    # int lsopenappend_( integer *iunit, char *name, integer *inform, ftnlen name_len )

    # int lsclose_( integer *iunit )
