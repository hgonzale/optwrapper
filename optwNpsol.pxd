##
## optwNpsol.pxd
##

from optwF2c cimport *

DEF STR_WARM_START = "Warm start"
DEF STR_PRINT_FILE = "Print file"
DEF STR_MAJOR_PRINT_LEVEL = "Major print level"
DEF STR_FEASIBILITY_TOLERANCE = "Feasibility tolerance"
DEF STR_OPTIMALITY_TOLERANCE = "Optimality tolerance"
DEF STR_MINOR_ITERATIONS_LIMIT = "Minor iterations limit"

ctypedef int (*funcon_fp)( integer*, integer*, integer*,
                           integer*, integer*, doublereal*,
                           doublereal*, doublereal*, integer* )
ctypedef int (*funobj_fp)( integer*, integer*, doublereal*,
                           doublereal*, doublereal*, integer* )

cdef extern from "npsol.h":
    int npsol_( integer *n, integer *nclin, integer *ncnln,
                integer *lda, integer *ldju, integer *ldr,
                doublereal *a, doublereal *bl, doublereal *bu,
                funcon_fp funcon, funobj_fp funobj,
                integer *inform__, integer *itern, integer *istate,
                doublereal *c__, doublereal *cjacu,
                doublereal *clamda, doublereal *objf, doublereal *gradu,
                doublereal *r__, doublereal *x,
                integer *iw, integer *leniw, doublereal *w, integer *lenw )

    int npfilewrapper_( char *name__, integer *inform__, ftnlen name_len )

    int npfile_( integer *ioptns, integer *inform__ )

    int npoptn_( char *string, ftnlen string_len )

    int npopti_( char *string, integer *ivalue, ftnlen string_len )

    int npoptr_(char *string, doublereal *rvalue, ftnlen string_len)

    ## npfilewrapper.h
    int npopenappend_( integer *iunit, char *name, integer *inform, ftnlen name_len )

    int npclose_( integer *iunit )
