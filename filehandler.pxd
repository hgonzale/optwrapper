from f2ch cimport *

cdef extern from "filehandler.h":
    int openfile_( integer *iunit, char *name__, integer *inform__, ftnlen name_len )

    int closefile_( integer *iunit )
