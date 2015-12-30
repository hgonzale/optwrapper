## Definitions taken from f2c.h
## Currently used by all Fortran-based libraries, i.e. lssol, npsol, and snopy

ctypedef long int integer
ctypedef unsigned long int uinteger
ctypedef char* address
ctypedef short int shortint
ctypedef float real
ctypedef double doublereal
# ctypedef struct { real r, i; } complex
# ctypedef struct { doublereal r, i; } doublecomplex
ctypedef long int logical
ctypedef short int shortlogical
ctypedef char logical1
ctypedef char integer1

ctypedef long int flag
ctypedef long int ftnlen
ctypedef long int ftnint
