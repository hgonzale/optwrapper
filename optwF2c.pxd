##
## optwF2c.pyx
## A few typedef declarations from f2c.h
##

cdef extern from *:
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
