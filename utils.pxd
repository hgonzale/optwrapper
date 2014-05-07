cimport numpy as np

cdef np.ndarray wrapPtr( void* array, np.ndarray dims, int typenum )
cdef np.ndarray wrap1dPtr( void* array, int length, int typenum )
cdef np.ndarray wrap2dPtr( void* array, int rows, int cols, int typenum )
cdef void* getPtr( np.ndarray input )
cdef np.ndarray convFortran( np.ndarray input )
cdef np.ndarray convIntFortran( np.ndarray input )
cpdef int isInt( object obj )
cpdef int isFloat( object obj )
cpdef int isString( object obj )
