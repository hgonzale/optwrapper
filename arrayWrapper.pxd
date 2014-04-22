cimport numpy as np

cdef np.ndarray wrapPtr( void* array, int size, int typenum )
cdef void* getPtr( np.ndarray[double, ndim=2, mode="fortran"] input )
