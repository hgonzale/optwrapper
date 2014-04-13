cimport numpy as np

cdef np.ndarray wrapPtr( void* array, int size, int typenum )
cdef void* retPtr( np.ndarray[double, ndim=2, mode="fortran"] input )
