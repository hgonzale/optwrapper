cimport numpy as np

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef np.ndarray wrapPtr( void* array, int size, int typenum ):
    cdef np.ndarray ndarray
    cdef np.npy_intp shape[1]

    shape[0] = <np.npy_intp> size
    # Create a 1D array, of length 'size'
    ndarray = np.PyArray_SimpleNewFromData( 1, shape, typenum, array )

    return ndarray
