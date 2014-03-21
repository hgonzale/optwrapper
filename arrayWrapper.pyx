from libc.stdlib cimport free
from cpython cimport PyObject, Py_INCREF
import numpy as np
cimport numpy as np

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

# We need to build an array-wrapper class to deallocate our array when
# the Python object is deleted.

cdef class ArrayWrapper:
    cdef void* data_ptr
    cdef int size

    cdef set_data( self, void* data_ptr, int size ):
        """ Set the data of the array
            This cannot be done in the constructor as it must recieve C-level
            arguments.

            Parameters:
            -----------
            int size: Length of the array.
            void* data_ptr: Pointer to the data
        """
        self.data_ptr = data_ptr
        self.size = size

    def __array__( self ):
        """ Here we use the __array__ method, that is called when numpy
            tries to get an array from the object.
        """
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.size
        # Create a 1D array, of length 'size'
        ndarray = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, self.data_ptr)
        return

    # def __dealloc__( self ):
    #     """ Frees the array. This is called by Python when all the
    #         references to the object are gone.
    #     """
    # free(<void*>self.data_ptr)


cdef np.ndarray wrapPtr( void* array, int size, int typenum ):
    cdef np.ndarray ndarray
    cdef np.npy_intp shape[1]

    shape[0] = <np.npy_intp> size
    # Create a 1D array, of length 'size'
    ndarray = np.PyArray_SimpleNewFromData( 1, shape, typenum, array )

    # arr_wrap = ArrayWrapper()
    # arr_wrap.set_data( array, size )
    # ndarray = np.array( arr_wrap, copy=False )
    # ndarray.base = <PyObject*> arr_wrap
    # Py_INCREF( arr_wrap )

    return ndarray
