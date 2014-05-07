import numpy as np
cimport numpy as np

## Numpy must be initialized. When using numpy from C or Cython you must
## _always_ do that, or you will have segfaults
np.import_array()

cdef np.ndarray wrapPtr( void* array, np.ndarray dims, int typenum ):
    if( not dims.flags["C_CONTIGUOUS"] or
        not dims.flags["ALIGNED"] or
        not dims.dtype == np.intp ):
        print( "'dims' was not appropriate" )
        print( dims.flags )
        print( dims.dtype )
        dims = np.require( dims.flat, dtype=np.intp, requirements=['C', 'A'] )

    return convFortran( np.PyArray_SimpleNewFromData( dims.size,
                                                      <np.npy_intp *> np.PyArray_GETPTR1( dims, 0 ),
                                                      typenum, array ) )


cdef np.ndarray wrap1dPtr( void* array, int length, int typenum ):
    cdef np.npy_intp dims[1]
    dims[0] = <np.npy_intp> length

    return convFortran( np.PyArray_SimpleNewFromData( 1, dims, typenum, array ) )


cdef np.ndarray wrap2dPtr( void* array, int rows, int cols, int typenum ):
    cdef np.npy_intp dims[2]
    dims[0] = <np.npy_intp> rows
    dims[1] = <np.npy_intp> cols

    return convFortran( np.PyArray_SimpleNewFromData( 2, dims, typenum, array ) )


cdef void* getPtr( np.ndarray array ):
    if( not array.flags["F_CONTIGUOUS"] or
        not array.flags["ALIGNED"] ):
        raise ValueError( "Array array must be 'F_CONTIGUOUS' and 'ALIGNED'" )

    if( array.ndim == 1 ):
        return np.PyArray_GETPTR1( array, 0 )
    elif( array.ndim == 2 ):
        return np.PyArray_GETPTR2( array, 0, 0 )
    else: ## HG: One day I will implement the case for n-dim arrays
        raise ValueError( "Array array must be at most 2-dimensional" )


cdef np.ndarray convFortran( np.ndarray array ):
    return np.require( array, dtype=np.float64, requirements=['F', 'A'] )


cdef np.ndarray convIntFortran( np.ndarray array ):
    return np.require( array, dtype=np.int_, requirements=['F', 'A'] )


cpdef int isInt( object obj ):
    try:
        int( obj )
        return True
    except ValueError:
        return False


cpdef int isFloat( object obj ):
    try:
        float( obj )
        return True
    except ValueError:
        return False


cpdef int isString( object obj ):
    try:
        str( obj )
        return True
    except ValueError:
        return False
