cimport numpy as cnp
from libc.stdint cimport int32_t, int64_t

cdef cnp.ndarray wrapPtr( void* array, cnp.ndarray dims, int typenum )
cdef cnp.ndarray wrap1dPtr( void* array, int length, int typenum )
cdef cnp.ndarray wrap2dPtr( void* array, int rows, int cols, int typenum )
cdef void* getPtr( cnp.ndarray input )
cpdef cnp.ndarray convFortran( cnp.ndarray input )
cpdef cnp.ndarray convIntFortran( cnp.ndarray input )

cdef class sMatrix:
    cdef double *data
    cdef int64_t *rptr
    cdef int64_t *ridx
    cdef int64_t *cidx
    cdef readonly int64_t nnz
    cdef readonly int64_t nrows
    cdef readonly int64_t ncols
    cdef readonly tuple shape
    cdef int data_alloc

    cdef void setDataPtr( self, void *ptr )

    cdef void copyFortranIdxs( self, int64_t* ridx, int64_t* cidx,
                               int64_t roffset=*, int64_t coffset=* )

    cdef void copyFortranIdxs32( self, int32_t* ridx, int32_t* cidx,
                                 int32_t roffset=*, int32_t coffset=* )

    cdef void copyData( self, double* data )

    cdef double get_elem_at( self, int64_t row, int64_t col )

    cdef bint set_elem_at( self, int64_t row, int64_t col, double val )

    cdef cnp.broadcast key_to_bcast( self, object key )

