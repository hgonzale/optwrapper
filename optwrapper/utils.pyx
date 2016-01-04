import numpy as np
cimport numpy as cnp

from libc.stdint cimport int32_t, int64_t
from libc.string cimport memcpy, memset
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf

## Numpy must be initialized whenever is called from C or Cython
cnp.import_array()

cdef cnp.ndarray wrapPtr( void* array, cnp.ndarray dims, int typenum ):
    if( not dims.flags["C_CONTIGUOUS"] or
        not dims.flags["ALIGNED"] or
        dims.dtype != np.intp ):
        dims = np.require( dims.flat, dtype=np.intp, requirements=['C', 'A'] )

    return cnp.PyArray_SimpleNewFromData( dims.size,
                                          <cnp.npy_intp *> cnp.PyArray_GETPTR1( dims, 0 ),
                                          typenum, array )


cdef cnp.ndarray wrap1dPtr( void* array, int length, int typenum ):
    cdef cnp.npy_intp dims[1]
    dims[0] = <cnp.npy_intp> length

    return cnp.PyArray_SimpleNewFromData( 1, dims, typenum, array )


cdef cnp.ndarray wrap2dPtr( void* array, int rows, int cols, int typenum ):
    cdef cnp.npy_intp dims[2]
    dims[0] = <cnp.npy_intp> cols
    dims[1] = <cnp.npy_intp> rows

    ## hack to return fortran-ordered array by transposing the c-ordered one
    return cnp.PyArray_SimpleNewFromData( 2, dims, typenum, array ).T


cdef void* getPtr( cnp.ndarray array ):
    if( not array.flags["F_CONTIGUOUS"] or
        not array.flags["ALIGNED"] ):
        raise ValueError( "Array array must be 'F_CONTIGUOUS' and 'ALIGNED'" )

    if( array.ndim == 1 ):
        return cnp.PyArray_GETPTR1( array, 0 )
    elif( array.ndim == 2 ):
        return cnp.PyArray_GETPTR2( array, 0, 0 )
    else: ## HG: I promise that someday I'll implement the case for n-dim arrays
        raise ValueError( "Array array must be at most 2-dimensional" )


cpdef cnp.ndarray convFortran( cnp.ndarray array ):
    return np.require( array, dtype=np.float64, requirements=['F', 'A'] )


cpdef cnp.ndarray convIntFortran( cnp.ndarray array ):
    return np.require( array, dtype=np.int64, requirements=['F', 'A'] )


## sMatrix helper to create arrays with valid indices out of keys with heterogeneous datatypes
cdef cnp.ndarray key_to_array( object key, int64_t limit ):
    if( isinstance( key, slice ) ):
        return np.arange( *key.indices( limit ), dtype=np.int64 )
    else:
        try:
            key = np.asarray( key, dtype=np.int64 )
        except:
            raise TypeError( "key cannot be converted to an integer array" )
        ## http://docs.scipy.org/doc/numpy/reference/arrays.nditer.html#modifying-array-values
        for val in np.nditer( key, op_flags=[ "readwrite" ] ):
            if( val >= limit or val < -limit ):
                raise IndexError( "index {0} is out of bounds, limit is {1}".format( val, limit ) )
            if( val < 0 ):
                val[...] = limit + val
        return key


cdef class sMatrix:
    def __cinit__( self, arr, int copy_data=False ):
        self.data_alloc = True

        try:
            arr = np.atleast_2d( np.asarray( arr, dtype=np.float64 ) )
        except:
            raise TypeError( "argument must be an array" )

        if( arr.ndim > 2 ):
            raise ValueError( "argument can have at most two dimensions" )

        ( self.nrows, self.ncols ) = self.shape = arr.shape
        self.nnz = np.count_nonzero( arr )

        self.data = <double *> malloc( self.nnz * sizeof( double ) )

        self.rptr = <int64_t *> malloc( ( self.nrows + 1 ) * sizeof( int64_t ) )
        self.ridx = <int64_t *> malloc( self.nnz * sizeof( int64_t ) )
        self.cidx = <int64_t *> malloc( self.nnz * sizeof( int64_t ) )

        ## populate ridx, cidx, and rptr by walking through arr in C order
        cdef int64_t row, col, k
        cdef double tmp
        k = 0
        self.rptr[0] = 0
        for row in range( self.nrows ):
            for col in range( self.ncols ):
                tmp = arr[row,col]
                if( tmp != 0.0 ):
                    self.ridx[k] = row
                    self.cidx[k] = col
                    if( copy_data ):
                        self.data[k] = tmp
                    k += 1
            self.rptr[row+1] = k

        ## zero out data if we didn't copy
        if( not copy_data ):
            memset( self.data, 0, self.nnz * sizeof( double ) )


    def print_debug( self ):
        """
        print internal C arrays containing representation data of this sparse matrix, which
        cannot be accessed using Python

        """

        print( "nrows: {0} - ncols: {1} - nnz: {2} - data_alloc: {3}".format( self.nrows,
                                                                              self.ncols,
                                                                              self.nnz,
                                                                              self.data_alloc ) )

        print( "rptr: [ " ),
        for k in range( self.nrows+1 ):
            printf( "%d ", self.rptr[k] )
        print( "]" )

        print( "ridx: [ " ),
        for k in range( self.nnz ):
            printf( "%d ", self.ridx[k] )
        print( "]" )

        print( "cidx: [ " ),
        for k in range( self.nnz ):
            printf( "%d ", self.cidx[k] )
        print( "]" )

        print( "data: [ " ),
        for k in range( self.nnz ):
            printf( "%f ", self.data[k] )
        print( "]" )


    def __dealloc__( self ):
        if( self.data_alloc ):
            free( self.data )
        free( self.rptr )
        free( self.ridx )
        free( self.cidx )


    cdef void setDataPtr( self, void *ptr ):
        if( self.data_alloc ):
            free( self.data )
            self.data_alloc = False

        self.data = <double *> ptr


    cdef void copyFortranIdxs( self, int64_t* ridx, int64_t* cidx,
                               int64_t roffset=0, int64_t coffset=0 ):
        memcpy( ridx, self.ridx, self.nnz * sizeof( int64_t ) )
        memcpy( cidx, self.cidx, self.nnz * sizeof( int64_t ) )
        for k in range( self.nnz ): ## have to add one because Fortran
            ridx[k] += roffset + 1
            cidx[k] += coffset + 1


    cdef void copyFortranIdxs32( self, int32_t* ridx, int32_t* cidx,
                                 int32_t roffset=0, int32_t coffset=0 ):
        for k in range( self.nnz ): ## have to add one because Fortran
            ridx[k] = ( <int32_t> self.ridx[k] ) + roffset + 1
            cidx[k] = ( <int32_t> self.cidx[k] ) + coffset + 1


    cdef void copyData( self, double* data ):
        memcpy( data, self.data, self.nnz * sizeof( double ) )
        for k in range( self.nnz ):
            data[k] = self.data[k]


    cdef double get_elem_at( self, int64_t row, int64_t col ):
        cdef int64_t first, last, midpoint

        ## binary search
        first = self.rptr[row]
        last = self.rptr[row+1]-1

        while( first <= last ):
            midpoint = (first + last)//2
            if( self.cidx[midpoint] == col ):
                return self.data[midpoint]
            else:
                if( col < self.cidx[midpoint] ):
                    last = midpoint-1
                else:
                    first = midpoint+1

        return 0


    cdef bint set_elem_at( self, int64_t row, int64_t col, double val ):
        cdef int64_t first, last, midpoint

        ## binary search
        first = self.rptr[row]
        last = self.rptr[row+1]-1

        while( first <= last ):
            midpoint = (first + last)//2
            if( self.cidx[midpoint] == col ):
                self.data[midpoint] = val
                return True
            else:
                if( col < self.cidx[midpoint] ):
                    last = midpoint-1
                else:
                    first = midpoint+1

        return False


    cdef cnp.broadcast key_to_bcast( self, object key ):
        if( self.nrows > 1 ):
            if( isinstance( key, tuple ) and len(key) == 2 ):
                rowiter = key_to_array( key[0], self.nrows )
                coliter = key_to_array( key[1], self.ncols )
                if( ( isinstance( key[0], slice ) and isinstance( key[1], slice ) ) or
                    ( isinstance( key[0], slice ) and coliter.size > 1 ) or
                    ( isinstance( key[1], slice ) and rowiter.size > 1 ) ): ## slices form meshes
                    ( rowiter, coliter ) = np.ix_( rowiter, coliter )
            else:
                rowiter = key_to_array( key, self.nrows )
                coliter = key_to_array( slice( None, None, None ), self.ncols )
                if( rowiter.size > 1 ):  ## slices form meshes
                    ( rowiter, coliter ) = np.ix_( rowiter, coliter )

        elif( self.nrows == 1 ):
            rowiter = np.array( 0 )
            coliter = key_to_array( key, self.ncols )
        else:
            raise TypeError( "key cannot be applied to this sMatrix" )

        ## here is where all the magic happens to figure out new dimensions
        return np.broadcast( rowiter, coliter )


    def __setitem__( self, key, value ):
        try:
            value = np.asarray( value, dtype=np.float64 )
        except:
            raise TypeError( "value cannot be converted into a float array" )

        bcast = self.key_to_bcast( key )
        origshape = value.shape

        ## this algorithm follow the rules listed here:
        ## http://docs.scipy.org/doc/numpy/reference/ufuncs.html#broadcasting

        ## shave extra dimensions we can't deal with
        while( len( bcast.shape ) < len( value.shape ) ):
            if( value.shape[0] > 1 ):
                raise ValueError( "could not broadcast value array from shape " +
                                  "{0} into shape {1}".format( origshape, bcast.shape ) )
            value = np.squeeze( value, axis=0 )

        ## add size 1 dimensions to the left
        while( len( bcast.shape ) > len( value.shape ) ):
            value = value[np.newaxis]

        ## now try to match different dimensions
        for k in range( len( value.shape ) ):
            if( bcast.shape[-k-1] != value.shape[-k-1] ):
                if( value.shape[-k-1] != 1 ):
                    raise ValueError( "could not broadcast value array from shape " +
                                      "{0} into shape {1}".format( origshape, bcast.shape ) )
                value = np.tile( value, (bcast.shape[-k-1],) + (1,) * k )

        ## finally set values, assuming (!) bcast is C-ordered
        for ( (row,col), val ) in zip( bcast, np.nditer( value, order='C' ) ):
            self.set_elem_at( row, col, val )


    def __getitem__( self, key ):
        bcast = self.key_to_bcast( key )

        ## cool trick copied from
        ## http://docs.scipy.org/doc/numpy/reference/generated/numpy.broadcast.html
        out = np.empty( bcast.shape )
        out.flat = [ self.get_elem_at( row, col ) for (row,col) in bcast ]

        return out


    def __str__( self ):
        return str( self[:] )
