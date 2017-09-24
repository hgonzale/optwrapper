# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp
# cython: boundscheck=False
# cython: wraparound=False

from __future__ import division
import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange
from libc.stdint cimport int32_t, int64_t
from libc.string cimport memcpy, memset
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf

## Numpy must be initialized whenever is called from C or Cython
cnp.import_array()

cdef cnp.ndarray wrapPtr( void* array, cnp.ndarray dims, int typenum ):
    if( not dims.flags[ "C_CONTIGUOUS" ] or
        not dims.flags[ "ALIGNED" ] or
        dims.dtype != np.intp ):
        dims = np.require( dims.flat, dtype=np.intp, requirements=[ "C", "A" ] )

    return cnp.PyArray_SimpleNewFromData( dims.size,
                                          <cnp.npy_intp *> cnp.PyArray_GETPTR1( dims, 0 ),
                                          typenum, array )


cdef cnp.ndarray wrap1dPtr( void* array, int length, int typenum ):
    cdef cnp.npy_intp dims[1]
    dims[0] = <cnp.npy_intp> length

    return cnp.PyArray_SimpleNewFromData( 1, dims, typenum, array )


cdef cnp.ndarray wrap2dPtr( void* array, int rows, int cols, int typenum, int fortran=False ):
    cdef cnp.npy_intp dims[2]
    cdef cnp.ndarray arr

    dims[0] = <cnp.npy_intp> cols
    dims[1] = <cnp.npy_intp> rows
    arr = cnp.PyArray_SimpleNewFromData( 2, dims, typenum, array )

    if( fortran ): ## hack to return fortran-ordered array by transposing the c-ordered one
        return np.transpose( arr )
    return arr


cdef void* getPtr( cnp.ndarray array ):
    if( not array.flags[ "FORC" ] or
        not array.flags[ "ALIGNED" ] ):
        raise ValueError( "Array array must be contiguous and aligned" )

    if( array.ndim == 1 ):
        return cnp.PyArray_GETPTR1( array, 0 )
    elif( array.ndim == 2 ):
        return cnp.PyArray_GETPTR2( array, 0, 0 )
    else: ## HG: I promise that someday I'll implement the case for n-dim arrays
        raise ValueError( "Array array must be at most 2-dimensional" )


cdef cnp.ndarray arraySanitize( object array, type dtype=None,
                                int fortran=False, int writtable=False ):
    reqs = [ "A" ]
    if( fortran ):
        reqs.append( "F" )
    else:
        reqs.append( "C" )
    if( writtable ):
        reqs.append( "W" )

    return np.require( np.asarray( array ), dtype=dtype, requirements=reqs )


###
### sMatrix
###
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
    def __cinit__( self, object arr, int copy_data=False ):
        self.data_alloc = True

        if( arr is None ):
            arr = np.empty( (1,0), dtype=np.float64 )

        try:
            arr = np.atleast_2d( np.asarray( arr, dtype=np.float64 ) )
        except:
            raise TypeError( "argument must be an array" )

        if( not np.all( np.isfinite( arr ) ) ):
            raise ValueError( "argument must have finite elements" )

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
        print_debug()

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


    cdef void setDataPtr( self, double* ptr ):
        if( self.data_alloc ):
            free( self.data )
            self.data_alloc = False

        self.data = ptr


    cdef inline void copyFortranIdxs( self, int64_t* ridx, int64_t* cidx,
                                      int64_t roffset=0, int64_t coffset=0 ):
        ## have to add one to offsets because Fortran
        self.copyIdxs( ridx, cidx, roffset + 1, coffset + 1 )


    cdef void copyIdxs( self, int64_t* ridx, int64_t* cidx,
                        int64_t roffset=0, int64_t coffset=0 ):
        memcpy( ridx, self.ridx, self.nnz * sizeof( int64_t ) )
        memcpy( cidx, self.cidx, self.nnz * sizeof( int64_t ) )

        if( roffset > 0 or coffset > 0 ):
            for k in range( self.nnz ):
                ridx[k] += roffset
                cidx[k] += coffset


    cdef void copyIdxs32( self, int32_t* ridx, int32_t* cidx,
                          int32_t roffset=0, int32_t coffset=0 ):
        for k in range( self.nnz ):
            ridx[k] = ( <int32_t> self.ridx[k] ) + roffset
            cidx[k] = ( <int32_t> self.cidx[k] ) + coffset


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


    cpdef cnp.ndarray dot( self, double[:] x ):
        """
        A.dot(x)

        Matrix-vector product, x must be a 1-D numpy float64 array

        """

        cdef double[:] out
        cdef int64_t row, idx
        out = np.zeros( (self.nrows,), dtype=np.float64 )

        for row in prange( self.nrows, nogil=True ):
            for idx in range( self.rptr[row], self.rptr[row+1] ):
                out[row] += self.data[idx] * x[ self.cidx[idx] ]

        return np.asarray( out )


    cpdef add_sparse( self, sMatrix A ):
        """
        A.add_sparse(B)

        Compute A = A + B, where B is an sMatrix.
        Zero entries in A are left unchanged.

        """

        cdef int64_t row, aidx
        cdef int64_t *sidx

        if( self.nrows != A.nrows or
            self.ncols != A.ncols ):
            raise ValueError( "Argument must have same dimensions as original matrix" )

        sidx = <int64_t *> malloc( self.nrows * sizeof( int64_t ) )
        memcpy( sidx, self.rptr, self.nrows * sizeof( int64_t ) )

        for row in prange( self.nrows, nogil=True ):
            for aidx in range( A.rptr[row], A.rptr[row+1] ):
                while( sidx[row] < self.rptr[row+1] and
                       self.cidx[ sidx[row] ] <= A.cidx[aidx] ):
                    if( self.cidx[ sidx[row] ] == A.cidx[aidx] ):
                        self.data[ sidx[row] ] += A.data[aidx]
                        break
                    else:
                        sidx[row] += 1

        free( sidx )


    def toarray( self ):
        """
        M = A.toarray()

        Transforms A into a dense numpy array M.

        """
        return self[:]


    def __repr__( self ):
        out = "{0}x{1}: {2} elems, {3:.1%} sparsity, ".format( self.nrows, self.ncols, self.nnz,
                                                               self.nnz/(self.nrows*self.ncols) )
        if( self.data_alloc ):
            out += "owns data"
        else:
            out += "does not own data"

        return out


    def __str__( self ):
        return str( self.toarray() )


    def __format__( self, str fmt ):
        if( str( fmt ).lower() == "r" ):
            return self.__repr__()

        return str( self )


###
### Options
###
cdef datatype checkType( object val ):
    if( isinstance( val, bool ) ):
        return BOOL
    elif( isinstance( val, int ) ):
        return INT
    elif( isinstance( val, float ) ):
        return DOUBLE
    elif( isinstance( val, str ) ):
        return STR
    else:
        return NONE


cdef class OptPair:
    def __init__( self, value, dtype ):
        self.value = value
        self.dtype = dtype


    def __str__( self ):
        return str( self.value )


    def __richcmp__( self, object value, int op ):
        if( op == 0 ):
            return ( self.value < value )
        elif( op == 1 ):
            return ( self.value <= value )
        elif( op == 2 ):
            return ( self.value == value )
        elif( op == 3 ):
            return ( self.value != value )
        elif( op == 4 ):
            return ( self.value > value )
        elif( op == 5 ):
            return ( self.value >= value )


    def __repr__( self ):
        tmp = ""
        if( self.dtype == BOOL ):
            tmp = "bool"
        elif( self.dtype == INT ):
            tmp = "int"
        elif( self.dtype == DOUBLE ):
            tmp = "float"
        elif( self.dtype == STR ):
            tmp = "str"
        else:
            tmp = "none"

        return "{0} ({1})".format( self.value, tmp )


    def __format__( self, fmt ):
        if( str( fmt ).lower() == "r" ):
            return self.__repr__()

        return str( self )


cdef class Options:
    def __init__( self, dict legacy=None, int case_sensitive=False ):
        self.legacy = dict()
        self.data = dict()
        self.case_sensitive = case_sensitive

        if( legacy is not None ):
            self.legacyInsert( legacy )


    def __setitem__( self, key, value ):
        cdef datatype dtype = checkType( value )
        cdef str mykey

        if( not isinstance( key, str ) ):
            raise TypeError( "key must be a string" )

        if( value is not None and
            dtype == NONE ):
            raise TypeError( "invalid datatype" )

        if( dtype == STR and not self.case_sensitive ):
            value = value.lower()

        mykey = self.sanitizeKey( key )
        self.data[ mykey ] = OptPair( value, dtype )


    def __getitem__( self, key ):
        cdef str mykey

        if( not isinstance( key, str ) ):
            raise TypeError( "key must be a string" )

        mykey = self.sanitizeKey( key )

        if( mykey not in self.data ):
            # raise KeyError( "unknown key" )
            return OptPair( None, NONE )

        return self.data[ mykey ]


    def __delitem__( self, key ):
        if( not isinstance( key, str ) ):
            raise KeyError( "key must be a string" )

        mykey = self.sanitizeKey( key )

        if( mykey not in self.data ):
            return

        del self.data[ mykey ]


    def __iter__( self ):
        return iter( self.data )


    def __len__( self ):
        return len( self.data )


    def __contains__( self, key ):
        cdef str mykey = self.sanitizeKey( key )
        return bool( mykey in self.data and self.data[mykey] )


    def __str__( self ):
        return str( self.data )


    def __repr__( self ):
        return repr( self.data )


    cpdef legacyInsert( self, dict legacy ):
        for (key,value) in legacy.iteritems():
            if( not self.case_sensitive ):
                self.legacy[ key.lower() ] = value.lower()
            else:
                self.legacy.update( legacy )


    cdef str sanitizeKey( self, str key ):
        cdef str lkey = key
        if( not self.case_sensitive ):
            lkey = key.lower()

        if( lkey in self.legacy ):
            return self.legacy[ lkey ]

        return lkey


    def toDict( self ):
        return { key:self.data[key].value for key in self.data }
