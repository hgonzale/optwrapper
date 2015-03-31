from scipy.sparse import csr_matrix

class csr_vector( csr_matrix ):
    def __init__( self, arg1, shape=None, dtype=None, copy=False ):
        csr_matrix.__init__( self, arg1, shape, dtype, copy )

        if( self.shape[0] != 1 ):
            raise TypeError( "csr_vectors must have shape[0] = 1, but shape[0] = " +
                             str( self.shape[0] ) )

    def __setitem__( self, key, value ):
        csr_matrix.__setitem__( self, (0, key), value )

    def __getitem__( self, key ):
        return csr_matrix.__getitem__( self, (0, key) )

    def toarray( self ):
        return csr_matrix.toarray( self )[0]
