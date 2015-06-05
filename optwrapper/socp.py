from __future__ import division
import numpy as np
import types
from optwrapper import nlp, ocp

class Problem( ocp.Problem ):
    """
    Switched Optimal Control Problem

    """

    def __init__( self, Nstates, Ninputs, Nmodes, Ncons ):
        """
        arguments:
        Nstates: number of continuous states
        Ninputs: number of continuous inputs
        Nmodes:  number of modes (or discrete inputs)
        Ncons:   number of inequality constraints
        """

        try:
            self.Nmodes = int( Nmodes )
        except:
            raise ValueError( "Nmodes must be an integer" )

        if( self.Nmodes < 1 ):
            raise ValueError( "Nmodes must be larger than zero" )

        ocp.Problem.__init__( self, Nstates, Ninputs, Ncons )


    def costInstant( self, icost, dxpattern=None, dupattern=None ):
        """
        set the instant cost functions and their gradients

        arguments:
        icost:     tuple of instant cost functions
        dxpattern: tuple of binary pattern of the state gradient for sparse nlp (optional)
        dupattern: tuple of binary pattern of the input gradient for sparse nlp (optional)

        """

        try:
            self.icost = tuple( icost )
        except:
            raise ValueError( "icost must be a tuple" )

        if( len( self.icost ) != self.Nmodes ):
            raise ValueError( "icost must have Nmodes elements" )

        if( dxpattern is None or dupattern is None ):
            return

        tmpdx = list( dxpattern )
        for k in range( self.Nmodes ):
            tmpdx[k] = np.asfortranarray( tmpdx[k], dtype=np.int )
            if( tmpdx[k].shape != ( self.Nstates, ) ):
                raise ValueError( "Each dxpattern must have size (" +
                                  str(self.Nstates) + ",)." )
        self.icostdxpattern = tuple( tmpdx )

        tmpdu = list( dupattern )
        for k in range( self.Nmodes ):
            tmpdu[k] = np.asfortranarray( tmpdu[k], dtype=np.int )
            if( tmpdu[k].shape != ( self.Ninputs, ) ):
                raise ValueError( "Each dupattern must have size (" +
                                  str(self.Ninputs) + ",)." )
        self.icostdupattern = tuple( tmpdu )


    def vectorField( self, vfield, dxpattern=None, dupattern=None ):
        """
        sets the vector field function and its gradients

        arguments:
        vfield:    vector field function
        dxpattern: binary pattern of the state gradient for sparse nlp (optional)
        dupattern: binary pattern of the input gradient for sparse nlp (optional)

        """

        try:
            self.vfield = tuple( vfield )
        except:
            raise ValueError( "vfield must be a tuple" )

        if( len( self.vfield ) != self.Nmodes ):
            raise ValueError( "vfield must have Nmodes elements" )

        if( dxpattern is None or dupattern is None ):
            return

        tmpdx = list( dxpattern )
        for k in range( self.Nmodes ):
            tmpdx[k] = np.asfortranarray( tmpdx[k], dtype=np.int )
            if( tmpdx[k].shape != ( self.Nstates, self.Nstates ) ):
                raise ValueError( "Each dxpattern must have size (" +
                                  str(self.Nstates) + "," + str(self.Nstates) + ")." )

        self.vfielddxpattern = tuple( tmpdx )

        tmpdu = list( dupattern )
        for k in range( self.Nmodes ):
            tmpdu[k] = np.asfortranarray( tmpdu[k], dtype=np.int )
            if( tmpdu[k].shape != ( self.Nstates, self.Ninputs ) ):
                raise ValueError( "Each dupattern must have size (" +
                                  str(self.Nstates) + "," + str(self.Ninputs) + ")." )
        self.vfielddupattern = tuple( tmpdu )


    def discForwardEuler( self, Nsamples ):
        """
        transforms this switched optimal control problem into a nonlinear programming problem using
        the Forward Euler ODE approximation with uniform sampling, and relaxing the discrete inputs
        as vectors in the simplex with dimension Nmodes

        Arguments:
        Nsamples: number of discrete time samples

        Returns:
        feuler:     nlp.SparseProblem instance
        solnDecode: function to decode final nonlinear programming vector

        """

        ## index helpers
        stidx = np.arange( 0, self.Nstates * ( Nsamples + 1 ) ).reshape(
            ( self.Nstates, Nsamples + 1 ), order='F' )
        uidx = ( stidx.size +
                 np.arange( 0, self.Ninputs * Nsamples ).reshape( ( self.Ninputs, Nsamples ),
                                                                  order='F' ) )
        didx = ( stidx.size + uidx.size +
                 np.arange( 0, self.Nmodes * Nsamples ).reshape( ( self.Nmodes, Nsamples ),
                                                                 order='F' ) )
        dconsidx = stidx
        iconsidx = ( dconsidx.size +
                     np.arange( 0, self.Ncons * Nsamples ).reshape( ( self.Ncons, Nsamples ),
                                                                    order='F' ) )

        deltaT = ( self.tf - self.t0 ) / ( Nsamples + 1 )

        feuler = nlp.SparseProblem( N = stidx.size + uidx.size + didx.size,
                                    Ncons = dconsidx.size + iconsidx.size,
                                    Nconslin = Nsamples )


        def encode( st, u, d ):
            """
            ( state, continuous input, discrete input ) -> optimization vector

            """

            s = np.zeros( ( feuler.N, ) )

            if( st.ndim == 2 and st.shape == ( self.Nstates, Nsamples+1 ) ):
                s[ stidx.ravel( order='F' ) ] = st.ravel( order='F' )
            elif( st.ndim == 1 and st.shape == ( self.Nstates, ) ):
                s[ stidx.ravel( order='F' ) ] = np.tile( st, ( Nsamples+1, ) )
            else:
                raise ValueError( "unknown state vector format." )

            if( u.ndim == 2 and u.shape == ( self.Ninputs, Nsamples ) ):
                s[ uidx.ravel( order='F' ) ] = u.ravel( order='F' )
            elif( u.ndim == 1 and u.shape == ( self.Ninputs, ) ):
                s[ uidx.ravel( order='F' ) ] = np.tile( u, ( Nsamples, ) )
            else:
                raise ValueError( "unknown continuous input vector format." )

            if( d.ndim == 2 and d.shape == ( self.Nmodes, Nsamples ) ):
                s[ didx.ravel( order='F' ) ] = d.ravel( order='F' )
            elif( d.ndim == 1 and d.shape == ( self.Nmodes, ) ):
                s[ didx.ravel( order='F' ) ] = np.tile( d, ( Nsamples, ) )
            else:
                raise ValueError( "unknown discrete input vector format." )

            return s


        def decode( s ):
            """
            optimization vector -> ( state, continuous input, discrete input )

            """

            st = s[ stidx ]
            u = s[ uidx ]
            d = s[ didx ]

            return ( st, u, d )


        def solnDecode( s ):
            """
            decodes final solution vector

            Arguments:
            s: nonlinear programming optimization vector

            Returns:
            st: state matrix with dimension (Nstates,Nsamples+1)
            u:  continuous input matrix with dimension (Ninputs,Nsamples)
            d:  discrete input matrix with dimension (Nmodes,Nsamples)
            t:  time vector with dimension (Nsamples+1,)

            """

            return decode( s ) + ( np.linspace( self.t0, self.tf, Nsamples + 1 ), )


        def objf( out, s ):
            """
            discretized nonlinear programming cost function

            """

            ( st, u, d ) = decode( s )

            out[0] = 0
            for k in range( Nsamples ):
                for idx in range( self.Nmodes ):
                    out[0] += d[idx,k] * self.icost[idx]( st[:,k], u[:,k], grad=False )

            out[0] *= deltaT
            out[0] += self.fcost( st[:,Nsamples], grad=False )


        def objg( out, s ):
            """
            discretized nonlinear programming cost gradient

            """

            ( st, u, d ) = decode( s )

            for k in range( Nsamples ):
                for idx in range( self.Nmodes ):
                    ( fx, dx, du ) = self.icost[idx]( st[:,k], u[:,k] )
                    out[ stidx[:,k] ] += d[idx,k] * dx * deltaT
                    out[ uidx[:,k] ] += d[idx,k] * du * deltaT
                    out[ didx[idx,k] ] = fx * deltaT

            out[ stidx[:,Nsamples] ] = self.fcost( st[:,Nsamples] )[1]


        def objgpattern():
            """
            binary pattern of the sparse cost gradient

            """

            if( self.icostdupattern is None or
                self.icostdxpattern is None or
                self.fcostdxpattern is None ):
                return None

            out = np.zeros( ( feuler.N, ), dtype=np.int )

            for k in range( Nsamples ):
                for idx in range( self.Nmodes ):
                    out[ stidx[:,k] ] += self.icostdxpattern[idx]
                    out[ uidx[:,k] ] += self.icostdupattern[idx]
                    out[ didx[idx,k] ] = 1

            out[ stidx[:,Nsamples] ] = self.fcostdxpattern

            return out


        def consf( out, s ):
            """
            discretized nonlinear programming constraint function

            """

            ( st, u, d ) = decode( s )

            ## initial condition collocation constraint
            out[ dconsidx[:,0] ] = st[:,0] - self.init
            for k in range( Nsamples ):
                ## Forward Euler collocation equality constraints
                fx = 0
                for idx in range( self.Nmodes ):
                    fx += d[idx,k] * self.vfield[idx]( st[:,k], u[:,k], grad=False )
                out[ dconsidx[:,k+1] ] = st[:,k+1] - st[:,k] - deltaT * fx

                ## inequality constraints
                if( self.Ncons > 0 ):
                    out[ iconsidx[:,k] ] = self.cons( st[:,k+1], grad=False )


        def consg( out, s ):
            """
            discretized nonlinear programming constraint gradient

            """

            ( st, u, d ) = decode( s )

            out[ np.ix_( dconsidx[:,0], stidx[:,0] ) ] = np.identity( self.Nstates )
            for k in range( Nsamples ):
                out[ np.ix_( dconsidx[:,k+1], stidx[:,k] ) ] = - np.identity( self.Nstates )
                out[ np.ix_( dconsidx[:,k+1], stidx[:,k+1] ) ] = np.identity( self.Nstates )
                for idx in range( self.Nmodes ):
                    ( fx, dyndx, dyndu ) = self.vfield[idx]( st[:,k], u[:,k] )
                    out[ np.ix_( dconsidx[:,k+1], stidx[:,k] ) ] += - d[idx,k] * dyndx * deltaT
                    out[ np.ix_( dconsidx[:,k+1], uidx[:,k] ) ] += - d[idx,k] * dyndu * deltaT
                    out[ dconsidx[:,k+1], didx[idx,k] ] = - fx * deltaT

                if( self.Ncons > 0 ):
                    out[ np.ix_( iconsidx[:,k], stidx[:,k+1] ) ] = self.cons( st[:,k+1] )[1]


        def consgpattern():
            """
            binary pattern of the sparse nonlinear constraint gradient

            """

            if( self.vfielddxpattern is None or
                self.vfielddupattern is None or
                ( self.Ncons > 0 and self.consdxpattern is None ) ):
                return None

            out = np.zeros( ( feuler.Ncons, feuler.N ), dtype=np.int )

            out[ np.ix_( dconsidx[:,0], stidx[:,0] ) ] = np.identity( self.Nstates )
            for k in range( Nsamples ):
                out[ np.ix_( dconsidx[:,k+1], stidx[:,k] ) ] = np.identity( self.Nstates )
                out[ np.ix_( dconsidx[:,k+1], stidx[:,k+1] ) ] = np.identity( self.Nstates )
                for idx in range( self.Nmodes ):
                    out[ np.ix_( dconsidx[:,k+1], stidx[:,k] ) ] += self.vfielddxpattern[idx]
                    out[ np.ix_( dconsidx[:,k+1], uidx[:,k] ) ] += self.vfielddupattern[idx]
                    out[ dconsidx[:,k+1], didx[idx,k] ] = np.ones( (self.Nstates,) )

                if( self.Ncons > 0 ):
                    out[ np.ix_( iconsidx[:,k], stidx[:,k+1] ) ] = self.consdxpattern

            return out


        def conslinA():
            """
            create linear matrix for discrete input simplex constraint

            """

            A = np.zeros( (Nsamples, feuler.N) )
            for k in range( Nsamples ):
                A[ k, didx[:,k] ] = np.ones( (self.Nmodes,) )

            return A


        ## setup feuler now that all the functions are defined
        feuler.initPoint( encode( self.init,
                                  np.zeros( (self.Ninputs,) ),
                                  1/self.Nmodes * np.ones( (self.Nmodes,) ) ) )
        feuler.consBox( encode( self.consstlb, self.consinlb, np.zeros( (self.Nmodes,) ) ),
                        encode( self.consstub, self.consinub, np.ones( (self.Nmodes,) ) ) )
        feuler.consLinear( conslinA(), np.ones( (Nsamples,) ), np.ones( (Nsamples,) ) )
        feuler.objFctn( objf )
        feuler.objGrad( objg, pattern=objgpattern() )
        if( self.Ncons > 0 ):
            feuler.consFctn( consf,
                             np.concatenate( ( np.zeros( ( dconsidx.size, ) ),
                                               np.tile( self.conslb, ( Nsamples, ) ) ) ),
                             np.concatenate( ( np.zeros( ( dconsidx.size, ) ),
                                               np.tile( self.consub, ( Nsamples, ) ) ) ) )
        else:
            feuler.consFctn( consf,
                             np.zeros( ( dconsidx.size, ) ),
                             np.zeros( ( dconsidx.size, ) ) )

        feuler.consGrad( consg, pattern=consgpattern() )

        return ( feuler, solnDecode )


def haarWaveletApprox( t, arr, N ):
    """
    computes Haar Wavelet approximation of a nonuniform-sampled function

    Arguments:
    t:   array of time samples of dimension (Nsamples+1,)
    arr: array of function samples of dimension (Ndims,Nsamples)
    N:   new sampling rate of the wavelet transform

    Returns:
    out: Haar wavelet transform of arr with dimension (Ndims,2**N)

    """

    ## sanity checks
    try:
        ( Ndims, Nsamples ) = arr.shape
    except:
        raise TypeError( "arr must be a two-dimensional array" )

    try:
        N = int( N )
    except:
        raise TypeError( "N must be an integer" )

    if( t.size - 1 != Nsamples ):
        raise TypeError( "t must have length {0}".format( Nsamples + 1 ) )

    if( t[-1] <= t[0] ):
        raise ValueError( "final time is smaller or equal than initial time" )

    ## add extra Haar sampling times to t and arr
    t = ( t - t[0] ) / ( t[-1] - t[0] ) ## normalize time
    deltaT = 1 / (2**N)
    tnew = np.zeros( (2**N + Nsamples,) )
    arrnew = np.zeros( (Ndims, tnew.size - 1) )

    oidx = 0 ## old array index
    hidx = 1 ## haar sampling index

    for k in range( 0, tnew.size ):
        if( oidx < Nsamples and t[oidx] < deltaT * hidx ): ## copy samples from original arrays
            tnew[k] = t[oidx]
            arrnew[:,k] = arr[:,oidx]
            oidx += 1
        else:                                              ## need to insert new haar sample
            tnew[k] = deltaT * hidx
            if( k < tnew.size - 1 ):                       ## tnew is one longer than arrnew
                arrnew[:,k] = arr[:,oidx-1]
            hidx += 1

    ## compute Haar Wavelet coefficients
    coeff = np.zeros( (Ndims,2**N) )
    dtnew = np.diff( tnew )

    coeff[:,0] = np.sum( arrnew * dtnew, axis=1 )

    cidx = 1 ## coeff index
    for k in range( N ):
        for j in range( 2**k ): ## TODO: this loop can be improved a bit
            idx = 0
            while( tnew[idx] < j / 2**k ):
                idx += 1

            while( tnew[idx] < (j + 0.5) / 2**k ):
                coeff[:,cidx] += dtnew[idx] * arrnew[:,idx]
                idx += 1

            while( tnew[idx] < (j + 1) / 2**k ):
                coeff[:,cidx] -= dtnew[idx] * arrnew[:,idx]
                idx += 1

            cidx += 1

    ## create matrix with Haar basis functions values at each sampling point
    tmpp = np.zeros( (2**N,) )
    tmpq = np.zeros( (2**N,) )
    H = np.zeros( (2**N, 2**N) )
    H[0,:] = np.ones( (2**N,) )

    tmpq[1] = 1
    for k in range( 1, N ):
        tmpp[2**k:2**(k+1)] = k * np.ones( (2**k,) )
        tmpq[2**k:2**(k+1)] = np.arange( 1, (2**k) + 1 )

    for k in range( 1, 2**N ):
        P = tmpp[k]
        Q = tmpq[k]
        for j in range( int( 2**N * (Q-1) / 2**P ), int( 2**N * (Q-0.5) / 2**P ) ):
            H[k,j] = 2**P
        for j in range( int( 2**N * (Q-0.5) / 2**P ), int( 2**N * Q / 2**P ) ):
            H[k,j] = -2**P

    return np.dot( coeff, H )


def pwmTransform( t, u, d ):
    """
    computes the PWM transformation of a uniformly sampled discrete input matrix

    Arguments:
    t: array of uniform time samples of dimension (Nsamples+1,)
    u: array of continuous input samples of dimension (Ninputs,Nsamples)
    d: array of discrete input samples of dimension (Nmodes,Nsamples)

    Returns:
    tpwm: array of nonuniform PWM time samples
    upwm: array of resampled continuous input using tpwm
    dpwm: array with PWM transformation of discrete input

    """

    ## sanity checks
    try:
        ( Nmodes, Nsamples ) = d.shape
    except:
        raise TypeError( "d must be a two-dimensional array" )

    try:
        Ninputs = u[:,0].size
    except:
        raise TypeError( "u must be a two-dimensional array" )

    if( t.size - 1 != Nsamples ):
        raise TypeError( "t must have length {0}".format( Nsamples + 1 ) )

    if( t[-1] <= t[0] ):
        raise ValueError( "final time is smaller or equal than initial time" )

    tpwm = np.zeros( ( Nmodes*Nsamples + 1, ) )
    upwm = np.zeros( ( Ninputs, Nmodes*Nsamples ) )
    dpwm = np.zeros( ( Nmodes, Nmodes*Nsamples ) )

    deltaT = (t[-1] - t[0]) / Nsamples

    tpwm[0] = t[0]
    pwmidx = 1
    for k in range( Nsamples ):
        for j in range( Nmodes ):
            if( d[j,k] > 0 ):
                tpwm[pwmidx] = t[k] + deltaT * np.sum( d[:j+1,k] )
                upwm[:,pwmidx-1] = u[:,k]
                dpwm[j,pwmidx-1] = 1
                pwmidx += 1

    return ( tpwm[:pwmidx], upwm[:,:pwmidx-1], dpwm[:,:pwmidx-1] )
