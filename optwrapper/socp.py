import numpy as np
import types
from optwrapper import ocp

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

        tmpdx = list( dxpattern )
        for k in range( self.Nmodes ):
            tmpdx[k] = np.asfortranarray( tmpdx[k], dtype=np.int )
            if( tmpdx[k].shape != ( self.Nstates, ) ):
                raise ValueError( "Each dxpattern must have size (" +
                                  str(self.Nstates) + ",)." )
        self.dxpattern = tuple( tmpdx )

        tmpdu = list( dupattern )
        for k in range( self.Nmodes ):
            tmpdu[k] = np.asfortranarray( tmpdu[k], dtype=np.int )
            if( tmpdu[k].shape != ( self.Ninputs, ) ):
                raise ValueError( "Each dupattern must have size (" +
                                  str(self.Ninputs) + ",)." )
        self.dupattern = tuple( tmpdu )


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

        tmpdx = list( dxpattern )
        for k in range( self.Nmodes ):
            tmpdx[k] = np.asfortranarray( tmpdx[k], dtype=np.int )
            if( tmpdx[k].shape != ( self.Nstates, self.Nstates ) ):
                raise ValueError( "Each dxpattern must have size (" +
                                  str(self.Nstates) + "," + str(self.Nstates) + ")." )

            self.dxpattern = tuple( tmpdx )

        tmpdu = list( dupattern )
        for k in range( self.Nmodes ):
            tmpdu[k] = np.asfortranarray( tmpdu[k], dtype=np.int )
            if( tmpdu[k].shape != ( self.Nstates, self.Ninputs ) ):
                raise ValueError( "Each dupattern must have size (" +
                                  str(self.Nstates) + "," + str(self.Ninputs) + ")." )
        self.dupattern = tuple( tmpdu )


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
            optimization vector -> ( state, input )

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
            u:  input matrix with dimension (Ninputs,Nsamples)
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
                out[ iconsidx[:,k] ] = self.cons( st[:,k+1], grad=False )


        def consg( out, s ):
            """
            discretized nonlinear programming constraint gradient

            """

            ( st, u, d ) =  decode( s )

            out[ np.ix_( dconsidx[:,0], stidx[:,0] ) ] = np.identity( self.Nstates )
            for k in range( Nsamples ):
                out[ np.ix_( dconsidx[:,k+1], stidx[:,k] ) ] = - np.identity( self.Nstates )
                out[ np.ix_( dconsidx[:,k+1], stidx[:,k+1] ) ] = np.identity( self.Nstates )
                for idx in range( self.Nmodes ):
                    ( fx, dyndx, dyndu ) = self.vfield[idx]( st[:,k], u[:,k] )
                    out[ np.ix_( dconsidx[:,k+1], stidx[:,k] ) ] += - d[idx,k] * dyndx * deltaT
                    out[ np.ix_( dconsidx[:,k+1], uidx[:,k] ) ] += - d[idx,k] * dyndu * deltaT
                    out[ np.ix_( dconsidx[:,k+1], didx[idx,k] ) ] = - fx * deltaT

                out[ np.ix_( iconsidx[:,k], stidx[:,k+1] ) ] = self.cons( st[:,k+1] )[1]


        def consgpattern():
            """
            binary pattern of the sparse nonlinear constraint gradient

            """

            if( self.vfielddxpattern is None or
                self.vfielddupattern is None or
                self.consdxpattern is None ):
                return None

            out = np.zeros( ( feuler.Ncons, feuler.N ), dtype=np.int )

            out[ np.ix_( dconsidx[:,0], stidx[:,0] ) ] = np.identity( self.Nstates )
            for k in range( Nsamples ):
                out[ np.ix_( dconsidx[:,k+1], stidx[:,k] ) ] = np.identity( self.Nstates )
                out[ np.ix_( dconsidx[:,k+1], stidx[:,k+1] ) ] = np.identity( self.Nstates )
                for idx in range( self.Nmodes ):
                    out[ np.ix_( dconsidx[:,k+1], stidx[:,k] ) ] += self.vfielddxpattern[idx]
                    out[ np.ix_( dconsidx[:,k+1], uidx[:,k] ) ] += self.vfielddupattern[idx]
                    out[ np.ix_( dconsidx[:,k+1], didx[idx,k] ) ] = np.ones( (self.Nstates,) )

                out[ np.ix_( iconsidx[:,k], stidx[:,k+1] ) ] = self.consdxpattern

            return out


        ## setup feuler now that all the functions are defined
        feuler.initPoint( encode( self.init,
                                  np.zeros( (self.Ninputs,) ),
                                  1/self.Nmodes * np.ones( (self.Nmodes,) ) ) )
        feuler.consBox( encode( self.consstlb, self.consinlb, np.zeros( (self.Nmodes,) ) ),
                        encode( self.consstub, self.consinub, np.ones( (self.Nmodes,) ) ) )
        feuler.objFctn( objf )
        feuler.objGrad( objg, pattern=objgpattern() )

        ###############
        ###############
        
        feuler.consFctn( consf,
                         np.concatenate( ( np.zeros( ( dconsidx.size, ) ),
                                           np.tile( self.conslb, ( Nsamples, ) ) ) ),
                         np.concatenate( ( np.zeros( ( dconsidx.size, ) ),
                                           np.tile( self.consub, ( Nsamples, ) ) ) ) )
        feuler.consGrad( consg, pattern=consgpattern() )

        return ( feuler, solnDecode )

        # def setLinCons():
        #     A = np.zeros( (Nsamples, N ) )
        #     for k in range( Nsamples ):
        #         A[ k, Nstates*(Nsamples+1) + Ninputs*Nsamples + k*Nmodes : Nstates*(Nsamples+1) + Ninputs*Nsamples + k*Nmodes + Nmodes] = 1

        #     self.conslinA = A
        #     self.conslinlb = np.ones( (Nsamples,) )
        #     self.conslinub = np.ones( (Nsamples,) )

        self.objf = objectiveFctn
        self.objg = objectiveGrad
        self.consf = constraintFctn
        self.consg = constraintGrad
        setBounds() ## this fctn sets self.lb and self.ub
        setBoundsCons() ## this fctn sets self.conslb and self.consub
        setLinCons()
        self.mixedCons = False
        self.init = np.zeros( ( self.N, ) )
