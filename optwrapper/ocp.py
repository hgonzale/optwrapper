from __future__ import division
import numpy as np
import types
from optwrapper import nlp

class Problem:

    """
    Optimal Control Problem

    """

    def __init__( self, Nstates, Ninputs, Ncons ):
        """
        Arguments:
        Nstates: number of states
        Ninputs: number of inputs
        Ncons:   number of inequality constraints

        """

        try:
            self.Nstates = int( Nstates )
        except:
            raise ValueError( "Nstates must be an integer" )
        if( self.Nstates <= 0 ):
            raise ValueError( "Nstates must be strictly positive" )

        try:
            self.Ninputs = int( Ninputs )
        except:
            raise ValueError( "Ninputs must be an integer" )
        if( self.Ninputs < 0 ):
            raise ValueError( "Ninputs must be positive" )

        try:
            self.Ncons = int( Ncons )
        except:
            raise ValueError( "Ncons must be an integer" )
        if( self.Ncons < 0 ):
            raise ValueError( "Ncons must be positive" )

        self.init = np.zeros( (self.Nstates,) )
        self.t0 = 0
        self.tf = 1
        self.icost = None
        self.icostdxpattern = None
        self.icostdupattern = None
        self.fcost = None
        self.fcostdxpattern = None
        self.vfield = None
        self.vfielddxpattern = None
        self.vfielddupattern = None
        self.cons = None
        self.consdxpattern = None
        self.conslb = None
        self.consub = None
        self.consstlb = None
        self.consstub = None
        self.consinlb = None
        self.consinub = None


    def checkSetup( self ):
        out = ( self.icost is not None and
                self.fcost is not None and
                self.vfield is not None and
                self.consstlb is not None and
                self.consstub is not None and
                self.consinlb is not None and
                self.consinub is not None )

        if( self.Ncons > 0 ):
            out = out and ( self.cons is not None and
                            self.conslb is not None and
                            self.consub is not None )

        return out


    def initCond( self, init ):
        """
        sets the initial point for the optimization problem

        Arguments:
        init: initial condition, a 1-D array of size Nstates, defaults to an array of zeros

        """

        self.init = np.asfortranarray( init )

        if( self.init.shape != (self.Nstates, ) ):
            raise ValueError( "Argument must have size (" + str(self.Nstates) + ",)." )


    def timeHorizon( self, t0, tf ):
        """
        sets the initial and final times for the optimization problem

        Arguments:
        t0: initial time of the optimization problem
        tf: final time of the optimization problem

        """

        self.t0 = float( t0 )
        self.tf = float( tf )

        if( self.t0 >= self.tf ):
            raise ValueError( "Final time must be larger than initial time." )


    def costInstant( self, icost, dxpattern=None, dupattern=None ):
        """
        set the instant cost function and its gradients

        Arguments:
        icost:     instant cost function
        dxpattern: binary pattern of the state gradient for sparse nlp (optional)
        dupattern: binary pattern of the input gradient for sparse nlp (optional)

        """

        if( type(icost) != types.FunctionType ):
            raise ValueError( "Argument must be a function" )

        self.icost = icost
        if( not dxpattern is None and not dupattern is None ):
            self.icostdxpattern = np.asfortranarray( dxpattern, dtype=np.int )
            self.icostdupattern = np.asfortranarray( dupattern, dtype=np.int )
            if( self.icostdxpattern.shape != ( self.Nstates, ) ):
                raise ValueError( "Argument 'dxpattern' must have size (" +
                                  str(self.Nstates) + ",)." )
            if( self.icostdupattern.shape != ( self.Ninputs, ) ):
                raise ValueError( "Argument 'dupattern' must have size (" +
                                  str(self.Ninputs) + ",)." )


    def costFinal( self, fcost, dxpattern=None ):
        """
        sets the final cost and its gradient with respect to the states

        Arguments:
        fcost: the final cost

        """

        if ( type(fcost) != types.FunctionType ):
            raise ValueError( "Argument must be a function" )

        self.fcost = fcost
        if( not dxpattern is None ):
            self.fcostdxpattern = np.asfortranarray( dxpattern, dtype=np.int )
            if( self.fcostdxpattern.shape != ( self.Nstates, ) ):
                raise ValueError( "Argument 'dxpattern' must have size (" +
                                  str(self.Nstates) + ",)." )


    def vectorField( self, vfield, dxpattern=None, dupattern=None ):
        """
        sets the vector field function and its gradients

        Arguments:
        vfield:    vector field function
        dxpattern: binary pattern of the state gradient for sparse nlp (optional)
        dupattern: binary pattern of the input gradient for sparse nlp (optional)

        """

        if ( type(vfield) != types.FunctionType ):
            raise ValueError( "Argument must be a function" )

        self.vfield = vfield
        if( not dxpattern is None and not dupattern is None ):
            self.vfielddxpattern = np.asfortranarray( dxpattern, dtype=np.int )
            self.vfielddupattern = np.asfortranarray( dupattern, dtype=np.int )
            if( self.vfielddxpattern.shape != ( self.Nstates, self.Nstates ) ):
                raise ValueError( "Argument 'dxpattern' must have size (" +
                                  str(self.Nstates) + "," + str(self.Nstates) + ")." )
            if( self.vfielddupattern.shape != ( self.Nstates, self.Ninputs ) ):
                raise ValueError( "Argument 'dupattern' must have size (" +
                                  str(self.Nstates) + "," + str(self.Ninputs) + ")." )


    def consNonlinear( self, cons, lb=None, ub=None, dxpattern=None ):

        """
        sets the instantaneous nonlinear constraint function and its gradients

        Arguments:
        cons:      nonlinear constraint function
        lb:        lower bound of the nonlinear inequality constraint (default: -inf)
        ub:        upper bound of the nonlinear inequality constraint (default: 0)
        dxpattern: binary pattern of the gradient for sparse nlp (optional)

        """

        if( self.Ncons == 0 ):
            raise RuntimeError( "Cannot use nonlinear constraints when Ncons is zero" )

        if( type(cons) != types.FunctionType ):
            raise ValueError( "Argument must be a function" )

        if( lb is None ):
            lb = -np.inf * np.ones( self.Ncons )

        if( ub is None ):
            ub = np.zeros( self.Ncons )

        self.cons = cons
        self.conslb = np.asfortranarray( lb )
        self.consub = np.asfortranarray( ub )
        if( self.conslb.shape != (self.Ncons, ) or
            self.consub.shape != (self.Ncons, ) ):
            raise ValueError( "Bounds must have size (" + str(self.Ncons) + ",)." )

        if( not dxpattern is None ):
            self.consdxpattern = np.asfortranarray( dxpattern, dtype=np.int )
            if( self.consdxpattern.shape != ( self.Ncons, self.Nstates ) ):
                raise ValueError( "Argument 'dxpattern' must have size (" +
                                  str(self.Ncons) + "," + str(self.Nstates) + ")." )


    def consBoxState( self, lb, ub ):
        """
        sets the state box constraints

        Arguments:
        lb: lower bound of the state inequality constraint
        ub: upper bound of the state inequality constraint

        """

        self.consstlb = np.asfortranarray( lb )
        self.consstub = np.asfortranarray( ub )

        if( self.consstlb.shape != (self.Nstates, ) or
            self.consstub.shape != (self.Nstates, ) ):
            raise ValueError( "Arguments must have size (" + str(self.Nstates) + ",)." )


    def consBoxInput( self, lb, ub ):
        """
        sets the input box constraints

        Arguments:
        lb: lower bound of the input inequality constraint
        ub: upper bound of the input inequality constraint

        """

        self.consinlb = np.asfortranarray( lb )
        self.consinub = np.asfortranarray( ub )

        if( self.consinlb.shape != (self.Ninputs, ) or
            self.consinub.shape != (self.Ninputs, ) ):
            raise ValueError( "Arguments must have size (" + str(self.Ninputs) + ",)." )


    def discForwardEuler( self, Nsamples ):
        """
        transforms this optimal control problem into a nonlinear programming problem using the
        Forward Euler ODE approximation and uniform sampling

        Arguments:
        Nsamples: number of discrete time samples

        Returns:
        feuler:     nlp.SparseProblem instance
        solnDecode: function to decode final nonlinear programming vector

        """

        if( not self.checkSetup() ):
            raise ValueError( "Problem has not been properly configured" )

        ## index helpers
        stidx = np.arange( 0, self.Nstates * ( Nsamples + 1 ),
                           dtype=np.int ).reshape( ( self.Nstates, Nsamples + 1 ), order='F' )
        uidx = ( stidx.size +
                 np.arange( 0, self.Ninputs * Nsamples,
                            dtype=np.int ).reshape( ( self.Ninputs, Nsamples ), order='F' ) )

        dconsidx = stidx
        iconsidx = ( dconsidx.size +
                     np.arange( 0, self.Ncons * Nsamples,
                                dtype=np.int ).reshape( ( self.Ncons, Nsamples ), order='F' ) )

        deltaT = ( self.tf - self.t0 ) / ( Nsamples + 1 )

        feuler = nlp.SparseProblem( N = stidx.size + uidx.size,
                                    Ncons = dconsidx.size + iconsidx.size,
                                    Nconslin = 0 )


        def encode( st, u ):
            """
            ( state, input ) -> optimization vector

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
                raise ValueError( "unknown input vector format." )

            return s


        def decode( s ):
            """
            optimization vector -> ( state, input )

            """

            st = s[ stidx ]
            u = s[ uidx ]

            return ( st, u )


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


        def initPointEncode( st, u ):
            """
            encodes initial optimization vector

            Arguments:
            st: initial state guess with dimension (Nstates,) or (Nstates,Nsamples+1)
            u:  initial input guess with dimension (Ninputs,) or (Ninputs,Nsamples)

            Returns:
            s: nonlinear programming optimization vector

            """

            return encode( np.asarray( st ), np.asarray( u ) )


        def objf( out, s ):
            """
            discretized nonlinear programming cost function

            """

            ( st, u ) = decode( s )

            tmp = 0
            for k in range( Nsamples ):
                tmp += self.icost( st[:,k], u[:,k], grad=False )

            out[0] = tmp * deltaT + self.fcost( st[:,Nsamples], grad=False )


        def objg( out, s ):
            """
            discretized nonlinear programming cost gradient

            """

            ( st, u ) = decode( s )

            for k in range( Nsamples ):
                ( dx, du ) = self.icost( st[:,k], u[:,k] )[1:]
                out[ stidx[:,k] ] = dx * deltaT
                out[ uidx[:,k] ] = du * deltaT

            out[ stidx[:,Nsamples] ] = self.fcost( st[:,Nsamples] )[1]


        def objgpattern():
            """
            binary pattern of the sparse cost gradient

            """

            icdxpat = ( self.icostdxpattern
                        if self.icostdxpattern is not None
                        else np.ones( (self.Nstates,) ) )
            icdupat = ( self.icostdupattern
                        if self.icostdupattern is not None
                        else np.ones( (self.Ninputs,) ) )
            fcdxpat = ( self.fcostdxpattern
                        if self.fcostdxpattern is not None
                        else np.ones( (self.Nstates,) ) )

            out = np.zeros( ( feuler.N, ), dtype=np.int )

            for k in range( Nsamples ):
                out[ stidx[:,k] ] = icdxpat
                out[ uidx[:,k] ] = icdupat

            out[ stidx[:,Nsamples] ] = fcdxpat

            return out


        def consf( out, s ):
            """
            discretized nonlinear programming constraint function

            """

            ( st, u ) = decode( s )

            ## initial condition collocation constraint
            for k in range( Nsamples ):
                ## Forward Euler collocation equality constraints
                out[ dconsidx[:,k+1] ] = - deltaT * self.vfield( st[:,k], u[:,k], grad=False )

                ## inequality constraints
                if( self.Ncons > 0 ):
                    out[ iconsidx[:,k] ] = self.cons( st[:,k+1], grad=False )


        def consA():
            """
            Linear mixed constraints

            """
            out = np.zeros( ( feuler.Ncons, feuler.N ) )

            out[ dconsidx[:,0], stidx[:,0] ] = 1.0
            for k in range( Nsamples ):
                out[ dconsidx[:,k+1], stidx[:,k+1] ] = 1.0
                out[ dconsidx[:,k+1], stidx[:,k] ] = -1.0

            return out


        def conslb():
            """
            constraints lower bounds

            """
            if( self.Ncons > 0 ):
                out = np.concatenate( ( np.zeros( (dconsidx.size,) ),
                                        np.tile( self.conslb, (Nsamples,) ) ) )
            else:
                out = np.zeros( (dconsidx.size,) )
            out[ dconsidx[:,0] ] = self.init

            return out


        def consub():
            """
            constraints upper bounds

            """
            if( self.Ncons > 0 ):
                out = np.concatenate( ( np.zeros( (dconsidx.size,) ),
                                        np.tile( self.consub, (Nsamples,) ) ) )
            else:
                out = np.zeros( (dconsidx.size,) )
            out[ dconsidx[:,0] ] = self.init

            return out


        def consg( out, s ):
            """
            discretized nonlinear programming constraint gradient

            """

            ( st, u ) =  decode( s )

            for k in range( Nsamples ):
                ( dyndx, dyndu ) = self.vfield( st[:,k], u[:,k] )[1:]
                out[ np.ix_( dconsidx[:,k+1], stidx[:,k] ) ] = - deltaT * dyndx
                out[ np.ix_( dconsidx[:,k+1], uidx[:,k] ) ] = - deltaT * dyndu

                if( self.Ncons > 0 ):
                    out[ np.ix_( iconsidx[:,k], stidx[:,k+1] ) ] = self.cons( st[:,k+1] )[1]


        def consgpattern():
            """
            binary pattern of the sparse nonlinear constraint gradient

            """

            vfdxpat = ( self.vfielddxpattern
                        if self.vfielddxpattern is not None
                        else np.ones( (self.Nstates,self.Nstates) ) )
            vfdupat = ( self.vfielddupattern
                        if self.vfielddupattern is not None
                        else np.ones( (self.Nstates,self.Ninputs) ) )
            if( self.Ncons > 0 ):
                consdxpat = ( self.consdxpattern
                              if self.consdxpattern is not None
                              else np.ones( (self.Ncons,self.Ninputs) ) )

            out = np.zeros( ( feuler.Ncons, feuler.N ), dtype=np.int )

            for k in range( Nsamples ):
                out[ np.ix_( dconsidx[:,k+1], stidx[:,k] ) ] = vfdxpat
                out[ np.ix_( dconsidx[:,k+1], uidx[:,k] ) ] = vfdupat

                if( self.Ncons > 0 ):
                    out[ np.ix_( iconsidx[:,k], stidx[:,k+1] ) ] = consdxpat

            return out


        ## setup feuler now that all the functions are defined
        feuler.consBox( encode( self.consstlb, self.consinlb ),
                        encode( self.consstub, self.consinub ) )
        feuler.objFctn( objf )
        feuler.objGrad( objg, pattern=objgpattern() )
        feuler.consFctn( consf,
                         lb=conslb(),
                         ub=consub(),
                         A=consA() )

        feuler.consGrad( consg, pattern=consgpattern() )

        return ( feuler, initPointEncode, solnDecode )
