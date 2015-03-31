import numpy as np
import types
from optwrapper import nlp

class Problem:

    """
    Optimal Control Programming Problem
    """

    def __init__( self, Nstates, Ninputs, Ncons ):
        """
        arguments:
        Nstates = the number of states
        Ninputs = the number of inputs
        Ncons = the number of inequality constraints
        """

        try:
            self.Nstates = int( Nstates )
        except:
            raise ValueError( "Nstates must be an integer" )

        if( self.Nstates <= 0 ):
            raise ValueError( "Nstates must be strictly positive" )

        self.init = np.zeros( ( self.Nstates, 1 ) )
        self.Nstates = Nstates
        self.Ninputs = Ninputs
        self.Ncons = Ncons
        self.t0 = 0
        self.tf = 1
        self.icostf = None
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


    def initCond( self, init ):
        """
        sets the initial point for the optimization problem

        arguments:
        init: initial condition, a 1-D array of size Nstates, defaults to an array of zeros

        """

        self.init = np.asfortranarray( init )

        if( self.init.shape != (self.Nstates, ) ):
            raise ValueError( "Argument must have size (" + str(self.Nstates) + ",)." )


    def timeHorizon( self, t0, tf ):
        """
        sets the initial and final times for the optimization problem

        arguments:
        t0: the initial time of the optimization problem
        tf: the final time of the optimization problem

        """

        self.t0 = float( t0 )
        self.tf = float( tf )

        if( self.t0 >= self.tf ):
            raise ValueError( "Final time must be larger than initial time." )


    def costInstant( self, icost, dxpattern=None, dupattern=None ):
        """
        set the instant cost and its gradients with respect to the states and input

        arguments:
        icost: the instant cost function

        """

        if ( type(icost) != types.FunctionType ):
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

        arguments:
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
        sets the vfield function and its gradients

        arguments:
        vfield: the vector field function

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
        defines the constraints and its gradients

        arguments:
        cons: the matrix of constraints, must have size Ninputseqcons

        """

        if ( type(cons) != types.FunctionType ):
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
        defines the box constraints on the states and inputs

        arguments:
        consstlb: the lower bound box constraints on the state; a vector of size Nstates
        consstub: the upper bound box constraints on the state; a vector of size Nstates
        consinplb: the lower bound box constraints on the input; a vector of size Ninputs
        consinpub: the upper bound box constraints on the input; a vector of size Ninputs

        """

        self.consstlb = np.asfortranarray( lb )
        self.consstub = np.asfortranarray( ub )

        if( self.consstlb.shape != (self.Nstates, ) or
            self.consstub.shape != (self.Nstates, ) ):
            raise ValueError( "Arguments must have size (" + str(self.Nstates) + ",)." )


    def consBoxInput( self, lb, ub ):
        """
        defines the box constraints on the states and inputs

        arguments:
        consstlb: the lower bound box constraints on the state; a vector of size Nstates
        consstub: the upper bound box constraints on the state; a vector of size Nstates
        consinplb: the lower bound box constraints on the input; a vector of size Ninputs
        consinpub: the upper bound box constraints on the input; a vector of size Ninputs

        """

        self.consinlb = np.asfortranarray( lb )
        self.consinub = np.asfortranarray( ub )

        if( self.consinlb.shape != (self.Ninputs, ) or
            self.consinub.shape != (self.Ninputs, ) ):
            raise ValueError( "Arguments must have size (" + str(self.Ninputs) + ",)." )


    ## TODO: use mixed constraints
    def discForwardEuler( self, Nsamples ):
        """
        this constructor will transform an optimal control problem into a non-linear programming problem

        Arguments:
        ocp: an instance from the OCP (optimal control problem) class
        Nsamples: the number of samples in the approximated version of the OCP
        """

        ## index helpers
        stidx = np.arange( 0, self.Nstates * ( Nsamples + 1 ) ).reshape(
            ( self.Nstates, Nsamples + 1 ),
            order='F' )
        uidx = stidx.size + np.arange( 0, self.Ninputs * Nsamples ).reshape(
            ( self.Ninputs, Nsamples ),
            order='F' )
        dconsidx = stidx
        iconsidx = dconsidx.size + np.arange( 0, self.Ncons * Nsamples ).reshape(
            ( self.Ncons, Nsamples ),
            order='F' )

        deltaT = ( self.tf - self.t0 ) / ( Nsamples + 1 )

        feuler = nlp.SparseProblem( N = stidx.size + uidx.size,
                                    Ncons = dconsidx.size + iconsidx.size,
                                    Nconslin = 0 )


        def encode( st, u ):
            """
            this function creates one big vector of all of the states and inputs

            Arguments:
            st: the matrix of states at all times tk for k=0...Nsamples
            u: the matrix of inputs at all times tk for k=0...Nsamples-1
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
            st = s[ stidx ]
            u = s[ uidx ]

            return ( st, u )


        def solnDecode( s ):
            return decode( s ) + ( np.linspace( self.t0, self.tf, Nsamples + 1 ), )


        def objf( out, s ):
            """
            this function uses the instant cost and the final cost from the ocp problem and
            creates the objective function for the nlp problem

            Arguments:
            s: the optimization vector

            """

            ( st, u ) = decode( s )

            tmp = 0
            for k in range( Nsamples ):
                tmp += self.icost( st[:,k], u[:,k], grad=False )

            out[0] = tmp * deltaT + self.fcost( st[:,Nsamples], grad=False )


        def objg( out, s ):
            """
            this function returns the gradient of the objective function with respect to a vector
            s, which is a concatenated vector of all of the states and inputs

            Arguments:
            s: the optimization problem vector

            """

            ( st, u ) = decode( s )

            for k in range( Nsamples ):
                ( dx, du ) = self.icost( st[:,k], u[:,k] )[1:3]
                out[ 0, stidx[:,k] ] = dx * deltaT
                out[ 0, uidx[:,k] ] = du * deltaT

            out[ 0, stidx[:,Nsamples] ] = self.fcost( st[:,Nsamples] )[1]


        def objgpattern():
            if( self.icostdupattern is None or
                self.icostdxpattern is None or
                self.fcostdxpattern is None ):
                return None

            out = np.zeros( ( feuler.N, ), dtype=np.int )

            for k in range( Nsamples ):
                out[ stidx[:,k] ] = self.icostdxpattern
                out[ uidx[:,k] ] = self.icostdupattern

            out[ stidx[:,Nsamples] ] = self.fcostdxpattern

            return out


        def consf( out, s ):
            """
            this function returns the Forward Euler constraints and the inequality constraints
            from the ocp for the nlp problem

            Arguments:
            s: the optimization vector

            """

            ( st, u ) = decode( s )

            ## Forward Euler collocation equality constraints
            out[ dconsidx[:,0] ] = st[:,0] - self.init
            for k in range( Nsamples ):
                out[ dconsidx[:,k+1] ] = ( st[:,k+1] - st[:,k] -
                                           deltaT * self.vfield( st[:,k], u[:,k], grad=False ) )

            ## inequality constraints
            for k in range( 0, Nsamples ):
                out[ iconsidx[:,k] ] = self.cons( st[:,k+1], grad=False )


        def consg( out, s ):
            """
            this function returns the gradient of the constraint function

            Arguments:
            s: the optimization vector

            """

            ( st, u ) =  decode( s )

            out[ np.ix_( dconsidx[:,0], stidx[:,0] ) ] = np.identity( self.Nstates )
            for k in range( Nsamples ):
                ( dyndx, dyndu ) = self.vfield( st[:,k], u[:,k] )[1:3]
                out[ np.ix_( dconsidx[:,k+1], stidx[:,k] ) ] = (
                    - np.identity( self.Nstates ) - deltaT * dyndx )
                out[ np.ix_( dconsidx[:,k+1], stidx[:,k+1] ) ] = np.identity( self.Nstates )
                out[ np.ix_( dconsidx[:,k+1], uidx[:,k] ) ] = - deltaT * dyndu

                out[ np.ix_( iconsidx[:,k], stidx[:,k+1] ) ] = self.cons( st[:,k+1] )[1]


        def consgpattern():
            if( self.vfielddxpattern is None or
                self.vfielddupattern is None or
                self.consdxpattern is None ):
                return None

            out = np.zeros( ( feuler.Ncons, feuler.N ), dtype=np.int )

            out[ np.ix_( dconsidx[:,0], stidx[:,0] ) ] = np.identity( self.Nstates )
            for k in range( Nsamples ):
                out[ np.ix_( dconsidx[:,k+1], stidx[:,k] ) ] = (
                    np.logical_or( np.identity( self.Nstates ), self.vfielddxpattern ) )
                out[ np.ix_( dconsidx[:,k+1], stidx[:,k+1] ) ] = np.identity( self.Nstates )
                out[ np.ix_( dconsidx[:,k+1], uidx[:,k] ) ] = self.vfielddupattern

                out[ np.ix_( iconsidx[:,k], stidx[:,k+1] ) ] = self.consdxpattern

            return out


        ## setup feuler now that all the functions are defined
        feuler.initPoint( encode( self.init, np.zeros( ( self.Ninputs, ) ) ) )
        feuler.consBox( encode( self.consstlb, self.consinlb ),
                        encode( self.consstub, self.consinub ) )
        feuler.objFctn( objf )
        feuler.objGrad( objg, pattern=objgpattern() )
        feuler.consFctn( consf,
                         np.concatenate( ( np.zeros( ( dconsidx.size, ) ),
                                           np.tile( self.conslb, ( Nsamples, ) ) ) ),
                         np.concatenate( ( np.zeros( ( dconsidx.size, ) ),
                                           np.tile( self.consub, ( Nsamples, ) ) ) ) )
        feuler.consGrad( consg, pattern=consgpattern() )

        return ( feuler, solnDecode )
