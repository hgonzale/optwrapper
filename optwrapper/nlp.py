import types
import numpy as np

#how do I ultimately assign my function outputs to what's in the second init function?

class Problem:
    """
    General nonlinear programming optimization problem.
    Requires a nonlinear objective function and its gradient.
    Accepts box, linear, and nonlinear constraints.
    """

    def __init__( self, ocp, Nsamples):
        """
        this constructor will transform an optimal control problem into a non-linear programming problem 

        Arguments:
        ocp: an instance from the OCP (optimal control problem) class 
        Nsamples: the number of samples in the approximated version of the OCP 

        NOTES FROM 11/25 MEETING:
        -N in the nlp problem is the size of s 
            -you need to set this 
            -this will be super helpful to use throughout your code 
        -s should be the input to all of your functions, then use decode to get the st/inp matrices 

        """

        def encode(st, inp):
            """
            this function creates one big vector of all of the states and inputs 

            Arguments:
            st: the matrix of states at all times tk for k=0...Nsamples
            inp: the matrix of inputs at all times tk for k=0...Nsamples-1
            """
            s = np.zeros( N ) 

            for k in range( Nsamples ):
                s[ k*Nst : k*Nst+Nst ] = st[:,k]
                s[ Nst*(Nsamples+1)+k*Ninp : Nst*(Nsamples+1)+k*Ninp+Ninp] = inp[:,k]


            s[ Nst * Nsamples : Nst * (Nsamples + 1) ] = st[:,Nsamples]

            return s 

        def decode(s):
            #note that the inputs Nst, Ninp, and Nsamples will not be necessary when you put this function in nlp.py
            st = np.zeros( ( Nst, Nsamples + 1 ) )
            inp = np.zeros( ( Ninp, Nsamples ) )

            for k in range( Nsamples ):
                st[:,k] = s[ k*Nst : k*Nst+Nst]
                inp[:,k] = s[ Nst*(Nsamples+1)+k*Ninp : Nst*(Nsamples+1)+k*Ninp+Ninp ]

            st[:,Nsamples] = s[ Nst * Nsamples : Nst * (Nsamples + 1) ]

            return (st, inp)


        def setBounds( ):
            """
            this function sets the lower and upper bounds on the optimization vector  

            Arguments:
            s: the optimization vector

            """

            #initialize the lb/ub vectors with all zeros 
            lb = np.zeros( N )
            ub = np.zeros( N )

            #set all lower bounds on the optimization vector to match the box constraints
            #for states and inputs given in OCP
            for k in range( Nsamples ):
                lb[ k*Nst : k*Nst+Nst ] = ocp.consstlb 
                lb[ Nst*(Nsamples+1)+k*Ninp : Nst*(Nsamples+1)+k*Ninp+Ninp] = ocp.consinplb 
                ub[ k*Nst : k*Nst+Nst ] = ocp.consstub 
                ub[ Nst*(Nsamples+1)+k*Ninp : Nst*(Nsamples+1)+k*Ninp+Ninp] = ocp.consinpub 

            lb[ Nst * Nsamples : Nst * (Nsamples + 1) ] = ocp.consstlb
            ub[ Nst * Nsamples : Nst * (Nsamples + 1) ] = ocp.consstub

            self.ub = ub
            self.lb = lb

        def setBoundsCons( ):
            """
            this function sets the lower and upper bounds on the constraints 

            """

            conslb = np.zeros ( self.Ncons )
            consub = np.zeros ( self.Ncons )

            conslb[Nst*(Nsamples + 1): Nst*(Nsamples + 1) + Nsamples * ocp.Nineqcons ] = -np.inf

            self.conslb = conslb
            self.consub = consub 

            

        def objectiveFctn( s ):
            """
            this function uses the instant cost and the final cost from the ocp problem and 
            creates the objective function for the nlp problem  

            Arguments:
            s: the optimization vector

            """

            (st, inp) = decode(s) 

            objfruncost = 0 
            
            for k in range( Nsamples ):
                objfruncost = objfruncost + ocp.instcost(st[:,k],inp[:,k])*deltaT

            objffincost = ocp.fincost(st[:,Nsamples])

            return objfruncost + objffincost 

        def objectiveGrad( s ):
            """
            this function returns the gradient of the objective function with respect to a vector
            s, which is a concatenated vector of all of the states and inputs 

            Arguments:
            s: the optimization problem vector 

            """

            (st, inp) = decode(s)

            objg = np.zeros( N )

            for k in range( Nsamples ):
                objg[ k*Nst : k*Nst+Nst ] = ocp.instcostgradst(st[:,k]) * deltaT
                objg[ Nst*(Nsamples+1)+k*Ninp : Nst*(Nsamples+1)+k*Ninp+Ninp] = ocp.instcostgradin(inp[:,k]) * deltaT

            objg[ Nst * Nsamples : Nst * (Nsamples + 1) ] = ocp.fincostgradst(st[:,Nsamples])

            return objg

        def constraintFctn( s ):
            """
            this function returns the Forward Euler constraints and the inequality constraints
            from the ocp for the nlp problem 

            Arguments:
            s: the optimization vector 

            """

            (st, inp) = decode(s)

            consf = np.zeros( self.Ncons )

            #this for loop puts all of the forward Euler constraints into consf 
            for k in range(0, Nsamples):
                consf[k*Nst:k*Nst+Nst] = st[:,k+1] - st[:,k] - deltaT*ocp.dynamics(st[:,k],inp[:,k])

            #this line puts the initial condition into the constraint function 
            consf[Nst*Nsamples:Nst*(Nsamples + 1)] = ocp.init 

            #this loop puts all of the inequality constraints into consf 
            for k in range(1, Nsamples+1):
                consf[Nst*(Nsamples + 1) + (k-1)*ocp.Nineqcons : Nst*(Nsamples + 1) + (k-1)*ocp.Nineqcons + ocp.Nineqcons] = ocp.cons(st[:,k])

            return consf

        def constraintGrad( s ):
            """
            this function returns the gradient of the constraint function

            Arguments:
            s: the optimization vector 

            """

            (st,inp) =  decode(s)

            #rows from forward Euler constraints: rows = Nst*Nsamples
            #rows from ocp inequality constraints: rows = Nineqconst*Nsamples 
            rows = Nst*Nsamples + Nst + ocp.Nineqcons*Nsamples
            columns = Nsamples*(Nst+Ninp)+ Nst 
            consg = np.zeros( (rows, columns) )
            #counter = 0 

            for k in range(0,Nsamples):
                consg[ k*Nst: k*Nst + Nst, k*Nst: k*Nst+Nst] = -np.identity(Nst) - deltaT*ocp.dynamicsgradst(st[:,k])
                consg[ k*Nst : k*Nst + Nst, Nst*(k+1) : Nst*(k+1) + Nst ] = np.identity(Nst)
                consg[ k*Nst : k*Nst + Nst, Nst*(Nsamples+1) + k*Ninp : Nst*(Nsamples+1) + k*Ninp + Ninp ] = -deltaT*ocp.dynamicsgradin(inp[:,k]) 

            consg[Nst*Nsamples : Nst*Nsamples + Nst, 0: Nst ] = np.identity(Nst)

            for k in range(0,Nsamples):
                consg[Nst*Nsamples + Nst + k*ocp.Nineqcons : Nst*Nsamples + Nst + k*ocp.Nineqcons + ocp.Nineqcons, (k+1)*Nst: (k+1)*Nst + Nst] = ocp.consgradst(st[:,k+1])

            return consg

        Nst = ocp.Nst 
        Ninp = ocp.Ninp 
        self.N = Nst*(Nsamples+1) + Ninp*Nsamples 
        N = self.N
        deltaT = (ocp.tf - ocp.t0) / Nsamples
        self.Nconslin = 0 #we do not have any linear constraints in this problem 
        self.Ncons = Nst*(Nsamples + 1) + Nsamples*ocp.Nineqcons
        

        Ncons = self.Ncons
        Nconslin = self.Nconslin
        mixedCons = False


        self.objf = objectiveFctn
        self.objg = objectiveGrad
        self.consf = constraintFctn
        self.consg = constraintGrad
        setBounds() ## this fctn sets self.lb and self.ub 
        setBoundsCons() ## this fctn sets self.conslb and self.consub



#    def __init__( self, N, Ncons=0, Nconslin=0, mixedCons=False ):
        """
        Arguments:
        N         number of optimization variables (required).
        Nconslin  number of linear constraints (default: 0).
        Ncons     number of constraints (default: 0).

        prob = optProblem( N=2, Ncons=2, Nconslin=3 )
        """

        try:
            self.N = int( N )
        except:
            raise ValueError( "N must be an integer" )

        if( self.N <= 0 ):
            raise ValueError( "N must be strictly positive" )

        try:
            self.Nconslin = int( Nconslin )
        except:
            raise ValueError( "Nconslin was not provided or was not an integer" )

        if( self.Nconslin < 0 ):
            raise ValueError( "Nconslin must be positive" )

        try:
            self.Ncons = int( Ncons )
        except:
            raise ValueError( "Ncons was not provided or was not an integer" )

        if( self.Ncons < 0 ):
            raise ValueError( "Ncons must be positive" )

        if( mixedCons and Nconslin != Ncons ):
            raise ValueError( "If constrained are mixed type then Nconslin must be equal to Ncons" )

        self.init = np.zeros( self.N )
        self.lb = None  
        self.ub = None  
        self.objf = None  
        self.objg = None 
        self.consf = None 
        self.consg = None
        self.conslb = None  
        self.consub = None  
        self.conslinA = None  
        self.conslinlb = None  
        self.conslinub = None 
        self.soln = None
        self.mixedCons = mixedCons


    def initPoint( self, init ):
        """
        Sets initial value for optimization variable.

        Arguments:
        init  initial condition, must be a one-dimensional array of size N
              (default: vector of zeros).

        prob.initCond( [ 1.0, 1.0 ] )
        """
        self.init = np.asfortranarray( init )

        if( self.init.shape != ( self.N, ) ):
            raise ValueError( "Argument must have size (" + str(self.N) + ",)." )


    def consBox( self, lb, ub ):
        """
        Defines box constraints.

        Arguments:
        lb  lower bounds, one-dimensional array of size N.
        ub  upper bounds, one-dimensional array of size N.

        prob.consBox( [-1,-2], [1,2] )
        """
        self.lb = np.asfortranarray( lb )
        self.ub = np.asfortranarray( ub )

        if( self.lb.shape != ( self.N, ) or
            self.ub.shape != ( self.N, ) ):
            raise ValueError( "Bound must have size (" + str(self.N) + ",)." )


    def consLinear( self, A, lb=None, ub=None ):
        """
        Defines linear constraints.

        Arguments:
        A   linear constraint matrix, two-dimensional array of size (Nconslin,N).
        lb  lower bounds, one-dimensional array of size Nconslin.
        ub  upper bounds, one-dimensional array of size Nconslin.

        prob.consLinear( [[1,-1],[1,1]], [-1,-2], [1,2] )
        """
        self.conslinA = np.asfortranarray( A )

        if( self.conslinA.shape != ( self.Nconslin, self.N ) ):
            raise ValueError( "Argument 'A' must have size (" + str(self.Nconslin)
                              + "," + str(self.N) + ")." )

        if( not self.mixedCons ):
            if( lb == None ):
                lb = -np.inf * np.ones( self.Nconslin )

            if( ub == None ):
                ub = np.zeros( self.Nconslin )

            self.conslinlb = np.asfortranarray( lb )
            self.conslinub = np.asfortranarray( ub )

            if( self.conslinlb.shape != ( self.Nconslin, ) or
                self.conslinub.shape != ( self.Nconslin, ) ):
                raise ValueError( "Bounds must have size (" + str(self.Nconslin) + ",)." )


    def objFctn( self, objf ):
        """
        Set objective function.

        Arguments:
        objf  objective function, must return a scalar.

        def objf(x):
            return x[1]
        prob.objFctn( objf )
        """
        if( type(objf) != types.FunctionType ):
            raise ValueError( "Argument must be a function" )

        self.objf = objf


    def objGrad( self, objg ):
        """
        Set objective gradient.

        Arguments:
        objg  gradient function, must return a one-dimensional array of size N.

        def objg(x):
            return np.array( [2,-1] )
        prob.objGrad( objg )
        """
        if( type(objg) != types.FunctionType ):
            raise ValueError( "Argument must be a function" )

        self.objg = objg


    def consFctn( self, consf, lb=None, ub=None ):
        """
        Set nonlinear constraints function.

        Arguments:
        consf  constraint function, must return a one-dimensional array of
               size Ncons.
        lb     lower bounds, one-dimensional array of size Ncons (default: vector
               of -inf).
        ub     upper bounds, one-dimensional array of size Ncons (default: vector
               of zeros).

        def consf(x):
            return np.array( [ x[0] - x[1],
                               x[0] + x[1] ] )
        prob.consFctn( consf )
        """
        if( type(consf) != types.FunctionType ):
            raise ValueError( "Argument must be a function" )

        if( lb == None ):
            lb = -np.inf * np.ones( self.Ncons )

        if( ub == None ):
            ub = np.zeros( self.Ncons )

        self.consf = consf
        self.conslb = np.asfortranarray( lb )
        self.consub = np.asfortranarray( ub )

        if( self.conslb.shape != ( self.Ncons, ) or
            self.consub.shape != ( self.Ncons, ) ):
            raise ValueError( "Bound must have size (" + str(self.Ncons) + ",)." )


    def consGrad( self, consg ):
        """
        Set nonlinear constraints gradient.

        Arguments:
        consg  constraint gradient, must return a two-dimensional array of
               size (Ncons,N), where entry [i,j] is the derivative of i-th
               constraint w.r.t. the j-th variables.

        def consg(x):
            return np.array( [ [ 2*x[0], 8*x[1] ],
                               [ 2*(x[0]-2), 2*x[1] ] ] )
        prob.consGrad( consg )
        """
        if( type(consg) != types.FunctionType ):
            raise ValueError( "Argument must be a function" )

        self.consg = consg


    def checkGrad( self, h=1e-5, etol=1e-4, point=None, debug=False ):
        """
        Checks if user-defined gradients are correct using finite
        differences.

        Arguments:
        h      optimization variable variation step size (default: 1e-5).
        etol   error tolerance (default: 1e-4).
        point  evaluation point one-dimensional array of size N (default:
               initial condition).
        debug  boolean to enable extra debug information (default: False).

        isCorrect = prob.checkGrad( h=1e-6, etol=1e-5, point, debug=False )
        """
        if( self.objf == None or
            self.objg == None ):
            raise StandardError( "Objective must be set before gradients are checked." )
        if( self.Ncons > 0 and
            ( self.consf == None or self.consg == None ) ):
            raise StandardError( "Constraints must be set before gradients are checked." )

        if( point == None ):
            point = self.init
        else:
            point = np.asfortranarray( point )
            if( point.shape != ( self.N, ) ):
                raise ValueError( "Argument 'point' must have size (" + str(self.N) + ",)." )

        usrgrad = np.zeros( [ self.Ncons + 1, self.N ] )
        numgrad = np.zeros( [ self.Ncons + 1, self.N ] )

        fph = np.zeros( self.Ncons + 1 )
        fmh = np.zeros( self.Ncons + 1 )
        for k in range( 0, self.N ):
            hvec = np.zeros( self.N )
            hvec[k] = h

            fph[0] = self.objf( point + hvec )
            fmh[0] = self.objf( point - hvec )
            fph[1:] = self.consf( point + hvec )
            fmh[1:] = self.consf( point - hvec )

            if( np.any( np.isnan( fph ) ) or np.any( np.isnan( fmh ) ) or
                np.any( np.isinf( fph ) ) or np.any( np.isinf( fmh ) ) ):
                raise ValueError( "Function returned NaN or inf at iteration " + str(k) )

            delta = ( fph - fmh ) / 2.0 / h
            numgrad[:,k] = delta

        usrgrad[0,:] = self.objg( point )
        usrgrad[1:,:] = self.consg( point )
        if( np.any( np.isnan( usrgrad ) ) or
            np.any( np.isinf( usrgrad ) ) ):
            raise ValueError( "Gradient returned NaN or inf." )

        errgrad = abs( usrgrad - numgrad )
        if( errgrad.max() < etol ):
            return( True, errgrad.max(), errgrad )
        else:
            if( debug ):
                idx = np.unravel_index( np.argmax(err), err.shape )
                if( idx[0] == 0 ):
                    print( "Objective gradient incorrect in element=("
                           + str(idx[1]) + ")" )
                else:
                    print( "Constraint gradient incorrect in element=( "
                           + str(idx[0]-1) + "," + str(idx[1]) + ")" )
            return( False, errgrad.max(), errgrad )


    def check( self, debug=False ):
        """
        General checks required before solver is executed.

        Arguments:
        debug  boolean to enable extra debug information (default: False).

        isCorrect = prob.check()
        """

        if( self.lb == None or
            self.ub == None or
            np.any( self.lb > self.ub ) ):
            if( debug ):
                print( "Box constraints not set or lower bound larger than upper bound." )
            return False

        if( np.any( self.lb > self.init ) or
            np.any( self.init > self.ub ) ):
            if( debug ):
                print( "Initial condition not set or violates box constraints." )
            return False

        if( self.objf == None or
            np.any( np.isnan( self.objf( self.init ) ) ) or
            np.any( np.isinf( self.objf( self.init ) ) ) ):
            if( debug ):
                print( "Objective function not set or return NaN/inf for initial condition." )
            return False

        if( self.objf( self.init ).shape != (1,) ):
            if( debug ):
                print( "Objective function must return a scalar array." )
            return False

        if( self.objg == None or
            np.any( np.isnan( self.objg( self.init ) ) ) or
            np.any( np.isinf( self.objg( self.init ) ) ) ):
            if( debug ):
                print( "Objective gradient not set or return NaN/inf for initial condition." )
            return False

        if( self.objg( self.init ).shape != ( self.N, ) ):
            if( debug ):
                print( "Objective gradient must return array of size (" + str(self.N) + ",)." )
            return False

        if( Nconslin > 0 ):
            if( self.conslinlb == None or
                self.conslinub == None or
                np.any( self.conslinlb > self.conslinub ) ):
                if( debug ):
                    print( "Linear constraint bounds not set or lower bound larger than upper bound." )
                return False

        if( Ncons > 0 ):
            if( self.conslb == None or
                self.consub == None or
                np.any( self.conslb > self.consub ) ):
                if( debug ):
                    print( "Constraint bounds not set or lower bound larger than upper bound." )
                return False

            if( self.consf == None or
                np.any( np.isnan( self.consf( self.init ) ) ) or
                np.any( np.isinf( self.consf( self.init ) ) ) ):
                if( debug ):
                    print( "Constraint function not set or return NaN/inf for initial condition." )
                return False

            if( self.consf( self.init ).shape != ( self.Ncons, ) ):
                if( debug ):
                    print( "Constraint function must return array of size (" + str(self.Ncons) + ",)." )
                return False

            if( self.consg == None or
                np.any( np.isnan( self.consg( self.init ) ) ) or
                np.any( np.isinf( self.consg( self.init ) ) ) ):
                if( debug ):
                    print( "Constraint gradient not set or return NaN/inf for initial condition." )
                return False

            if( self.consg( self.init ).shape != ( self.Ncons, self.N ) ):
                if( debug ):
                    print( "Constraint gradient must return array of size ("
                           + str(self.Ncons) + "," + str(self.N) + ")." )
                return False

        return True



class SparseProblem( Problem ):
    """
    General nonlinear programming optimization problem.
    Requires a nonlinear objective function and its gradient.
    Accepts box, linear, and nonlinear constraints.
    """

    def __init__( self, N, Ncons=0, Nconslin=0, mixedCons=False ):
        Problem.__init__( self, N, Ncons, Nconslin, mixedCons )

        self.objgpattern = None
        self.consgpattern = None


    def objGrad( self, objg, pattern=None ):
        """
        Set objective gradient.

        Arguments:
        objg  gradient function, must return a one-dimensional array of size N.

        def objg(x):
            return np.array( [2,-1] )
        prob.objGrad( objg )
        """
        Problem.objGrad( self, objg )

        if( pattern != None ):
            self.objgpattern = np.asfortranarray( pattern, dtype=np.int )

            if( self.objgpattern.shape != ( self.N, ) ):
                raise ValueError( "Argument 'pattern' must have size (" + str(self.N) + ",)." )


    def consGrad( self, consg, pattern=None ):
        """
        Set nonlinear constraints gradient.

        Arguments:
        consg  constraint gradient, must return a two-dimensional array of
               size (Ncons,N), where entry [i,j] is the derivative of i-th
               constraint w.r.t. the j-th variables.

        def consg(x):
            return np.array( [ [ 2*x[0], 8*x[1] ],
                               [ 2*(x[0]-2), 2*x[1] ] ] )
        prob.consGrad( consg )
        """
        Problem.consGrad( self, consg )

        if( pattern != None ):
            self.consgpattern = np.asfortranarray( pattern, dtype=np.int )

            if( self.consgpattern.shape != ( self.Ncons, self.N ) ):
                raise ValueError( "Argument 'pattern' must have size (" + str(self.Ncons)
                                + "," + str(self.N) + ")." )


    def checkPatterns( self ):
        ## TODO
        pass
