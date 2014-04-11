import types
import numpy as np
# import nlopt
# import ipopt
from optw_snopt import SnoptSolver
from optw_npsol import NpsolSolver

class optProblem:
    """
    General (finite-dimensional) optimization problem
    """

    def __init__( self, N, Ncons=0, isMax=False ):
        """
        Arguments:
        N      number of optimization variables (required). N > 0.
        Ncons  number of constraints. Ncons >= 0.
        isMax  boolean indicating whether the objective should be
               maximized.

        prob = optProblem( N=2, Ncons=2, isMax=False )
        """

        try:
            self.N = int( N )
            if( self.N <= 0 ):
                print( "Usage: " )
                print( self.__init__.__doc__ )
                raise ValueError( "Argument 'N' must be strictly positive" )
        except:
            print( "Usage: " )
            print( self.__init__.__doc__ )
            raise ValueError( "Argument 'N' must be an integer" )

        try:
            self.Ncons = int( Ncons )
            if( self.Ncons < 0 ):
                print( "Usage: " )
                print( self.__init__.__doc__ )
                raise ValueError( "Ncons must be positive" )
        except:
            print( "Usage: " )
            print( self.__init__.__doc__ )
            raise ValueError( "Ncons was not provided or was not an integer" )

        try:
            self.isMax = bool( isMax )
        except:
            print( "Usage: " )
            print( self.__init__.__doc__ )
            raise ValueError( "max must be a boolean" )

        self.stopOpts = { "stopval":None,
                          "maxeval":1000,
                          "maxtime":1.0,
                          "xtol":1e-6,
                          "ftol":1e-6 }
        self.printOpts = { "print_level":0,
                           "summary_level":0,
                           "options_file":None,
                           "print_file":None }
        self.solveOpts = { "warm_start":False,
                           "constraint_violation":1e-8 }
        self.init = None
        self.lb = None
        self.ub = None
        self.objf = None
        self.objg = None
        self.consf = None
        self.consg = None
        self.conslb = None
        self.consub = None


    def initCond( self, init ):
        """
        Sets initial condition for optimization variable.

        Arguments:
        init  initial condition, must be a one-dimensional array of size N.

        prob.initCond( [ 1.0, 1.0 ] )
        """
        if( not len(init) == self.N ):
            print( "Usage: " )
            print( self.set_start_x.__doc__ )
            raise ValueError( "Argument must be have size N=" + str(self.N) )
        self.init = np.asarray( init )


    def consBox( self, lb, ub ):
        """
        Defines bounds for optimization variable.

        Arguments:
        lb  lower bounds, one-dimensional array of size N.
        ub  upper bounds, one-dimensional array of size N.

        prob.consBox( [-1,-2], [1,2] )
        """
        if( len(lb) != self.N or
            len(up) != self.N ):
            print( "Usage: " )
            print( self.consBox.__doc__ )
            raise ValueError( "Argument sizes must be equal to N=" + str(self.N) )

        self.lb = np.asarray( lb )
        self.ub = np.asarray( ub )


    # def constraint_bounds( self, low, high ):
    #     """
    #     Defines upper and lower bounds for each constraint

    #     prob.constraint_bounds( [-1,-2], [1,2] )
    #     """
    #     if( not len(low) == self.nconstraint or
    #         not len(high) == self.nconstraint ):
    #         print( "Usage: " )
    #         print( self.constraint_bounds.__doc__ )
    #         raise ValueError( "constraint_bounds size does not match problem size(" + str(self.nconstraint) + ")" )
    #     self.Flow = np.asarray( low )
    #     self.Fupp = np.asarray( high )


    def objFctn( self, objf ):
        """
        Set objective function.

        Arguments:
        objf  objective function, must return a scalar.

        def objf(x):
            return x[1]
        prob.objFctn( objf )
        """
        if( not type(objf) == types.FunctionType ):
            print( "Usage: " )
            print( self.objFctn.__doc__ )
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
        if( not type(objg) == types.FunctionType ):
            print( "Usage: " )
            print( self.objGrad.__doc__ )
            raise ValueError( "Argument must be a function" )
        self.objgrad = objg


    def consFctn( self, consf, lb=None, ub=None ):
        """
        Set nonlinear constraints function.

        Arguments:
        consf  constraint function, must return a one-dimensional array of
               size Ncons.
        lb  lower bounds, one-dimensional array of size Ncons (default: vector
            of -inf).
        ub  upper bounds, one-dimensional array of size Ncons (default: vector
            of zeros).

        def consf(x):
            return np.array( [ x[0] - x[1],
                               x[0] + x[1] ] )
        prob.consFctn( consf )
        """
        if( not type(consf) == types.FunctionType ):
            print( "Usage: " )
            print( self.consFctn.__doc__ )
            raise ValueError( "Argument must be a function" )

        if( lb == None ):
            lb = -inf * np.ones( self.Ncons )

        if( ub == None ):
            ub = np.zeros( self.Ncons )

        if( len(lb) != self.Ncons or
            len(up) != self.Ncons ):
            print( "Usage: " )
            print( self.consFctn.__doc__ )
            raise ValueError( "Argument sizes must be equal to Ncons=" + str(self.Ncons) )

        self.consf = consf
        self.conslb = np.asarray( lb )
        self.consub = np.asarray( ub )


    def consGrad( self, consg ):
        """
        Set nonlinear constraints gradient.

        Arguments:
        consg  constraint gradient, must return a two-dimensional array of
               size [Ncons,N], where entry [i,j] is the derivative of i-th
               constraint w.r.t. the j-th variables.

        def consg(x):
            return np.array( [ [ 2*x[0], 8*x[1] ],
                               [ 2*(x[0]-2), 2*x[1] ] ] )
        prob.consGrad( consg )
        """
        if( not type(consg) == types.FunctionType ):
            print( "Usage: " )
            print( self.consGrad.__doc__ )
            raise ValueError( "Argument must be a function" )
        # self.jacobian_style = "dense"
        self.consg = consg


    # def set_sparse_constraint_gradient( self, i_indices, j_indices, gg ):
    #     """
    #     i_indices, j_indices, are arrays of length nj
    #     where nj is the number of non-zero elements in
    #     the jacobian
    #     gg is a function that returns an array A which contains
    #     the non-zero jacobian elements with corresponding indices
    #     specified by i_indices and j_indices, that is,
    #     A[p] is the partial derivative of the i_indices[p] constraint
    #     with respect to the j_indices[p] variable

    #     to define the jacobian as a dense matrix,
    #     consider using set_constraint_gradient(gg)

    #     i_indices = [ 0, 0, 1, 1 ]
    #     j_indices = [ 0, 1, 0, 1 ]
    #     def gg(x):
    #         return np.array( [ 2*x[0], 8*x[1],
    #                            2*(x[0]-2), 2*x[1] ] )
    #     prob.set_sparse_constraint_gradient( i_indices, j_indices, gg )
    #     """
    #     if( not len(i_indices) == len(j_indices) ):
    #         print( "Usage: " )
    #         print( self.set_sparse_constraint_gradient.__doc__ )
    #         raise ValueError( "mismatched length of i_indices and j_indices" )
    #     if( not type(gg) == types.FunctionType ):
    #         print( "Usage: " )
    #         print( self.set_sparse_constraint_gradient.__doc__ )
    #         raise ValueError( "input gg must be a function" )
    #     self.iG = i_indices
    #     self.jG = j_indices
    #     self.jacobian_style = "sparse"
    #     self.congrad = gg


    def checkStopOpts( self ):
        """
        Check if dictionary self.stopOpts is valid.
        """
        if( not type(val) == types.DictType or
            not val.has_key("xtol") or
            not val.has_key("ftol") or
            not val.has_key("stopval") or
            not val.has_key("maxeval") or
            not val.has_key("maxtime") ):
            return False
        else:
            return True


    def checkPrintOpts( self ):
        """
        Check if dictionary self.printOpts is valid.
        """
        # print_level: verbosity of file output (0-11 for SNOPT, 0-30 for NPSOL, 0-12 for IPOPT)
        # summary_level: verbosity of output to screen (0-1 for SNOPT, 0-12 for IPOPT)
        if( not type(val) == types.DictType or
            not val.has_key("print_level") or
            not val.has_key("summary_level") or
            not val.has_key("options_file") or
            not val.has_key("print_file") ):
            return False
        else:
            return True


    def checkSolveOpts( self ):
        """
        Check if dictionary self.solveOpts is valid.
        """
        if( not type(val) == types.DictType or
            not val.has_key("constraint_violation") or
            not val.has_key("warm_start") ):
            return False
        else:
            return True


    def checkGrad( self, h=1e-5, etol=1e-4, point=None, debug=False ):
        """
        Checks if user-defined gradients are correct using finite
        differences.

        Arguments:
        h      optimization variable variation step size (default: 1e-5).
        etol   error tolerance (default: 1e-4).
        point  evaluation point one-dimensional array of size N (default:
               self.init, or that is not available, vector of zeros).
        debug  boolean to enable extra debug information (default: False).

        isCorrect = prob.checkGrad( h=1e-6, etol=1e-5, point, debug=False )
        """
        if( self.objf == None or
            self.objg == None ):
            raise StandardError( "Objective function must be set before gradients are checked." )
        if( self.Ncons > 0 and
            ( self.consf == None or self.consg == None ) ):
            raise StandardError( "Constraint function must be set before gradients are checked." )

        if( point == None ):
            if( self.init == None ):
                point = np.zeros( self.N )
            else:
                point = self.init

        usrgrad = np.zeros( [ self.Ncons + 1, self.N ] )
        numgrad = np.zeros( [ self.Ncons + 1, self.N ] )

        fph = np.zeros( self.Ncons + 1 )
        fmh = np.zeros( self.Ncons + 1 )
        for k in range( 0, self.N ):
            hvec = np.zeros( self.N )
            hvec[k] = h
            fph[0] = self.objf( point + hvec )
            fph[1:] = self.consf( point + hvec )
            fmh[0] = self.objf( point - hvec )
            fmh[1:] = self.consf( point - hvec )
            if( np.any( np.isnan( fph ) ) or np.any( np.isnan( fmh ) ) ):
                raise ValueError( "NaN found for index " + str(k) )
            delta = ( fph - fmh ) / 2.0 / h
            numgrad[:,k] = delta

        usrgrad[0,:] = self.objg( point )
        usrgrad[1:,:] = self.congrad( point )
        # if( self.jacobian_style == "sparse" ):
        #     A = self.congrad( point )
        #     for p in range( 0, len(self.iG) ):
        #         usrgrad[ self.iG[p]+1, self.jG[p] ] = A[p]
        # else:

        errgrad = abs( usrgrad - numgrad )
        if( errgrad.max() < tol ):
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


    def checkErrors( self, debug=False ):
        """
        General checks required before solver is executed.

        Arguments:
        debug  boolean to enable extra debug information (default: False).

        isCorrect = prob.checkErrors()
        """

        if( self.lb == None or
            self.ub == None or
            np.any( self.lb > self.ub ) ):
            if( debug ):
                print( "Box constraints not set or lower bound larger than upper bound." )
            return False

        if( self.init == None or
            np.any( self.lb > self.init ) or
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

        if( self.objf( self.init ).shape != () ):
            if( debug ):
                print( "Objective function must return a scalar." )
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

            if( self.objg( self.init ).shape != ( self.Ncons, self.N ) ):
                if( debug ):
                    print( "Objective gradient must return array of size (" + str(self.N) + ","
                           + str(self.Ncons) + ")." )
                return False

        return True


    def solve( self, solver ):
        """
        Solves the problem using the specified solver
        returns a tuple consisting of
        objective_value, x_values, solve_status
        if using a specific algorithm, separate the
        name of the algorithm and the name of the solver
        by a space

        obj,x,st = prob.solve( "NLOPT MMA" )
        """
        if( solver == "SNOPT" ):
            if( self.jacobian_style == "sparse" ):
                lenG = len(self.iG) + self.n
            else:
                lenG = ( self.nconstraint + 1 ) * self.n
            prob = SnoptSolver( n = self.n, neF = ( self.nconstraint + 1 ),
                                lenA = 0, lenG = lenG,
                                summaryLevel = self.print_options["summary_level"],
                                printLevel = self.print_options["print_level"],
                                printFile = self.print_options["print_file"],
                                maximize = int(self.maximize) )
            prob.x_bounds( self.xlow, self.xupp )
            prob.F_bounds( np.append( [-1e20], self.Flow), np.append( [1e20], self.Fupp ) )
            prob.set_x( self.x )
            if( self.jacobian_style == "sparse" ):
                indGx = [1 for j in range( 1, self.n+1 ) ]
                indGy = range( 1, self.n + 1 )
                indGx.extend( [ x+2 for x in self.iG ] )
                indGy.extend( [ y+1 for y in self.jG ] )
            else:
                indGx = [ i for i in range( 1, self.nconstraint + 2 ) for j in range( 1, self.n + 1 ) ]
                indGy = [ j for i in range( 1, self.nconstraint + 2 ) for j in range( 1, self.n + 1 ) ]
            print( indGx )
            print( indGy )
            prob.G_indices( indGx, indGy )

            ## Page 23, Section 3.6, of SNOPT"s manual
            def snoptcallback( status, n, x, needF, neF, F, needG, neG, G ):
                if( needF > 0 ):
                    F[0] = self.objf(x)
                    con = self.con(x)
                    for i in range( 0, self.nconstraint ):
                        F[i+1] = con[i]

                if( needG > 0 ):
                    objgrad = self.objgrad(x)
                    for i in range( 0, self.n ):
                        G[i] = objgrad[i]

                    if( self.jacobian_style == "sparse" ):
                        congrad = self.congrad(x)
                    else:
                        congrad = np.asarray( self.congrad(x) ).reshape(-1)

                    for i in range( 0, len(congrad) ):
                        G[i + self.n] = congrad[i]

                print( "F: " + str(F) + " G: " + str(G) )
                assert( len(F) == neF and len(G) == neG ) # Just being a bit paranoid.

            prob.set_funobj( snoptcallback )
            prob.set_options( int( self.solve_options["warm_start"] ),
                              self.stop_options["maxeval"],
                              self.solve_options["constraint_violation"],
                              self.stop_options["ftol"] )

            answer = prob.solve()

            finalX = prob.get_x()
            status = prob.get_status()
            finalXArray = [ finalX[i] for i in range( 0, len(finalX) ) ]
            return answer, finalXArray, status

        elif( solver == "NPSOL" ):
            prob = NpsolSolver( n = self.n, nclin = 0, ncnln = self.nconstraint,
                                printLevel = self.print_options["print_level"],
                                printFile = self.print_options["print_file"],
                                maximize = int(self.maximize) )
            prob.set_bounds( self.xlow, self.xupp, None, None, self.Flow, self.Fupp )
            prob.set_x( self.x )

            def objcallback( x, f, g ):
                of = self.objf(x)
                og = self.objgrad(x)
                if( self.maximize ):
                    f[0] = -of
                    for i in range( 0, self.n ):
                        g[i] = -og[i]
                else:
                    f[0] = of
                    for i in range( 0, self.n ):
                        g[i] = og[i]

            if( self.jacobian_style == "sparse" ):
                def concallback( x, c, j ):
                    con = self.con(x)
                    for i in range( 0, self.nconstraint ):
                        c[i] = con[i]
                    conm = np.zeros( [ self.nconstraint, self.n ] )
                    A = self.congrad(x)
                    for p in range( 0, len(self.iG) ):
                        conm[ self.iG[p], self.jG[p] ] = A[p]
                    conm = np.asarray( conm.transpose() ).reshape(-1)
                    for i in range( 0, len(conm) ):
                        j[i] = conm[i]
            else:
                def concallback( x, c, j ):
                    con = self.con(x)
                    for i in range( 0, self.nconstraint ):
                        c[i] = con[i]
                    congrad = np.asarray( self.congrad(x).transpose() ).reshape(-1)
                    for i in range( 0, len(congrad) ):
                        j[i] = congrad[i]
            prob.set_user_function( concallback, objcallback )
            prob.set_options( int( self.solve_options["warm_start"] ), self.stop_options["maxeval"],
                              self.solve_options["constraint_violation"], self.stop_options["ftol"] )
            answer = prob.solve()
            finalX = prob.get_x()
            status = prob.get_status()
            finalXArray = [ finalX[i] for i in range( 0, len(finalX) ) ]
            return answer, finalXArray, status

        elif( solver.startswith("NLOPT") ):
            algorithm = solver[6:]
            if( algorithm == "" or algorithm == "MMA" ):
                #this is the default
                opt = nlopt.opt( nlopt.LD_MMA, self.n )
            elif( algorithm == "SLSQP" ):
                opt = nlopt.opt( nlopt.LD_SLSQP, self.n )
            elif( algorithm == "AUGLAG" ):
                opt = nlopt.opt( nlopt.LD_AUGLAG, self.n )
            else:
                ## other algorithms do not support vector constraints
                ## auglag does not support stopval
                raise ValueError( "invalid solver" )

            def objfcallback( x, grad ):
                if( grad.size > 0 ):
                    grad[:] = self.objgrad(x)
                return self.objf(x)

            if( self.jacobian_style == "sparse" ):
                def confcallback( res, x, grad ):
                    if( grad.size > 0 ):
                        conm = np.zeros( [ self.nconstraint, self.n ] )
                        A = self.congrad(x)
                        for p in range( 0, len(self.iG) ):
                            conm[ self.iG[p], self.jG[p] ] = A[p]
                        conm = np.asarray(conm)
                        grad[:] = np.append( -1*conm, conm, axis=0 )
                    conf = np.asarray( self.con(x) )
                    res[:] = np.append( self.Flow - conf, conf - self.Fupp )
            else:
                def confcallback( res, x, grad ):
                    if( grad.size > 0 ):
                        conm = np.asarray( self.congrad(x) )
                        grad[:] = np.append( -1*conm, conm, axis=0 )
                    conf = np.asarray( self.con(x) )
                    res[:] = np.append( self.Flow - conf, conf - self.Fupp )

            if( self.maximize ):
                opt.set_max_objective( objfcallback )
            else:
                opt.set_min_objective( objfcallback )

            opt.add_inequality_mconstraint( confcallback,
                        np.ones( self.nconstraint*2 ) * self.solve_options["constraint_violation"] )
            opt.set_lower_bounds( self.xlow )
            opt.set_upper_bounds( self.xupp )
            if( not self.stop_options["xtol"] == None ):
                opt.set_xtol_rel( self.stop_options["xtol"] )
            if( not self.stop_options["ftol"] == None ):
                opt.set_ftol_rel( self.stop_options["ftol"] )
            if( not self.stop_options["stopval"] == None ):
                opt.set_stopval( self.stop_options["stopval"] )
            if( not self.stop_options["maxeval"] == None ):
                opt.set_maxeval( self.stop_options["maxeval"] )
            if( not self.stop_options["maxtime"] == None ):
                opt.set_maxtime( self.stop_options["maxtime"] )

            finalX = opt.optimize( self.x )
            answer = opt.last_optimum_value()
            status = opt.last_optimize_result()

            if( status == 1 ):
                status = "generic success"
            elif( status == 2 ):
                status = "stopval reached"
            elif( status == 3 ):
                status = "ftol reached"
            elif( status == 4 ):
                status = "xtol reached"
            elif( status == 5 ):
                status = "maximum number of function evaluations exceeded"
            elif( status == 6 ):
                status = "timed out"
            else:
                #errors will be reported as thrown exceptions
                status = "invalid return code"
            return answer, finalX, status

        elif( solver.startswith("IPOPT") ):
            algorithm=solver[6:]
            if( not ( algorithm == "" or
                      algorithm == "ma27" or
                      algorithm == "ma57" or
                      algorithm == "ma77" or
                      algorithm == "ma86" or
                      algorithm == "ma97" or
                      algorithm == "pardiso" or
                      algorithm == "wsmp" or
                      algorithm == "mumps") ):
                raise ValueError("invalid solver")

            class DummyWrapper:
                pass

            usrfun = DummyWrapper()
            usrfun.objective = self.objf
            usrfun.gradient = self.objgrad
            usrfun.constraints = self.con
            if( self.jacobian_style == "sparse" ):
                usrfun.jacobianstructure = lambda: (self.iG,self.jG)
            usrfun.jacobian = lambda b: np.asarray( self.congrad(b) ).reshape(-1)
            usrfun.hessianstructure = lambda: ( range( 0 , self.n ), range( 0, self.n ) )
            usrfun.hessian = lambda b,c,d: np.ones( self.n )

            nlp = ipopt.problem( n = self.n, m = self.nconstraint,
                                 problem_obj = usrfun,
                                 lb = self.xlow, ub = self.xupp,
                                 cl = self.Flow, cu = self.Fupp )

            if( not self.stop_options["ftol"] == None ):
                nlp.addOption( "tol", self.stop_options["ftol"] )
            if( not self.stop_options["maxeval"] == None ):
                nlp.addOption( "max_iter", self.stop_options["maxeval"] )
            if( not self.stop_options["maxtime"]==None ):
                nlp.addOption( "max_cpu_time", self.stop_options["maxtime"] )
            if( not algorithm=="" ):
                nlp.addOption( "linear_solver", algorithm )
            if( not self.solve_options["constraint_violation"] == None ):
                nlp.addOption( "constr_viol_tol", self.solve_options["constraint_violation"] )
            if( not self.print_options["summary_level"] == None ):
                nlp.addOption( "print_level", self.print_options["summary_level"] )
            if( not self.print_options["print_level"] == None ):
                nlp.addOption( "file_print_level", self.print_options["print_level"] )
            if( not self.print_options["print_file"]==None ):
                nlp.addOption( "output_file", self.print_options["print_file"] )
            if( self.maximize ):
                nlp.addOption( "obj_scaling_factor", -1.0 )
            if( self.solve_options["warm_start"] ):
                nlp.addOption( "warm_start_init_point", "yes" )

            res, info = nlp.solve( self.x )
            return info["obj_val"], res, info["status_msg"]

        else:
            raise ValueError( "invalid solver" )
