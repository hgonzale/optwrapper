import types
import numpy as np
# import nlopt
# import ipopt
from optw_snopt import SnoptSolver
from optw_npsol import NpsolSolver

class OptProblem:
    '''
    Wrapper for an optimization problem
    '''

    def __init__( self, **kwargs ):
        '''
        required keyword arguments:
        n: an integer representing number of variables
        nconstraint: an integer representing number of constraints
        maximize: a boolean indicating whether the objective should be
                  maximized(True) or minimized(False)

        prob = OptProblem( n=2, nconstraint=2, maximize=False )
        '''
        try:
            self.n = int( kwargs['n'] )
            assert (self.n > 0 )
        except:
            print( 'Usage: ' )
            print( self.__init__.__doc__ )
            raise ValueError( 'input n must be positive integer' )

        try:
            self.nconstraint = int( kwargs['nconstraint'] )
            assert( self.nconstraint >= 0 )
        except:
            print( 'Usage: ' )
            print( self.__init__.__doc__ )
            raise ValueError( 'input nconstraint must be positive integer' )

        try:
            self.maximize = bool( kwargs['maximize'] )
        except ValueError:
            print( 'Usage: ' )
            print( self.__init__.__doc__ )
            raise ValueError( 'input maximize must be either True or False' )

        self.stop_options = { 'stopval':None, 'maxeval':1000,
                              'maxtime':1.0, 'xtol':1e-4, 'ftol':1e-4 }
        self.print_options={ 'print_level':0, 'summary_level':0,
                            'options_file':None, 'print_file':None }
        self.solve_options={ 'warm_start':False, 'constraint_violation':1e-8 }

    def x_bounds( self, low, high ):
        '''
        Defines upper and lower bounds for x

        prob.x_bounds( [-1,-2], [1,2] )
        '''
        if( not len(low) == self.n or
            not len(high) == self.n ):
            print( 'Usage: ' )
            print( self.x_bounds.__doc__ )
            raise ValueError( 'x_bounds size does not match problem size(' + str(self.n) + ')' )
        self.xlow = np.asarray( low )
        self.xupp = np.asarray( high )

    def constraint_bounds( self, low, high ):
        '''
        Defines upper and lower bounds for each constraint

        prob.constraint_bounds( [-1,-2], [1,2] )
        '''
        if( not len(low) == self.nconstraint or
            not len(high) == self.nconstraint ):
            print( 'Usage: ' )
            print( self.constraint_bounds.__doc__ )
            raise ValueError( 'constraint_bounds size does not match problem size(' + str(self.nconstraint) + ')' )
        self.Flow = np.asarray( low )
        self.Fupp = np.asarray( high )

    def set_objective( self, f ):
        '''
        f is a function that returns the calculated objective (scalar)

        def f(x):
            return x[1]
        prob.set_objective( f )
        '''
        if( not type(f) == types.FunctionType ):
            print( 'Usage: ' )
            print( self.set_objective.__doc__ )
            raise ValueError( 'input must be a function' )
        self.objf = f

    def set_objective_gradient( self, fg ):
        '''
        fg is a function that returns
        an array (dense) or a dictionary(sparse)

        def fg(x):
            return np.array( [0,1] )
        def fg(x):
            return { 0:0, 1:1 }
        prob.set_objective_gradient( fg )
        '''
        if( not type(fg) == types.FunctionType ):
            print( 'Usage: ' )
            print( self.set_objective_gradient.__doc__ )
            raise ValueError( 'input must be a function' )
        self.objgrad = fg

    def set_constraint( self, g ):
        '''
        g is a function that returns an array

        def g(x):
            return np.array( [ x[0] - x[1],
                               x[0] + x[1] ] )
        prob.set_constraint( g )
        '''
        if( not type(g) == types.FunctionType ):
            print( 'Usage: ' )
            print( self.set_constraint.__doc__ )
            raise ValueError( 'input must be a function' )
        self.con = g

    def set_constraint_gradient( self, gg ):
        '''
        gg is a function that returns
        a matrix (dense) or a dictionary(sparse)
        the matrix M is such that
        M[i,j] is the partial derivative of the ith constraint
        with respect to the jth variable

        to define the jacobian as a sparse matrix, consider using
        set_sparse_constraint gradient(self,i_indices,j_indices,gg)

        def gg(x):
            return np.matrix( [ [ 2*x[0], 8*x[1] ],
                                [ 2*(x[0]-2), 2*x[1] ] ] )
        prob.set_constraint_gradient( gg )
        '''
        if( not type(gg) == types.FunctionType ):
            print( 'Usage: ' )
            print( self.set_constraint_gradient.__doc__ )
            raise ValueError( 'input must be a function' )
        self.jacobian_style = 'dense'
        self.congrad = gg

    def set_sparse_constraint_gradient( self, i_indices, j_indices, gg ):
        '''
        i_indices, j_indices, are arrays of length nj
        where nj is the number of non-zero elements in
        the jacobian
        gg is a function that returns an array A which contains
        the non-zero jacobian elements with corresponding indices
        specified by i_indices and j_indices, that is,
        A[p] is the partial derivative of the i_indices[p] constraint
        with respect to the j_indices[p] variable

        to define the jacobian as a dense matrix,
        consider using set_constraint_gradient(gg)

        i_indices = [ 0, 0, 1, 1 ]
        j_indices = [ 0, 1, 0, 1 ]
        def gg(x):
            return np.array( [ 2*x[0], 8*x[1],
                               2*(x[0]-2), 2*x[1] ] )
        prob.set_sparse_constraint_gradient( i_indices, j_indices, gg )
        '''
        if( not len(i_indices) == len(j_indices) ):
            print( 'Usage: ' )
            print( self.set_sparse_constraint_gradient.__doc__ )
            raise ValueError( 'mismatched length of i_indices and j_indices' )
        if( not type(gg) == types.FunctionType ):
            print( 'Usage: ' )
            print( self.set_sparse_constraint_gradient.__doc__ )
            raise ValueError( 'input gg must be a function' )
        self.iG = i_indices
        self.jG = j_indices
        self.jacobian_style = 'sparse'
        self.congrad = gg

    def get_stop_options( self ):
        '''
        returns a dictionary containing all stop options
        '''
        return self.stop_options

    def set_stop_options( self, val ):
        '''
        defines a dictionary containing all stop options

        stop = prob.get_stop_options()
        stop['maxtime'] = 5.0
        prob.set_stop_options(stop)
        '''
        if( not type(val) == types.DictType or
            not val.has_key('xtol') or
            not val.has_key('ftol') or
            not val.has_key('stopval') or
            not val.has_key('maxeval') or
            not val.has_key('maxtime') ):
            print( 'Usage: ' )
            print( self.set_stop_options.__doc__ )
            raise ValueError( 'input must be a dictionary defining all stop options' )
        else :
            self.stop_options = val

    def get_print_options( self ):
        '''
        returns a dictionary containing all print options
        '''
        return self.print_options

    def set_print_options( self, val ):
        '''
        defines a dictionary containing all print options
        print_level: verbosity of file output (0-11 for SNOPT, 0-30 for NPSOL, 0-12 for IPOPT)
        summary_level: verbosity of output to screen (0-1 for SNOPT, 0-12 for IPOPT)

        prt = prob.get_print_options()
        prt['print_file'] = 'ipopt.out'
        prob.set_print_options(prt)
        '''
        if( not type(val) == types.DictType or
            not val.has_key('print_level') or
            not val.has_key('summary_level') or
            not val.has_key('options_file') or
            not val.has_key('print_file') ):
            print( 'Usage: ' )
            print( self.set_print_options.__doc__ )
            raise ValueError( 'input must be a dictionary defining all print options' )
        else:
            self.print_options = val

    def get_solve_options( self ):
        '''
        returns a dictionary containing all solve options
        '''
        return self.solve_options

    def set_solve_options( self, val ):
        '''
        defines a dictionary containing all solve options

        sv = prob.get_solve_options()
        sv['warm_start'] = True
        prob.set_solve_options( sv )
        '''
        if( not type(val) == types.DictType or
            not val.has_key('constraint_violation') or
            not val.has_key('warm_start') ):
            print( 'Usage: ' )
            print( self.set_solve_options.__doc__ )
            raise ValueError( 'input must be a dictionary defining all solve options' )
        else :
            self.solve_options = val

    def set_start_x( self, val ):
        '''
        Defines an array that contains an initial guess for
        values of x

        prob.set_start_x( [ 1.0, 1.0 ] )
        '''
        if( not len(val) == self.n ):
            print( 'Usage: ' )
            print( self.set_start_x.__doc__ )
            raise ValueError( 'start_x size does not match problem size(' + str(self.n) + ')' )
        self.x = np.asarray( val )

    def check_gradient( self, h=1e-6, tol=1e-4 ):
        '''
        Checks the user-defined gradients for the problem
        using a finite difference method with:
        h: small change in x (default 1e-6)
        tol: error tolerance (default 1e-4)

        is_correct = prob.check_gradient( h=1e-6, tol=1e-5 )
        '''
        usrgrad = np.zeros( [ self.nconstraint+1, self.n ] )
        numgrad = np.zeros( [ self.nconstraint+1, self.n ] )
        fph = np.zeros( self.nconstraint + 1 )
        fmh = np.zeros( self.nconstraint + 1 )
        for k in range( 0, self.n ):
            hvec = np.zeros( self.n )
            hvec[k] = h
            fph[0] = self.objf( self.x + hvec )
            fph[1:] = self.con( self.x + hvec )
            fmh[0] = self.objf( self.x - hvec )
            fmh[1:] = self.con( self.x - hvec )
            if( np.any( np.isnan( fph ) ) or np.any( np.isnan( fmh ) ) ):
                raise ValueError( 'NaN found for column ' + str(k) )
            delta = ( fph - fmh ) / 2.0 / h
            numgrad[:,k] = delta
        usrgrad[0,:] = self.objgrad( self.x )
        if( self.jacobian_style == 'sparse' ):
            A = self.congrad( self.x )
            for p in range( 0, len(self.iG) ):
                usrgrad[ self.iG[p]+1, self.jG[p] ] = A[p]
        else:
            usrgrad[1:,:] = self.congrad( self.x )
        err = abs( usrgrad - numgrad ).max()
        err_index = int( abs( usrgrad - numgrad ).argmax() )
        print( 'Check gradient: maximum error in gradient = ' + str(err) )
        if( err < tol ):
            return True
        else:
            print( 'Gradient may be incorrect for Row: ' +
                    str( err_index / ( self.nconstraint + 1 ) ) + ', Column: ' +
                    str( err_index % ( self.nconstraint + 1 ) ) )
            return False

    def check_errors( self ):
        '''
        Checks whether the variables,constraints,
        bounds, and user-defined functions
        are defined properly

        is_correct = prob.check_errors()
        '''
        ## accessing undefined attributes will raise AttributeError
        try:
            for i in range( 0, self.n ):
                if( self.xupp[i] < self.xlow[i] ):
                    print( 'upper bound is less than lower bound for x(' + str(i) + ')' )
                    return False
                if( self.x[i] > self.xupp[i] or self.x[i] < self.xlow[i] ):
                    print( 'x(' + str(i) + ') is out of bounds' )
                    return False
            for i in range( 0, self.nconstraint ):
                if( self.Fupp[i] < self.Flow[i] ):
                    print( 'upper bound is less than lower bound for constraint('
                           + str(i) + ')' )
                    return False
            if( not np.array( self.objf( self.x ) ).shape == () ):
                print( 'The objective function should return a scalar' )
                return False
            if( not np.array( self.objgrad( self.x ) ).shape == ( self.n, ) ):
                print( 'The objective gradient function should return an array of size '
                       + str(self.n) )
                return False
            if( not np.array( self.con( self.x ) ).shape == ( self.nconstraint, ) ):
                print( 'The constraint function should return an array of size '
                       + str(self.nconstraint) )
                return False
            if( self.jacobian_style == 'sparse' ):
                if( not len( self.congrad( self.x ) ) == len(self.iG) ):
                    print( 'The constraint gradient function should return an array of size '
                           + str( len(self.iG) ) )
                    return False
            else:
                if( not np.mat( self.congrad( self.x ) ).shape == ( self.nconstraint, self.n ) ):
                    print( 'The constraint gradient function should return a matrix of shape ('
                           + str(self.nconstraint) + ',' + str(self.n) + ')' )
                    return False
        except AttributeError as e1:
            print( 'The attribute ' + str(e1).split("'")[-2] + ' has not been defined yet')
            return False
        return True

    def solve( self, solver ):
        '''
        Solves the problem using the specified solver
        returns a tuple consisting of
        objective_value, x_values, solve_status
        if using a specific algorithm, separate the
        name of the algorithm and the name of the solver
        by a space

        obj,x,st = prob.solve( 'NLOPT MMA' )
        '''
        if( solver == 'SNOPT' ):
            if( self.jacobian_style == 'sparse' ):
                lenG = len(self.iG) + self.n
            else:
                lenG = ( self.nconstraint + 1 ) * self.n
            prob = SnoptSolver( n = self.n, neF = ( self.nconstraint + 1 ),
                                lenA = 0, lenG = lenG,
                                summaryLevel = self.print_options['summary_level'],
                                printLevel = self.print_options['print_level'],
                                printFile = self.print_options['print_file'],
                                maximize = int(self.maximize) )
            prob.x_bounds( self.xlow, self.xupp )
            prob.F_bounds( np.append([-1e6], self.Flow), np.append([1e6], self.Fupp ) )
            prob.set_x( self.x )
            if( self.jacobian_style == 'sparse' ):
                indGx = [1 for j in range( 1, self.n+1 ) ]
                indGy = range( 1, self.n + 1 )
                indGx.extend( [ x+2 for x in self.iG ] )
                indGy.extend( [ y+1 for y in self.jG ] )
            else:
                indGx = [ i for i in range( 1, self.nconstraint + 2 ) for j in range( 1, self.n + 1 ) ]
                indGy = [ j for i in range( 1, self.nconstraint + 2 ) for j in range( 1, self.n + 1 ) ]
            prob.G_indices( indGx, indGy )

            ## Page 23, Section 3.6, of SNOPT's manual
            def callback( status, n, x, needF, neF, F, needG, neG, G ):
                if( needF > 0 ):
                    F[0] = self.objf(x)
                    con = self.con(x)
                    for i in range( 1, self.nconstraint + 1 ):
                        F[i] = con[i-1]
                if( needG > 0 ):
                    objgrad = self.objgrad(x)
                    for i in range( 0, self.n ):
                        G[i] = objgrad[i]
                    if( self.jacobian_style == 'sparse' ):
                        congrad = self.congrad(x)
                    else:
                        congrad = np.asarray( self.congrad(x) ).reshape(-1)
                    for i in range( 0, len(congrad) ):
                        G[i + self.n] = congrad[i]

            prob.set_funobj( callback )
            prob.set_options( int( self.solve_options['warm_start'] ), self.stop_options['maxeval'],
                              self.solve_options['constraint_violation'], self.stop_options['ftol'] )
            answer = prob.solve()
            finalX = prob.get_x()
            status = prob.get_status()
            finalXArray = [ finalX[i] for i in range( 0, len(finalX) ) ]
            return answer, finalXArray, status

        elif( solver == 'NPSOL' ):
            prob = NpsolSolver( n = self.n, nclin = 0, ncnln = self.nconstraint,
                                printLevel = self.print_options['print_level'],
                                printFile = self.print_options['print_file'],
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

            if( self.jacobian_style == 'sparse' ):
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
            prob.set_options( int( self.solve_options['warm_start'] ), self.stop_options['maxeval'],
                              self.solve_options['constraint_violation'], self.stop_options['ftol'] )
            answer = prob.solve()
            finalX = prob.get_x()
            status = prob.get_status()
            finalXArray = [ finalX[i] for i in range( 0, len(finalX) ) ]
            return answer, finalXArray, status

        elif( solver.startswith('NLOPT') ):
            algorithm = solver[6:]
            if( algorithm == '' or algorithm == 'MMA' ):
                #this is the default
                opt = nlopt.opt( nlopt.LD_MMA, self.n )
            elif( algorithm == 'SLSQP' ):
                opt = nlopt.opt( nlopt.LD_SLSQP, self.n )
            elif( algorithm == 'AUGLAG' ):
                opt = nlopt.opt( nlopt.LD_AUGLAG, self.n )
            else:
                ## other algorithms do not support vector constraints
                ## auglag does not support stopval
                raise ValueError( 'invalid solver' )

            def objfcallback( x, grad ):
                if( grad.size > 0 ):
                    grad[:] = self.objgrad(x)
                return self.objf(x)

            if( self.jacobian_style == 'sparse' ):
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
                        np.ones( self.nconstraint*2 ) * self.solve_options['constraint_violation'] )
            opt.set_lower_bounds( self.xlow )
            opt.set_upper_bounds( self.xupp )
            if( not self.stop_options['xtol'] == None ):
                opt.set_xtol_rel( self.stop_options['xtol'] )
            if( not self.stop_options['ftol'] == None ):
                opt.set_ftol_rel( self.stop_options['ftol'] )
            if( not self.stop_options['stopval'] == None ):
                opt.set_stopval( self.stop_options['stopval'] )
            if( not self.stop_options['maxeval'] == None ):
                opt.set_maxeval( self.stop_options['maxeval'] )
            if( not self.stop_options['maxtime'] == None ):
                opt.set_maxtime( self.stop_options['maxtime'] )

            finalX = opt.optimize( self.x )
            answer = opt.last_optimum_value()
            status = opt.last_optimize_result()

            if( status == 1 ):
                status = 'generic success'
            elif( status == 2 ):
                status = 'stopval reached'
            elif( status == 3 ):
                status = 'ftol reached'
            elif( status == 4 ):
                status = 'xtol reached'
            elif( status == 5 ):
                status = 'maximum number of function evaluations exceeded'
            elif( status == 6 ):
                status = 'timed out'
            else:
                #errors will be reported as thrown exceptions
                status = 'invalid return code'
            return answer, finalX, status

        elif( solver.startswith('IPOPT') ):
            algorithm=solver[6:]
            if( not ( algorithm == '' or
                      algorithm == 'ma27' or
                      algorithm == 'ma57' or
                      algorithm == 'ma77' or
                      algorithm == 'ma86' or
                      algorithm == 'ma97' or
                      algorithm == 'pardiso' or
                      algorithm == 'wsmp' or
                      algorithm == 'mumps') ):
                raise ValueError('invalid solver')

            class DummyWrapper:
                pass

            usrfun = DummyWrapper()
            usrfun.objective = self.objf
            usrfun.gradient = self.objgrad
            usrfun.constraints = self.con
            if( self.jacobian_style == 'sparse' ):
                usrfun.jacobianstructure = lambda: (self.iG,self.jG)
            usrfun.jacobian = lambda b: np.asarray( self.congrad(b) ).reshape(-1)
            usrfun.hessianstructure = lambda: ( range( 0 , self.n ), range( 0, self.n ) )
            usrfun.hessian = lambda b,c,d: np.ones( self.n )

            nlp = ipopt.problem( n = self.n, m = self.nconstraint,
                                 problem_obj = usrfun,
                                 lb = self.xlow, ub = self.xupp,
                                 cl = self.Flow, cu = self.Fupp )

            if( not self.stop_options['ftol'] == None ):
                nlp.addOption( 'tol', self.stop_options['ftol'] )
            if( not self.stop_options['maxeval'] == None ):
                nlp.addOption( 'max_iter', self.stop_options['maxeval'] )
            if( not self.stop_options['maxtime']==None ):
                nlp.addOption( 'max_cpu_time', self.stop_options['maxtime'] )
            if( not algorithm=='' ):
                nlp.addOption( 'linear_solver', algorithm )
            if( not self.solve_options['constraint_violation'] == None ):
                nlp.addOption( 'constr_viol_tol', self.solve_options['constraint_violation'] )
            if( not self.print_options['summary_level'] == None ):
                nlp.addOption( 'print_level', self.print_options['summary_level'] )
            if( not self.print_options['print_level'] == None ):
                nlp.addOption( 'file_print_level', self.print_options['print_level'] )
            if( not self.print_options['print_file']==None ):
                nlp.addOption( 'output_file', self.print_options['print_file'] )
            if( self.maximize ):
                nlp.addOption( 'obj_scaling_factor', -1.0 )
            if( self.solve_options['warm_start'] ):
                nlp.addOption( 'warm_start_init_point', 'yes' )

            res, info = nlp.solve( self.x )
            return info['obj_val'], res, info['status_msg']

        else:
            raise ValueError( 'invalid solver' )
