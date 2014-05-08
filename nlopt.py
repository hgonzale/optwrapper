class nloptSolver( optwSolver ):
    def solve( self ):
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
