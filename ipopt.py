class ipoptSolver( optwSolver ):
    def solve( self ):
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
