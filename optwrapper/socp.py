import numpy as np
import types

class Problem:
    """
    this transforms a switched optimal control problem (socp) into a non-linear programming problem
    (nlp) so that the socp can be solved with a solver

    requires a non-linear objective function and its gradient
    requires constraints (box, linear, and non-linear)

    """

    def __init__(self, socp, Nsamples):
        """
        arguments:
        socp: an instance from the SOCP (switched optimal control problem) class
        Nsamples: the number of samples in the approximated version of the SOCP

        """

        Nst = socp.Nst
        Ninpcont = socp.Ninpcont
        Nmodes = socp.Nmodes
        N = Nst*(Nsamples + 1) + Ninpcont*Nsamples + Nmodes*Nsamples #number of opt variables
        deltaT = (socp.tf - socp.t0) / Nsamples
        self.N = N
        self.Nconslin = Nsamples #linear constraint for the relaxed discrete mode input
        self.Ncons = Nst * (Nsamples + 1) + Nsamples*socp.Nineqcons

        def encode(st, inpcont, inpmode):
            """
            this function creates one big vector of all the states, continuous inputs, and mode inputs for all times

            arguments:
            st: a matrix of all the states at all times tk = 0...Nsamples
            inpcont: a matrix of all the continuous inputs at all times tk = 0...Nsamples - 1
            inpmode: a matrix of all the mode inputs at all times tk = 0...Nsamples
            """

            s = np.zeros(N)

            for k in range( Nsamples ):
                s[ k*Nst : k*Nst+Nst ] = st[:,k]
                s[ Nst*(Nsamples+1)+k*Ninpcont : Nst*(Nsamples+1)+k*Ninpcont+Ninpcont] = inpcont[:,k]
                s[ Nst*(Nsamples+1) + Ninpcont*Nsamples + k*Nmodes : Nst*(Nsamples+1) + Ninpcont*Nsamples + k*Nmodes + Nmodes] = inpmode[:,k]

            s[ Nst * Nsamples : Nst * (Nsamples + 1) ] = st[:,Nsamples]

            return s

        def decode(s):
            """
            this function creates the st, inpcont, and inpmode matrices from the optimizaion vector, s

            arguments:
            s: the optimizaion vector of size N

            """

            st = np.zeros( (Nst, Nsamples+1) )
            inpcont = np.zeros( (Ninpcont, Nsamples) )
            inpmode = np.zeros( (Nmodes, Nsamples) )

            for k in range(Nsamples):
                st[:,k] = s[ k*Nst : k*Nst+Nst]
                inpcont[:,k] = s[ Nst*(Nsamples+1)+k*Ninpcont : Nst*(Nsamples+1)+k*Ninpcont+Ninpcont ]
                inpmode[:,k] = s[ Nst*(Nsamples+1) + Ninpcont*Nsamples + k*Nmodes : Nst*(Nsamples+1) + Ninpcont*Nsamples + k*Nmodes + Nmodes]

            st[:,Nsamples] = s[ Nst * Nsamples : Nst * (Nsamples + 1) ]

            return (st, inpcont, inpmode)

        def setBounds():
            """
            this function sets the lower and upper bounds on the optimization vector
            """

            lb = np.zeros(N)
            ub = np.zeros(N)

            for k in range( Nsamples ):
                lb[ k*Nst : k*Nst+Nst ] = socp.consstlb
                lb[ Nst*(Nsamples+1)+k*Ninpcont : Nst*(Nsamples+1)+k*Ninpcont+Ninpcont] = socp.consinpcontlb
                lb[ Nst*(Nsamples+1) + Ninpcont*Nsamples + k*Nmodes : Nst*(Nsamples+1) + Ninpcont*Nsamples + k*Nmodes + Nmodes] = 0
                ub[ k*Nst : k*Nst+Nst ] = socp.consstub
                ub[ Nst*(Nsamples+1)+k*Ninpcont : Nst*(Nsamples+1)+k*Ninpcont+Ninpcont] = socp.consinpcontub
                ub[ Nst*(Nsamples+1) + Ninpcont*Nsamples + k*Nmodes : Nst*(Nsamples+1) + Ninpcont*Nsamples + k*Nmodes + Nmodes] = 1

            lb[ Nst * Nsamples : Nst * (Nsamples + 1) ] = socp.consstlb
            ub[ Nst * Nsamples : Nst * (Nsamples + 1) ] = socp.consstub

            self.ub = ub
            self.lb = lb

        def setBoundsCons():
            """
            this function sets the lower and upper bounds on the constraints

            """

            conslb = np.zeros ( self.Ncons )
            consub = np.zeros ( self.Ncons )

            conslb[Nst*(Nsamples + 1): Nst*(Nsamples + 1) + Nsamples * socp.Nineqcons ] = -np.inf

            self.conslb = conslb
            self.consub = consub

        def setLinCons():
            A = np.zeros( (Nsamples, N ) )
            for k in range( Nsamples ):
                A[ k, Nst*(Nsamples+1) + Ninpcont*Nsamples + k*Nmodes : Nst*(Nsamples+1) + Ninpcont*Nsamples + k*Nmodes + Nmodes] = 1

            self.conslinA = A
            self.conslinlb = np.ones( (Nsamples,) )
            self.conslinub = np.ones( (Nsamples,) )

        def objectiveFctn( s ):
            """
            this function uses the instant cost functions for the various modes from the and the final cost
            function from the socp problem and creates the objective function for the nlp problem

            arguments:
            s: the optimization vector

            """

            (st, inpcont, inpmode) = decode(s)

            objfruncost = 0

            for i in range(Nmodes):
                for k in range(Nsamples):
                    instcostarray = socp.instcost[i](st[:,k], inpcont[:,k])
                    objfruncost = objfruncost + (inpmode[i,k] * instcostarray * deltaT)

            objffincost = socp.fincost(st[:,Nsamples])

            return objfruncost + objffincost

        def objectiveGrad( s ):
            """
            this function returns the gradient of the objective function with respect to the optimization
            vector, s, which is a concatenated vector of all the states, continuous inputs, and modal inputs

            arguments:
            s: optimization vector

            """

            (st, inpcont, inpmode) = decode(s)

            objg = np.zeros(N)

            for k in range(Nsamples):
                gradstprev = 0
                gradinpprev = 0
		gradmode = np.zeros( Nmodes  )
                for i in range(Nmodes):
                    gradstcurrent = inpmode[i,k] * deltaT * socp.instcostgradst[i](st[:,k])
                    gradstprev = gradstprev + gradstcurrent
                    gradinpcurrent = inpmode[i,k] * deltaT * socp.instcostgradinpcont[i](inpcont[:,k])
                    gradinpprev = gradinpprev + gradinpcurrent
		    gradmode[i] = socp.instcost[i](st[:,k], inpcont[:,k]) * deltaT 
                objg[ k*Nst : k*Nst+Nst ] = gradstprev
                objg[ Nst*(Nsamples+1)+k*Ninpcont : Nst*(Nsamples+1)+k*Ninpcont+Ninpcont] = gradinpprev
                objg[ Nst*(Nsamples+1) + Ninpcont*Nsamples + k*Nmodes : Nst*(Nsamples+1) + Ninpcont*Nsamples + k*Nmodes + Nmodes ] = gradmode 

            objg[ Nst * Nsamples : Nst * (Nsamples + 1) ] = socp.fincostgradst(st[:,Nsamples])

            return objg

        def constraintFctn( s ):
            """
            this function returns the Forward Euler Constraints, the initial condition imposement, and the
            inequality constraints from socp for the nlp problem

            arguments:
            the optimization vector s

            """

            (st, inpcont, inpmode) = decode(s)

            consf = np.zeros( self.Ncons )

            for k in range(Nsamples):
                dynnew = 0
                for i in range(Nmodes):
                    dyninter = inpmode[i,k] * socp.dynamics[i](st[:,k], inpcont[:,k])
                    dynnew = dyninter + dynnew
                consf[k*Nst:k*Nst+Nst] = st[:,k+1] - st[:,k] - deltaT*dynnew

            consf[Nst*Nsamples:Nst*(Nsamples + 1)] = socp.init

            for k in range(1, Nsamples+1):
                consf[Nst*(Nsamples + 1) + (k-1)*socp.Nineqcons : Nst*(Nsamples + 1) + (k-1)*socp.Nineqcons + socp.Nineqcons] = socp.cons(st[:,k])

            return consf

        def constraintGrad( s ):
            """
            this function returns the gradient of the constraint function

            arguments:
            s: the optimization vector

            """

            (st, inpcont, inpmode) = decode(s)

            #rows from forward Euler constraints: Nst * Nsamples
            rows = Nst * Nsamples + Nst + socp.Nineqcons * Nsamples
            columns = Nst*(Nsamples+1) + Ninpcont*Nsamples + Nmodes*Nsamples
            consg = np.zeros( (rows, columns) )

            #this for loop puts the forward euler constraints into the consg matrix
            for k in range(Nsamples):
                cgradstprev = 0
                cgradinpprev = 0
                for i in range(Nmodes):
                    cgradstcurrent = inpmode[i,k] *  socp.dynamicsgradst[i](st[:,k])
                    cgradstprev = cgradstprev + cgradstcurrent
                    cgradinpcurrent = inpmode[i,k] * socp.dynamicsgradinp[i](inpcont[:,k])
                    cgradinpprev = cgradinpprev + cgradinpcurrent
                #fwd euler cons grad wrt states
                consg[ k*Nst : k*Nst + Nst, k*Nst: k*Nst+Nst] = -np.identity(Nst) - deltaT*cgradstprev
                consg[ k*Nst : k*Nst + Nst, Nst*(k+1) : Nst*(k+1) + Nst ] = np.identity(Nst)
                consg[ k*Nst : k*Nst + Nst, Nst*(Nsamples+1) + k*Ninpcont : Nst*(Nsamples+1) + k*Ninpcont + Ninpcont ] = -deltaT*cgradinpprev
                for i in range(Nmodes):
		    dyn_array = socp.dynamics[i](st[:,k], inpcont[:,k])
                    consg[k*Nst : k*Nst + Nst, Nst*(Nsamples+1) + Ninpcont*Nsamples + k*Nmodes + i] = -deltaT * dyn_array

            #this puts the initial condition constraint into consg
            consg[ Nst*Nsamples : Nst*Nsamples + Nst, 0: Nst ] = np.identity(Nst)

            #this for loop puts the constraint function gradients into consg
            for k in range(Nsamples):
                consg[ Nst * Nsamples + Nst + k*socp.Nineqcons : Nst * Nsamples + Nst + k*socp.Nineqcons + socp.Nineqcons, (k+1)*Nst: (k+1)*Nst + Nst ] = socp.consgradst(st[:,k+1])


            return consg

        self.objf = objectiveFctn
        self.objg = objectiveGrad
        self.consf = constraintFctn
        self.consg = constraintGrad
        setBounds() ## this fctn sets self.lb and self.ub
        setBoundsCons() ## this fctn sets self.conslb and self.consub
        setLinCons()
        self.mixedCons = False
        self.init = np.zeros( ( self.N, ) )


class SparseProblem:
    pass
