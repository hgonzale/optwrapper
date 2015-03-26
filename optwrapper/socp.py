import numpy as np 
import types 

class Problem:

    """ 
    Switched Optimal Control Problem 
    """

    def __init__(self, Nst, Ninpcont, Nmodes, Nineqcons):
        """
        arguments:
        Nst: the number of continuous states
        Ninpcont: the number of continuous inputs
        Nmodes: the number of discrete inputs (the number of modes)
        Nineqcons: the number of inequality constraints 
        """

        try:
            self.Nst = int(Nst)
        except:
            raise ValueError("Nst must be an integer")

        try:
            self.Ninpcont = int(Ninpcont)
        except:
            raise ValueError("Ninpcont must be an integer")

        try:
            self.Nmodes = int(Nmodes)
        except:
            raise ValueError("Nmodes must be an integer")

        if (self.Nst <= 0):
            raise ValueError("Nst must be strictly positive")

        if (self.Ninpcont <= 0):
            raise ValueError("Ninpcont must be strictly positive")

        if (self.Nmodes <= 0):
            raise ValueError("Nmodes must be strictly positive")

        self.init = np.zeros(self.Nst)
        self.Nst = Nst 
        self.Ninpcont = Ninpcont
        self.Nmodes = Nmodes
        self.Nineqcons = Nineqcons
        self.t0 = None
        self.tf = None
        self.instcost = None 
        self.instcostgradst = None
        self.instcostgradinp = None    
        self.fincost = None
        self.fincostgradst = None 

        self.dynamics = None
        self.dynamicsgradst = None
        self.dynamicsgradinp = None

        self.cons = None
        self.consgradst = None 

        self.consstlb = None
        self.consstub = None
        self.consinpcontlb = None
        self.consinpcontub = None 

    def initPoint(self, init):
        """
        sets the initial point for the optimization problem 

        arguments:
        init: the initial condition, a 1D array of size Nst, defaults to an array of zeros 

        """

        self.init = np.array(init)

    def initialFinalTime(self, t0, tf):
        """
        sets the initial and final times for the optimization problem 

        arguments:
        t0: initial time of the optimization problem
        tf: final time of the optimization problem

        """

        self.t0 = t0
        self.tf = tf 

    def instantCost(self, instcost, instcostgradst, instcostgradinpcont):
        """
        sets the instant cost and gradients with respect to the state and continuous input for each mode 

        arguments:
        instcost: a list of functions of size Nmodes of the instantaneous cost for each mode 
        instcostgradst: a list of functions of size Nmodes of the gradients of the instantaneous cost with respect to the states for each mode 
        instcostgradinp: a list of functions of size Nmodes of the gradients of the instantaneous cost with respect to the continuous inputs for each mode 
        """

        self.instcost = instcost
        self.instcostgradst = instcostgradst
        self.instcostgradinpcont = instcostgradinpcont

        # for k in range(self.Nmodes):
        #     self.instcost[k] = instcost[k]
        #     self.instcostgradst[k] = instcostgradst[k]
        #     self.instcostgradinpcont[k] = instcostgradinpcont[k]

    def finalCost(self,fincost, fincostgradst):
        """
        sets the final cost and its gradient with respect to the states 

        arguments:
        fincost: the final cost function 
        fincostgradst: the gradient of the final cost with respect to the states 

        """

        self.fincost = fincost
        self.fincostgradst = fincostgradst 

        if ( type(fincost) != types.FunctionType ):
            raise ValueError("the final cost must be a function")

    def dynamicsFctn(self, dynamics, dynamicsgradst, dynamicsgradinp):
        """
        sets the dynamics functions and gradients with respect to the state and continuous input for each mode 

        arguments:
        dynamics: a list of functions of size Nmodes of the dynamics for each mode 
        dynamicsgradst: a list of functions of size Nmodes of the gradient of the dynamics with respect to the states for each mode 
        dynamicsgradinp: a list of functions of size Nmodes of the gradient of the dynamics with respect to the continuous inputs for each mode

        """

        self.dynamics = dynamics
        self.dynamicsgradst = dynamicsgradst
        self.dynamicsgradinp = dynamicsgradinp

        # for k in range(self.Nmodes):
        #     self.dynamics[k] = dynamics[k]
        #     self.dynamicsgradst[k] = dynamicsgradst[k]
        #     self.dynamicsgradinp[k] = dynamicsgradinp[k]

    def consFctn(self, cons, consgradst):
        """
        defines the constraints for the switched optimal control problem 

        arguments:
        cons: the matrix of constraints, must have size Nineqcons
        consgradst: the gradient of the constraints with respect to the states 

        """

        self.cons = cons
        self.consgradst = consgradst 

    def consBox(self, consstlb, consstub, consinpcontlb, consinpcontub):
        """
        defines the box constraints on the states and continuous inputs 
        note: these constraints are independent of the mode 

        arguments:
        consstlb: the lower bound box constraints on the state; a vector of size Nst 
        consstub: the upper bound box constraints on the state; a vector of size Nst 
        consinpcontlb: the lower bound box constraints on the continuous input; a vector of size Ninpcont
        consinpcontub: the upper bound box constraints on the continuous input; a vector of size Ninpcont

        """

        self.consstlb = consstlb 
        self.consstub = consstub
        self.consinpcontlb = consinpcontlb
        self.consinpcontub = consinpcontub



