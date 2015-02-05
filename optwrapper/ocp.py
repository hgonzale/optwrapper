import numpy as np 
import types

class Problem:

	""" 
	Optimal Control Programming Problem
	"""

	def __init__(self, Nst, Ninp, Nineqcons):
		""" 
		arguments:
		Nst = the number of states
		Ninp = the number of inputs  
		Nineqcons = the number of inequality constraints 
		"""

		try:
			self.Nst = int(Nst)
		except:
			raise ValueError("m must be an integer")

		if (self.Nst <= 0):
			raise ValueError("m must be strictly positive")

		self.init = np.zeros(self.Nst)
		self.t0 = None
		self.tf = None
		self.instcost = None
		self.instcostgradst = None
		self.instcostgradin = None
		self.fincost = None
		self.fincostgradst = None
		self.dynamics = None
		self.dynamicsgradst = None
		self.dynamicsgradin = None
		self.cons = None
		self.consgradst = None
		self.consstlb = None
		self.consstub = None
		self.consinplb = None 
		self.consinpub = None

		

	def initPoint(self,init):
		"""
		sets the initial point for the optimization problem
		
		arguments:
		init: initial condition, a 1-D array of size Nst, defaults to an array of zeros 

		"""

		self.init = np.array(init)

		if (self.init.shape != (self.Nst,) ):
			raise ValueError("the initial point must have size (" + str(self.Nst) + ",)." ) 

	def initialFinalTime(self,t0,tf):
		"""
		sets the initial and final times for the optimization problem 

		arguments:
		t0: the initial time of the optimization problem
		tf: the final time of the optimization problem

		"""

		self.t0 = t0
		self.tf = tf 

	def instantCost(self,instcost,instcostgradst,instcostgradin):
		"""
		set the instant cost and its gradients with respect to the states and input

		arguments:
		instcost: the instant cost function
		instcostgradst: the gradient with respect to the states of the instant cost function
		instcostgradin: the gradient with respect to the input of the instant cost function

		"""

		self.instcost = instcost 
		self.instcostgradst = instcostgradst
		self.instcostgradin = instcostgradin 

		if ( type(instcost) != types.FunctionType ):
			raise ValueError("argument must be a function")

	
	def finalCost(self,fincost,fincostgradst):
		"""
		sets the final cost and its gradient with respect to the states

		arguments:
		fincost: the final cost
		fincostgradst: the gradient of the final cost with respect to the states

		"""

		self.fincost = fincost
		self.fincostgradst = fincostgradst

		if ( type(fincost) != types.FunctionType ):
			raise ValueError("argument must be a function")


	def dynamicsFctn(self,dynamics,dynamicsgradst,dynamicsgradin):
		"""
		sets the dynamics function and its gradients

		arguments:
		dynamics: the dynamics function 
		dynamicsgradst: the derivative of the dynamics function with respect to the states
		dynamicsgradin: the derivative of the dynamics function with respect to the input

		"""

		self.dynamics = dynamics
		self.dynamicsgradst = dynamicsgradst
		self.dynamicsgradin = dynamicsgradin 

		if ( type(dynamics) != types.FunctionType ):
			raise ValueError("argument must be a function")


	def consFctn(self,cons,consgradst):

		""" 
		defines the constraints and its gradients

		arguments:
		cons: the matrix of constraints, must have size Nineqcons
		consgradst: the gradient of the constraints with respect to the states

		"""

		self.cons = cons; 
		self.consgradst = consgradst; 

		if ( type(cons) != types.FunctionType ):
			raise ValueError("argument must be a function")

		# if (self.cons(self.init).shape != (self.Nineqcons,) ):
		# 	raise ValueError("the constraints must have size (" + str(self.Nineqcons) + ",)." ) 

	def consBox(self,consstlb, consstub, consinplb, consinpub):
		"""
		defines the box constraints on the states and inputs 

		arguments:
		consstlb: the lower bound box constraints on the state; a vector of size Nst 
		consstub: the upper bound box constraints on the state; a vector of size Nst
		consinplb: the lower bound box constraints on the input; a vector of size Ninp
		consinpub: the upper bound box constraints on the input; a vector of size Ninp

		"""

		self.consstlb = consstlb; 
		self.consstub = consstub; 
		self.consinplb = consinplb; 
		self.consinpub = consinpub; 

	# def checkGradSt(self, fctn, grad, fctnpoint, fctnin, h_step=1e-5 ,etol=1e-4 ,debug=False):
	# 	"""
	# 	checks if the user defined gradients are correct using finite differences 

	# 	arguments:
	# 	fctn: the function you want to check 
	# 	grad: the gradient of the function you are checking
	# 	fctnpoint: evaluation point of size Nst; defaults to the init condition 
	# 	fctnin: the input for the function 
	# 	h_step: the optimization variable variation step size, defaults to 1e-5
	# 	etol: the error tolerance, defaults to 1e-4
	# 	debug: boolean to enable extra debugging information 

	# 	"""

	# 	if (fctnpoint == None):
	# 		fctnpoint = self.init 
	# 	# else:
	# 	# 	if (fctnpoint.shape != (self.Nst) ):
	# 	# 		raise ValueError("the point must have size (" + str(self.Nineqcons) + ",)." ) 


	# 	if ( type(fctn) != types.FunctionType ):
	# 		raise ValueError("argument must be a function")

	# 	if ( type(grad) != types.FunctionType ):
	# 		raise ValueError("argument must be a function")

	# 	try:
	# 		size = len( np.squeeze( fctn( fctnpoint, fctnin ) ) )
	# 	except TypeError:
	# 		size = 1


	# 	usrgrad = np.zeros( [size, self.Nst] )
	# 	numgrad = np.zeros( [size, self.Nst] )

	# 	fph = np.zeros( size )
	# 	fmh = np.zeros( size )

	# 	for k in range(0,self.Nst):
	# 		hvec = np.zeros( self.Nst )
	# 		hvec[k] = h_step 

	# 		fph = fctn((fctnpoint + hvec), fctnin )
	# 		fmh = fctn((fctnpoint - hvec), fctnin )

	# 		if( np.any (np.isnan (fph) ) or np.any ( np.isnan (fmh) ) or np.any( np.isinf ( fph ) ) 
	# 			or np.any( np.isinf (fmh) ) ):
	# 			raise ValueError("function returned NaN or inf at iteration" + str(k) )

	# 		delta = (fph - fmh ) / 2.0 / h_step
	# 		numgrad[:,k] = delta 

	# 	usrgrad = grad(fctnpoint)
		
	# 	if (np.any (np.isnan ( usrgrad) ) or np.any (np.isinf (usrgrad) ) ):
	# 		raise ValueError("gradient returned NaN or inf")

	# 	errgrad = abs(numgrad - usrgrad)
	# 	if (errgrad.max() < etol ):
	# 		return(True, errgrad.max(), usrgrad)
	# 	else:
	# 		if (debug):
	# 			idx = np.unravel_index(np.argmax(errgrad), errgrad.shape)
	# 			print "gradient incorrect in element  (" + str(idx[1]) + ",)."
	# 		return(False, errgrad.max(), usrgrad)

	# def checkGradIn(self, fctn, grad, fctnpoint, fctnin, h_step=1e-5 ,etol=1e-4 ,debug=False):
	# 	"""
	# 	checks if the user defined gradients are correct using finite differences 

	# 	arguments:
	# 	fctn: the function you want to check 
	# 	grad: the gradient of the function you are checking
	# 	size: the size of the function 
	# 	fctnpoint: evaluation point of size Nst; defaults to the init condition 
	# 	fctnin: the function's input   
	# 	h_step: the optimization variable variation step size, defaults to 1e-5
	# 	etol: the error tolerance, defaults to 1e-4
	# 	debug: boolean to enable extra debugging information 

	# 	"""

	# 	if (fctnpoint == None):
	# 		fctnpoint = self.init 
	# 	# else:
	# 	# 	if (point.shape != (self.Nst) ):
	# 	# 		raise ValueError("the point must have size (" + str(self.Nineqcons) + ",)." ) 


	# 	if ( type(fctn) != types.FunctionType ):
	# 		raise ValueError("argument must be a function")

	# 	if ( type(grad) != types.FunctionType ):
	# 		raise ValueError("argument must be a function")

	# 	try:
	# 		size = len( np.squeeze( fctn( fctnpoint, fctnin ) ) )
	# 	except TypeError:
	# 		size = 1

	# 	try:
	# 		p = len( np.squeeze( fctnin ) )
	# 	except TypeError:
	# 		p = 1

	# 	usrgrad = np.zeros( [size, p] )
	# 	numgrad = np.zeros( [size, p] )

	# 	fph = np.zeros( size )
	# 	fmh = np.zeros( size )

	# 	for k in range(0,p):
	# 		hvec = np.zeros( p )
	# 		hvec[k] = h_step 

	# 		fph = fctn( fctnpoint, (fctnin + hvec) )
	# 		fmh = fctn( fctnpoint, (fctnin - hvec) )

	# 		if( np.any (np.isnan (fph) ) or np.any ( np.isnan (fmh) ) or np.any( np.isinf ( fph ) ) 
	# 			or np.any( np.isinf (fmh) ) ):
	# 			raise ValueError("function returned NaN or inf at iteration" + str(k) )

	# 		delta = (fph - fmh ) / 2.0 / h_step
	# 		numgrad[:,k] = delta 

	# 	usrgrad = grad( fctnin )
		
	# 	if (np.any (np.isnan ( usrgrad) ) or np.any (np.isinf (usrgrad) ) ):
	# 		raise ValueError("gradient returned NaN or inf")

	# 	errgrad = abs(numgrad - usrgrad)
	# 	if (errgrad.max() < etol ):
	# 		return(True, errgrad.max(), usrgrad)
	# 	else:
	# 		if (debug):
	# 			idx = np.unravel_index(np.argmax(errgrad), errgrad.shape)
	# 			print "gradient incorrect in element  (" + str(idx[1]) + ",)."
	# 		return(False, errgrad.max(), usrgrad)







