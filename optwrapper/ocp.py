import numpy as np 
import types

class Problem:

	""" 
	Optimal Control Programming Problem
	"""

	def __init__(self, m, Ncons):
		""" 
		arguments:
		m = the number of optimization variables 
		Ncons = the number of constraints in h 
		"""

		try:
			self.m = int(m)
		except:
			raise ValueError("m must be an integer")

		if (self.m <= 0):
			raise ValueError("m must be strictly positive")

		self.init = np.zeros(self.m)
		self.L = None
		self.dLdx = None
		self.dLdu = None
		self.Phi = None
		self.dPhidx = None
		self.x_dot = None
		self.dfdx = None
		self.dfdu = None
		self.h = None
		self.dhdx = None

	def initPoint(self,init):
		"""
		sets the initial point for the optimization problem
		
		arguments:
		init: initial condition, a 1-D array of size m, defaults to an array of zeros 

		"""

		self.init = init

		if (self.init.shape != (self.m) ):
			raise ValueError("the initial point must have size (" + str(self.m) + ",)." ) 

	def instant_cost(self,L,dLdx,dLdu):
		"""
		set the instantaneous cost, L, and its gradients, dLdx and dLdu

		arguments:
		L: the cost function
		dLdx: the gradient with respect to x of the cost function
		dLdu: the gradient with respect to u of the cost function

		"""

		self.L = L 
		self.dLdx = dLdx
		self.dLdu = dLdu 

		if ( type(L) != types.FunctionType ):
			raise ValueError("argument must be a function")

	
	def final_cost(self,Phi,dPhidx):
		"""
		sets the final cost, Phi and its gradient dPhidx

		arguments:
		Phi: the final cost
		dPhidx: the gradient of the final cost with respect to x 

		"""

		self.Phi = Phi
		self.dPhidx = dPhidx

		if ( type(Phi) != types.FunctionType ):
			raise ValueError("argument must be a function")


	def dynamics(self,x_dot,dfdx,dfdu):
		"""
		sets the dynamics function, x_dot, and its gradients, dfdx and dfdu

		arguments:
		x_dot: the dynamics function 
		dfdx: the derivative of the dynamics function with respect to x 
		dfdu: the derivative of the dynamics function with respect to u 

		"""

		self.x_dot = x_dot
		self.dfdx = dfdx
		self.dfdu = dfdu  

		if ( type(x_dot) != types.FunctionType ):
			raise ValueError("argument must be a function")


	def cons(self,h,dhdx):

		""" 
		defines the constraints and its gradients

		arguments:
		h: the matrix of constraints, must have size Ncons
		dhdx: the gradient of the constraints with respect to x

		"""

		self.h = h; 
		self.dhdx = dhdx; 

		if ( type(h) != types.FunctionType ):
			raise ValueError("argument must be a function")

		# if (self.h(self.init).shape != (self.Ncons) ):
		# 	raise ValueError("the constraints must have size (" + str(self.Ncons) + ",)." ) 

	def checkGrad(self, func, grad, size, point, u, h_step=1e-5 ,etol=1e-4 ,debug=False):
		"""
		checks if the user defined gradients are correct using finite differences 

		arguments:
		func: the function you want to check 
		grad: the gradient of the function you are checking
		size: the size of the function 
		point: evaluation point of size m; defaults to the init condition 
		u: the input 
		h_step: the optimization variable variation step size, defaults to 1e-5
		etol: the error tolerance, defaults to 1e-4
		debug: boolean to enable extra debugging information 

		"""

		if (point == None):
			point = self.init 
		# else:
		# 	if (point.shape != (self.m) ):
		# 		raise ValueError("the point must have size (" + str(self.Ncons) + ",)." ) 


		if ( type(func) != types.FunctionType ):
			raise ValueError("argument must be a function")

		if ( type(grad) != types.FunctionType ):
			raise ValueError("argument must be a function")


		usrgrad = np.zeros( [size, self.m] )
		numgrad = np.zeros( [size, self.m] )

		fph = np.zeros( size )
		fmh = np.zeros( size )

		for k in range(0,self.m):
			hvec = np.zeros( self.m )
			hvec[k] = h_step 

			fph = func((point + hvec),u)
			fmh = func((point - hvec),u)

			if( np.any (np.isnan (fph) ) or np.any ( np.isnan (fmh) ) or np.any( np.isinf ( fph ) ) 
				or np.any( np.isinf (fmh) ) ):
				raise ValueError("function returned NaN or inf at iteration" + str(k) )

			delta = (fph - fmh ) / 2.0 / h_step
			numgrad[:,k] = delta 

		usrgrad = grad(point)
		
		if (np.any (np.isnan ( usrgrad) ) or np.any (np.isinf (usrgrad) ) ):
			raise ValueError("gradient returned NaN or inf")

		errgrad = abs(numgrad - usrgrad)
		if (errgrad.max() < etol ):
			return(True, errgrad.max(), errgrad)
		else:
			if (debug):
				idx = np.unravel_index(np.argmax(errgrad), errgrad.shape)
				print "gradient incorrect in element  (" + str(idx[1]) + ",)."
			return(False, errgrad.max(), errgrad)

	def checkGrad_u(self, func, grad, point, u, h_step=1e-5 ,etol=1e-4 ,debug=False):
		"""
		checks if the user defined gradients are correct using finite differences 

		arguments:
		func: the function you want to check 
		grad: the gradient of the function you are checking
		size: the size of the function 
		point: evaluation point of size m; defaults to the init condition 
		u: the input 
		p: the size of the input 
		h_step: the optimization variable variation step size, defaults to 1e-5
		etol: the error tolerance, defaults to 1e-4
		debug: boolean to enable extra debugging information 

		"""

		if (point == None):
			point = self.init 
		# else:
		# 	if (point.shape != (self.m) ):
		# 		raise ValueError("the point must have size (" + str(self.Ncons) + ",)." ) 


		if ( type(func) != types.FunctionType ):
			raise ValueError("argument must be a function")

		if ( type(grad) != types.FunctionType ):
			raise ValueError("argument must be a function")

		try:
			size = len( np.squeeze( func( point, u ) ) )
		except TypeError:
			size = 1

		try:
			p = len( np.squeeze(u) )
		except TypeError:
			p = 1

		usrgrad = np.zeros( [size, p] )
		numgrad = np.zeros( [size, p] )

		fph = np.zeros( size )
		fmh = np.zeros( size )

		for k in range(0,p):
			hvec = np.zeros( p )
			hvec[k] = h_step 

			fph = func( point, (u+hvec) )
			fmh = func( point, (u-hvec) )

			if( np.any (np.isnan (fph) ) or np.any ( np.isnan (fmh) ) or np.any( np.isinf ( fph ) ) 
				or np.any( np.isinf (fmh) ) ):
				raise ValueError("function returned NaN or inf at iteration" + str(k) )

			delta = (fph - fmh ) / 2.0 / h_step
			numgrad[:,k] = delta 

		usrgrad = grad(u)
		
		if (np.any (np.isnan ( usrgrad) ) or np.any (np.isinf (usrgrad) ) ):
			raise ValueError("gradient returned NaN or inf")

		errgrad = abs(numgrad - usrgrad)
		if (errgrad.max() < etol ):
			return(True, errgrad.max(), errgrad)
		else:
			if (debug):
				idx = np.unravel_index(np.argmax(errgrad), errgrad.shape)
				print "gradient incorrect in element  (" + str(idx[1]) + ",)."
			return(False, errgrad.max(), errgrad)







