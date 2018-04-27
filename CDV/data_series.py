import numpy as np
from matplotlib import pyplot as plt

from data_gen import CdV


class CdV_base_dynamics(object):
	"""
	This object contains functions that generate POD-space trajectories 
	(full-dimensional and truncated).

	"""

	def __init__(self, params, dt=.01):
		self.dt = dt
		self.L_y, self.b_y = params['L_y'], params['b_y']
		self.xm, self.w = params['xm'], params['W']
		self.y_std = params['y_std']

	def get_original(self,y):
		return self.xm+np.matmul(y*self.y_std,self.w.T)

	def true_dynamics(self,y):
		### returns true dynamics for a given state (vector type) ###
		X = self.xm+np.matmul(y*self.y_std,self.w.T)
		return np.matmul(CdV.dynamics(X),self.w)/self.y_std

	# def known_dynamics(self,y):
	# 	### returns known (approximate) dynamics for a given state ###
	# 	return np.matmul(y,L_y) + b_y

	def generate_series_true(self, init_states,steps):
		""" return time series integrated based on true dynamics defined 
			result is 3D np array ([time,sample,features], made to be 
			more conveniently used as training input to rnn models) 	"""
		y = init_states.copy()
		Y,dYdt = [],[]
		for step in range(steps):
			Y.append(y.copy())
			dydt = self.true_dynamics(y)
			y += self.dt*dydt
			# Y.append(y.copy())
			dYdt.append(dydt.copy())
		Y, dYdt = np.array(Y), np.array(dYdt)

		return Y,dYdt

	def generate_series_trunc(self,init_states,steps,trunc_dim):
		""" return time series integrated based on truncated dynamics defined 
			result is 3D np array ([time,sample,features], made to be 
			more conveniently used as training input to rnn models) 	"""
		y = init_states.copy()
		y[:,-trunc_dim:] = 0
		Y,dYdt = [],[]
		for step in range(steps):
			Y.append(y.copy())
			dydt = self.true_dynamics(y)
			dydt[:,-trunc_dim:] = 0
			y += self.dt*dydt
			# Y.append(y.copy())
			dYdt.append(dydt.copy())
		Y, dYdt = np.array(Y), np.array(dYdt)

		return Y,dYdt
	
	# def generate_series_known(self,init_states,steps):
	# 	""" return time series integrated based on known dynamics defined 
	# 		result is 3D np array ([sample,time,features], made to be 
	# 		more conveniently used for error calculation and plotting) 	"""
	# 	pred = []
	# 	y0 = init_states.copy()
	# 	for step in range(steps):
	# 		dydt = self.known_dynamics(y)
	# 		y0 += self.dt*dydt
	# 		pred.append(y0.copy())

	# 	pred = np.array(pred)
	# 	if len(init_states.shape) == 2:
	# 		pred = np.swapaxes(pred,0,1)

	# 	return pred 	


