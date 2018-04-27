import os
import numpy as np

from keras import layers
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_series import CdV_base_dynamics
from data_gen import CdV
from error_func import traj_err
from model_arch1 import lstm_hybrid


"""
model architecture 2 built just for prediction; no training option
for 5-d POD on CdV, training on architecture 1 alone proves to be sufficient

"""


class lstm_hybrid_pred(lstm_hybrid):

	dt = .01

	def __init__(self, trunc_dim, s, l, hid_units_x, hid_units_B, params, savepath):

		self.dim = self.fdim - trunc_dim
		self.s = s 								# length of setup series length
		self.l = l 								# length of training series
		self.hist = None
		
		self.savepath = savepath
		if not os.path.exists(savepath):
			os.makedirs(savepath)
		
		# input layers
		self.inputs = [layers.Input(shape=(1,self.dim), name='InitSt'+str(v)) for v in range(s)]
		self.inputs_p = [layers.Input(shape=(1,self.dim), name='ys'), layers.Input(shape=(hid_units_x,), name='xs_1'), 
							layers.Input(shape=(hid_units_x,), name='xs_2')]
		
		# trainable layers
		self.x_lstm = layers.recurrent.LSTM(hid_units_x, input_shape=(None, self.fdim), 
							return_state=True, name='x_lstm', implementation=2)
		self.Bxy_hidden = layers.Dense(units=hid_units_B, activation='relu', name='Bxy_hidden')
		self.Bxy_output = layers.Dense(units=self.dim, activation='linear', name='Bxy_output')
		self.Ly_lambda = layers.core.Lambda(function=self.trunc_dynamics, output_shape=(self.dim,),
								arguments={'params':params}, name='Ly_lambda')
		
		# non-trainable operational layers
		self.dt_mult = layers.core.Lambda(lambda x: x*self.dt, name='dt_mult')
		self.expand_dims_layer = layers.core.Lambda(lambda x: K.expand_dims(x, axis=1), name='ExpandDims')
		self.squeeze_layer = layers.core.Lambda(lambda x: K.squeeze(x, axis=-2), name='Squeeze')
		self.dynamics_add = layers.Add()
		self.integrate_add = layers.Add()
		self.xy_concat = layers.Concatenate(axis=-1)


	def time_integrate(self, y0_3d, x0):

		y0 = self.squeeze_layer(y0_3d)
		xy0 = self.xy_concat([x0, y0])
		Bxy0 = self.Bxy_hidden(xy0)
		Bxy0 = self.Bxy_output(Bxy0)
		Ly0 = self.Ly_lambda(y0)
		dydt0 = self.dynamics_add([Ly0, Bxy0])
		dy0 = self.dt_mult(dydt0)
		y1 = self.integrate_add([y0, dy0])
		y1_3d = self.expand_dims_layer(y1)

		return y1_3d, Bxy0, dydt0


	def compile(self):
		
		print('Building graph...')
		
		init_ySeq = layers.concatenate(self.inputs, axis=1)
		lstm_out = self.x_lstm(init_ySeq)
		x0, xlstm_states = lstm_out[0], lstm_out[1:]
		y0 = self.inputs[-1]
		y0, Bxy0, dy0dt = self.time_integrate(y0, x0)

		self.model_s = Model(inputs=self.inputs, outputs=[Bxy0, dy0dt, y0] + xlstm_states)
		self.model_s.compile(optimizer='adam', loss='mse')

		# print(self.inputs[0].shape)
		# print(xlstm_states[0].shape)
		# print(xlstm_states[1].shape)
		
		# ys, xs = y0, xlstm_states
		y0, xlstm_states = self.inputs_p[0], self.inputs_p[1:]
		Bxy_list, dydt_list, y_list = [], [], []
		
		for step in range(self.l):

			lstm_out = self.x_lstm(y0, initial_state=xlstm_states)
			x0, xlstm_states = lstm_out[0], lstm_out[1:]
			y0, Bxy0, dy0dt = self.time_integrate(y0, x0)
			
			Bxy_list.append(Bxy0)
			dydt_list.append(dy0dt)
			y_list.append(y0)

		self.model_p = Model(inputs=self.inputs_p, outputs=Bxy_list+dydt_list+y_list+xlstm_states)
		self.model_p.compile(optimizer='adam', loss='mse')
		

	def predict(self, inputs, steps, loadWeights=None):
		""" 
		inputs = 4D np array with dimensions (s,nb_samples,1,features)

		"""		
		if loadWeights is not None:
			print('Loading weights...')
			self.model_s.load_weights(loadWeights)
			self.model_p.load_weights(loadWeights)

		print('Running predictions - Setup stage...')
		nb_runs = int(np.ceil(steps/self.l))
		records = inputs.copy()
		s_outputs = self.model_s.predict(list(inputs))										# set-up stage
		Bs, dysdt, ys1, xs = s_outputs[0], s_outputs[1], s_outputs[2], s_outputs[3:]		# B_s, y_{s+1}
		B_pred, dydt_pred, y_pred = [Bs.copy()], [dysdt.copy()], [ys1.copy()]

		# prediction stage
		print('Prediction stage...')
		for step in range(nb_runs):

			p_outputs = self.model_p.predict([ys1] + xs)
			
			Bl, dydtl = p_outputs[:self.l], p_outputs[self.l:2*self.l]
			yl, xs = p_outputs[2*self.l:3*self.l], p_outputs[3*self.l:]
			ys1 = yl[-1]
			
			B_pred.extend(list(Bl))				# use list() to make a copy
			dydt_pred.extend(list(dydtl))
			y_pred.extend(list(yl))

		y_pred = np.swapaxes(np.squeeze(np.array(y_pred[:steps])), 0, 1)
		B_pred = np.swapaxes(np.array(B_pred[:steps]), 0, 1)
		dydt_pred = np.swapaxes(np.array(dydt_pred[:steps]), 0, 1)

		return y_pred, B_pred, dydt_pred


	@staticmethod
	def data_proc(trunc_dim, s, l, loadFile, idx_start=9000, idx_end=10000):
		
		dt = lstm_hybrid_pred.dt

		npzfile = np.load(loadFile)
		phys_model = CdV_base_dynamics(npzfile, dt)
		y0 = npzfile['y'][idx_start:idx_end]
		inputs = [np.expand_dims(y0[:,:-trunc_dim].copy(), axis=1)]		# model data

		for i in range(s-1):
			dy0dt = phys_model.true_dynamics(y0)
			y0 += dt*dy0dt
			inputs.append(np.expand_dims(y0[:,:-trunc_dim].copy(), axis=1))
		inputs = np.array(inputs)

		## y0 = true system state (6D full); y0t = truncated y0 (5D); 
		## y1 = system state evolved with known dynamics only (6D with last dim forced to 0)
		traj0, traj1 = [], []
		dydt_traj0, B_traj0 = [], []										
		y1 = y0.copy()							
		y1[:,-trunc_dim:] = 0
		
		for i in range(l):
			
			y0t = y0.copy()
			y0t[:,-trunc_dim:] = 0
			
			dy0dt = phys_model.true_dynamics(y0)
			dy0tdt = phys_model.true_dynamics(y0t)
			B0 = dy0dt - dy0tdt
			y0 += dt*dy0dt
			traj0.append(y0[:,:-trunc_dim].copy())
			B_traj0.append(B0[:,:-trunc_dim].copy())
			dydt_traj0.append(dy0dt[:,:-trunc_dim].copy())

			dy1dt = phys_model.true_dynamics(y1)
			dy1dt[:,-trunc_dim:] = 0
			y1 += dt*dy1dt
			traj1.append(y1[:,:-trunc_dim].copy())
		
		traj0 = np.swapaxes(np.array(traj0), 0, 1)
		traj1 = np.swapaxes(np.array(traj1), 0, 1)
		B_traj0 = np.swapaxes(np.array(B_traj0), 0, 1)
		dydt_traj0 = np.swapaxes(np.array(dydt_traj0), 0, 1)
		traj_dict = {'traj0':traj0, 'traj1':traj1, 'B_traj0':B_traj0, 'dydt_traj0':dydt_traj0}

		return inputs, traj_dict


def main():
	
	loadFile = './data/modal_coords_10k_std.npz'
	npzfile = np.load(loadFile)

	nx, nB = 1, 16    # number of hidden units in layers
	trunc_dim = 1
	s, l = 50, 100
	sp = './logs/lstm_arch2/test1/'

	## compile model
	lstm_model = lstm_hybrid_pred(trunc_dim, s, l, hid_units_x=nx, hid_units_B=nB, params=npzfile, savepath=sp)
	lstm_model.compile()
	# lstm_model.model.summary()
	# plot_model(lstm_model.model, to_file=sp+'model.png', show_shapes=True)
	
	## load trained weights and run predictions
	steps = 4000
	test_inputs, traj_dict = lstm_hybrid_pred.data_proc(trunc_dim, s, steps, loadFile, idx_start=9000)
	
	weights = './logs/lstm_arch1/test1/trained_weights.h5'    ### enter path for weights obtained using model_arch1 ###
	y_pred, B_pred, dydt_pred = lstm_model.predict(test_inputs, steps, loadWeights=weights)
	case_idx = 135
	tt = np.linspace(.01, .01*steps, steps)

	## plot predictions
	f, axarr = plt.subplots(1, 6 - trunc_dim)
	for dim in range(6 - trunc_dim):
		axarr[dim].plot(tt,traj_dict['traj0'][case_idx,:,dim],'r-.',label='true')
		axarr[dim].plot(tt,y_pred[case_idx,:,dim],'b-',label='pred')
		axarr[dim].plot(tt,traj_dict['traj1'][case_idx,:,dim],'g-',label='trunc')
		axarr[dim].set_xlim([0,.01*steps])
		axarr[dim].set_title('mode '+str(dim+1))
	# axarr[dim].legend(frameon=False)
	plt.tight_layout()
	plt.savefig(lstm_model.savepath+'tc'+str(case_idx)+'_series.png', dpi=300)
	plt.savefig('./figures_performance/tc'+str(case_idx)+'_series.png', dpi=300)



if __name__ == '__main__':
	main()

