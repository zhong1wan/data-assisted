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

"""
implementation of fully data-driven model (architecture 1) for CdV

"""

class lstm_data(object):
	
	fdim = 6

	def __init__(self, trunc_dim, l, hid_units_x, hid_units_B, params, savepath):

		self.dim = self.fdim - trunc_dim
		self.l = l 								# length of training series
		self.hist = None
		
		self.savepath = savepath
		if not os.path.exists(savepath):
			os.makedirs(savepath)
		
		self.inputs = layers.Input(shape=(l,self.dim), name='ySeq')
		self.x_lstm = layers.recurrent.LSTM(hid_units_x, input_shape=(None, self.fdim), 
							return_sequences=True, name='x_lstm', implementation=2)
		self.Bxy_hidden = layers.Dense(units=hid_units_B, activation='relu', name='Bxy_hidden')
		self.Bxy_output = layers.Dense(units=self.dim, activation='linear', name='Bxy_output')


	def compile(self):
		
		x_series = self.x_lstm(self.inputs)
		
		xy = layers.concatenate([x_series, self.inputs], axis=-1)
		Bxy = layers.TimeDistributed(self.Bxy_hidden)(xy)
		Bxy = layers.TimeDistributed(self.Bxy_output)(Bxy)

		self.outputs = Bxy 					# this is the total dynamics

		self.model = Model(inputs=[self.inputs], outputs=[self.outputs])
		self.model.compile(optimizer='adam', loss='mse')


	def train(self, inputs, outputs, pretrain=None, val_ratio=.1, 
				batch_sz=256, epochs=50, saveWeights=True):
		""" 
		inputs, outputs = 3D np array with dimensions (d,nb_samples,features)

		"""
		nb_samples = inputs.shape[0]
		idx_cut = int(nb_samples * (1 - val_ratio))

		inputs_train, inputs_val = inputs[:idx_cut,:,:], inputs[idx_cut:,:,:]
		outputs_train, outputs_val = outputs[:idx_cut,:,:], outputs[idx_cut:,:,:]

		print('Training...')
		if pretrain is not None:				# load pretrained weights if provided
			self.model.load_weights(pretrain)
		self.hist = self.model.fit([inputs_train], [outputs_train], epochs=epochs, batch_size=batch_sz,
								validation_data=(inputs_val, outputs_val), verbose=2)

		if saveWeights:
			self.model.save_weights(self.savepath+'trained_weights.h5')

		return self.hist
		

	def predict(self, y_trunc, loadWeights=None):

		if loadWeights is not None:
			self.model.load_weights(loadWeights)
		
		return self.model.predict([y_trunc])


	def plot_history(self, saveFig=True):
		
		assert self.hist is not None, 'No training history found.'
		print('Plotting history...')

		plt.figure()
		plt.semilogy(self.hist.history['loss'],'b-',label='training')
		plt.semilogy(self.hist.history['val_loss'],'r-',label='validation')
		plt.legend(frameon=False)
		plt.title('Loss Curve - Overall')
		if saveFig:
			plt.savefig(self.savepath+'loss_overall.png',dpi=300)


def data_proc(trunc_dim, l, idx_start=0, idx_end=10000, dt=.01):
	
	npzfile = np.load('./data/modal_coords_10k_std.npz')
	phys_model = CdV_base_dynamics(npzfile, dt)
	y0 = npzfile['y'][idx_start:idx_end].copy()
	inputs, outputs = [],[]
	total_dyn, trunc_dyn = [], []

	for i in range(l):
		
		inputs.append(y0[:,:-trunc_dim].copy())
		dy0dt = phys_model.true_dynamics(y0)
		
		y0_trunc = y0.copy()
		y0_trunc[:,-trunc_dim:] = 0
		dy0dt_trunc = phys_model.true_dynamics(y0_trunc)
		dy0dt_diff = dy0dt - dy0dt_trunc

		y0 += dt*dy0dt
		outputs.append(dy0dt_diff[:,:-trunc_dim].copy())
		total_dyn.append(dy0dt[:,:-trunc_dim].copy())
		trunc_dyn.append(dy0dt_trunc[:,:-trunc_dim].copy())
	
	inputs, outputs = np.array(inputs), np.array(outputs)
	inputs = np.swapaxes(inputs, 0, 1)
	outputs = np.swapaxes(outputs, 0, 1)
	total_dyn = np.swapaxes(np.array(total_dyn), 0, 1)
	trunc_dyn = np.swapaxes(np.array(trunc_dyn), 0, 1)

	return inputs, outputs, total_dyn, trunc_dyn


def main():
	
	npzfile = np.load('./data/modal_coords_10k_std.npz')
	
	nx, nB = 1, 16    # number of hidden units in layers
	trunc_dim = 1
	l = 200
	sp = './logs/lstm_data/test1/'

	train_inputs, _, train_outputs, _ = data_proc(trunc_dim, l, idx_end=9000)
	test_inputs, _, test_outputs, test_trunc = data_proc(trunc_dim, l, idx_start=9000)

	lstm_model = lstm_data(trunc_dim, l, hid_units_x=nx, hid_units_B=nB, params=npzfile, savepath=sp)
	lstm_model.compile()
	lstm_model.train(train_inputs, train_outputs, epochs=1000, pretrain=None)
	lstm_model.plot_history()

	weights = sp + 'trained_weights.h5'    
	test_pred = lstm_model.predict(test_inputs, loadWeights=weights)
	
	case_idx = 100
	tt = np.linspace(.01,.01*l,l)
	R,C = 1, 6-trunc_dim
	f, axarr = plt.subplots(R, C)
	for row in range(R):
		for col in range(C):
			dim = row*C + col
			axarr[row,col].plot(tt, test_outputs[case_idx,:,dim], 'r-.', label='true')
			axarr[row,col].plot(tt, test_pred[case_idx,:,dim], 'b-', label='pred')
			axarr[row,col].plot(tt, test_trunc[case_idx,:,dim], 'g-', label='trunc')
			axarr[row,col].set_xlim([0,.01*l])
			axarr[row,col].set_title('mode '+str(dim+1))
	axarr[row,col].legend(frameon=False)
	plt.tight_layout()
	plt.savefig(lstm_model.savepath+'tc'+str(case_idx)+'_total_l'+str(l)+'.png', dpi=300)


if __name__ == '__main__':
	main()
