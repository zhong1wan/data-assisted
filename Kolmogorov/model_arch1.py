import os
import numpy as np
import pickle

# from tqdm import tqdm

from keras import layers, optimizers
from keras import backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from kol2d_trunc import Kol2D_trunc

"""
lowD version: only models the dynamics of a few modes
			  lstm layer is assumed to have more units than input dimensions
			  and is directly fed into the dense layer without concatenating
			  with the input

lstm implementation param: use 0 for CPU, 2 for GPU

"""

class lstm_hybrid(object):

	def __init__(self, trunc_dim, l, hid_units_x, hid_units_B, savepath):

		self.dim = trunc_dim
		self.l = l 								# length of training series
		self.hist = None
		
		self.savepath = savepath
		if not os.path.exists(savepath):
			os.makedirs(savepath)
		
		self.inputs = layers.Input(shape=(self.l, self.dim), name='ySeq')
		self.x_lstm = layers.recurrent.LSTM(hid_units_x, input_shape=(None, self.dim), 
							return_sequences=True, name='x_lstm', implementation=2)
		self.Bxy_hidden = layers.Dense(units=hid_units_B, activation='relu', name='Bxy_hidden')
		self.Bxy_output = layers.Dense(units=self.dim, activation='linear', name='Bxy_output')


	def compile(self):
		
		print('Compiling model...')

		x_series = self.x_lstm(self.inputs)
		
		Bxy = layers.TimeDistributed(self.Bxy_hidden)(x_series)
		Bxy = layers.TimeDistributed(self.Bxy_output)(Bxy)

		self.outputs = Bxy
		self.model = Model(inputs=[self.inputs], outputs=[self.outputs])
		adam = optimizers.adam(lr=1e-3, epsilon=.01)
		# sgd = optimizers.SGD(lr=.01, momentum=.9, nesterov=True)
		self.model.compile(optimizer=adam, loss='mse', sample_weight_mode='temporal')		# weight time steps differently
		# self.predictor = K.function(inputs=[self.inputs], outputs=[self.outputs])


	def train(self, inputs, outputs, pretrain=None, val_ratio=.05, setup_ratio=.2,
				batch_sz=256, epochs=50, saveWeights=True):
		""" 
		inputs, outputs = 3D np array with dimensions (d,nb_samples,features)

		"""
		nb_samples = inputs.shape[0]
		idx_cut = int(nb_samples * (1 - val_ratio))

		inputs_train, inputs_val = inputs[:idx_cut,:,:], inputs[idx_cut:,:,:]
		outputs_train, outputs_val = outputs[:idx_cut,:,:], outputs[idx_cut:,:,:]

		# define weights of the time steps on the loss function (initial steps weighted much more lightly)
		time_cut = int(setup_ratio*self.l)

		# max_weight, min_weight = 1, .2
		# c = np.log(max_weight - min_weight + 1)/(self.l - time_cut - 1)
		# w = max_weight + 1 - np.exp(c*range(self.l - time_cut))
		# w = np.concatenate((0*np.ones((time_cut,)), w))
		# sample_weight = np.tile(w, (idx_cut, 1))
		sample_weight = np.concatenate((.01*np.ones((idx_cut, time_cut)), np.ones((idx_cut, self.l - time_cut))), axis=1)

		print('Training...')
		# load pretrained weights if provided
		if pretrain is not None:				
			self.model.load_weights(pretrain)

		# callbacks: keep a copy of the best model so far
		ckpt = ModelCheckpoint(self.savepath+'weights.best.hdf5', verbose=1, save_best_only=True, period=10)
		cb_list = [ckpt]

		self.hist = self.model.fit([inputs_train], [outputs_train], epochs=epochs, batch_size=batch_sz,
								validation_data=(inputs_val, outputs_val), verbose=2, sample_weight=sample_weight, callbacks=cb_list)

		if saveWeights:
			self.model.save_weights(self.savepath+'weights.final.h5')
			with open(self.savepath+'history.pkl', 'wb') as file_pi:
				pickle.dump(self.hist.history, file_pi)

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


def data_proc(steps, init_cond, dynamics_true, dynamics_trunc, vectorizer, dt,
				uvec_mean=0, uvec_std=1):
	"""
	function that prepares the training data to be used by lstm_hybrid
	Input parameters:
	
	steps: 			INT number of steps to run
	init_cond: 		NP ARRAY containing n UNTRUNCATED initial conditions 
					where n = init_cond.shape[0]
	dynamics_true: 	FUNCTION that operates on untruncated states and returns the true dynamics
					that have the exact same dimensions
	dynamics_trunc:	FUNCTION that operates on untruncated states and returns the truncated dyn-
					namics that have zeros in the truncated modes (steps: a. replaces truncated
					dimensions with 0s in the state, b. run dynamics and c. replaces truncated
					dimensions with 0s in the resulting dynamics)
	vectorizer:		FUNCTION that converts the states (untruncated or truncated but filled with
					appropriate 0s) to vectors that can be used as input to the model
	dt:				FLOAT that specifies the 1st order Euler explicit integration time step
	uvec_mean, 
	uvec_std:		ARRAY used to standardize the vectorized states

	"""

	print('Processing data...')

	u0 = init_cond.copy()
	inputs, outputs = [],[]
	tot_dynamics, trunc_dynamics = [], []

	# standardization
	u0_vec = vectorizer(u0)
	# uvec_mean, uvec_std = np.mean(u0_vec, axis=0), np.std(u0_vec, axis=0)

	for i in range(steps):

		print('%d of %d steps completed'%(i,steps))
		inputs.append((vectorizer(u0) - uvec_mean)/uvec_std)

		du0dt = dynamics_true(u0)
		du0dt_trunc = dynamics_trunc(u0)
		du0dt_diff = du0dt - du0dt_trunc

		u0 += dt*du0dt

		outputs.append(vectorizer(du0dt_diff)/uvec_std)
		tot_dynamics.append(vectorizer(du0dt)/uvec_std)
		trunc_dynamics.append(vectorizer(du0dt_trunc)/uvec_std)
	
	inputs = np.swapaxes(np.array(inputs), 0, 1)
	outputs = np.swapaxes(np.array(outputs), 0, 1)
	tot_dynamics = np.swapaxes(np.array(tot_dynamics), 0, 1)
	trunc_dynamics = np.swapaxes(np.array(trunc_dynamics), 0, 1)

	return inputs, outputs, tot_dynamics, trunc_dynamics


def main():

	## set up physical model
	wn0 = [[0,4], [1,0], [1,4]]							# wave numbers we are interested in predicting
	wn = wn0
	wn_idx = Kol2D_trunc.find_wn(wn, wn0)				# find indices of wn0 in wn
	trunc_dim = 2*len(wn)
	phys_trunc_model = Kol2D_trunc(N=8, kwn=wn)


	## model parameters
	hid_units_x = 70
	hid_units_B = 38
	l = 200
	sp = './logs/lstm_hybrid_1/'
	pretrain = './logs/lstm_hybrid_1/weights.best.hdf5'


	## load training data from file: input is standardized, output remains unscaled
	datafile = np.load('./data/ktriad_l200_dt4step.npz')
	train_inputs, train_outputs = datafile['inputs'][:50000], datafile['outputs'][:50000]


	## initiate lstm model and perform training
	lstm_model = lstm_hybrid(trunc_dim, l, hid_units_x, hid_units_B, savepath=sp)
	lstm_model.compile()
	lstm_model.train(train_inputs, train_outputs, batch_sz=250, epochs=1000, pretrain=pretrain)
	lstm_model.plot_history()


	## test trained model
	print('Model testing')

	test_inputs, test_outputs = datafile['inputs'][-10000::5], datafile['outputs'][-10000::5]
	# test_outputs = test_outputs*B_scale + B_mean 					# unscale test outputs

	weights = sp + 'weights.best.hdf5'
	test_pred = lstm_model.predict(test_inputs, loadWeights=weights)
	
	## plot test results
	case_idx = 80
	T = .005*l
	tt = np.linspace(.005, T, l)
	nb_plots = len(wn0)
	f, axarr = plt.subplots(1, 2*nb_plots, figsize=(18,3))
	for i in range(nb_plots):
		axarr[i].plot(tt, test_outputs[case_idx, :, wn_idx[i]], 'r-.', label='true')
		axarr[i].plot(tt, test_pred[case_idx, :, wn_idx[i]], 'b-', label='pred')
		axarr[i].set_xlim([0, T])
		axarr[i].set_title('mode '+np.array_str(np.array(wn[wn_idx[i]]))+' real part')
	
	for i in range(nb_plots):
		axarr[i+nb_plots].plot(tt, test_outputs[case_idx, :, wn_idx[i]+len(wn)], 'r-.', label='true')
		axarr[i+nb_plots].plot(tt, test_pred[case_idx, :, wn_idx[i]+len(wn)], 'b-', label='pred')
		axarr[i+nb_plots].set_xlim([0, T])
		axarr[i+nb_plots].set_title('mode '+np.array_str(np.array(wn0[i]))+' imag part')
	plt.tight_layout()
	plt.savefig(lstm_model.savepath+'tc'+str(case_idx)+'_Bseries.png', dpi=300)



if __name__ == '__main__':
	main()
