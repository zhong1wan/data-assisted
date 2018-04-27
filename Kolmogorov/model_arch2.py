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
from kol2d_Kdynamics import K_dynamics

"""
lowD version: only models the dynamics of a few modes
			  lstm layer is assumed to have more units than input dimensions
			  and is directly fed into the dense layer without concatenating
			  with the input

lstm implementation param: use 0 for CPU, 2 for GPU

"""

class lstm_series(object):

	def __init__(self, trunc_dim, s, l, dt, hid_units_x, hid_units_B, savepath):

		self.dim = trunc_dim
		self.s = s 						# length of the setup sequence
		self.l = l 						# length of the result sequence
		self.hist = None
		
		self.savepath = savepath
		if not os.path.exists(savepath):
			os.makedirs(savepath)
		
		# input
		self.inputs = layers.Input(shape=(self.s, self.dim), name='ySeq')

		# trainable
		self.x_lstm = layers.recurrent.LSTM(hid_units_x, input_shape=(None, self.dim), 
							return_state=True, name='x_lstm', implementation=2)
		self.Bxy_hidden = layers.Dense(units=hid_units_B, activation='relu', name='Bxy_hidden')
		self.Bxy_output = layers.Dense(units=self.dim, activation='linear', name='Bxy_output')

		# non-trainable
		self.dynamics_add = layers.Add()
		self.dt_mult = layers.core.Lambda(lambda x: x*dt, name='dt_mult')
		self.integrate_add = layers.Add()
		self.expand_dims_layer = layers.core.Lambda(lambda x: K.expand_dims(x, axis=1), name='ExpandDims')


	def K_setup(self, kol2d_t, Ym, Ys):
		"""  
		Kmodel contains all the necessary functions (in tensor implementation) and parameters
		for computing the known part of the dynamics as well as any helper functions (e.g. converting
		known and LSTM dynamics to the same scale)
	
		"""
		self.Kmodel = K_dynamics(kol2d_t, Ym, Ys)


	def compile_time_step(self, y_input, initial_state=None):
		
		xs, *final_states = self.x_lstm(y_input, initial_state=initial_state)
		Bs = self.Bxy_hidden(xs)
		Bs = self.Bxy_output(Bs) 						# LSTM dynamics in original scale

		ys = self.Kmodel.slicelayer(y_input)			# get the last y in the setup sequence
		Ls = self.Kmodel.dlayer(ys)						# compute known dynamics in original scale

		dydt = self.dynamics_add([Bs, Ls]) 				# add two parts of the dynamics
		dy = self.dt_mult(dydt)
		ys_o = self.Kmodel.rescaleylayer(ys) 			# revert y to original scale

		ys1_o = self.integrate_add([ys_o, dy])  		# integrate with time
		ys1 = self.Kmodel.scaleylayer(ys1_o) 			# scale time-stepped y

		ys1_o = self.expand_dims_layer(ys1_o) 			# expand to 3 dimension
		ys1 = self.expand_dims_layer(ys1) 				# expand to 3 dimension so that it can be used as LSTM input

		return ys1, ys1_o, Bs, dydt, final_states


	def compile(self):
		
		print('Compiling model...')

		self.Yseries, self.Bseries, self.dYseries = [], [], []
		ys, Ys, Bs, dYs, lstm_states = self.compile_time_step(self.inputs)
		
		self.Yseries.append(Ys)
		self.Bseries.append(Bs)
		self.dYseries.append(dYs)

		for _ in range(self.l):
			
			ys, Ys, Bs, dYs, lstm_states = self.compile_time_step(ys, lstm_states)
			self.Yseries.append(Ys)
			self.Bseries.append(Bs)
			self.dYseries.append(dYs)

		adam = optimizers.adam(lr=2e-6, epsilon=.01)
		loss_weights = list(.98**np.arange(self.l + 1))
		# loss_weights = list(np.ones(self.l + 1,))

		self.model = Model(inputs=[self.inputs], outputs=self.dYseries)
		self.model.compile(optimizer=adam, loss='mse', loss_weights=loss_weights)		
		self.Y_eval = K.function(inputs=[self.inputs], outputs=self.Yseries) 			# evaluate Y series (on its own scale)


	def train(self, inputs, outputs, pretrain=None, val_ratio=.05,
				batch_sz=200, epochs=50, saveWeights=True, val_inputs=None, val_outputs=None):
		""" 
		inputs = 3D np array with dimensions (nb_samples,s,features)
		outputs = 3D np array with dimensions (nb_samples,l+1,features)
					converted to list of (nb_samples, 1, features) arrays

		"""
		nb_samples = inputs.shape[0]
		idx_cut = int(nb_samples * (1 - val_ratio))

		inputs_train, inputs_val = inputs[:idx_cut,:,:], inputs[idx_cut:,:,:]
		outputs_train, outputs_val = outputs[:idx_cut,:,:], outputs[idx_cut:,:,:]
		outputs_train = [np.squeeze(i, axis=1) for i in np.split(outputs_train, self.l+1, axis=1)]
		outputs_val = [np.squeeze(i, axis=1) for i in np.split(outputs_val, self.l+1, axis=1)]

		if val_inputs is not None:
			inputs_val = val_inputs
			outputs_val = val_outputs

		print('Training...')
		# load pretrained weights if provided
		if pretrain is not None:				
			self.model.load_weights(pretrain)

		ckpt = ModelCheckpoint(self.savepath+'weights.best.hdf5', verbose=1, save_best_only=True, period=10)
		ckpt2 = ModelCheckpoint(self.savepath+'weights.{epoch:03d}-{val_loss:.2f}.hdf5', period=100)
		cb_list = [ckpt, ckpt2]

		### check if first epoch actually improves loss ###
		train_loss = self.model.evaluate(x=inputs_train, y=outputs_train)
		print('train loss: ', train_loss)
		###

		self.hist = self.model.fit([inputs_train], outputs_train, epochs=epochs, batch_size=batch_sz,
								validation_data=(inputs_val, outputs_val), verbose=2, callbacks=cb_list)

		if saveWeights:
			self.model.save_weights(self.savepath+'trained_weights.h5')
			with open(self.savepath+'history.pkl', 'wb') as file_pi:
				pickle.dump(self.hist.history, file_pi)

		return self.hist
		

	def predict(self, inputs_test, loadWeights=None):

		if loadWeights is not None:
			self.model.load_weights(loadWeights)
		
		return self.model.predict([inputs_test])


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



def main():

	# known dynamics model
	wn = [[0,4], [1,0], [1,4]]
	trunc_dim = 2*len(wn)
	kol2d_t = Kol2D_trunc(N=8, kwn=wn)	

	# define LSTM model parameters
	dt = .005
	hid_units_x = 70
	hid_units_B = 38
	s, l = 100, 100
	sp = './logs/lstm_seq2seq_4/step100/'
	pretrain = './logs/lstm_seq2seq_4/step50_1kepoch/weights.best.hdf5'
	# pretrain = './logs/lstm_hybrid_10/trained_weights.h5'

	# load data
	file = np.load('./data/ktriad_l300_totald.npz')
	train_inputs = file['inputs'][:50000, :s, :]
	train_outputs = file['outputs'][:50000, s-1:s+l, :]
	Ym, Ys = file['input_mean'], file['input_scale']
	dYm, dYs = file['output_mean'], file['output_scale']
	train_outputs = train_outputs*dYs + dYm

	# test data
	test_inputs = file['inputs'][-10000::5, :s, :]
	test_Y = file['inputs'][-10000::5, s:s+l+1, :]*Ys + Ym
	test_dY = file['outputs'][-10000::5, s-1:s+l, :]*dYs + dYm
	test_dY_list = [np.squeeze(i, axis=1) for i in np.split(test_dY, l+1, axis=1)]   	# for loss evaluation

	# initialize LSTM model
	model = lstm_series(trunc_dim, s, l, dt, hid_units_x, hid_units_B, savepath=sp)
	model.K_setup(kol2d_t, Ym, Ys) 														# set up known dynamics
	model.compile()
	model.model.load_weights(pretrain)

	# evaluate test loss
	test_loss = model.model.evaluate(x=test_inputs, y=test_dY_list)
	print('test loss: ', test_loss)
	
	# perform training
	model.train(train_inputs, train_outputs, pretrain, batch_sz=500, epochs=1000, val_inputs=test_inputs, val_outputs=test_dY_list)
	model.plot_history()
	
	weights = sp + 'weights.best.hdf5'
	# weights = './logs/lstm_hybrid_10/trained_weights.h5'
	dYdt_pred = model.predict(test_inputs, loadWeights=weights)
	Y_pred = model.Y_eval([test_inputs])

	# # evaluate train loss
	# train_dY_list = [np.squeeze(i, axis=1) for i in np.split(train_outputs, l+1, axis=1)]
	# train_loss = model.model.evaluate(x=train_inputs, y=train_dY_list)
	# print('training loss: ', train_loss)

	# evaluate test loss
	test_loss = model.model.evaluate(x=test_inputs, y=test_dY_list)
	print('test loss: ', test_loss)

	# reshape predictions
	Y_pred = np.swapaxes(np.squeeze(np.array(Y_pred)), 0, 1)
	dYdt_pred = np.swapaxes(np.array(dYdt_pred), 0, 1)

	# plot predicted series for a single test case
	case_idx = 80
	T = dt*l
	tt = np.linspace(0, T, l+1)
	nb_plots = trunc_dim
	f, axarr = plt.subplots(1, nb_plots, figsize=(18,3))
	for i in range(len(wn)):
		axarr[i].plot(tt, test_Y[case_idx, :, i], 'r-.', label='true')
		axarr[i].plot(tt, Y_pred[case_idx, :, i], 'b-', label='pred')
		axarr[i].set_xlim([0, T])
		axarr[i].set_title('mode ' + np.array_str(np.array(wn[i])) + ' real part')
	
	for i in range(len(wn), nb_plots):
		axarr[i].plot(tt, test_Y[case_idx, :, i], 'r-.', label='true')
		axarr[i].plot(tt, Y_pred[case_idx, :, i], 'b-', label='pred')
		axarr[i].set_xlim([0, T])
		axarr[i].set_title('mode ' + np.array_str(np.array(wn[i-len(wn)])) + ' imag part')
	plt.tight_layout()
	plt.savefig(sp+'tc'+str(case_idx)+'_yseries.png', dpi=300)

	# plot predicted dYdt series for a single test case
	f, axarr = plt.subplots(1, nb_plots, figsize=(18,3))
	for i in range(len(wn)):
		axarr[i].plot(tt, test_dY[case_idx, :, i], 'r-.', label='true')
		axarr[i].plot(tt, dYdt_pred[case_idx, :, i], 'b-', label='pred')
		axarr[i].set_xlim([0, T])
		axarr[i].set_title('mode ' + np.array_str(np.array(wn[i])) + ' real part')
	
	for i in range(len(wn), nb_plots):
		axarr[i].plot(tt, test_dY[case_idx, :, i], 'r-.', label='true')
		axarr[i].plot(tt, dYdt_pred[case_idx, :, i], 'b-', label='pred')
		axarr[i].set_xlim([0, T])
		axarr[i].set_title('mode ' + np.array_str(np.array(wn[i-len(wn)])) + ' imag part')
	plt.tight_layout()
	plt.savefig(sp+'tc'+str(case_idx)+'_dYdtseries.png', dpi=300)


if __name__ == '__main__':
	main()