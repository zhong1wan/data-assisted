import os
import numpy as np

from keras import layers
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from kol2d_trunc import Kol2D_trunc
from error_func import traj_err
from model_puredata_arch2 import lstm_series


"""
model built just for prediction; no training option
lstm implementation param: use 0 for CPU, 2 for GPU

"""


class lstm_pred(object):

	def __init__(self, trunc_dim, s, hid_units_x, hid_units_B, savepath):

		self.dim = trunc_dim
		self.s = s 								# length of setup series length
		self.hist = None
		
		self.savepath = savepath
		if not os.path.exists(savepath):
			os.makedirs(savepath)
		
		# input layers
		self.inputs_s = layers.Input(shape=(self.s, self.dim), name='InitSeq')
		self.inputs_p = [layers.Input(shape=(1, self.dim), name='ys'), layers.Input(shape=(hid_units_x,), name='xs_1'), 
							layers.Input(shape=(hid_units_x,), name='xs_2')]
		
		# trainable layers
		self.x_lstm = layers.recurrent.LSTM(hid_units_x, input_shape=(None, self.dim), 
							return_state=True, name='x_lstm', implementation=2)
		self.Bxy_hidden = layers.Dense(units=hid_units_B, activation='relu', name='Bxy_hidden')
		self.Bxy_output = layers.Dense(units=self.dim, activation='linear', name='Bxy_output')


	def compile(self):
		
		print('Building graph...')
		
		# set up model (long series, no initial cell/hidden states provided)
		lstm_out = self.x_lstm(self.inputs_s)
		x0, xlstm_states = lstm_out[0], lstm_out[1:]
		Bxy = self.Bxy_hidden(x0)
		Bxy = self.Bxy_output(Bxy)

		self.model_s = Model(inputs=self.inputs_s, outputs=[Bxy] + xlstm_states)
		self.model_s.compile(optimizer='adam', loss='mse')
		
		# prediction model (1 step, initial cell/hidden states provided)
		y0, xlstm_states = self.inputs_p[0], self.inputs_p[1:]

		lstm_out = self.x_lstm(y0, initial_state=xlstm_states)
		x0, xlstm_states = lstm_out[0], lstm_out[1:]
		Bxy = self.Bxy_hidden(x0)
		Bxy = self.Bxy_output(Bxy)

		self.model_p = Model(inputs=self.inputs_p, outputs=[Bxy] + xlstm_states)
		self.model_p.compile(optimizer='adam', loss='mse')
		

	def predict(self, inputs, steps, dynamics_trunc, dt, loadWeights,
				input_mean, input_scale):
		""" 
		inputs = 3D np array with dimensions (nb_samples,s,features)

		"""

		def time_step(y, B):
			
			# dynamics (trunc, complementary and full) in original scale
			Y = y*input_scale + input_mean

			dYdt_trunc = dynamics_trunc(Y)
			dYdt = dYdt_trunc + B

			Y1 = Y + dt*dYdt 								# time-marched state in original scale
			y1 = (Y1 - input_mean)/input_scale				# time-marched state in standardized scale

			return y1, Y1, dYdt


		print('Loading weights...')
		self.model_s.load_weights(loadWeights)
		self.model_p.load_weights(loadWeights)

		print('Running predictions - Setup stage...')
		
		y0 = inputs[:, -1, :]
		s_outputs = self.model_s.predict(inputs)										# set-up stage
		Bs, lstm_states = s_outputs[0], s_outputs[1:]
		y0, Y0, dy0dt = time_step(y0, Bs)

		B_pred, dydt_pred, y_pred = [Bs.copy()], [dy0dt.copy()], [Y0.copy()]		# B, dydt and y are all in original scale

		# prediction stage
		print('Prediction stage...')
		for step in range(steps - 1):

			print('%d of %d steps completed...'%(step+2, steps))
			p_outputs = self.model_p.predict([np.expand_dims(y0, axis=1)] + lstm_states)
			Bp, lstm_states = p_outputs[0], p_outputs[1:]

			y0, Y0, dy0dt = time_step(y0, Bp)
			
			B_pred.append(Bp.copy())
			dydt_pred.append(dy0dt.copy())
			y_pred.append(Y0.copy())

		y_pred = np.swapaxes(np.array(y_pred), 0, 1)
		B_pred = np.swapaxes(np.array(B_pred), 0, 1)
		dydt_pred = np.swapaxes(np.array(dydt_pred), 0, 1)

		return y_pred, B_pred, dydt_pred


def main():

	## set up physical model
	wn0 = [[0,4], [1,0], [1,4]]							# wave numbers we are interested in predicting
	wn = wn0
	wn_idx = Kol2D_trunc.find_wn(wn, wn0)				# find indices of wn0 in wn
	trunc_dim = 2*len(wn)
	phys_trunc_model = Kol2D_trunc(N=8, kwn=wn)

	## load scales
	npzfile = np.load('./data/ktriad_l200_standardized.npz')
	input_mean, input_scale = npzfile['input_mean'], npzfile['input_scale']
	output_mean, output_scale = npzfile['output_mean'], npzfile['output_scale']

	## model parameters
	dt = .005
	s = 100
	steps = 200
	# n = int(dt/.005)
	hid_units_x, hid_units_B = 70, 38
	# Ts = 1 					# time span of the setup stage
	# s1 = int(Ts/.005)			# first idx of setup sequence
	# s = int(Ts/dt) 					# number of setup steps
	sp = './logs/lstm_seq2seq_4/step50_1kepoch/'
	weights = './logs/lstm_seq2seq_4/step100/weights.best.hdf5'
	# steps = 100							# number of prediction steps
	# s2 = steps*n 							# last index of prediction sequence

	## load test data file
	npzfile = np.load('./data/ktriad_s100p1000_leadtimetest.npz')
	# test_inputs = npzfile['inputs'][:, -s1::n, :]
	# true_traj = npzfile['true_traj'][:, :s2:n, :]
	# trunc_traj = npzfile['trunc_traj'][:, :s2:n, :]
	# dydt_traj, Bxy_traj = npzfile['dydt_traj'][:, :s2:n, :], npzfile['Bxy_traj'][:, :s2:n, :]
	# Bxy_traj = Bxy_traj*output_scale + output_mean


	lstm_model = lstm_pred(trunc_dim, s, hid_units_x, hid_units_B, savepath=sp)
	lstm_model.compile()
	
	## run predictions
	# Yp, B_pred, dydt_pred = lstm_model.predict(test_inputs, steps, dynamics_trunc=phys_trunc_model.vec_dynamics, 
	# 												loadWeights=weights, dt=dt,
	# 												input_mean=input_mean, input_scale=input_scale,
	# 												true_traj=true_traj)
	
	## plot results for a single test case
	Tspan = 50
	t_Y = np.linspace(.005, Tspan, int(Tspan/.005))
	
	# construct true trajectory
	Y1 = npzfile['inputs'][:Tspan]*input_scale + input_mean
	Y2 = npzfile['true_traj'][:Tspan, :100, :]
	Y = np.concatenate((Y1, Y2), axis=1)

	Y = np.split(Y, Tspan, axis=0)
	Y = [np.squeeze(y, axis=0) for y in Y]
	Y = np.concatenate(Y, axis=0)

	a_mag = np.sqrt(Y[:, :3]**2 + Y[:, 3:]**2)/289


	# predictions (magnitude) based on projected dynamics
	steps = 200
	t_Yt = np.arange(Tspan) + .5 + steps*.005
	Yt = npzfile['trunc_traj'][:Tspan, steps - 1, :]
	a_mag_t = np.sqrt(Yt[:, :3]**2 + Yt[:, 3:]**2)/289


	# predictions from LSTM (data-assisted)
	test_inputs = npzfile['inputs'][:Tspan]
	Yp, *_ = lstm_model.predict(test_inputs, steps, dynamics_trunc=phys_trunc_model.vec_dynamics, 
									loadWeights=weights, dt=dt, input_mean=input_mean, input_scale=input_scale)
	a_mag_p = np.sqrt(Yp[:, steps - 1, :3]**2 + Yp[:, steps - 1, 3:]**2)/289


	# predictions from LSTM (purely data driven)
	lstm_data = lstm_series(trunc_dim, s, steps, dt, hid_units_x, hid_units_B, savepath=sp)
	lstm_data.K_setup(phys_trunc_model, input_mean, input_scale)
	lstm_data.compile()
	lstm_data.model.load_weights('./logs/lstm_seq2seq_5/step50/weights.best.hdf5')
	Yd = lstm_data.Y_eval([test_inputs])
	Yd = np.swapaxes(np.squeeze(np.array(Yd)), 0, 1)
	a_mag_d = np.sqrt(Yd[:, steps - 1, :3]**2 + Yd[:, steps - 1, 3:]**2)/289


	###
	###	3*3 subplots
	###
	plt.rc('axes', linewidth=1.5)
	plt.rc('text', usetex=True)
	plt.rc('text.latex', preamble=r'\usepackage{cmbright}')
	plt.rc('xtick', labelsize=6)
	plt.rc('ytick', labelsize=6)

	t_win = [20, 50]
	x_ticks = [20, 30, 40, 50]

	cf, ce = ['#f97306', 'b', 'g'], ['#f97306', 'b', 'g']	# color settings
	
	f, ax = plt.subplots(3, 3)

	## magnitude ##
	ax[0, 0].plot(t_Y, a_mag[:, 0], 'k-', linewidth=1)
	ax[0, 0].scatter(t_Yt, a_mag_t[:, 0], s=5, marker='^', edgecolors=ce[0], facecolors=cf[0], linewidths=.75)
	ax[0, 0].scatter(t_Yt, a_mag_p[:, 0], s=5, marker='o', edgecolors=ce[1], facecolors=cf[1], linewidths=.75)
	ax[0, 0].scatter(t_Yt, a_mag_d[:, 0], s=5, marker='s', edgecolors=ce[2], facecolors=cf[2], linewidths=.75)
	ax[0, 0].add_patch(patches.Rectangle((38.4, -3), 4.3, 6, color='r', alpha=0.15, lw=0))
	ax[0, 0].add_patch(patches.Rectangle((47.4, -3), 8.5, 6, color='r', alpha=0.15, lw=0))
	ax[0, 0].set_xticks(x_ticks)
	ax[0, 0].set_yticks([0, .2, .4])
	ax[0, 0].set_xlim(t_win)
	ax[0, 0].set_ylim([0, .4])
	ax[0, 0].set_title('modulus', fontsize=8)
	ax[0, 0].set_ylabel('mode [0, 4]', size=8)

	ax[1, 0].plot(t_Y, a_mag[:, 1], 'k-', linewidth=1)
	ax[1, 0].scatter(t_Yt, a_mag_t[:, 1], s=5, marker='^', edgecolors=ce[0], facecolors=cf[0], linewidths=.75)
	ax[1, 0].scatter(t_Yt, a_mag_p[:, 1], s=5, marker='o', edgecolors=ce[1], facecolors=cf[1], linewidths=.75)
	ax[1, 0].scatter(t_Yt, a_mag_d[:, 1], s=5, marker='s', edgecolors=ce[2], facecolors=cf[2], linewidths=.75)
	ax[1, 0].axhline(y=.4, xmin=0, xmax=50, c='r', ls=':', lw=.5)
	ax[1, 0].add_patch(patches.Rectangle((38.4, -3), 4.3, 6, color='r', alpha=0.15, lw=0))
	ax[1, 0].add_patch(patches.Rectangle((47.4, -3), 8.5, 6, color='r', alpha=0.15, lw=0))
	ax[1, 0].set_xticks(x_ticks)
	ax[1, 0].set_yticks([.2, .5, .8])
	ax[1, 0].set_xlim(t_win)
	ax[1, 0].set_ylim([.2, .8])
	ax[1, 0].set_ylabel('mode [1, 0]', size=8)

	ax[2, 0].plot(t_Y, a_mag[:, 2], 'k-', linewidth=1)
	ax[2, 0].scatter(t_Yt, a_mag_t[:, 2], s=5, marker='^', edgecolors=ce[0], facecolors=cf[0], linewidths=.75)
	ax[2, 0].scatter(t_Yt, a_mag_p[:, 2], s=5, marker='o', edgecolors=ce[1], facecolors=cf[1], linewidths=.75)
	ax[2, 0].scatter(t_Yt, a_mag_d[:, 2], s=5, marker='s', edgecolors=ce[2], facecolors=cf[2], linewidths=.75)
	ax[2, 0].add_patch(patches.Rectangle((38.4, -3), 4.3, 6, color='r', alpha=0.15, lw=0))
	ax[2, 0].add_patch(patches.Rectangle((47.4, -3), 8.5, 6, color='r', alpha=0.15, lw=0))
	ax[2, 0].set_xticks(x_ticks)
	ax[2, 0].set_yticks([0, .2, .4])
	ax[2, 0].set_xlim(t_win)
	ax[2, 0].set_ylim([0, .4])
	ax[2, 0].set_xlabel('time', size=8)
	ax[2, 0].set_ylabel('mode [1, 4]', size=8)

	
	## real part ##

	Y, Yt, Yp, Yd = Y/289, Yt/289, Yp/289, Yd/289

	ax[0, 1].plot(t_Y, Y[:, 0], 'k-', linewidth=1)
	ax[0, 1].scatter(t_Yt, Yt[:, 0], s=5, marker='^', edgecolors=ce[0], facecolors=cf[0], linewidths=.75)
	ax[0, 1].scatter(t_Yt, Yp[:, steps - 1, 0], s=5, marker='o', edgecolors=ce[1], facecolors=cf[1], linewidths=.75)
	ax[0, 1].scatter(t_Yt, Yd[:, steps - 1, 0], s=5, marker='s', edgecolors=ce[2], facecolors=cf[2], linewidths=.75)
	ax[0, 1].add_patch(patches.Rectangle((38.4, -3), 4.3, 6, color='r', alpha=0.15, lw=0))
	ax[0, 1].add_patch(patches.Rectangle((47.4, -3), 8.5, 6, color='r', alpha=0.15, lw=0))
	ax[0, 1].set_xticks(x_ticks)
	ax[0, 1].set_xlim(t_win)
	ax[0, 1].set_yticks([-.1, 0, .1])
	ax[0, 1].set_ylim([-.1, .1])
	ax[0, 1].set_title('real part', fontsize=8)

	ax[1, 1].plot(t_Y, Y[:, 1], 'k-', linewidth=1)
	ax[1, 1].scatter(t_Yt, Yt[:, 1], s=5, marker='^', edgecolors=ce[0], facecolors=cf[0], linewidths=.75)
	ax[1, 1].scatter(t_Yt, Yp[:, steps - 1, 1], s=5, marker='o', edgecolors=ce[1], facecolors=cf[1], linewidths=.75)
	ax[1, 1].scatter(t_Yt, Yd[:, steps - 1, 1], s=5, marker='s', edgecolors=ce[2], facecolors=cf[2], linewidths=.75)
	ax[1, 1].add_patch(patches.Rectangle((38.4, -3), 4.3, 6, color='r', alpha=0.15, lw=0))
	ax[1, 1].add_patch(patches.Rectangle((47.4, -3), 8.5, 6, color='r', alpha=0.15, lw=0))
	ax[1, 1].set_xticks(x_ticks)
	ax[1, 1].set_xlim(t_win)
	ax[1, 1].set_yticks([-.6, 0, .6])
	ax[1, 1].set_ylim([-.6, .6])

	ax[2, 1].plot(t_Y, Y[:, 2], 'k-', linewidth=1)
	ax[2, 1].scatter(t_Yt, Yt[:, 2], s=5, marker='^', edgecolors=ce[0], facecolors=cf[0], linewidths=.75)
	ax[2, 1].scatter(t_Yt, Yp[:, steps - 1, 2], s=5, marker='o', edgecolors=ce[1], facecolors=cf[1], linewidths=.75)
	ax[2, 1].scatter(t_Yt, Yd[:, steps - 1, 2], s=5, marker='s', edgecolors=ce[2], facecolors=cf[2], linewidths=.75)
	ax[2, 1].add_patch(patches.Rectangle((38.4, -3), 4.3, 6, color='r', alpha=0.15, lw=0))
	ax[2, 1].add_patch(patches.Rectangle((47.4, -3), 8.5, 6, color='r', alpha=0.15, lw=0))
	ax[2, 1].set_xticks(x_ticks)
	ax[2, 1].set_xlim(t_win)
	ax[2, 1].set_yticks([-.3, 0, .3])
	ax[2, 1].set_ylim([-.3, .3])
	ax[2, 1].set_xlabel('time', size=8)

	## imag part ##

	ax[0, 2].plot(t_Y, Y[:, 3], 'k-', linewidth=1)
	ax[0, 2].scatter(t_Yt, Yt[:, 3], s=5, marker='^', edgecolors=ce[0], facecolors=cf[0], linewidths=.75)
	ax[0, 2].scatter(t_Yt, Yp[:, steps - 1, 3], s=5, marker='o', edgecolors=ce[1], facecolors=cf[1], linewidths=.75)
	ax[0, 2].scatter(t_Yt, Yd[:, steps - 1, 3], s=5, marker='s', edgecolors=ce[2], facecolors=cf[2], linewidths=.75)
	ax[0, 2].add_patch(patches.Rectangle((38.4, -3), 4.3, 6, color='r', alpha=0.15, lw=0))
	ax[0, 2].add_patch(patches.Rectangle((47.4, -3), 8.5, 6, color='r', alpha=0.15, lw=0))
	ax[0, 2].set_xticks(x_ticks)
	ax[0, 2].set_xlim(t_win)
	ax[0, 2].set_yticks([-.4, -.2, 0])
	ax[0, 2].set_ylim([-.4, 0])
	ax[0, 2].set_title('imag part', fontsize=8)

	ax[1, 2].plot(t_Y, Y[:, 4], 'k-', linewidth=1)
	ax[1, 2].scatter(t_Yt, Yt[:, 4], s=5, marker='^', edgecolors=ce[0], facecolors=cf[0], linewidths=.75)
	ax[1, 2].scatter(t_Yt, Yp[:, steps - 1, 4], s=5, marker='o', edgecolors=ce[1], facecolors=cf[1], linewidths=.75)
	ax[1, 2].scatter(t_Yt, Yd[:, steps - 1, 4], s=5, marker='s', edgecolors=ce[2], facecolors=cf[2], linewidths=.75)
	ax[1, 2].add_patch(patches.Rectangle((38.4, -3), 4.3, 6, color='r', alpha=0.15, lw=0))
	ax[1, 2].add_patch(patches.Rectangle((47.4, -3), 8.5, 6, color='r', alpha=0.15, lw=0))
	ax[1, 2].set_xticks(x_ticks)
	ax[1, 2].set_xlim(t_win)
	ax[1, 2].set_yticks([-.6, -.4, -.2])
	ax[1, 2].set_ylim([-.6, -.2])

	ax[2, 2].plot(t_Y, Y[:, 5], 'k-', linewidth=1)
	ax[2, 2].scatter(t_Yt, Yt[:, 5], s=5, marker='^', edgecolors=ce[0], facecolors=cf[0], linewidths=.75)
	ax[2, 2].scatter(t_Yt, Yp[:, steps - 1, 5], s=5, marker='o', edgecolors=ce[1], facecolors=cf[1], linewidths=.75)
	ax[2, 2].scatter(t_Yt, Yd[:, steps - 1, 5], s=5, marker='s', edgecolors=ce[2], facecolors=cf[2], linewidths=.75)
	ax[2, 2].add_patch(patches.Rectangle((38.4, -3), 4.3, 6, color='r', alpha=0.15, lw=0))
	ax[2, 2].add_patch(patches.Rectangle((47.4, -3), 8.5, 6, color='r', alpha=0.15, lw=0))
	ax[2, 2].set_xticks(x_ticks)
	ax[2, 2].set_xlim(t_win)
	ax[2, 2].set_yticks([-.4, -.2, 0])
	ax[2, 2].set_ylim([-.4, 0])
	ax[2, 2].set_xlabel('time', size=8)

	plt.tight_layout()
	plt.savefig('./lead1.0.png', dpi=350)
	###


if __name__ == '__main__':
	main()

