import os
import numpy as np
import pickle

# from tqdm import tqdm

from keras import layers
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from kol2d_trunc import Kol2D_trunc


class K_dynamics(object):
	"""
	implement known dynamics in keras backend (tensors) and setup
	the necessary contants as well as the helper functions/layers
	
	"""
	def __init__(self, kol2d_t, Ym, Ys):

		self.np_model = kol2d_t 	# numpy model (contains resolution, wavenumber info)
		self.np_model.triad_setup()
		
		# scaling tensors
		self.Ym, self.Ys = K.constant(Ym), K.constant(Ys)

		#
		self.dlayer = layers.core.Lambda(function=self.dynamics, output_shape=(6,), name='dlayer')
		self.rescaleylayer = layers.core.Lambda(function=self.rescale_y, output_shape=(6,), name='rsc_y')
		self.scaleylayer = layers.core.Lambda(function=self.scale_y, output_shape=(6,), name='sc_y')
		self.slicelayer = layers.core.Lambda(function=self.slice, name='slice_last')


	def dynamics(self, y):
		""" y is scaled, returns known/truncated dynamics to the same scale """
		Y = y*self.Ys + self.Ym
		cc, Re, k, N = self.np_model.cc, self.np_model.Re, self.np_model.k, self.np_model.N

		dY_known = []
		dY_known.append(cc[0]*(Y[:,1]*Y[:,5] - Y[:,2]*Y[:,4]) - Y[:,0]/Re*k[0]**2)
		dY_known.append(cc[1]*(Y[:,0]*Y[:,5] - Y[:,2]*Y[:,3]) - Y[:,1]/Re*k[1]**2)
		dY_known.append(cc[2]*(-Y[:,0]*Y[:,4] - Y[:,1]*Y[:,3]) - Y[:,2]/Re*k[2]**2)
		dY_known.append(cc[0]*(-Y[:,1]*Y[:,2] - Y[:,4]*Y[:,5]) - Y[:,3]/Re*k[0]**2 - 0.5*(2*N + 1)**2)
		dY_known.append(cc[1]*(-Y[:,0]*Y[:,2] - Y[:,3]*Y[:,5]) - Y[:,4]/Re*k[1]**2)
		dY_known.append(cc[2]*(Y[:,0]*Y[:,1] - Y[:,3 ]*Y[:,4]) - Y[:,5]/Re*k[2]**2)
		dY_known = K.stack(dY_known, axis=-1)

		return dY_known


	def scale_y(self, y):
		
		return (y - self.Ym)/self.Ys

	def rescale_y(self, y):
		
		return y*self.Ys + self.Ym

	def slice(self, y):
		
		return y[:,-1,:]

	def test_setup(self):
	 	
		inputs = layers.Input(shape=(6,), name='inputs')
		dy = self.dlayer(inputs)
		self.model = K.function(inputs=[inputs], outputs=[dy])


## test the accuracy in the main program
def main():
	
	file = np.load('./data/ktriad_l200_standardized.npz')

	inputs = file['inputs']
	Ym, Ys = file['input_mean'], file['input_scale']

	wn = [[0,4], [1,0], [1,4]]
	kol2d_t = Kol2D_trunc(N=8, kwn=wn)
	
	model = K_dynamics(kol2d_t, Ym, Ys)
	model.test_setup()

	y0 = inputs[:1, 0, :]
	y1 = inputs[:1, 1, :]
	dydt = kol2d_t.vec_dynamics(y0*Ys + Ym)

	dydt2 = model.model([y0])

	print(dydt)
	print(dydt2)


if __name__ == '__main__':
	main()


