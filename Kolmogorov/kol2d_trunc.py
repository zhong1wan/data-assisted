import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from kol2d_odd import Kol2D_odd

"""
implements Kolmogorov dynamics truncated/projected to specified modes

"""


class Kol2D_trunc(Kol2D_odd):
	"""docstring for Kol2D_trunc"""
	def __init__(self, Re=40, n=4, N=8, kwn=None):
		"""
		kwn = list of wave numbers to keep; 
			  valid range (1) -N<=k1<=N, 1<=k2<=N (2) 1<=k1<=N, k2=0

		"""
		super(Kol2D_trunc, self).__init__(Re, n, N)
		self.kwn = kwn
	
		# p0 = matrix to multiply when truncating	
		self.p0 = np.zeros((2*N+1, 2*N+1), dtype=int)
		for k in kwn:
			self.p0[k[1]+self.N, k[0]+self.N] = 1 
		self.p0 += np.flipud(np.fliplr(self.p0))


	def trunc_dynamics(self, uh, vh):
		
		uh_t, vh_t = self.p0*uh, self.p0*vh
		duh_t, dvh_t = self.dynamics(uh_t, vh_t)

		return duh_t*self.p0, dvh_t*self.p0


	def trunc_dynamics_a(self, ah):
		
		ah_t = self.p0*ah
		dah_t = self.dynamics_a(ah_t)

		return self.p0*dah_t


	def mat2vec_by_wn(self, a, sep=True):
		""" 
		Takes a matrix and puts the Fourier coefficients corresponding to
		self.kwn into a vector; this is a truncation procedure that doesn't
		require multiplying self.p0

		sep: whether to separate real and imag parts 

		"""
		a_vec = []
		for k in self.kwn:
			a_vec.append(a[..., k[1]+self.N, k[0]+self.N])

		a_vec = np.array(a_vec)
		if a_vec.ndim > 1:
			a_vec = np.swapaxes(a_vec, 0, -1)

		if sep:
			a_vec = np.concatenate((a_vec.real, a_vec.imag), axis=-1)

		return a_vec


	def vec2mat_by_wn(self, a_vec, sep=True):
		""" 
		Takes vectors of Fourier coefficients corresponding to
		self.kwn and converts to matrix (inverse of mat2vec_by_wn)

		sep: whether a_vec has been separated into real and imag parts 

		"""
		a = np.zeros(a_vec.shape[:-1] + (2*self.N+1, 2*self.N+1), dtype=np.complex128)
		for i, k in enumerate(self.kwn):
			a[..., k[1]+self.N, k[0]+self.N] = a_vec[..., i]

		if sep:
			for i, k in enumerate(self.kwn):
				a[..., k[1]+self.N, k[0]+self.N] += 1j*a_vec[..., i + len(self.kwn)]

		a -= np.conj(a[..., ::-1, ::-1])		# here a(i,j) = -a(-i,-j).conj; different from u, v

		return a


	def vec_dynamics(self, a_vec):		# this dynamics is inherently truncated
		
		ah = self.vec2mat_by_wn(a_vec)
		dah = self.dynamics_a(ah)
		
		return self.mat2vec_by_wn(dah)


	def triad_setup(self):
		"""se upt the constants required for triad dynamics"""
		assert len(self.kwn) == 3, 'Number of modes is not 3.'

		kk = np.array(self.kwn)
		self.k = np.linalg.norm(kk, axis=-1)
		
		def c(p, q, r):
	 		coeff1 = (p[0]*q[1] - p[1]*q[0])*(r[0]*q[0] + r[1]*q[1])/np.prod(self.k)
	 		coeff2 = (q[0]*p[1] - q[1]*p[0])*(r[0]*p[0] + r[1]*p[1])/np.prod(self.k) 
	 		return (coeff1 + coeff2)/(2*self.N + 1)**2

		self.cc = [c(-kk[1], kk[2], kk[0]), c(-kk[0], kk[2], kk[1]), c(kk[0], kk[1], kk[2])]


	def vec_dynamics_triad(self, a_vec):
	 	"""
		only valid for kwn = [k1, k2, k3] such that k1 + k2 = k3

	 	"""
	 	self.triad_setup()
	 	da_vec = np.zeros(a_vec.shape)
	 	
	 	da_vec[:,0] = self.cc[0]*(a_vec[:,1]*a_vec[:,5] - a_vec[:,2]*a_vec[:,4]) - a_vec[:,0]/self.Re*self.k[0]**2
	 	da_vec[:,3] = self.cc[0]*(-a_vec[:,1]*a_vec[:,2] - a_vec[:,4]*a_vec[:,5]) - a_vec[:,3]/self.Re*self.k[0]**2 - 0.5*(2*self.N + 1)**2

	 	da_vec[:,1] = self.cc[1]*(a_vec[:,0]*a_vec[:,5] - a_vec[:,2]*a_vec[:,3]) - a_vec[:,1]/self.Re*self.k[1]**2
	 	da_vec[:,4] = self.cc[1]*(-a_vec[:,0]*a_vec[:,2] - a_vec[:,3]*a_vec[:,5]) - a_vec[:,4]/self.Re*self.k[1]**2

	 	da_vec[:,2] = self.cc[2]*(-a_vec[:,0]*a_vec[:,4] - a_vec[:,1]*a_vec[:,3]) - a_vec[:,2]/self.Re*self.k[2]**2
	 	da_vec[:,5] = self.cc[2]*(a_vec[:,0]*a_vec[:,1] - a_vec[:,3 ]*a_vec[:,4]) - a_vec[:,5]/self.Re*self.k[2]**2

	 	return da_vec

		
	@staticmethod

	def idx_gen(kt1, kt2=None):
		# generate idx (k1,k2) such that |k1| <= kt1, |k2| <= kt2
		if kt2 is None:
			kt2 = kt1

		k1, k2 = np.meshgrid(np.arange(-kt1, kt1+1), np.arange(1,kt2+1))
		k11, k22 = np.meshgrid(np.arange(1, kt1+1), 0)

		k1, k2 = k1.reshape(-1,), k2.reshape(-1,)
		k11, k22 = k11.reshape(-1,), k22.reshape(-1,)
		
		k1 = list(k1) + list(k11)
		k2 = list(k2) + list(k22)

		keep_wn = [list(k) for k in zip(k1,k2)]

		return keep_wn


	def find_wn(wn_list, wn):
		# find the index of each wavenumber in wn in wn_list
		wn_idx = []
		for k in wn:
			for idx,kk in enumerate(wn_list):
				if kk == k:
					wn_idx.append(idx)
					break
		return wn_idx


def main():
	
	keep_wn = [[0,4], [1,0], [1,4]]
	wn_list = Kol2D_trunc.idx_gen(kt1=3, kt2=4)
	wn_idx = Kol2D_trunc.find_wn(wn_list, keep_wn)
	print(wn_list)
	print(wn_idx)

	nb_wn = len(wn_list)
	kol2d_t = Kol2D_trunc(N=8, kwn=wn_list)
	kol2d = Kol2D_odd(N=8)

	datafile = 'data/traj_pt10k_dT1.npz'
	npz = np.load(datafile)
	uh0, vh0 = npz['Uh'][9890], npz['Vh'][9890]
	uht, vht = uh0*kol2d_t.p0, vh0*kol2d_t.p0

	dt = .005
	T = 20
	t = np.arange(0, T, dt)
	A0, At = [], []

	for tt in tqdm(t):

		duh0, dvh0 = kol2d.dynamics(uh0, vh0)
		uh0 += duh0*dt
		vh0 += dvh0*dt
		ah0 = kol2d.uv2a(uh0, vh0)
		A0.append(ah0)

		duht, dvht = kol2d_t.trunc_dynamics(uht, vht)
		uht += duht*dt
		vht += dvht*dt
		aht = kol2d.uv2a(uht, vht)
		At.append(aht)

	# extract interested Fourier coefficients
	A0, At = np.array(A0), np.array(At)
	a0 = kol2d_t.mat2vec_by_wn(A0)
	a0r, a0i = a0[:, :nb_wn], a0[:, -nb_wn:]
	at = kol2d_t.mat2vec_by_wn(At)
	atr, ati = at[:, :nb_wn], at[:, -nb_wn:]

	# plot comparison between true & truncated dynamics
	nb_plots = len(keep_wn)
	f, axarr = plt.subplots(1, 2*nb_plots, figsize=(18,3))
	for i in range(nb_plots):
		axarr[i].plot(t, a0r[:, wn_idx[i]], 'r-.', label='true')
		axarr[i].plot(t, atr[:, wn_idx[i]], 'g-', label='trunc')
		axarr[i].set_xlim([0, T])
		axarr[i].set_title('mode '+np.array_str(np.array(wn_list[wn_idx[i]]))+' real part')
	
	for i in range(nb_plots):
		axarr[i+nb_plots].plot(t, a0i[:, wn_idx[i]], 'r-.', label='true')
		axarr[i+nb_plots].plot(t, ati[:, wn_idx[i]], 'g-', label='trunc')
		axarr[i+nb_plots].set_xlim([0, T])
		axarr[i+nb_plots].set_title('mode '+np.array_str(np.array(keep_wn[i]))+' imag part')
	plt.tight_layout()

	plt.show()


if __name__ == '__main__':
	main()
