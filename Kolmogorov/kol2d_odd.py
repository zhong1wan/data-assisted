import numpy as np
import matplotlib.pyplot as plt

"""
Implements a pseudo-spectral solver for the Kolmogorov flow

"""

np.seterr(divide='ignore', invalid='ignore')

class Kol2D_odd(object):
	"""
	N: resolution of grid used; number of grids (single direction) = (2N+1)
	Re: Reynolds number
	n: wavernumber of external forcing in x direction

	wave numbers are arranged such that 0 is in the center	
	"""
	def __init__(self, Re=40, n=4, N=6):
		
		self.N = N
		self.grid_setup(N)
		self.grids = 2*N + 1
		self.Re = Re
		self.fx = np.fft.fftshift(np.fft.fft2(np.sin(n*self.yy)))

		# aa = np.fft.ifft2(np.fft.ifftshift(self.fx))
		# print(aa.real)
		# print(aa.imag)

	def grid_setup(self,N):
		
		# physical grid
		x = np.linspace(0, 2*np.pi, 2*N+2)
		x = x[:-1]
		self.xx, self.yy = np.meshgrid(x,x)

		# wavenumbers
		k = np.arange(-N, N+1)
		self.kk1, self.kk2 = np.meshgrid(k,k)
		self.kk = self.kk1**2 + self.kk2**2

		# parameters for divergence-free projection (Fourier domain)
		self.p1 = self.kk2**2/self.kk
		self.p2 = -self.kk1*self.kk2/self.kk
		self.p3 = self.kk1**2/self.kk

		# differentiation (Fourier domain)
		self.ddx = 1j*self.kk1
		self.ddy = 1j*self.kk2

		# matrix for converting u,v to a and vice versa: u = a*pu, v = a*pv
		self.pu = self.kk2/np.sqrt(self.kk)
		self.pu[self.N, self.N] = 0
		self.pv = -self.kk1/np.sqrt(self.kk)
		self.pv[self.N, self.N] = 0


	def proj_DF(self,fx_h,fy_h):	# divergence free projection
		
		ux_h = self.p1*fx_h + self.p2*fy_h
		uy_h = self.p2*fx_h + self.p3*fy_h

		# boundary conditions
		if fx_h.ndim == 2:
			ux_h[self.N, self.N] = 0
			uy_h[self.N, self.N] = 0

		elif fx_h.ndim == 3:
			ux_h[:, self.N, self.N] = 0
			uy_h[:, self.N, self.N] = 0

		return ux_h,uy_h


	def uv2a(self, u_h, v_h):	# unified Fourier coefficients a(x,t)
		
		a_h = u_h/self.pu
		a_v = v_h/self.pv

		if u_h.ndim == 2:
			a_h[self.N] = a_v[self.N]
			a_h[self.N, self.N] = 0
		elif u_h.ndim == 3:
			a_h[:, self.N, :] = a_v[:, self.N, :]
			a_h[:, self.N, self.N] = 0

		return a_h


	def a2uv(self, a_h):

		return a_h*self.pu, a_h*self.pv


	def vort(self,u_h,v_h):		# calculate vorticity
		
		return self.ddy*u_h - self.ddx*v_h


	def dissip(self,u_h,v_h):	# calculate dissipation
		
		w_h = self.vort(u_h,v_h)
		D = np.sum(w_h*w_h.conjugate(),axis=(-1,-2))
		D = np.squeeze(D)/self.Re/self.grids**4

		return D.real


	def dynamics(self,u_h,v_h):

		fx_h = -self.ddx*self.aap(u_h,u_h) - self.ddy*self.aap(u_h,v_h) + self.fx
		fy_h = -self.ddx*self.aap(u_h,v_h) - self.ddy*self.aap(v_h,v_h)

		Pfx_h,Pfy_h = self.proj_DF(fx_h,fy_h)

		du_h = -self.kk*u_h/self.Re + Pfx_h
		dv_h = -self.kk*v_h/self.Re + Pfy_h
	
		return du_h,dv_h


	def dynamics_a(self, a_h):
		
		u_h, v_h = self.a2uv(a_h)
		du_h, dv_h = self.dynamics(u_h, v_h)
		da_h = self.uv2a(du_h, dv_h)

		return da_h


	def random_field(self,A_std,A_mag,c1=0,c2=3):

		'''
			generate a random field whose energy is normally distributed
			in Fourier domain centered at wavenumber (c1,c2) with random phase
		'''
		
		A = A_mag*4*self.grids**2*np.exp(-(self.kk1-c1)**2-
				(self.kk2-c2)**2/2/A_std**2)/np.sqrt(2*np.pi*A_std**2)
		u_h = A*np.exp(1j*2*np.pi*np.random.rand(self.grids, self.grids))
		v_h = A*np.exp(1j*2*np.pi*np.random.rand(self.grids, self.grids))

		u = np.fft.irfft2(np.fft.ifftshift(u_h), s=u_h.shape[-2:])
		v = np.fft.irfft2(np.fft.ifftshift(v_h), s=v_h.shape[-2:])

		u_h = np.fft.fftshift(np.fft.fft2(u))
		v_h = np.fft.fftshift(np.fft.fft2(v))

		u_h,v_h = self.proj_DF(u_h,v_h)

		return u_h, v_h


	def plot_vorticity(self,u_h,v_h,wmax=None,subplot=False):
		
		w_h = self.vort(u_h,v_h)
		w = np.fft.ifft2(np.fft.ifftshift(w_h))
		w = w.real

		# calculate color axis limit if not specified
		if not wmax:
			wmax = np.ceil(np.abs(w).max())
		wmin = -wmax

		## plot with image
		tick_loc = np.array([0,.5,1,1.5,2])*np.pi
		tick_label = ['0','$\pi/2$','$\pi$','$3\pi/2$','$2\pi$']
		im = plt.imshow(w, cmap='RdBu', vmin=wmin, vmax=wmax,
					extent=[0,2*np.pi,0,2*np.pi],
					interpolation='spline36',origin='lower')
		plt.xticks(tick_loc,tick_label)
		plt.yticks(tick_loc,tick_label)
		if subplot:
			plt.colorbar(im,fraction=.046,pad=.04)
			plt.tight_layout()
		else:
			plt.colorbar()


	def plot_quiver(self,u_h,v_h):
		
		u = np.fft.ifft2(np.fft.ifftshift(u_h)).real
		v = np.fft.ifft2(np.fft.ifftshift(v_h)).real

		Q = plt.quiver(self.xx, self.yy, u, v, units='width')

		tick_loc = np.array([0,.5,1,1.5,2])*np.pi
		tick_label = ['0','$\pi/2$','$\pi$','$3\pi/2$','$2\pi$']

		plt.xticks(tick_loc,tick_label)
		plt.yticks(tick_loc,tick_label)


	def aap(self,f1,f2):		# anti-aliased product

		ndim = f1.ndim
		assert ndim < 4, 'input dimensions is greater than 3.'
		if ndim == 2:
			f1_h, f2_h = np.expand_dims(f1, axis=0).copy(), np.expand_dims(f2, axis=0).copy()
		elif ndim == 3:
			f1_h, f2_h = f1.copy(), f2.copy()
		
		sz2 = 4*self.N + 1
		ff1_h = np.zeros((f1_h.shape[0], sz2, sz2), dtype=np.complex128)
		ff2_h = np.zeros((f1_h.shape[0], sz2, sz2), dtype=np.complex128)

		idx1, idx2 = self.N, 3*self.N + 1
		ff1_h[:, idx1:idx2, idx1:idx2] = f1_h
		ff2_h[:, idx1:idx2, idx1:idx2] = f2_h

		ff1 = np.fft.irfft2(np.fft.ifftshift(ff1_h), s=ff1_h.shape[-2:])
		ff2 = np.fft.irfft2(np.fft.ifftshift(ff2_h), s=ff1_h.shape[-2:])  		# must take real part or use irfft2

		pp_h = (sz2/self.grids)**2*np.fft.fft2(ff1*ff2)
		pp_h = np.fft.fftshift(pp_h)

		p_h = pp_h[:, idx1:idx2, idx1:idx2]

		if ndim == 2:
			p_h = p_h[0,:,:]

		return p_h


def main():
	
	kol2d = Kol2D_odd(Re=40, n=4, N=16)	
	
	# initial condition: random
	u0h,v0h = kol2d.random_field(A_std=2, A_mag=.001)

	dt = .005		# integration time step
	dTr = 1		# recording time step

	# get past transients
	T = 100
	t = np.arange(dt,T,dt)
	print('Integrating past initial transients: ')
	for tt in t:
		du0h,dv0h = kol2d.dynamics(u0h,v0h)
		u0h += dt*du0h
		v0h += dt*dv0h
	u0h_r,v0h_r = u0h.copy(),v0h.copy()

	# data generation
	T = 10
	t1 = np.arange(0,T,dTr)
	t2 = np.arange(0,dTr,dt) 
	Tr = list()
	Uh,Vh = np.expand_dims(u0h_r,axis=0),np.expand_dims(v0h_r,axis=0)
	# dUh,dVh = np.expand_dims(du0h,axis=0),np.expand_dims(dv0h,axis=0)
	D = []

	print('Recording state and dynamics at freq = %f Hz'%(1/dTr))
	for tt1 in t1:
		
		for tt2 in t2:
			du0h,dv0h = kol2d.dynamics(u0h,v0h)
			u0h_r,v0h_r = u0h.copy(),v0h.copy()		# copy for recording purpose
			u0h += dt*du0h
			v0h += dt*dv0h
			d = kol2d.dissip(u0h,v0h)
			
		Tr.append(tt1)
		Uh = np.r_[Uh,[u0h_r]]		# record state and dynamics
		Vh = np.r_[Vh,[v0h_r]]
		# dUh = np.r_[dUh,[du0h]]
		# dVh = np.r_[dVh,[dv0h]]
		D.append(d)

	Ah = kol2d.uv2a(Uh[1:], Vh[1:])
	print(Ah.shape)

	# # save state and dynamics to file
	# np.savez('data/traj_pt100k_dT1.npz', Ah=Ah)

	# plot final state
	plt.figure()
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif', size=12)
	kol2d.plot_vorticity(u0h,v0h)
	
	# plot dissipation
	D = np.array(D)
	plt.figure()
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif', size=12)
	plt.plot(t1,D,'k-',linewidth=1.5)
	plt.xlim(0,T)

	plt.show()


if __name__ == '__main__':
	main()

