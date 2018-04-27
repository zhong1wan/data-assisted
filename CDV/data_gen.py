import numpy as np
from matplotlib import pyplot as plt

class CdV(object):

	x1s = .95
	x4s = -.76095
	C = .1
	beta = 1.25
	gamma = .2
	b = .5

	m = np.array([1,2])
	alpha = 8*np.sqrt(2)*m**2*(b**2+m**2-1)/np.pi/(4*m**2-1)/(b**2+m**2)
	beta = beta*b**2/(b**2+m**2)
	delta = 64*np.sqrt(2)*(b**2-m**2+1)/15/np.pi/(b**2+m**2)
	gamma_m = gamma*4*np.sqrt(2)*m**3*b/np.pi/(4*m**2-1)/(b**2+m**2)
	gamma_m_star = gamma*4*np.sqrt(2)*m*b/np.pi/(4*m**2-1)
	epsilon = 16*np.sqrt(2)/5/np.pi

	# linear part of the operator
	L = np.zeros((6,6))
	L[0,0],L[2,0] = -C,gamma_m_star[0]
	L[1,1],L[2,1] = -C,beta[0]
	L[0,2],L[1,2],L[2,2] = -gamma_m[0],-beta[0],-C
	L[3,3],L[5,3] = -C,gamma_m_star[1]
	L[4,4],L[5,4] = -C,beta[1]
	L[3,5],L[4,5],L[5,5] = -gamma_m[1],-beta[1],-C

	b = np.zeros((1,6))
	b[:,0],b[:,3] = C*x1s,C*x4s


	@staticmethod
	def NL(x):

		assert len(x.shape) == 2, 'Input needs to be a two-dimensional array.'
		Nx = np.zeros(x.shape)

		Nx[:,1] = -CdV.alpha[0]*x[:,0]*x[:,2] - CdV.delta[0]*x[:,3]*x[:,5]
		Nx[:,2] = CdV.alpha[0]*x[:,0]*x[:,1] + CdV.delta[0]*x[:,3]*x[:,4]
		Nx[:,3] = CdV.epsilon*(x[:,1]*x[:,5] - x[:,2]*x[:,4])
		Nx[:,4] = -CdV.alpha[1]*x[:,0]*x[:,5] - CdV.delta[1]*x[:,2]*x[:,3]
		Nx[:,5] = CdV.alpha[1]*x[:,0]*x[:,4] + CdV.delta[1]*x[:,3]*x[:,1]

		return Nx


	def dynamics(x):

		assert len(x.shape) == 2, 'Input needs to be a two-dimensional array.'
		# dxdt = np.zeros(x.shape)

		# dxdt[:,0] = CdV.gamma_m_star[0]*x[:,2] - CdV.C*(x[:,0] - CdV.x1s)
		# dxdt[:,1] = -(CdV.alpha[0]*x[:,0] - CdV.beta[0])*x[:,2] - CdV.C*x[:,1] - CdV.delta[0]*x[:,3]*x[:,5]
		# dxdt[:,2] = (CdV.alpha[0]*x[:,0] - CdV.beta[0])*x[:,1] - CdV.gamma_m[0]*x[:,0] - CdV.C*x[:,2] + CdV.delta[0]*x[:,3]*x[:,4]
		# dxdt[:,3] = CdV.gamma_m_star[1]*x[:,5] - CdV.C*(x[:,3] - CdV.x4s) + CdV.epsilon*(x[:,1]*x[:,5] - x[:,2]*x[:,4])
		# dxdt[:,4] = -(CdV.alpha[1]*x[:,0] - CdV.beta[1])*x[:,5] - CdV.C*x[:,4] - CdV.delta[1]*x[:,2]*x[:,3]
		# dxdt[:,5] = (CdV.alpha[1]*x[:,0] - CdV.beta[1])*x[:,4] - CdV.gamma_m[1]*x[:,3] - CdV.C*x[:,5] + CdV.delta[1]*x[:,3]*x[:,1]

		dxdt = np.matmul(x,CdV.L) + CdV.NL(x) + CdV.b

		return dxdt


def main():

	x0 = np.random.rand(1,6)
	x0 = np.array([[.11,.22,.33,.44,.55,.66]])
	X = []

	dt,T = .01,2000
	tt = np.arange(0,T,1)

	L = CdV.L
	v,w = np.linalg.eig(L)

	print('eigenvalues:',v)
	Linv = np.linalg.inv(L)
	x_linsol = np.matmul(CdV.b,Linv)
	print('Linear solution:',x_linsol)
	print('NL of that:', CdV.NL(x_linsol))

	# initial spin-up
	for t in np.arange(0,2000,dt):
		dxdt = CdV.dynamics(x0)
		x0 = x0 + dt*dxdt

	# formal stepping
	for t in range(T):
		for t2 in np.arange(0,1,dt):
			dxdt = CdV.dynamics(x0)
			x0 = x0 + dt*dxdt
		X.append(np.squeeze(x0))

	X = np.array(X)
	print(X.shape)

	## save data
	# np.savez('data/traj_pt10k_dt1.npz',X=X)

	## plot trajectory in (x1, x4) plane
	plt.figure(figsize=(2, 2))
	plt.scatter(X[:,0], X[:,3], s=5, marker='o', facecolor='k', edgecolor='none', )
	plt.xlim([0.7, 1])
	plt.ylim([-0.8, -0.1])

	plt.show()


if __name__ == '__main__':
	main()
