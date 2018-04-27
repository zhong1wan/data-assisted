import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.stats import gaussian_kde
from collections import Counter

from kol2d_odd import Kol2D_odd
from kol2d_trunc import Kol2D_trunc

def main():
	
	datafile = 'data/traj_pt100k_dT1.npz'
	npz = np.load(datafile)

	kol2d = Kol2D_odd(Re=40, n=4, N=8)
	wn =[[0, 4], [1, 0], [1, 4]]
	kol2d_t = Kol2D_trunc(N=8, kwn=wn)

	Ah = npz['Ah']
	Uh, Vh = kol2d.a2uv(Ah)
	D = kol2d.dissip(Uh, Vh)
	a10 = Ah[:, 8, 9]
	a10_mag = np.absolute(a10)/289

	plt.rc('axes', linewidth=1.5)
	plt.rc('text', usetex=True)
	plt.rc('text.latex', preamble=r'\usepackage{cmbright}')
	plt.rc('xtick', labelsize=8)
	plt.rc('ytick', labelsize=8)

	
	# 
	fig = plt.figure()

	ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=4, rowspan=2)
	ax1.plot(D, 'k-', lw=.75)
	ax1.set_ylabel('D', size=8)
	ax1.set_xlim([0, 1000])
	ax1.set_ylim([-.4, .4])
	ax1.set_yticks([0, .2, .4])
	ax1.text(-0.11, 1, '(A)', transform=ax1.transAxes, size=10, weight='bold')
	ax1.axhline(linewidth=1.5, color='k')
	ax1.yaxis.set_label_coords(-.075, .76)

	ax12 = ax1.twinx()
	# ax12 = plt.subplot2grid((4, 4), (1, 0), colspan=4, sharex=ax1)
	ax12.plot(a10_mag, color='b', lw=.75)
	ax12.set_ylim([0, 2])
	ax12.set_yticks([0, .5, 1])
	ax12.set_ylabel('$|a(1,0)|$', color='b', size=8)
	ax12.set_xlabel('$t$')
	ax12.set_xlim([0, 1000])
	for t in ax12.get_yticklabels():
		t.set_color('b')

	ax12.yaxis.set_label_coords(1.075, .26)
	ax12.add_patch(patches.Rectangle((95, -.1), 20, 2.2, color='r', alpha=0.15, lw=0))
	ax12.add_patch(patches.Rectangle((323, -.1), 22, 2.2, color='r', alpha=0.15, lw=0))
	ax12.add_patch(patches.Rectangle((510, -.1), 24, 2.2, color='r', alpha=0.15, lw=0))
	ax12.add_patch(patches.Rectangle((797, -.1), 15, 2.2, color='r', alpha=0.15, lw=0))
	ax12.add_patch(patches.Rectangle((824, -.1), 27, 2.2, color='r', alpha=0.15, lw=0))

	# energy spectrum
	r = np.absolute(Ah)
	spec = np.sqrt(np.mean(r**2, axis=0))/289
	bb = kol2d_t.mat2vec_by_wn(spec**2)
	print(spec**2)
	print(kol2d.kk1)
	print(2*np.sum(bb)/np.sum(spec**2))
	# plt.figure()
	# plt.rc('text', usetex=True)
	# plt.rc('font', family='serif', size=12)
	ax2 = plt.subplot2grid((4, 4), (2, 0), colspan=2, rowspan=2)
	p2 = ax2.pcolor(kol2d.kk1, kol2d.kk2, spec, cmap='Reds', vmin=0, vmax=0.8)
	ax2.set_xticks([-8, -4, 0, 4, 8])
	ax2.set_yticks([-8, -4, 0, 4, 8])
	ax2.set_title('$\sqrt{E[|a(\mathbf{k})|^2]}$', fontsize=8)
	plt.colorbar(p2, ticks=[0, .2, .4, .6, .8], fraction=.046, pad=.04)
	ax2.text(-0.25, 1.1, '(B)', transform=ax2.transAxes, size=10, weight='bold')
	ax2.set_xlabel('$k_x$')
	ax2.set_ylabel('$k_y$')

	# vorticity snapshot
	ax3 = plt.subplot2grid((4, 4), (2, 2), colspan=2, rowspan=2)
	
	idx = 90
	w = kol2d_t.vort(Uh[idx], Vh[idx])
	w = np.fft.ifft2(np.fft.ifftshift(w)).real

	tick_loc = np.array([0,1,2])*np.pi
	tick_label = ['0','$\pi$','$2\pi$']
	p3 = ax3.imshow(w, cmap='RdBu', vmin=-8, vmax=8,
				extent=[0,2*np.pi,0,2*np.pi], interpolation='spline36')
	ax3.set_xticks(tick_loc)
	ax3.set_xticklabels(tick_label)
	ax3.set_yticks(tick_loc)
	ax3.set_yticklabels(tick_label)
	ax3.set_title('$\omega(t = 90)$', fontsize=8)
	ax3.set_xlabel('$x$')
	ax3.set_ylabel('$y$')
	plt.colorbar(p3, ticks=[-8, -4, 0, 4, 8], fraction=.046, pad=.04)
	ax3.text(-0.29, 1.1, '(C)', transform=ax3.transAxes, size=10, weight='bold')
	
	plt.tight_layout()

	plt.savefig('./test.png', dpi=350)
	# plt.show()

if __name__ == '__main__':
	main()

