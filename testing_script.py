import numpy as np
import matplotlib.pyplot as plt

from TWA_Design_Tools import TWA_Design_Toolkit
    
# Set up my case 
f = 53e6 # 96 MHz
kz0 = 4
nstraps = 91
num_points = int(1e5)
myTWA = TWA_Design_Toolkit(num_straps=nstraps, f0=f, k_par_max=kz0, d_straps=0.25)
myTWA.print_key_params()
myTWA.set_strap_width(w=0.1)
zrange = 100
#myTWA.plot_J_of_z(J0=1, zmin=-zrange, zmax=zrange, num_points=num_points)
#myTWA.plot_J_k_versus_k(zmin=-zrange, zmax=zrange, kplotmin=3, kplotmax=5, num_pointsz=num_points, num_pointsk=1000, J0=1)

# S_mat = np.loadtxt('S_mat_real.txt') - 1j*np.loadtxt('S_mat_imag.txt')
#S_mat = np.loadtxt('S_mat_96MHz_real.txt') - 1j*np.loadtxt('S_mat_96MHz_imag.txt')
S_mat = 1j*(np.loadtxt('Smat_testing_R.txt') + 1j*np.loadtxt('Smat_testing_I.txt'))
myTWA.set_Smatrix(S_mat)
myTWA.get_Z_matrix(Z0_port=83.1201)
myTWA.plot_Smat_and_Zmat()
print('C0 = ', myTWA.calculate_C0()/1e-12, ' pF')

# # Test my Smatrix tool 
# # enter specs about the simulation 
# f = 53e6 # Hz
# mu0 = 4*np.pi*10**(-7)
# epsi0 = 8.85418e-12
# D = 0.016
# d = 0.004
# Z0 = (1/(2*np.pi))*np.sqrt(mu0/epsi0)*np.log(D/d)
# print('Z0: ', Z0)
# I = np.identity(smat_size, dtype=complex)
# Zmat = Z0*np.matmul((I + Smat),np.linalg.inv(I - Smat))
# np.abs(Zmat[0,0])

# TODO: you are to solve for the capacitance needed at each strap and the average. 

