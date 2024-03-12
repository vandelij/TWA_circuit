import numpy as np
import matplotlib.pyplot as plt

from TWA_Design_Tools import TWA_Design_Toolkit
    
# Set up my case 
f = 53e6 # 96 MHz
kz0 = 4
nstraps = 12
num_points = int(1e5)
myTWA = TWA_Design_Toolkit(num_straps=nstraps, f0=f, k_par_max=kz0)
myTWA.get_key_params(True)
myTWA.set_strap_width(w=0.01)
myTWA.plot_J_of_z(J0=1, zmin=-2, zmax=2, num_points=num_points)
myTWA.plot_J_k_versus_k(zmin=-2, zmax=2, num_points=num_points, J0=1)