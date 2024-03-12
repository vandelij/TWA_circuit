import numpy as np
import matplotlib.pyplot as plt

class TWA_Design_Toolkit:

    def __init__(self, num_straps, f0, k_par_max):
        self.f0 = f0
        self.w0 = 2*np.pi*f0
        self.num_straps = num_straps
        self.k_par_max = k_par_max

        # default values
        self.delta_phi = 1# TODO np.pi/2  # this may change if off-resonance
        self.clight = 3e8 # m/s 
        self.lamda0 = self.clight / self.f0

        # flags
        self.called_set_strap_width = False

        # function calls

        self.get_key_params(to_print=False)

    def get_key_params(self, to_print=False):
        delta_phi = 1# TODO np.pi/2  # self.delta_phi is NOT being used because d does not change with delta_phi: geometry of d is fixed 
        self.d = delta_phi / self.k_par_max
        if to_print:
            print(f'Distance between strap centers d = {self.d} m')
            print(f'The first null will be at {2*np.pi/(self.d*self.num_straps)} m^-1')
            print(r'Strap length should be less than lambda/4 = ' + f'{self.lamda0/4} m')
        
    def set_strap_width(self, w):
        self.wstr = w
        self.called_set_strap_width = True
        wstr = self.wstr
        nstr = self.num_straps
        d = self.d

        self.TWA_length = (nstr*wstr + (nstr - 1)*(d - wstr)) 
        print(f'The antenna length is {self.TWA_length} m long')

    def get_J_z(self, J0, z):
        """
        This function takes in the current magnitude for each strap assuming constant power
        and constructs a peicwise function determing J(z) per strap. 
        """
        if self.called_set_strap_width == False:
            raise ValueError('Error: trap width has not been set. Use set_strap_width to do this')
        
        wstr = self.wstr
        nstr = self.num_straps
        d = self.d

        h = (nstr*wstr + (nstr - 1)*(d - wstr) )  / 2
        g = d - wstr

        for n in range(nstr):
            if z >= (-h + n*g) and z < ((-h + n*g) + wstr):
                return J0*np.exp(1j*n*self.delta_phi)
            
        return 0 # if z was not in any of those ranges, return 0
            
    def plot_J_of_z(self, J0, zmin, zmax, num_points):
        zarray = np.linspace(zmin, zmax, num_points)
        Jarray = np.zeros_like(zarray,dtype=complex)

        for i in range(Jarray.shape[0]):
            Jarray[i] = self.get_J_z(J0, zarray[i])

        #plt.plot(zarray, np.real(Jarray), color='red', label='Real')
        #plt.plot(zarray, np.imag(Jarray), color='blue', label='Imaginary')
        plt.plot(zarray, np.abs(Jarray), color='purple', linestyle='--', label='Magnitude')
        plt.scatter(zarray, np.abs(Jarray), color='purple', marker='.', label='Magnitude')
        plt.legend()
        plt.xlabel('z [m]')
        plt.ylabel('J [A/m^2]')
        plt.show()

    def get_fft_of_J_of_z(self, zmin, zmax, num_points, J0):
        zarray = np.linspace(zmin, zmax, num_points)
        dz = zarray[1] - zarray[0]
        Jarray = np.zeros_like(zarray, dtype=complex)

        for i in range(Jarray.shape[0]):
            Jarray[i] = self.get_J_z(J0, zarray[i])

        # perform the fft of the current 
        J_k = np.fft.fftshift(np.fft.fft(Jarray)) * dz / (2 * np.pi) 

        # Generate k-space values
        dk = 2 * np.pi / (num_points * dz)
        k_values = np.fft.fftshift(np.fft.fftfreq(num_points, dz))

        return k_values, J_k
        

    def plot_J_k_versus_k(self, zmin, zmax, num_points, J0):
        k_values, J_k = self.get_fft_of_J_of_z(zmin, zmax, num_points, J0)
        #plt.plot(k_values, np.real(J_k), label='Real', color='red')
        #plt.plot(k_values, np.imag(J_k), label='Imagenary', color='blue')
        plt.plot(k_values, np.abs(J_k), label='Magintude', color='purple', linestyle='--')
        plt.xlabel(r'$k_z$ [m$^{-1}$]')
        plt.ylabel(r'J(k)')

        # now,plot expected curve 


        plt.show()



    def build_lumped_element_model(self, L, C, M, R0, Rt):
        self.L = L
        self.C = C
        self.M = M
        self.R0 = R0
        self.Rt = Rt
        # more to come 

    def get_Zmat(self, w):
        num_straps = self.num_straps 
        L = self.L
        C = self.C 
        M = self.M 
        R0 = self.R0 
        Rt = self.Rt 
        S = 1j*w*L - 1j/(w*C) + Rt
        Zmat = np.zeros((num_straps, num_straps), dtype=complex)
        for i in range(num_straps):
            for j in range(num_straps):
                if i == j:
                    Zmat[i,j] = S
                    if i !=0:
                        Zmat[i, j-1] = 1j*w*M
                    if i != num_straps-1:
                        Zmat[i,j+1] = 1j*w*M
        Zmat[-1,-1] = S + R0
        return Zmat

    def solve_TWA(self, f, Vin):
        num_straps = self.num_straps 
        L = self.L
        C = self.C 
        M = self.M 
        R0 = self.R0 
        Rt = self.Rt 
        w = 2*np.pi*f
        S = Rt + 1j*w*L - 1j/(w*C) # make function 
        V_vec= np.zeros((num_straps, 1), dtype=complex)
        V_vec[0] = Vin

        # calaculate Zin0 for the reflection coeffecient 
        Zmat0 = self.get_Zmat(num_straps, self.w0, L, C, M, R0, Rt) # Z matrix at resonance
        I_vec0 = np.matmul(np.linalg.inv(Zmat0), V_vec) 
        Zin0 = (Vin/I_vec0[0]) # TODO made this abs

        Zmat = self.get_Zmat(num_straps, w, L, C, M, R0, Rt)
        I_vec = np.matmul(np.linalg.inv(Zmat), V_vec) 
        Pt = 0.5*np.abs(I_vec[-1])**2*R0
        Pf = (0.5*np.real(np.conjugate(I_vec[0])*Vin)) 

        Zin = (Vin/I_vec[0]) 
        T = Pt/Pf
        rho = (Zin - Zin0)/(Zin + Zin0)
        R = (np.abs(rho)**2)
        A = 1 - (R + T)

        return T, R, A, Pt, Pf, I_vec, Zin