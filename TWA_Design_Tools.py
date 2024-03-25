import numpy as np
import matplotlib.pyplot as plt

class TWA_Design_Toolkit:

    def __init__(self, num_straps, f0, k_par_max, d_straps=0):
        self.f0 = f0
        self.w0 = 2*np.pi*f0
        self.num_straps = num_straps
        self.k_par_max = k_par_max

        # default values
        self.clight = 3e8 # m/s 
        self.lamda0 = self.clight / self.f0

        # flags
        self.called_set_strap_width = False

        # function calls
        if d_straps == 0:
            print(f"You are at resonance, so delta_phi = pi/2. Solving for d given k_par:")
            self.set_key_params_at_resonance()
            
        else: 
            print(f"You are off resonance, so d is supplied by you. Solving for delta_phi given k_par:")
            self.set_key_params_off_resonance(d_straps)

    def set_key_params_at_resonance(self):
        delta_phi = np.pi/2  # At TWA resonance, the phase difference is pi/2.  
        self.d = delta_phi / self.k_par_max
        self.delta_phi = delta_phi
    
    def set_key_params_off_resonance(self, d):
        # Here, d is used with k_par_max in order to get the delta_phi
        self.delta_phi =   d*self.k_par_max
        self.d = d

    def print_key_params(self):
        print('\n')
        print('----------------Parameter--Box-------------------')
        print(f'Distance between strap centers d = {self.d} m')
        print(f'The first null will be at {2*np.pi/(self.d*self.num_straps)} m^-1')
        print(f'delta_phi = {self.delta_phi/np.pi} pi')
        print(r'Strap length should be less than lambda/4 = ' + f'{self.lamda0/4} m')
        print('--------------END--Parameter--Box----------------')
        print('\n')

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
        and constructs a peice-wise function determing J(z) per strap. 
        """
        if self.called_set_strap_width == False:
            raise ValueError('Error: strap width has not been set. Use set_strap_width to do this')
        
        wstr = self.wstr
        nstr = self.num_straps
        d = self.d

        h = (nstr*wstr + (nstr - 1)*(d - wstr) )  / 2

        for n in range(nstr):
            if z >= (-h + n*d) and z < ((-h + n*d) + wstr):
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
        dz = (zarray[1] - zarray[0]) # TODO
        Jarray = np.zeros_like(zarray, dtype=complex)

        for i in range(Jarray.shape[0]):
            Jarray[i] = self.get_J_z(J0, zarray[i])

        # perform the fft of the current 
        J_k = np.fft.fftshift(np.fft.fft(Jarray)) 

        # Generate k-space values
        k_values = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(num_points, dz))

        return k_values, J_k
        

    def get_fft_analytic(self, k, P0):
        n = -k*self.clight/self.w0 # TODO
        if n == 0:
            return 1*P0
        else:
            term1 = np.sin(n*self.wstr*self.w0/(2*self.clight))**2 / n**2
            alpha = self.delta_phi + n*self.w0*self.d/self.clight
            term2 = np.sin(self.num_straps*alpha/2)**2/np.sin(alpha/2)**2
            return P0*term1 * term2
        
    def get_plot_fft_analytic(self, kmin, kmax, num_points, P0):
        karray = np.linspace(kmin, kmax, num_points)
        power_array = np.zeros_like(karray)
        for i in range(karray.shape[0]):
            power_array[i] = self.get_fft_analytic(karray[i], P0)
        plt.plot(karray, np.sqrt(power_array))
        plt.show()

        

    def plot_J_k_versus_k(self, zmin, zmax, kplotmin, kplotmax, num_pointsz, num_pointsk, J0):
        k_values, J_k = self.get_fft_of_J_of_z(zmin, zmax, num_pointsz, J0)
        #plt.plot(k_values, np.real(J_k), label='Real', color='red')
        #plt.plot(k_values, np.imag(J_k), label='Imagenary', color='blue')
        # plt.plot(k_values)
        # plt.show()
        indicies = np.where((k_values >= kplotmin) & (k_values <= kplotmax))
        peak = np.max(np.abs(J_k[indicies]))
        plt.plot(k_values, np.abs(J_k)/peak, label='fft Magnitude', color='purple')
        plt.xlabel(r'$k_z$ [m$^{-1}$]')
        plt.ylabel(r'J(k)')
        plt.xlim(kplotmin, kplotmax)
        plt.axvline(x=self.k_par_max, ymin=0, ymax=max(np.abs(J_k)/peak), color='black', linestyle='--')

        # now,plot expected curve


        karray = np.linspace(kplotmin, kplotmax, num_pointsk)
        power_array = np.zeros_like(karray)
        for i in range(karray.shape[0]):
            power_array[i] = self.get_fft_analytic(karray[i], J0**2)
        plt.plot(karray, np.sqrt(power_array)/np.max(np.sqrt(power_array)), color='red', label='analytic', linestyle='--')
        plt.legend()
        plt.show()

# area to get the needed capacitance per strap 




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