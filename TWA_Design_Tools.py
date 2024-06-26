import numpy as np
import matplotlib.pyplot as plt
import csv

class TWA_Design_Toolkit:

    def __init__(self, num_straps, f0, k_par_max, d_straps=0):
        self.f0 = f0
        self.w0 = 2*np.pi*f0
        self.num_straps = num_straps
        self.k_par_max = k_par_max

        # default values
        self.clight = 299792458 # m/s 
        self.mu0 = (4*np.pi)*1e-7  # vacuum permeability 
        self.epsi0 = 8.854e-12 # vacuum permitivity 
        self.lamda0 = self.clight / self.f0

        # flags
        self.called_set_strap_width = False
        self.S_matrix_set = False
        self.called_get_Z_matrix = False
        self.center_fed = False


        # function calls
        if d_straps == 0:
            print(f"You are at resonance, so delta_phi = pi/2. Solving for d given k_par:")
            self.set_key_params_at_resonance()
            
        else: 
            print(f"You are off resonance, so d is supplied by you. Solving for delta_phi given k_par:")
            self.set_key_params_off_resonance(d_straps)

    def set_center_fed(self, center_fed_bool):
        self.center_fed = center_fed_bool


    def set_key_params_at_resonance(self):
        # At TWA resonance, the phase difference is pi/2. Sign matches k_par, as del_phi/d = k_par, and d is a + value. 
        if self.k_par_max < 0:
            delta_phi = -np.pi/2  # At TWA resonance, the phase difference is pi/2.  #TODO change this to posotive 
        else: 
            delta_phi = np.pi/2 

        self.d = np.abs(delta_phi / self.k_par_max)
        self.delta_phi = delta_phi
    
    def set_key_params_off_resonance(self, d):
        # Here, d is used with k_par_max in order to get the delta_phi
        self.delta_phi =  d*self.k_par_max
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
    
    def get_J_z_centerfed(self, J0, z):
        """
        This function takes in the current magnitude for each strap assuming constant power
        and constructs a peice-wise function determing J(z) per strap, assuming center fed.
        Must be an even number of straps.  
        """
        if self.called_set_strap_width == False:
            raise ValueError('Error: strap width has not been set. Use set_strap_width to do this')
        
        if self.num_straps % 2 == 0:
            raise ValueError('To be center fed, you must have an even number of straps')
        
        wstr = self.wstr
        nstr = self.num_straps
        d = self.d
        h = (nstr*wstr + (nstr - 1)*(d - wstr) )  / 2
        m = (nstr-1)/2

        for n in range(nstr):
            if z >= (-h + n*d) and z < ((-h + n*d) + wstr):
                return J0*np.exp(1j*np.abs(m-n)*self.delta_phi)
            
        return 0 # if z was not in any of those ranges, return 0
            
    def plot_J_of_z(self, J0, zmin, zmax, num_points):
        zarray = np.linspace(zmin, zmax, num_points)
        Jarray = np.zeros_like(zarray,dtype=complex)

        for i in range(Jarray.shape[0]):
            if not self.center_fed:
                Jarray[i] = self.get_J_z(J0, zarray[i])
            elif self.center_fed: 
                Jarray[i] = self.get_J_z_centerfed(J0, zarray[i])

        plt.plot(zarray, np.abs(Jarray), color='purple', linestyle='--', label='Magnitude')
        plt.scatter(zarray, np.abs(Jarray), color='purple', marker='.', label='Magnitude')
        plt.legend()
        plt.xlabel('z [m]')
        plt.ylabel('J [A/m^2]')
        plt.grid()
        plt.show()

    def get_fft_of_J_of_z(self, zmin, zmax, num_points, J0):
        zarray = np.linspace(zmin, zmax, num_points)
        dz = (zarray[1] - zarray[0]) # TODO
        Jarray = np.zeros_like(zarray, dtype=complex)

        for i in range(Jarray.shape[0]):
            if not self.center_fed:
                Jarray[i] = self.get_J_z(J0, zarray[i])
            elif self.center_fed: 
                Jarray[i] = self.get_J_z_centerfed(J0, zarray[i])

        # perform the fft of the current 
        J_k = np.fft.fftshift(np.fft.fft(Jarray)) 

        # Generate k-space values
        k_values = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(num_points, dz))

        return k_values, J_k
        

    def get_fft_analytic(self, k, P0):
        n = k*self.clight/self.w0 # TODO why is this negative? 
        if n == 0:
            return 1*P0
        else:
            term1 = np.sin(n*self.wstr*self.w0/(2*self.clight))**2 / n**2
            alpha = -self.delta_phi + n*self.w0*self.d/self.clight  # this negative sign seems to fix this.. 
            # alpha = self.delta_phi + n*self.w0*self.d/self.clight
            term2 = np.sin(self.num_straps*alpha/2)**2/np.sin(alpha/2)**2
            return P0*term1 * term2
        
    def get_plot_fft_analytic(self, kmin, kmax, num_points, P0):
        karray = np.linspace(kmin, kmax, num_points)
        power_array = np.zeros_like(karray)
        for i in range(karray.shape[0]):
            power_array[i] = self.get_fft_analytic(karray[i], P0)
        plt.plot(karray, np.sqrt(power_array))
        plt.grid()
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
        plt.grid()
        plt.show()

    def plot_normalized_power_versus_k(self, zmin, zmax, kplotmin, kplotmax, num_pointsz, num_pointsk, J0):
        k_values, J_k = self.get_fft_of_J_of_z(zmin, zmax, num_pointsz, J0)
        #plt.plot(k_values, np.real(J_k), label='Real', color='red')
        #plt.plot(k_values, np.imag(J_k), label='Imagenary', color='blue')
        # plt.plot(k_values)
        # plt.show()
        indicies = np.where((k_values >= kplotmin) & (k_values <= kplotmax))
        peak = np.max(np.abs(J_k[indicies]))
        plt.plot(k_values, np.square(np.abs(J_k)/peak), label='fft Magnitude', color='blue')
        plt.xlabel(r'$k_z$ [m$^{-1}$]')
        plt.ylabel(r'P(k)/$P_0$')
        plt.xlim(kplotmin, kplotmax)
        plt.axvline(x=self.k_par_max, ymin=0, ymax=max(np.abs(J_k)/peak), color='black', linestyle='--')

        # now,plot expected curve


        karray = np.linspace(kplotmin, kplotmax, num_pointsk)
        power_array = np.zeros_like(karray)
        for i in range(karray.shape[0]):
            power_array[i] = self.get_fft_analytic(karray[i], J0**2)
        plt.plot(karray, power_array/np.max(power_array), color='black', label='analytic', linestyle='--')
        plt.legend()
        plt.grid()
        plt.show()
        
# area to get the needed capacitance per strap 
    def read_Smat_from_comsol_portscan_Stable(self, filename, return_flag=False):
        # this function reads in the csv file saved from runnnig a comsol 
        # port scan and outputs the Smatrix 
        data = []
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                data.append(row)

        data = data[5:]
        Smat_string  = []
        for row in data:
            Smat_string.append(row[1:])

        Smat = np.array([[complex(num.replace('i', 'j')) for num in row] for row in Smat_string], dtype=complex)

        # Smat = -np.imag(Smat) + 1j*np.real(Smat)# TODO: REMOVE ME!
        #Smat = np.real(Smat) - 1j*np.imag(Smat)# TODO: REMOVE ME!
        #Smat = 1j*Smat  # TODO: this also is sus and needs to be removed. 
        self.Smatrix = Smat
        self.S_matrix_set = True # set flag to set
        if return_flag:
            return Smat
      # This version is depricated and uses an inferior comsol output file format   
    def read_Smat_from_comsol_portscan_Stable_depricated(self, filename, return_flag=False):
        # this function reads in the csv file saved from runnnig a comsol 
        # port scan and outputs the Smatrix 
        data = []
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                data.append(row)
        #data[5:]
        complex_array = np.array([[complex(num.replace('i', 'j')) for num in row] for row in data[5:]], dtype=complex)[:, 2:]
        size_Smatrix = int(np.sqrt(complex_array.shape[1]))
        Smat = np.zeros((size_Smatrix, size_Smatrix), dtype=complex)
        colnum = 0

        # build up the S matrix
        for i in range(size_Smatrix):
            for j in range(size_Smatrix):
                Sij = complex_array[j, colnum]
                Smat[i,j] = Sij
                colnum += 1

        # Smat = -np.imag(Smat) + 1j*np.real(Smat)# TODO: REMOVE ME!
        #Smat = np.real(Smat) - 1j*np.imag(Smat)# TODO: REMOVE ME!
        #Smat = 1j*Smat  # TODO: this also is sus and needs to be removed. 
        self.Smatrix = Smat
        self.S_matrix_set = True # set flag to set
        if return_flag:
            return Smat

    def set_Smatrix(self, S_mat):
        # provides a manual way to set the S matrix 
        self.Smatrix = S_mat
        self.S_matrix_set = True # set flag to set

    def get_Z_matrix(self, Z0_port, return_flag=False): 
        S_mat = self.Smatrix
        smat_size = S_mat.shape[0]  # get the shape of the S matrix
        I = np.identity(smat_size, dtype=complex)
        self.Zmatrix = Z0_port*np.matmul((I + S_mat),np.linalg.inv(I - S_mat))
        self.called_get_Z_matrix = True
        if return_flag:
            return self.Zmatrix

    def get_coax_Z0(self, d_outer, d_inner):
        # Takes in the outer and inner diameter of the coax cable gap and returns the cable characteristic impedence
        # this assumes vaccuum in the coax. adjust epsi0 and mu0 with mu_r and epsi_r to model dielectric filled coax
        return (1/(2*np.pi))*np.sqrt(self.mu0/self.epsi0)*np.log(d_outer/d_inner)
    
    def plot_Smat_and_Zmat(self):
        if self.S_matrix_set  == False: 
            raise ValueError('You need to set the Smatrix first, which calculates the Zmatrix.') 
        
        Zmat = self.Zmatrix
        Smat = self.Smatrix
        fig, axs = plt.subplots(2, 3, figsize=(15, 15))

        s1 = axs[0,0].matshow(np.real(Smat))
        plt.colorbar(s1, ax=axs[0,0])

        s2 = axs[0,1].matshow(np.imag(Smat))
        plt.colorbar(s2, ax=axs[0,1])

        s3 = axs[0,2].matshow(np.abs(Smat))
        plt.colorbar(s3, ax=axs[0,2])

        s1.set_clim(vmin=-.8, vmax=.8)
        s2.set_clim(vmin=-.8, vmax=.8)
        s3.set_clim(vmin=-.8, vmax=.8)

        axs[0,0].axis('equal')
        axs[0,1].axis('equal')
        axs[0,2].axis('equal')

        axs[0,0].set_title('Re[S]')
        axs[0,1].set_title('Im[S]')
        axs[0,2].set_title('|S|')

        s4 = axs[1,0].matshow(np.real(Zmat))
        plt.colorbar(s4, ax=axs[1,0])

        s5 = axs[1,1].matshow(np.imag(Zmat))
        plt.colorbar(s5, ax=axs[1,1])

        s6 = axs[1,2].matshow(np.abs(Zmat))
        plt.colorbar(s6, ax=axs[1,2])

        axs[1,0].axis('equal')
        axs[1,1].axis('equal')
        axs[1,2].axis('equal')

        # s4.set_clim(vmin=-1, vmax=1)
        # s5.set_clim(vmin=-1, vmax=1)
        # s6.set_clim(vmin=-1, vmax=1)

        axs[1,0].set_title('Re[Z]')
        axs[1,1].set_title('Im[Z]')
        axs[1,2].set_title('|Z|')
        plt.show()

    def calculate_C0(self):
        # Uses the Smatrix -> Zmatrix to find the capacitance needed per strap to cancel the self-inductance to set the TWA in resonance 
        if self.called_get_Z_matrix == False: 
            raise ValueError(f'You need to set the Smatrix first, which calculates the Zmatrix. It is {self.called_get_Z_matrix}') 
        
        smat_size = self.Smatrix.shape[0]
        w_L_average = np.imag((1/smat_size)*np.trace(self.Zmatrix)) # note: this is the average iductive reactance, X = w0*L_self
        self.C0 = 1 / (self.w0*w_L_average)  # dont be confused: its w0 not w0^2 because the other w is hidden in the inductive reactance
        return self.C0
    
    def calculate_C1(self):
        # if you choose a T-feed instead of an L-feed, your circuit is a CLC series: 1/(jwC1) + 1/(jwC1) + jwL = 0 cancels reactive L
        return 2*self.calculate_C0() 
    
    def cap_area_given_gap(self, h):
        self.calculate_C0()
        # simple double plate cap with vacuum in the gap. hieght in meters  
        return self.C0*h/self.epsi0
    
    def cap_gap_given_area(self, A):
        self.calculate_C0()
        # simple double plate cap with vacuum in the gap. area in meters^2  
        return self.epsi0*A/self.C0
    
    def cap_area_given_gap_and_C(self, h, C):
        # simple double plate cap with vacuum in the gap. hieght in meters  
        return C*h/self.epsi0
    
    def cap_gap_given_area_and_C(self, A, C):
        # simple double plate cap with vacuum in the gap. area in meters^2  
        return self.epsi0*A/C


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