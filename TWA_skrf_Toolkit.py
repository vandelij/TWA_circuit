import numpy as np
import matplotlib.pyplot as plt
import csv
import skrf as rf

class TWA_skrf_Toolkit:

    def __init__(self, num_straps, f0, k_par_max, capz0, antz0):
        self.f0 = f0
        self.w0 = 2*np.pi*f0
        self.num_straps = num_straps
        self.k_par_max = k_par_max

        # default values
        self.clight = 299792458 # m/s 
        self.mu0 = (4*np.pi)*1e-7  # vacuum permeability 
        self.epsi0 = 8.854e-12 # vacuum permitivity 
        self.lamda0 = self.clight / self.f0
        if self.k_par_max < 0:
            self.delta_phi_rez = -np.pi/2  
        else: 
            self.delta_phi_rez = np.pi/2 
        
        self.s_rez = np.abs(self.delta_phi / self.k_par_max)
        
        # geometry
        self.geometry_dict = {} # the user can add any geometric parameter and it will get printed, along with defualts
        self.geometry_dict['num_straps'] = self.num_straps
        self.geometry_dict['s_rez'] = self.s_rez
        self.geometry_dict['lamda0'] = self.lamda0 # free space wavelength 

        # port properties
        self.cap_z0 = capz0  # the chopped cap rectangular z0
        self.ant_z0 = antz0  # the main antenna coax feed z0

    def print_geometry(self):
        for key in self.geometry_dict:
            print(f'{key}: ', self.geometry_dict[key])

    def get_cap_S_datatable(self, filename):
        data = []
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                data.append(row)
        headers = data[4]
        data = data[5:]
        fmat_string  = []
        for row in data:
            fmat_string.append(row)
        fmat = np.array([[complex(num.replace('i', 'j')) for num in row] for row in fmat_string], dtype=complex)
        return fmat, headers 

    def get_cap_S_given_f_and_lcap(self,filename, f, lcap, round_level=3):
        lcap = np.round(lcap, round_level)
        data, headers = self.get_cap_S_datatable(filename)
        freqs = np.real(data[:, 0])
        ufreqs = np.unique(freqs)
        num_freq = ufreqs.shape[0]
        lengths = np.round(np.real(data[:,1]), round_level)
        ulengths = np.unique(lengths)
        num_lengths = ulengths.shape[0]
        i_f = np.where(ufreqs == f)[0][0]
        i_length = np.where(ulengths == lcap)[0][0]
        rownum = i_f*num_lengths + i_length
        ffound = np.real(data[rownum, 0])
        lcapfound = np.real(data[rownum,1])
        S11 = data[rownum, 5] # TODO: was 3, now de-embedded
        S11db = np.real(data[rownum, 4])
        Z0_port = np.real(data[rownum, 7])
        VSWR = np.real(data[rownum, 8])
        #print('TODO: should test with a slightly larger dataset')
        return ffound, lcapfound, S11, S11db, Z0_port, VSWR

    def build_capnet_given_length(self, length, freqs, filename, round_level=3):
        """
        freqs: numpy array of the used frquencies [MHz]
        returns: skrf network object representing a 1-port cap given length
        """
        S11_array = np.zeros((freqs.shape[0], 1, 1), dtype='complex')
        for i in range(freqs.shape[0]):
            f = freqs[i]
            ffound, lcapfound, S11, S11db, Z0_port, VSWR = self.get_cap_S_given_f_and_lcap(filename=filename,
                                        f=f, lcap=length, round_level=round_level)
            S11_array[i, 0, 0] = S11


        # create the network object 
        capnet = rf.Network()
        capnet.frequency = rf.Frequency.from_f(freqs, unit='MHz')
        capnet.s = S11_array
        capnet.z0 = self.capz0
        capnet.name = str(lcapfound)
        return capnet
    

    def print_Znorm_and_capacitance(self, network, f, toprint=True):
        """
        f: frequency in MHz
        z0: required characteristic impedence 
        """
        freqs = network.frequency.f_scaled
        idx = np.where(freqs == f)
        Zcap = rf.s2z(network.s, z0=network.z0)[idx][0,0,0]
        C = (-np.imag(Zcap)*2*np.pi*f*1e6)**-1
        if toprint:
            print(f'Zcap:{Zcap}, z0: {network.z0[0]}, Zcap/z0: {Zcap/network.z0[0]}')
            print(f'C = {C*1e12} pF')
        return Zcap, C