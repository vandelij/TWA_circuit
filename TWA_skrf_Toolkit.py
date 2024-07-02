import numpy as np
import matplotlib.pyplot as plt
import csv
import skrf as rf
from scipy.interpolate import interp2d, interp1d

class TWA_skrf_Toolkit:

    def __init__(self, num_straps, f0, k_par_max, capz0, antz0, freqs_for_fullant, capfile, antfile):
        """
        freqs_for_fullant: frequency np array in [MHz] 
        capfile: path to comsol output csv file containing freq, length, S11... on each row
        antfile: path to comsol output csv file contaiing freq, Smat data 
        """
        self.f0 = f0
        self.w0 = 2*np.pi*f0
        self.num_straps = num_straps
        self.k_par_max = k_par_max
        self.freqs_for_fullant = freqs_for_fullant
        self.capfile = capfile
        self.antfile = antfile 

        # default values
        self.clight = 299792458 # m/s 
        self.mu0 = (4*np.pi)*1e-7  # vacuum permeability 
        self.epsi0 = 8.854e-12 # vacuum permitivity 
        self.lamda0 = self.clight / self.f0
        if self.k_par_max < 0:
            self.delta_phi_rez = -np.pi/2  
        else: 
            self.delta_phi_rez = np.pi/2 
        
        self.s_rez = np.abs(self.delta_phi_rez / self.k_par_max)
        
        # geometry
        self.geometry_dict = {} # the user can add any geometric parameter and it will get printed, along with defualts
        self.geometry_dict['num_straps'] = self.num_straps
        self.geometry_dict['s_rez'] = self.s_rez
        self.geometry_dict['lamda0'] = self.lamda0 # free space wavelength 

        # port properties
        self.capz0 = capz0  # the chopped cap rectangular z0
        self.antz0 = antz0  # the main antenna coax feed z0

        # set the internal data tables
        self.set_capacitor_data()
        self.set_ant_data()

        # set the smat interpolator matricies for the cap and ant interpolators
        self.set_interpolators_cap_data()
        self.set_ant_Smat_interpolator()

    def print_geometry(self):
        maxstring = 1
        for key in self.geometry_dict:
            if len(key) > maxstring:
                maxstring = len(key)

        for key in self.geometry_dict:
            gapstring = ' '*(maxstring - len(key))
            print(f'{key}: {gapstring}{self.geometry_dict[key]}')

    def add_to_geometry(self, key, value):
        self.geometry_dict[key] = value

    def get_comsol_datatable(self, filename):
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
    
    def set_capacitor_data(self):
        self.captable, headers = self.get_comsol_datatable(filename=self.capfile)

    def set_ant_data(self):
        self.ant_table, headers = self.get_comsol_datatable(filename=self.antfile)

    def get_cap_S_given_f_and_lcap(self,filename, f, lcap, round_level=3):
        lcap = np.round(lcap, round_level)
        data, headers = self.get_comsol_datatable(filename)
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
        capnet.name = 'l = ' + str(lcapfound)
        return capnet
    

    def print_znorm_and_capacitance(self, network, f, toprint=True):
        """
        f: frequency in MHz
        """
        freqs = network.frequency.f_scaled
        idx = np.where(freqs == f)
        Zcap = rf.s2z(network.s, z0=network.z0)[idx][0,0,0]
        C = (-np.imag(Zcap)*2*np.pi*f*1e6)**-1
        if toprint:
            print(f'Zcap:{Zcap}, z0: {network.z0[0]}, Zcap/z0: {Zcap/network.z0[0]}')
            print(f'C = {C*1e12} pF')
        return Zcap, C
        
    def get_ant_Smat_given_f(self, filename, f):
        data, headers = self.get_comsol_datatable(filename)
        num_ports = data[0,:].shape[0] - 1
        self.num_ports_chopped_ant = num_ports - 2 # the number of "chopped" cap ports, one per strap 
        if self.num_ports_chopped_ant != self.num_straps:
            raise ValueError(f'The number of straps {self.num_straps} does not match nnumber of chopped ports {self.num_ports_chopped_ant}')
        
        freqs = np.real(data[:, 0])
        ufreqs = np.unique(freqs)
        num_freqs = ufreqs.shape[0]
        i_f = np.where(ufreqs == f)[0][0]
        start_idx = i_f*(self.num_straps + 2)
        Smat = data[start_idx:(start_idx+self.num_straps + 2), 1:]
        return Smat # this is the smat for a given frequency 
    
    def build_antnet_chopped(self, freqs, filename, name=None):
        """
        freqs: a numpy array of frquencies in MHz
        """
        z0s = [self.antz0, self.antz0] + [self.capz0]*self.num_straps # create a list of the antenna port z0s. The two coax ports are first
        smat_test =  self.get_ant_Smat_given_f(filename, freqs[0])
        numports = smat_test.shape[0]
        smat_versus_freq = np.zeros((freqs.shape[0], numports, numports), dtype='complex') # the (nb_f, N, N) shaped s matrix
        for i in range(freqs.shape[0]):
            smat =  self.get_ant_Smat_given_f(filename, freqs[i])
            smat_versus_freq[i, :, :] = smat

        # create the network object 
        antnet = rf.Network()
        antnet.frequency = rf.Frequency.from_f(freqs, unit='MHz')
        antnet.s = smat_versus_freq
        antnet.z0 = z0s
        antnet.name = str(name)
        return antnet
    
    def set_antnet_chopped(self, freqs, filename, name=None):
        self.antnet_chopped = self.build_antnet(freqs, filename, name)

    def get_full_TWA_network_S11_S21(self, fullnet, f):
        """
        fullnet: network object for a full antenna
        f: the frequency you want the S parameters for
        """
        freqs = fullnet.frequency.f_scaled
        i_f = np.where(freqs == f)[0][0]
        return fullnet.s[i_f][0,0], fullnet.s[i_f][0,1]

    def get_fullant_given_one_length(self, length):
        """
        length: length of all capactors (assumes caps all have same length) [cm]
        """
        freqs = self.freqs_for_fullant

        # antenna network 
        antnet1 = self.build_antnet_chopped(self.freqs_for_fullant, self.antfile, name='chopped ant network')

        # capacitor network 
        cap_list = []
        for i_port in range(self.num_straps):
            capname = '_cap_' + str(i_port+1) + f'_port_{i_port + 2}'
            cap_list.append(self.build_capnet_given_length(length=length, freqs=freqs, filename=self.capfile, round_level=3))
            cap_list[i_port].name = 'l_' + cap_list[i_port].name + capname

        portf = rf.Frequency.from_f(freqs, unit='MHz')

        port_in = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='input')
        port_out = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='output')


        # wire them together 
        connections = [
            [(antnet1, 0), (port_in, 0)],
            [(antnet1, 1), (port_out, 0)]]
        
        for i in range(self.num_straps):
            connections = connections + [[(antnet1, i+2), (cap_list[i], 0)]]

        circuit_model = rf.Circuit(connections)
        full_network = circuit_model.network
        return full_network
    
    def get_fullant_S11_S12_given_one_length(self, length, f):
        fullnet = self.get_fullant_given_one_length(length)
        return self.get_full_TWA_network_S11_S21(fullnet, f)
    
    def get_fullant_given_C_via_caps(self, C):
        """
        C: capacitance [F] of strap caps 
        """
        freqs = self.freqs_for_fullant

        rf_freq_object = rf.Frequency.from_f(freqs, unit='MHz')

        # antenna network 
        antnet1 = self.build_antnet_chopped(self.freqs_for_fullant, self.antfile, name='chopped ant network')

        # capacitor network 
        capZ = 1 / (1j * 2 * rf.pi * rf_freq_object.f * C) # this is a numpy array
        Z = np.zeros((rf_freq_object.f.shape[0],1,1), dtype='complex') # this impedence needs to be shape number of frequencies, 1, 1 
        Z[:,0,0] = capZ
        cap_list = []
        for i_port in range(self.num_straps):
            capname = 'C_' + str(i_port+1) + f'_port_{i_port + 2}'
            cap_list.append(rf.Network(frequency=rf_freq_object, z=Z, z0=self.capz0, name=capname))

        portf = rf.Frequency.from_f(freqs, unit='MHz')

        port_in = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='input')
        port_out = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='output')


        # wire them together 
        connections = [
            [(antnet1, 0), (port_in, 0)],
            [(antnet1, 1), (port_out, 0)]]
        
        for i in range(self.num_straps):
            connections = connections + [[(antnet1, i+2), (cap_list[i], 0)]]

        circuit_model = rf.Circuit(connections)
        full_network = circuit_model.network
        return full_network
    
    def get_fullant_S11_S12_given_C(self, C, f):
        """
        C: capacitance [F]
        f: [MHz]
        """
        fullnet = self.get_fullant_given_C_via_caps(C)
        return self.get_full_TWA_network_S11_S21(fullnet, f)
    
    def plot_abs_S11_S21_l_scan(self, ls, f, return_data=False):
        """
        ls: a numpy array of capacitor lengths [m]
        f: the frequency you want to look at 
        """
        S11v = np.array([])
        S21v = np.array([])

        for i in range(ls.shape[0]):
            l = ls[i]
            S11, S21 = self.get_fullant_S11_S12_given_one_length(l, f)
            S11abs = np.abs(S11)
            S21abs = np.abs(S21)
            S11v = np.append(S11v, S11abs)
            S21v = np.append(S21v, S21abs)

        fig, axs = plt.subplots(1, 3, figsize=(12, 5))

        axs[0].plot(ls*100, S11v, label='|S11|')
        axs[0].set_ylabel('|S|')
        axs[0].set_xlabel('Cap length [cm]')
        axs[0].grid()
        axs[0].legend()

        axs[1].plot(ls*100, S21v, label='|S21|', color='red')
        axs[1].set_ylabel('|S|')
        axs[1].set_xlabel('Cap length [cm]')
        axs[1].grid()
        axs[1].legend()
        axs[1].set_title(f'Frequency: {f} MHz')

        axs[2].plot(ls*100, S11v, label='|S11|')
        axs[2].plot(ls*100, S21v, label='|S12|', color='red')
        axs[2].set_ylabel('|S|')
        axs[2].set_xlabel('Cap length [cm]')
        axs[2].grid()
        axs[2].legend()     
        #plt.show()

        if return_data:
            return S11v, S21v, axs
        
    # copies of functions that are faster due to not reading the file every time 
    def get_cap_S_given_f_and_lcap_from_internal_datatable(self, f, lcap, round_level=3):
        lcap = np.round(lcap, round_level)
        data = self.captable
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

    def build_capnet_given_length_from_internal_datatable(self, length, freqs, round_level=3):
        """
        freqs: numpy array of the used frquencies [MHz]
        returns: skrf network object representing a 1-port cap given length
        """
        S11_array = np.zeros((freqs.shape[0], 1, 1), dtype='complex')
        for i in range(freqs.shape[0]):
            f = freqs[i]
            ffound, lcapfound, S11, S11db, Z0_port, VSWR = self.get_cap_S_given_f_and_lcap_from_internal_datatable(
                                        f=f, lcap=length, round_level=round_level)
            S11_array[i, 0, 0] = S11


        # create the network object 
        capnet = rf.Network()
        capnet.frequency = rf.Frequency.from_f(freqs, unit='MHz')
        capnet.s = S11_array
        capnet.z0 = self.capz0
        capnet.name = 'l = ' + str(lcapfound)
        return capnet
    
    def get_ant_Smat_given_f_from_internal_datatable(self, f):
        data = self.ant_table
        num_ports = data[0,:].shape[0] - 1
        self.num_ports_chopped_ant = num_ports - 2 # the number of "chopped" cap ports, one per strap 
        if self.num_ports_chopped_ant != self.num_straps:
            raise ValueError(f'The number of straps {self.num_straps} does not match nnumber of chopped ports {self.num_ports_chopped_ant}')
        
        freqs = np.real(data[:, 0])
        ufreqs = np.unique(freqs)
        num_freqs = ufreqs.shape[0]
        i_f = np.where(ufreqs == f)[0][0]
        start_idx = i_f*(self.num_straps + 2)
        Smat = data[start_idx:(start_idx+self.num_straps + 2), 1:]
        return Smat # this is the smat for a given frequency 

    def build_antnet_chopped_from_internal_datatable(self, freqs, name=None):
        """
        freqs: a numpy array of frquencies in MHz
        """
        z0s = [self.antz0, self.antz0] + [self.capz0]*self.num_straps # create a list of the antenna port z0s. The two coax ports are first
        smat_test =  self.get_ant_Smat_given_f_from_internal_datatable(freqs[0])
        numports = smat_test.shape[0]
        smat_versus_freq = np.zeros((freqs.shape[0], numports, numports), dtype='complex') # the (nb_f, N, N) shaped s matrix
        for i in range(freqs.shape[0]):
            smat =  self.get_ant_Smat_given_f_from_internal_datatable(freqs[i])
            smat_versus_freq[i, :, :] = smat

        # create the network object 
        antnet = rf.Network()
        antnet.frequency = rf.Frequency.from_f(freqs, unit='MHz')
        antnet.s = smat_versus_freq
        antnet.z0 = z0s
        antnet.name = str(name)
        return antnet
    
    def get_fullant_given_one_length_from_internal_datatable(self, length):
        """
        length: length of all capactors (assumes caps all have same length) [cm]
        """
        freqs = self.freqs_for_fullant

        # antenna network 
        antnet1 = self.build_antnet_chopped_from_internal_datatable(self.freqs_for_fullant, name='chopped ant network')

        # capacitor network 
        cap_list = []
        for i_port in range(self.num_straps):
            capname = '_cap_' + str(i_port+1) + f'_port_{i_port + 2}'
            cap_list.append(self.build_capnet_given_length_from_internal_datatable(length=length, freqs=freqs, round_level=3))
            cap_list[i_port].name = 'l_' + cap_list[i_port].name + capname

        portf = rf.Frequency.from_f(freqs, unit='MHz')

        port_in = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='input')
        port_out = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='output')


        # wire them together 
        connections = [
            [(antnet1, 0), (port_in, 0)],
            [(antnet1, 1), (port_out, 0)]]
        
        for i in range(self.num_straps):
            connections = connections + [[(antnet1, i+2), (cap_list[i], 0)]]

        circuit_model = rf.Circuit(connections)
        full_network = circuit_model.network
        return full_network

    def get_fullant_S11_S12_given_one_length_from_internal_datatable(self, length, f):
        fullnet = self.get_fullant_given_one_length_from_internal_datatable(length)
        return self.get_full_TWA_network_S11_S21(fullnet, f)
    
    def get_fullant_given_C_via_caps_from_internal_datatable(self, C):
        """
        C: capacitance [F] of strap caps 
        """
        freqs = self.freqs_for_fullant

        rf_freq_object = rf.Frequency.from_f(freqs, unit='MHz')

        # antenna network 
        antnet1 = self.build_antnet_chopped_from_internal_datatable(self.freqs_for_fullant, name='chopped ant network')

        # capacitor network 
        capZ = 1 / (1j * 2 * rf.pi * rf_freq_object.f * C) # this is a numpy array
        Z = np.zeros((rf_freq_object.f.shape[0],1,1), dtype='complex') # this impedence needs to be shape number of frequencies, 1, 1 
        Z[:,0,0] = capZ
        cap_list = []
        for i_port in range(self.num_straps):
            capname = 'C_' + str(i_port+1) + f'_port_{i_port + 2}'
            cap_list.append(rf.Network(frequency=rf_freq_object, z=Z, z0=self.capz0, name=capname))

        portf = rf.Frequency.from_f(freqs, unit='MHz')

        port_in = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='input')
        port_out = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='output')


        # wire them together 
        connections = [
            [(antnet1, 0), (port_in, 0)],
            [(antnet1, 1), (port_out, 0)]]
        
        for i in range(self.num_straps):
            connections = connections + [[(antnet1, i+2), (cap_list[i], 0)]]

        circuit_model = rf.Circuit(connections)
        full_network = circuit_model.network
        return full_network

    def get_fullant_S11_S12_given_C_from_internal_datatable(self, C, f):
        """
        C: capacitance [F]
        f: [MHz]
        """
        fullnet = self.get_fullant_given_C_via_caps_from_internal_datatable(C)
        return self.get_full_TWA_network_S11_S21(fullnet, f)
    
    def plot_abs_S11_S21_l_scan_from_internal_datatable(self, ls, f, return_data=False):
        """
        ls: a numpy array of capacitor lengths [m]
        f: the frequency you want to look at 
        """
        S11v = np.array([])
        S21v = np.array([])

        for i in range(ls.shape[0]):
            l = ls[i]
            S11, S21 = self.get_fullant_S11_S12_given_one_length_from_internal_datatable(l, f)
            S11abs = np.abs(S11)
            S21abs = np.abs(S21)
            S11v = np.append(S11v, S11abs)
            S21v = np.append(S21v, S21abs)

        fig, axs = plt.subplots(1, 3, figsize=(12, 5))

        axs[0].plot(ls*100, S11v, label='|S11|')
        axs[0].set_ylabel('|S|')
        axs[0].set_xlabel('Cap length [cm]')
        axs[0].grid()
        axs[0].legend()

        axs[1].plot(ls*100, S21v, label='|S21|', color='red')
        axs[1].set_ylabel('|S|')
        axs[1].set_xlabel('Cap length [cm]')
        axs[1].grid()
        axs[1].legend()
        axs[1].set_title(f'Frequency: {f} MHz')

        axs[2].plot(ls*100, S11v, label='|S11|')
        axs[2].plot(ls*100, S21v, label='|S12|', color='red')
        axs[2].set_ylabel('|S|')
        axs[2].set_xlabel('Cap length [cm]')
        axs[2].grid()
        axs[2].legend()     
        #plt.show()

        if return_data:
            return S11v, S21v, axs
        
    def plot_abs_S11_S21_f_scan_from_internal_datatable(self, fs, l, return_data=False):
        """
        ls: a numpy array of capacitor lengths [m]
        f: the frequency you want to look at 
        """
        S11v = np.array([])
        S21v = np.array([])

        for i in range(fs.shape[0]):
            f = fs[i]
            S11, S21 = self.get_fullant_S11_S12_given_one_length_from_internal_datatable(l, f)
            S11abs = np.abs(S11)
            S21abs = np.abs(S21)
            S11v = np.append(S11v, S11abs)
            S21v = np.append(S21v, S21abs)

        fig, axs = plt.subplots(1, 3, figsize=(12, 5))

        axs[0].plot(fs, S11v, label='|S11|')
        axs[0].set_ylabel('|S|')
        axs[0].set_xlabel('f [MHz]')
        axs[0].grid()
        axs[0].legend()

        axs[1].plot(fs, S21v, label='|S21|', color='red')
        axs[1].set_ylabel('|S|')
        axs[1].set_xlabel('f [MHz]')
        axs[1].grid()
        axs[1].legend()
        axs[1].set_title(f'Cap Length: {l*100} cm')

        axs[2].plot(fs, S11v, label='|S11|')
        axs[2].plot(fs, S21v, label='|S12|', color='red')
        axs[2].set_ylabel('|S|')
        axs[2].set_xlabel('f [MHz]')
        axs[2].grid()
        axs[2].legend()     
        #plt.show()

        if return_data:
            return S11v, S21v, axs
    
    def set_interpolators_cap_data(self):
        capdata = self.captable
        round_level = 3
        fs = np.real(np.unique(capdata[:,0]))
        ls = np.round(np.real(np.unique(capdata[:,1])), round_level)
        S11_real = np.real(capdata[:,5]).reshape(fs.shape[0], ls.shape[0]) # this is the de-embeded collumn 
        S11_imag = np.imag(capdata[:,5]).reshape(fs.shape[0], ls.shape[0]) # this is the de-embeded collumn 

        fsmesh, lsmesh = np.meshgrid(fs, ls, indexing='ij')

        self.S11_real_interpolator = interp2d(fsmesh, lsmesh, S11_real)
        self.S11_imag_interpolator = interp2d(fsmesh, lsmesh, S11_imag)
        
    def interpolate_cap_data(self, f, l):
        S11 = self.S11_real_interpolator(f, l) + 1j*self.S11_imag_interpolator(f,l)
        return S11[0]
    
    def set_ant_Smat_interpolator(self):
        num_ports = self.num_straps + 2
        fs = self.freqs_for_fullant
        s = self.build_antnet_chopped_from_internal_datatable(fs, name=None).s
        interp_matrix_real = []
        interp_matrix_imag = []


        for i in range(num_ports):
            interp_row_list_real = []
            interp_row_list_imag = []
            for j in range(num_ports):
                interp_row_list_real.append(interp1d(fs, np.real(s[:,i,j]))) # interpolate the real part 
                interp_row_list_imag.append(interp1d(fs, np.imag(s[:,i,j]))) # interpolate the imag part

            interp_matrix_real.append(interp_row_list_real)
            interp_matrix_imag.append(interp_row_list_imag)

        self.interp_matrix_real = interp_matrix_real
        self.interp_matrix_imag = interp_matrix_imag

    
    def interpolate_sant_for_any_f(self, f):
        interp_matrix_real = self.interp_matrix_real
        interp_matrix_imag = self.interp_matrix_imag

        Smat = np.zeros((self.num_straps+2, self.num_straps+2), dtype='complex') # initialize the s matrix 

        for i in range(self.num_straps+2):
            for j in range(self.num_straps+2):
                Smat[i,j] = interp_matrix_real[i][j](f) + 1j*interp_matrix_imag[i][j](f)
        
        return Smat


    def build_capnet_given_length_interpolated(self, length, freqs):
        """
        freqs: numpy array of the used frquencies [MHz]
        returns: skrf network object representing a 1-port cap given any interpolated length
        """
        S11_array = np.zeros((freqs.shape[0], 1, 1), dtype='complex')
        for i in range(freqs.shape[0]):
            f = freqs[i]
            S11 = self.interpolate_cap_data(f, length)
            S11_array[i, 0, 0] = S11


        # create the network object 
        capnet = rf.Network()
        capnet.frequency = rf.Frequency.from_f(freqs, unit='MHz')
        capnet.s = S11_array
        capnet.z0 = self.capz0
        capnet.name = 'l = ' + str(length)
        return capnet


    def get_fullant_given_lengths_from_internal_datatable(self, lengths, symetric_mode=False):
        """
        lengths: a list of lengths, must be same number as the number of ports/2 for even num straps, (n-1)/2 for odd
        if running in symetric mode. If not, then must be the same length as the number of straps 
        """
        freqs = self.freqs_for_fullant

        # antenna network 
        antnet1 = self.build_antnet_chopped_from_internal_datatable(self.freqs_for_fullant, name='chopped ant network')

        # capacitor network 
        if symetric_mode:
            lengths_reverse = lengths.copy()
            lengths_reverse.reverse()
            if self.num_straps % 2 == 0:
                if len(lengths_reverse) != int(self.num_straps/2):
                    raise ValueError('The lengths array is not the correct length')
                for i in range(len(lengths_reverse)):
                    lengths.append(lengths_reverse[i])

            elif self.num_straps % 2 != 0:
                if len(lengths_reverse) != int((self.num_straps+1)/2):
                    raise ValueError('The lengths array is not the correct length')
                for i in range(1,len(lengths_reverse)):
                    lengths.append(lengths_reverse[i])
        else:
            if self.num_straps != len(lengths):
                raise ValueError('The lengths array is not the correct length')

        cap_list = []
        for i_port in range(self.num_straps):
            capname = '_cap_' + str(i_port+1) + f'_port_{i_port + 2}'
            cap_list.append(self.build_capnet_given_length_interpolated(length=lengths[i_port], freqs=freqs))
            cap_list[i_port].name = 'l_' + cap_list[i_port].name + capname

        portf = rf.Frequency.from_f(freqs, unit='MHz')

        port_in = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='input')
        port_out = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='output')


        # wire them together 
        connections = [
            [(antnet1, 0), (port_in, 0)],
            [(antnet1, 1), (port_out, 0)]]
        
        for i in range(self.num_straps):
            connections = connections + [[(antnet1, i+2), (cap_list[i], 0)]]

        circuit_model = rf.Circuit(connections)
        full_network = circuit_model.network
        print(lengths)
        return full_network

        

