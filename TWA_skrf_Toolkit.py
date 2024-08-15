import numpy as np
import matplotlib.pyplot as plt
import csv
import skrf as rf
from scipy.interpolate import interp2d, interp1d, RectBivariateSpline, PchipInterpolator
from scipy.optimize import minimize, differential_evolution

class TWA_skrf_Toolkit:

    def __init__(self, num_straps, f0, k_par_max, capz0, antz0, freqs_for_fullant, capfile, antfile, center_fed_mode=False):
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
        self.center_fed_mode = center_fed_mode

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
        if self.freqs_for_fullant.shape[0] > 1:
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

        if self.center_fed_mode == True:
            self.num_ports_chopped_ant = num_ports - 3 # the number of "chopped" cap ports, one per strap 
            if self.num_ports_chopped_ant != self.num_straps:
                raise ValueError(f'The number of straps {self.num_straps} does not match number of chopped ports {self.num_ports_chopped_ant}')
            
            freqs = np.real(data[:, 0])
            ufreqs = np.unique(freqs)
            num_freqs = ufreqs.shape[0]
            i_f = np.where(ufreqs == f)[0][0]
            start_idx = i_f*(self.num_straps + 3)
            Smat = data[start_idx:(start_idx+self.num_straps + 3), 1:]

        else:
            self.num_ports_chopped_ant = num_ports - 2 # the number of "chopped" cap ports, one per strap 
            if self.num_ports_chopped_ant != self.num_straps:
                raise ValueError(f'The number of straps {self.num_straps} does not match number of chopped ports {self.num_ports_chopped_ant}')
            
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
        if self.center_fed_mode == True:
            z0s = [self.antz0, self.antz0, self.antz0] + [self.capz0]*self.num_straps # create a list of the antenna port z0s. The two coax ports are first
        else:
            z0s = [self.antz0, self.antz0] + [self.capz0]*self.num_straps

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
        returns: if self.center_fed_mode == True, then it will return S11, S21, S31. Else, it will return S11 and S21.
        """
        freqs = fullnet.frequency.f_scaled
        i_f = np.where(freqs == f)[0][0]

        if self.center_fed_mode == True:
            return fullnet.s[i_f][0,0], fullnet.s[i_f][0,1], fullnet.s[i_f][0,2]
        else:
            return fullnet.s[i_f][0,0], fullnet.s[i_f][0,1]

    def get_fullant_given_one_length(self, length):
        """
        length: length of all capactors (assumes caps all have same length) [cm]
        """
        freqs = self.freqs_for_fullant

        # antenna network 
        antnet1 = self.build_antnet_chopped(self.freqs_for_fullant, self.antfile, name='chopped ant network')


        portf = rf.Frequency.from_f(freqs, unit='MHz')
        cap_list = []

        if self.center_fed_mode == True:
            # capacitor network 
            for i_port in range(self.num_straps):
                capname = '_cap_' + str(i_port+2) + f'_port_{i_port + 3}'
                cap_list.append(self.build_capnet_given_length(length=length, freqs=freqs, filename=self.capfile, round_level=3))
                cap_list[i_port].name = 'l_' + cap_list[i_port].name + capname

            port_in = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='input')
            port_out1 = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='output1')
            port_out2 = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='output2')


            # wire them together 
            connections = [
                [(antnet1, 0), (port_in, 0)],
                [(antnet1, 1), (port_out1, 0)],
                [(antnet1, 2), (port_out2, 0)]]
            
            for i in range(self.num_straps):
                connections = connections + [[(antnet1, i+3), (cap_list[i], 0)]]

            circuit_model = rf.Circuit(connections)
            full_network = circuit_model.network

        else: 
            # capacitor network 
            for i_port in range(self.num_straps):
                capname = '_cap_' + str(i_port+1) + f'_port_{i_port + 2}'
                cap_list.append(self.build_capnet_given_length(length=length, freqs=freqs, filename=self.capfile, round_level=3))
                cap_list[i_port].name = 'l_' + cap_list[i_port].name + capname

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
        """
        Will return S11, S21, S31 if self.center_fed_mode == True 
        """
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

        if self.center_fed_mode == True:
            for i_port in range(self.num_straps):
                capname = 'C_' + str(i_port+2) + f'_port_{i_port + 3}'
                cap_list.append(rf.Network(frequency=rf_freq_object, z=Z, z0=self.capz0, name=capname))

            portf = rf.Frequency.from_f(freqs, unit='MHz')

            port_in = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='input')
            port_out1 = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='output1')
            port_out2 = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='output2')

            # wire them together 
            connections = [
                [(antnet1, 0), (port_in, 0)],
                [(antnet1, 1), (port_out1, 0)],
                [(antnet1, 2), (port_out2, 0)]]
            
            for i in range(self.num_straps):
                connections = connections + [[(antnet1, i+3), (cap_list[i], 0)]]

            circuit_model = rf.Circuit(connections)
            full_network = circuit_model.network

        else:
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

        if self.center_fed_mode:
            S11v = np.array([])
            S21v = np.array([])
            S31v = np.array([])

            for i in range(ls.shape[0]):
                l = ls[i]
                S11, S21, S31 = self.get_fullant_S11_S12_given_one_length(l, f)
                S11abs = np.abs(S11)
                S21abs = np.abs(S21)
                S31abs = np.abs(S31)
                S11v = np.append(S11v, S11abs)
                S21v = np.append(S21v, S21abs)
                S31v = np.append(S31v, S31abs)

            fig, axs = plt.subplots(1, 3, figsize=(12, 5))

            axs[0].plot(ls*100, S11v, label='|S11|')
            axs[0].set_ylabel('|S|')
            axs[0].set_xlabel('Cap length [cm]')
            axs[0].grid()
            axs[0].legend()

            axs[1].plot(ls*100, S21v, label='|S21|', color='red')
            axs[1].plot(ls*100, S31v, label='|S31|', color='orange')
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
                return S11v, S21v, S31v, axs
        else:
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

        if self.center_fed_mode == True:
            self.num_ports_chopped_ant = num_ports - 3 # the number of "chopped" cap ports, one per strap 
            if self.num_ports_chopped_ant != self.num_straps:
                raise ValueError(f'The number of straps {self.num_straps} does not match nnumber of chopped ports {self.num_ports_chopped_ant}')
            
            freqs = np.real(data[:, 0])
            ufreqs = np.unique(freqs)
            num_freqs = ufreqs.shape[0]
            i_f = np.where(ufreqs == f)[0][0]
            start_idx = i_f*(self.num_straps + 3)
            Smat = data[start_idx:(start_idx+self.num_straps + 3), 1:]

        else:
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
        if self.center_fed_mode == True:
            z0s = [self.antz0, self.antz0, self.antz0] + [self.capz0]*self.num_straps # create a list of the antenna port z0s. The two coax ports are first
        
        else:
            z0s = [self.antz0, self.antz0] + [self.capz0]*self.num_straps

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

        if self.center_fed_mode == True:
            for i_port in range(self.num_straps):
                capname = '_cap_' + str(i_port+2) + f'_port_{i_port + 3}'
                cap_list.append(self.build_capnet_given_length_from_internal_datatable(length=length, freqs=freqs, round_level=3))
                cap_list[i_port].name = 'l_' + cap_list[i_port].name + capname

            portf = rf.Frequency.from_f(freqs, unit='MHz')

            port_in = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='input')
            port_out1 = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='output1')
            port_out2 = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='output2')


            # wire them together 
            connections = [
                [(antnet1, 0), (port_in, 0)],
                [(antnet1, 1), (port_out1, 0)],
                [(antnet1, 2), (port_out2, 0)]]
            
            for i in range(self.num_straps):
                connections = connections + [[(antnet1, i+3), (cap_list[i], 0)]]

            circuit_model = rf.Circuit(connections)
            full_network = circuit_model.network

        else:
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
        """
        Will return S11, S21, S31 if self.center_fed_mode == True 
        """
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


        if self.center_fed_mode == True:
            for i_port in range(self.num_straps):
                capname = 'C_' + str(i_port+2) + f'_port_{i_port + 3}'
                cap_list.append(rf.Network(frequency=rf_freq_object, z=Z, z0=self.capz0, name=capname))

            portf = rf.Frequency.from_f(freqs, unit='MHz')

            port_in = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='input')
            port_out1 = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='output1')
            port_out2 = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='output2')


            # wire them together 
            connections = [
                [(antnet1, 0), (port_in, 0)],
                [(antnet1, 1), (port_out1, 0)],
                [(antnet1, 2), (port_out2, 0)]]
            
            for i in range(self.num_straps):
                connections = connections + [[(antnet1, i+3), (cap_list[i], 0)]]

            circuit_model = rf.Circuit(connections)
            full_network = circuit_model.network
        else:
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
        if self.center_fed_mode == True: 
            S11v = np.array([])
            S21v = np.array([])
            S31v = np.array([])

            for i in range(ls.shape[0]):
                l = ls[i]
                S11, S21, S31 = self.get_fullant_S11_S12_given_one_length_from_internal_datatable(l, f)
                S11abs = np.abs(S11)
                S21abs = np.abs(S21)
                S31abs = np.abs(S31)
                S11v = np.append(S11v, S11abs)
                S21v = np.append(S21v, S21abs)
                S31v = np.append(S31v, S31abs)

            fig, axs = plt.subplots(1, 3, figsize=(12, 5))

            axs[0].plot(ls*100, S11v, label='|S11|')
            axs[0].set_ylabel('|S|')
            axs[0].set_xlabel('Cap length [cm]')
            axs[0].grid()
            axs[0].legend()

            axs[1].plot(ls*100, S21v, label='|S21|', color='red')
            axs[1].plot(ls*100, S31v, label='|S31|', color='orange')
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
                return S11v, S21v, S31v, axs
        else:
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
        if self.center_fed_mode == True:
            S11v = np.array([])
            S21v = np.array([])
            S31v = np.array([])

            for i in range(fs.shape[0]):
                f = fs[i]
                S11, S21, S31 = self.get_fullant_S11_S12_given_one_length_from_internal_datatable(l, f)
                S11abs = np.abs(S11)
                S21abs = np.abs(S21)
                S31abs = np.abs(S31)
                S11v = np.append(S11v, S11abs)
                S21v = np.append(S21v, S21abs)
                S31v = np.append(S31v, S31abs)

            fig, axs = plt.subplots(1, 3, figsize=(12, 5))

            axs[0].plot(fs, S11v, label='|S11|')
            axs[0].set_ylabel('|S|')
            axs[0].set_xlabel('f [MHz]')
            axs[0].grid()
            axs[0].legend()

            axs[1].plot(fs, S21v, label='|S21|', color='red')
            axs[1].plot(fs, S31v, label='|S31|', color='orange')
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
                return S11v, S21v, S31v, axs
        else:
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

        self.S11_real_interpolator = RectBivariateSpline(fs, ls, S11_real)
        self.S11_imag_interpolator = RectBivariateSpline(fs, ls, S11_imag)
        # fsmesh, lsmesh = np.meshgrid(fs, ls, indexing='ij')

        # self.S11_real_interpolator = interp2d(fsmesh, lsmesh, S11_real, kind='linear') # TODO: switching from cubic to linear
        # self.S11_imag_interpolator = interp2d(fsmesh, lsmesh, S11_imag, kind='linear')
        
    def interpolate_cap_data(self, f, l):
        S11 = self.S11_real_interpolator(f, l) + 1j*self.S11_imag_interpolator(f,l)
        return S11[0]
    
    def set_ant_Smat_interpolator(self):
        if self.center_fed_mode:
            num_ports = self.num_straps + 3
        else:
            num_ports = self.num_straps + 2

        fs = self.freqs_for_fullant
        s = self.build_antnet_chopped_from_internal_datatable(fs, name=None).s
        interp_matrix_real = []
        interp_matrix_imag = []


        for i in range(num_ports):
            interp_row_list_real = []
            interp_row_list_imag = []
            for j in range(num_ports):  #TODO: changed from interp1d to PchipInterpolator
                interp_row_list_real.append(PchipInterpolator(fs, np.real(s[:,i,j]))) # interpolate the real part 
                interp_row_list_imag.append(PchipInterpolator(fs, np.imag(s[:,i,j]))) # interpolate the imag part

            interp_matrix_real.append(interp_row_list_real)
            interp_matrix_imag.append(interp_row_list_imag)

        self.interp_matrix_real = interp_matrix_real
        self.interp_matrix_imag = interp_matrix_imag

    
    def interpolate_sant_for_any_f(self, f):
        interp_matrix_real = self.interp_matrix_real
        interp_matrix_imag = self.interp_matrix_imag

        if self.center_fed_mode == True:
            Smat = np.zeros((self.num_straps+3, self.num_straps+3), dtype='complex') # initialize the s matrix 

            for i in range(self.num_straps+3):
                for j in range(self.num_straps+3):
                    Smat[i,j] = interp_matrix_real[i][j](f) + 1j*interp_matrix_imag[i][j](f)

        else:
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


    def get_fullant_given_lengths_from_internal_datatable(self, lengths, symetric_mode=False,
                                                          one_cap_type_mode=False,
                                                          end_cap_mode=False,
                                                          return_circ=False):
        """
        lengths: a list of lengths, must be same number as the number of ports/2 for even num straps, (n-1)/2 for odd
        if running in symetric mode. If running in one_cap_type mode, then should be a list of one value [length1].
        if not either of these, then must be the same length as the number of straps manually specifying all caps.
        """
        freqs = self.freqs_for_fullant
        if type(lengths) != list:
            lengths = lengths.tolist()
        # print('lengths:', lengths)
        # print('len(lengths):', len(lengths))
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
                    raise ValueError('The lengths array is not the correct length for symetric mode')
                for i in range(1,len(lengths_reverse)):
                    lengths.append(lengths_reverse[i])
        elif one_cap_type_mode:
            if len(lengths) != 1:
                raise ValueError('The lengths array is not the correct length for one_cap_type mode')
            lengths = lengths*self.num_straps  # make the lengths array an array full of the same value

        elif end_cap_mode:
            if len(lengths) != 2:
                raise ValueError('The lengths array is not the correct length for end_cap_mode')
            
            endcaps = lengths[0]
            midcaps = lengths[1]
            lengths = [endcaps] + (self.num_straps - 2)*[midcaps] + [endcaps]
        else:
            if self.num_straps != len(lengths):
                raise ValueError('The lengths array is not the correct length for this mode')

        if self.center_fed_mode == True:
            cap_list = []
            for i_port in range(self.num_straps):
                capname = '_cap_' + str(i_port+2) + f'_port_{i_port + 3}'
                cap_list.append(self.build_capnet_given_length_interpolated(length=lengths[i_port], freqs=freqs))
                cap_list[i_port].name = 'l_' + cap_list[i_port].name + capname

            portf = rf.Frequency.from_f(freqs, unit='MHz')

            port_in = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='input')
            port_out1 = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='output1')
            port_out2 = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='output2')


            # wire them together 
            connections = [
                [(antnet1, 0), (port_in, 0)],
                [(antnet1, 1), (port_out1, 0)],
                [(antnet1, 2), (port_out2, 0)]]
            
            for i in range(self.num_straps):
                connections = connections + [[(antnet1, i+3), (cap_list[i], 0)]]

            circuit_model = rf.Circuit(connections)
            full_network = circuit_model.network

        else:
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


        # if the user requests the full circuit, return the circuit
        if return_circ:
            return circuit_model
        else:
            return full_network
    
    
    def get_fullant_given_Cs_via_caps_from_internal_datatable(self, Cs, symetric_mode=False, one_cap_type_mode=False, end_cap_mode=False, different_freq=False, new_freqs=np.array([])):
        """
        Cs: capacitance list [F] of strap caps, an array of size numstraps 
        """
        if different_freq == True:
            freqs = new_freqs
        else:
            freqs = self.freqs_for_fullant

        if type(Cs) != list:
            Cs = Cs.tolist()

        rf_freq_object = rf.Frequency.from_f(freqs, unit='MHz')

        # antenna network 
        antnet1 = self.build_antnet_chopped_from_internal_datatable(self.freqs_for_fullant, name='chopped ant network')

        # capacitor network 
        if symetric_mode:
            Cs_reverse = Cs.copy()
            Cs_reverse.reverse()
            if self.num_straps % 2 == 0:
                if len(Cs_reverse) != int(self.num_straps/2):
                    raise ValueError('The Cs array is not the correct length')
                for i in range(len(Cs_reverse)):
                    Cs.append(Cs_reverse[i])

            elif self.num_straps % 2 != 0:
                if len(Cs_reverse) != int((self.num_straps+1)/2):
                    raise ValueError('The Cs array is not the correct length for symetric mode')
                for i in range(1,len(Cs_reverse)):
                    Cs.append(Cs_reverse[i])
        elif one_cap_type_mode:
            if len(Cs) != 1:
                raise ValueError('The Cs array is not the correct length for one_cap_type mode')
            Cs = Cs*self.num_straps  # make the lengths array an array full of the same value

        elif end_cap_mode:
            if len(Cs) != 2:
                raise ValueError('The Cs array is not the correct length for end_cap_mode')
            
            endcaps = Cs[0]
            midcaps = Cs[1]
            Cs = [endcaps] + (self.num_straps - 2)*[midcaps] + [endcaps]

        else:
            if self.num_straps != len(Cs):
                raise ValueError('The Cs array is not the correct length for this mode')
        capZs = []
        for i in range(len(Cs)):
            capZ = 1 / (1j * 2 * rf.pi * rf_freq_object.f * Cs[i]) # this is a numpy array of length frequency list 
            Z = np.zeros((rf_freq_object.f.shape[0],1,1), dtype='complex') # this impedence needs to be shape number of frequencies, 1, 1 
            Z[:,0,0] = capZ
            capZs.append(Z)

        cap_list = []

        if self.center_fed_mode == True:
            for i_port in range(self.num_straps):
                capname = 'C_' + str(i_port+2) + f'_port_{i_port + 3}'
                cap_list.append(rf.Network(frequency=rf_freq_object, z=capZs[i_port], z0=self.capz0, name=capname))

            portf = rf.Frequency.from_f(freqs, unit='MHz')

            port_in = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='input')
            port_out1 = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='output1')
            port_out2 = rf.Circuit.Port(frequency=portf, z0=self.antz0, name='output2')


            # wire them together 
            connections = [
                [(antnet1, 0), (port_in, 0)],
                [(antnet1, 1), (port_out1, 0)],
                [(antnet1, 2), (port_out2, 0)]]
            
            for i in range(self.num_straps):
                connections = connections + [[(antnet1, i+3), (cap_list[i], 0)]]

            circuit_model = rf.Circuit(connections)
            full_network = circuit_model.network

        else:
            for i_port in range(self.num_straps):
                capname = 'C_' + str(i_port+1) + f'_port_{i_port + 2}'
                cap_list.append(rf.Network(frequency=rf_freq_object, z=capZs[i_port], z0=self.capz0, name=capname))

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
    
    def op_info(self, i_iter, p):
        """
        Print information during the fitting procedure
        """
        print("-" * 40)
        print(f"i_iter = {i_iter}")
        print("New simulation.")
        print(f"Point is: {p}")

    def run_optimization(self, initial_guess, length_bounds, S11_db_cutouff, freq_bounds, method, options, symetric_mode=False, one_cap_type_mode=False, end_cap_mode=False):
        self.i_iter = 0
        self.prms = []
        self.errors = []
        self.symetric_mode = symetric_mode
        self.one_cap_type_mode = one_cap_type_mode
        self.end_cap_mode = end_cap_mode
        self.freq_bounds_for_optimization = freq_bounds
        self.S11_db_cutouff = S11_db_cutouff
        res = minimize(self.error_function,
               initial_guess,
               bounds=length_bounds,
               method=method,
               options=options)
        
        return res


    def error_function(self, prm):

        self.prms.append(prm)
        self.i_iter += 1
        self.op_info(self.i_iter, prm)
        # Filter the results if a negative value is found
        if any([e < 0 for e in prm]):
            return 1e30
        
        network = self.get_fullant_given_lengths_from_internal_datatable(lengths=prm, symetric_mode=self.symetric_mode, 
                                                                         one_cap_type_mode=self.one_cap_type_mode,
                                                                         end_cap_mode=self.end_cap_mode) 

        S11_array = np.zeros_like(self.freqs_for_fullant, dtype='complex')

        for i in range(S11_array.shape[0]):
            S11 = self.get_full_TWA_network_S11_S21(fullnet=network, f=self.freqs_for_fullant[i])[0] # TODO: changed howe first return is handled
            S11_array[i] = S11


        err = 0

        for i in range(S11_array.shape[0]):
            
            S11_mag = np.abs(S11_array[i])
            S11_db = 20*np.log10(S11_mag)
            # only contribute to error if we are between the desired frequency range 
            if self.freqs_for_fullant[i] >= self.freq_bounds_for_optimization[0] and self.freqs_for_fullant[i] <= self.freq_bounds_for_optimization[1]:
            
                if S11_db <= self.S11_db_cutouff: 
                    err = err + 0
                else:
                    err = err + (S11_db - self.S11_db_cutouff)**2 # squared error if the value of S11 is above -30 
        
        print(f"Average absolute error is : {err:.2e}")
        self.errors.append(err)
        return err    


    def run_optimization_explicitC(self, initial_guess, cap_bounds, S11_db_cutouff, freq_bounds, method, options, symetric_mode=False, one_cap_type_mode=False, end_cap_mode=False):
        self.i_iter = 0
        self.prms = []
        self.errors = []
        self.symetric_mode = symetric_mode
        self.one_cap_type_mode = one_cap_type_mode
        self.end_cap_mode = end_cap_mode
        self.freq_bounds_for_optimization = freq_bounds
        self.S11_db_cutouff = S11_db_cutouff
        res = minimize(self.error_function_explicitC,
               initial_guess,
               bounds=cap_bounds,
               method=method,
               options=options)
        
        return res


    def error_function_explicitC(self, prm):

        self.prms.append(prm)
        self.i_iter += 1
        self.op_info(self.i_iter, prm)
        # Filter the results if a negative value is found
        if any([e < 0 for e in prm]):
            return 1e30
        
        network = self.get_fullant_given_Cs_via_caps_from_internal_datatable(Cs=prm, symetric_mode=self.symetric_mode, 
                                                                         one_cap_type_mode=self.one_cap_type_mode,
                                                                         end_cap_mode=self.end_cap_mode) 

        S11_array = np.zeros_like(self.freqs_for_fullant, dtype='complex')

        for i in range(S11_array.shape[0]):
            S11 = self.get_full_TWA_network_S11_S21(fullnet=network, f=self.freqs_for_fullant[i])[0]
            S11_array[i] = S11


        err = 0

        for i in range(S11_array.shape[0]):
            
            S11_mag = np.abs(S11_array[i])
            S11_db = 20*np.log10(S11_mag)
            # only contribute to error if we are between the desired frequency range 
            if self.freqs_for_fullant[i] >= self.freq_bounds_for_optimization[0] and self.freqs_for_fullant[i] <= self.freq_bounds_for_optimization[1]:
            
                if S11_db <= self.S11_db_cutouff: 
                    err = err + 0
                else:
                    err = err + (S11_db - self.S11_db_cutouff)**2 # squared error if the value of S11 is above -30 
        
        print(f"Average absolute error is : {err:.2e}")
        self.errors.append(err)
        return err 
    
    def run_differential_evolution_global_op(self, 
                                            length_bounds,
                                            S11_db_cutouff,
                                            freq_bounds,
                                            strategy='best1bin',
                                            symetric_mode=False,
                                            one_cap_type_mode=False,
                                            end_cap_mode=False):
        self.i_iter = 0
        self.prms = []
        self.errors = []
        self.symetric_mode = symetric_mode
        self.one_cap_type_mode = one_cap_type_mode
        self.end_cap_mode = end_cap_mode
        self.freq_bounds_for_optimization = freq_bounds
        self.S11_db_cutouff = S11_db_cutouff
        res = differential_evolution(self.error_function, bounds=length_bounds, strategy=strategy)
        
        return res
        
    def run_differential_evolution_global_op_explicitC(self, 
                                            cap_bounds,
                                            S11_db_cutouff,
                                            freq_bounds,
                                            strategy='best1bin',
                                            symetric_mode=False,
                                            one_cap_type_mode=False,
                                            end_cap_mode=False):
        self.i_iter = 0
        self.prms = []
        self.errors = []
        self.symetric_mode = symetric_mode
        self.one_cap_type_mode = one_cap_type_mode
        self.end_cap_mode = end_cap_mode
        self.freq_bounds_for_optimization = freq_bounds
        self.S11_db_cutouff = S11_db_cutouff
        res = differential_evolution(self.error_function_explicitC, bounds=cap_bounds, strategy=strategy)
        
        return res

    def run_differential_evolution_global_op_l_matters(self, 
                                            length_bounds,
                                            S11_db_cutouff,
                                            freq_bounds,
                                            beta_length_op,
                                            strategy='best1bin',
                                            symetric_mode=False,
                                            one_cap_type_mode=False,
                                            end_cap_mode=False):
        """
        This version of the optimization also aims to force the lengths to be similar with weight beta
        """
        self.i_iter = 0
        self.prms = []
        self.errors = []
        self.symetric_mode = symetric_mode
        self.one_cap_type_mode = one_cap_type_mode
        self.end_cap_mode = end_cap_mode
        self.freq_bounds_for_optimization = freq_bounds
        self.S11_db_cutouff = S11_db_cutouff
        self.beta_length_op = beta_length_op
        res = differential_evolution(self.error_function_l_matters, bounds=length_bounds, strategy=strategy)
        
        return res
     
    def error_function_l_matters(self, prm):

        """
        This error function uses a new parameter, self.beta_length_op, to add error when the cap lengths are not
        close to eachother, which is beta*sum((length - average length)^2) divided by the square of the average length. 
        """

        self.prms.append(prm)
        self.i_iter += 1
        self.op_info(self.i_iter, prm)
        # Filter the results if a negative value is found
        if any([e < 0 for e in prm]):
            return 1e30
        
        network = self.get_fullant_given_lengths_from_internal_datatable(lengths=prm, symetric_mode=self.symetric_mode, 
                                                                         one_cap_type_mode=self.one_cap_type_mode,
                                                                         end_cap_mode=self.end_cap_mode) 

        S11_array = np.zeros_like(self.freqs_for_fullant, dtype='complex')

        for i in range(S11_array.shape[0]):
            S11 = self.get_full_TWA_network_S11_S21(fullnet=network, f=self.freqs_for_fullant[i])[0]
            S11_array[i] = S11


        err = 0

        for i in range(S11_array.shape[0]):
            
            S11_mag = np.abs(S11_array[i])
            S11_db = 20*np.log10(S11_mag)
            # only contribute to error if we are between the desired frequency range 
            if self.freqs_for_fullant[i] >= self.freq_bounds_for_optimization[0] and self.freqs_for_fullant[i] <= self.freq_bounds_for_optimization[1]:
            
                if S11_db <= self.S11_db_cutouff: 
                    err = err + 0
                else:
                    err = err + (S11_db - self.S11_db_cutouff)**2 # squared error if the value of S11 is above -30 

            # New section with error created via the lengths not being similar 
            if self.symetric_mode:
                length_sum = 0
                for i in range(len(prm)-1):
                    length_sum += 2*prm[i]
                length_av = (length_sum + prm[-1])/self.num_straps 

                length_error = np.sum(2*(np.array(prm[:-1]) - length_av)**2) + (prm[-1] - length_av)**2
                length_error = self.beta_length_op*length_error/length_av**2

            elif self.one_cap_type_mode:
                raise ValueError('You cannot use this function with one_cap_type_mode==True: the caps are all the same')
            
            else:
                length_sum = 0

                for i in range(len(prm)):
                    length_sum += prm[i]

                length_av = (length_sum)/self.num_straps
                length_error = np.sum((np.array(prm) - length_av)**2)
                length_error = self.beta_length_op*length_error/length_av**2  

            err += length_error            


        
        print(f"Average absolute error is : {err:.2e}")
        self.errors.append(err)
        return err 

    def analytic_power_spectrum_general_phase_diff(self, npar, w_strap, d_strap, freq, phase_array):
        """
        npar: n|| = k|| c / omega
        w_strap: strap width in meters
        d_strap: strap seperation in meters
        freq: frequency in Hz of the spectrum
        phase_array: numpy array of all the strap phases in radians 

        from equation 1.38 in Greg Wallace's thesis Behavior of Lower Hybrid Waves in the Scrape-Off Layer of a Diverted Tokamak
        """

        w = 2*np.pi*freq
        insin = npar*w_strap*w/(2*self.clight)
        
        # deal with the sinc function zero 
        if npar == 0:
            term1 = (w_strap*w/(2*self.clight))**2 # the lim of sin^2(a*x)/x^2 = a^2 as x -> 0
        else: 
            term1 = np.sin(insin)**2 / npar**2

        sum1 = 0
        j = 0
        for phase in phase_array:
            beta = npar*(w/self.clight)*(j*d_strap) # this ignores w_strap/2, as discussed in Greg Wallace's thesis 
            sum1 += np.exp(-1j*(beta + phase))
            j += 1

        sum2 = 0
        j = 0
        for phase in phase_array:
            beta = npar*(w/self.clight)*(j*d_strap) # this ignores w_strap/2, as discussed in Greg Wallace's thesis
            sum2 += np.exp(+1j*(beta + phase))
            j += 1

        return term1 * sum1 * sum2 
    
    def get_phase(self, complexn):
        real = np.real(complexn)
        imag = np.imag(complexn)
        angle_rad = np.arctan2(imag, real)
        return angle_rad
    
    def get_peak_npar_spectrum(self, lengths, npar_bounds, freq, num_npars=1000, power=[1,0], phase=[0,0],
                                symetric_mode=True,
                                one_cap_type_mode=False,
                                end_cap_mode=False):
        """
        lengths: list of capacitor lengths
        npar_bounds: iterable of size 2: the upper and lower bounds of the npar spectrum 
        num_npars: the number of npars used to plot to find the maximum 
        freq: frquency of specturm in Hz
        power: iterable size 2, contains the external port excitation 
        phase: same as above but for the phase 
        """
        full_circ = self.get_fullant_given_lengths_from_internal_datatable(lengths, symetric_mode,
                                                          one_cap_type_mode,
                                                          end_cap_mode,
                                                          return_circ=True)
        
        idx = np.where(full_circ.frequency.f_scaled == freq/1e6) # get the index of the frequency in question 
        #print('BOOM: here u go, the f you want is:', full_circ.frequency.f_scaled[idx])
        
        if self.center_fed_mode:
            strap_current_array = full_circ.currents(power,phase)[idx,:].reshape(self.num_straps + 3,2)[:,1][3:]  # remove double counting, remove three or two external ports
        else:
            strap_current_array = full_circ.currents(power,phase)[idx,:].reshape(self.num_straps + 2,2)[:,1][2:]

        strap_phases = self.get_phase(strap_current_array)
        npar_array = np.linspace(npar_bounds[0], npar_bounds[1], num_npars)
        result_circ_model = np.array([], dtype='complex')
        
        for i in range(npar_array.shape[0]):
            power = self.analytic_power_spectrum_general_phase_diff(npar_array[i], w_strap=self.geometry_dict['wstrap'],
                                                                        d_strap=self.geometry_dict['d'],
                                                                        freq=freq,
                                                                        phase_array=strap_phases)
            result_circ_model = np.append(result_circ_model, power)

        npar_max = npar_array[np.where(result_circ_model == np.max(result_circ_model))[0][0]]

        power_max = self.analytic_power_spectrum_general_phase_diff(npar_max, w_strap=self.geometry_dict['wstrap'],
                                                                        d_strap=self.geometry_dict['d'],
                                                                        freq=freq,
                                                                        phase_array=strap_phases)
        
        power_zero = self.analytic_power_spectrum_general_phase_diff(0, w_strap=self.geometry_dict['wstrap'],
                                                                        d_strap=self.geometry_dict['d'],
                                                                        freq=freq,
                                                                        phase_array=strap_phases)        

        return npar_max, np.real(power_max), np.real(power_zero) 
    
    # now, define a new cost function that allows the user to control how strong the effect of the npar, max is.

    def run_differential_evolution_global_op_npar_match(self, 
                                            length_bounds,
                                            S11_db_cutouff,
                                            freq, # in Hz
                                            freq_bounds,
                                            alpha_npar_op,
                                            target_npar,
                                            npar_bounds, # for finding the maximum for the optimization 
                                            num_npars,
                                            strategy='best1bin',
                                            symetric_mode=False,
                                            one_cap_type_mode=False,
                                            end_cap_mode=False):
        """
        This version of the optimization also aims to try and also match the desired npar performance. 
        """
        self.i_iter = 0
        self.prms = []
        self.errors = []
        self.symetric_mode = symetric_mode
        self.one_cap_type_mode = one_cap_type_mode
        self.end_cap_mode = end_cap_mode
        self.freq_bounds_for_optimization = freq_bounds
        self.S11_db_cutouff = S11_db_cutouff
        self.alpha_npar_op = alpha_npar_op
        self.target_npar = target_npar # this sets the target npar 
        self.npar_bounds_for_npar_op = npar_bounds
        self.num_npars_for_npar_op = num_npars
        self.freq_for_npar_op = freq
        res = differential_evolution(self.error_function_npar_match, bounds=length_bounds, strategy=strategy)
        
        return res
     
    def error_function_npar_match(self, prm):

        """
        This error function uses a new parameter, self.beta_length_op, to add error when the cap lengths are not
        close to eachother, which is beta*sum((length - average length)^2) divided by the square of the average length. 
        """

        self.prms.append(prm)
        self.i_iter += 1
        self.op_info(self.i_iter, prm)
        # Filter the results if a negative value is found
        if any([e < 0 for e in prm]):
            return 1e30
        
        network = self.get_fullant_given_lengths_from_internal_datatable(lengths=prm, symetric_mode=self.symetric_mode, 
                                                                         one_cap_type_mode=self.one_cap_type_mode,
                                                                         end_cap_mode=self.end_cap_mode) 

        S11_array = np.zeros_like(self.freqs_for_fullant, dtype='complex')

        for i in range(S11_array.shape[0]):
            S11 = self.get_full_TWA_network_S11_S21(fullnet=network, f=self.freqs_for_fullant[i])[0]
            S11_array[i] = S11


        err = 0

        for i in range(S11_array.shape[0]):
            
            S11_mag = np.abs(S11_array[i])
            S11_db = 20*np.log10(S11_mag)
            # only contribute to error if we are between the desired frequency range 
            if self.freqs_for_fullant[i] >= self.freq_bounds_for_optimization[0] and self.freqs_for_fullant[i] <= self.freq_bounds_for_optimization[1]:
            
                if S11_db <= self.S11_db_cutouff: 
                    err = err + 0
                else:
                    err = err + (S11_db - self.S11_db_cutouff)**2 # squared error if the value of S11 is above the cuttoff

                # New section with error created via the npar peak not being in the correct place.  
                if self.center_fed_mode == True:
                    found_npar_peak = self.get_peak_npar_spectrum(lengths=prm,
                                                                npar_bounds=self.npar_bounds_for_npar_op,
                                                                freq=self.freq_for_npar_op,
                                                                num_npars=self.num_npars_for_npar_op,
                                                                power=[1,0,0],
                                                                phase=[0,0,0],
                                                                symetric_mode=self.symetric_mode,
                                                                one_cap_type_mode=self.one_cap_type_mode,
                                                                end_cap_mode=self.end_cap_mode)[0]
                else:
                    found_npar_peak = self.get_peak_npar_spectrum(lengths=prm,
                                                                npar_bounds=self.npar_bounds_for_npar_op,
                                                                freq=self.freq_for_npar_op,
                                                                num_npars=self.num_npars_for_npar_op,
                                                                power=[1,0],
                                                                phase=[0,0],
                                                                symetric_mode=self.symetric_mode,
                                                                one_cap_type_mode=self.one_cap_type_mode,
                                                                end_cap_mode=self.end_cap_mode)[0]
                
                # npar_error = self.alpha_npar_op*(found_npar_peak + 2.05 - self.target_npar)**2/(self.target_npar**2)  # TODO: the 2.05 here is found manually and not convinced it applies everywhere 
                npar_error = self.alpha_npar_op*(found_npar_peak - self.target_npar)**2/(self.target_npar**2)  # TODO: the 2.05 here is found manually and not convinced it applies everywhere
                err += npar_error            


        
        print(f"Average absolute error is : {err:.2e}")
        self.errors.append(err)
        return err 
 
    
    def run_differential_evolution_global_op_npar_match_low_npar_zero(self, 
                                            length_bounds,
                                            S11_db_cutouff,
                                            freq, # in Hz
                                            freq_bounds,
                                            alpha_npar_op,
                                            gamma_npar_op,
                                            target_npar,
                                            npar_bounds, # for finding the maximum for the optimization 
                                            num_npars,
                                            strategy='best1bin',
                                            symetric_mode=False,
                                            one_cap_type_mode=False,
                                            end_cap_mode=False):
        """
        This version of the optimization also aims to try and also match the desired npar performance. both the
        peak location (weighted by alpha) and the peak at zero (weighted by gamma) are added to the cost 
        """
        self.i_iter = 0
        self.prms = []
        self.errors = []
        self.symetric_mode = symetric_mode
        self.one_cap_type_mode = one_cap_type_mode
        self.end_cap_mode = end_cap_mode
        self.freq_bounds_for_optimization = freq_bounds
        self.S11_db_cutouff = S11_db_cutouff
        self.alpha_npar_op = alpha_npar_op
        self.gamma_npar_op = gamma_npar_op
        self.target_npar = target_npar # this sets the target npar 
        self.npar_bounds_for_npar_op = npar_bounds
        self.num_npars_for_npar_op = num_npars
        self.freq_for_npar_op = freq
        res = differential_evolution(self.error_function_npar_match_low_npar_zero, bounds=length_bounds, strategy=strategy)
        
        return res
     
    def error_function_npar_match_low_npar_zero(self, prm):

        """
        This error function uses a new parameter, self.beta_length_op, to add error when the cap lengths are not
        close to eachother, which is beta*sum((length - average length)^2) divided by the square of the average length. 
        """

        self.prms.append(prm)
        self.i_iter += 1
        self.op_info(self.i_iter, prm)
        # Filter the results if a negative value is found
        if any([e < 0 for e in prm]):
            return 1e30
        
        network = self.get_fullant_given_lengths_from_internal_datatable(lengths=prm, symetric_mode=self.symetric_mode, 
                                                                         one_cap_type_mode=self.one_cap_type_mode,
                                                                         end_cap_mode=self.end_cap_mode) 

        S11_array = np.zeros_like(self.freqs_for_fullant, dtype='complex')

        for i in range(S11_array.shape[0]):
            S11 = self.get_full_TWA_network_S11_S21(fullnet=network, f=self.freqs_for_fullant[i])[0]
            S11_array[i] = S11


        err = 0

        for i in range(S11_array.shape[0]):
            
            S11_mag = np.abs(S11_array[i])
            S11_db = 20*np.log10(S11_mag)
            # only contribute to error if we are between the desired frequency range 
            if self.freqs_for_fullant[i] >= self.freq_bounds_for_optimization[0] and self.freqs_for_fullant[i] <= self.freq_bounds_for_optimization[1]:
            
                if S11_db <= self.S11_db_cutouff: 
                    err = err + 0
                else:
                    err = err + (S11_db - self.S11_db_cutouff)**2 # squared error if the value of S11 is above the cuttoff

                # New section with error created via the npar peak not being in the correct place.  
                if self.center_fed_mode == True:
                    found_npar_peak, power_max, power_zero = self.get_peak_npar_spectrum(lengths=prm,
                                                                npar_bounds=self.npar_bounds_for_npar_op,
                                                                freq=self.freq_for_npar_op,
                                                                num_npars=self.num_npars_for_npar_op,
                                                                power=[1,0,0],
                                                                phase=[0,0,0],
                                                                symetric_mode=self.symetric_mode,
                                                                one_cap_type_mode=self.one_cap_type_mode,
                                                                end_cap_mode=self.end_cap_mode)
                else:
                    found_npar_peak, power_max, power_zero = self.get_peak_npar_spectrum(lengths=prm,
                                                                npar_bounds=self.npar_bounds_for_npar_op,
                                                                freq=self.freq_for_npar_op,
                                                                num_npars=self.num_npars_for_npar_op,
                                                                power=[1,0],
                                                                phase=[0,0],
                                                                symetric_mode=self.symetric_mode,
                                                                one_cap_type_mode=self.one_cap_type_mode,
                                                                end_cap_mode=self.end_cap_mode)
                
                # npar_error = self.alpha_npar_op*(found_npar_peak + 2.05 - self.target_npar)**2/(self.target_npar**2)  # TODO: the 2.05 here is found manually and not convinced it applies everywhere 
                npar_error = self.alpha_npar_op*(found_npar_peak - self.target_npar)**2/(self.target_npar**2)  # TODO: the 2.05 here is found manually and not convinced it applies everywhere
                
                # add the P(n||=0) error
                npar_zero_error = self.gamma_npar_op * power_zero / power_max

                err += npar_error + npar_zero_error           


        
        print(f"Average absolute error is : {err:.2e}")
        self.errors.append(err)
        return err 
    # The below function does not work because the antenna network can not be wired to a capacitor of arb. f. 
    # def plot_S11_S21_v_f_using_caps_to_increase_f_range(self, lengths, new_f_range, f0, symetric_mode=False, one_cap_type_mode=False):
    #     """
    #     lengths: the optimized lengths form the narrower f range
    #     new_f_range: the larger frequency range one wishes to plot over, a numpy array
    #     f0: the center frequency in units of MHz. This is the f at which the C of the capacitor is extracted from 
    #     """
    #     caps = []
    #     Cs = []

    #     for i in range(len(lengths)):
    #         caps.append(self.build_capnet_given_length_interpolated(length=lengths[i], freqs=self.freqs_for_fullant))
        
    #     for j in range(len(caps)):
    #         Z, C = self.print_znorm_and_capacitance(caps[j], f0, toprint=False)
    #         Cs.append(C)

    #     ant_with_Cs = self.get_fullant_given_Cs_via_caps_from_internal_datatable(Cs, 
    #                                                             symetric_mode=symetric_mode,
    #                                                             one_cap_type_mode=one_cap_type_mode,
    #                                                             different_freq=True,
    #                                                             new_freqs=new_f_range)
        
    #     ant_with_lengths = self.get_fullant_given_lengths_from_internal_datatable(lengths,
    #                                                                               symetric_mode=symetric_mode,
    #                                                                               one_cap_type_mode=one_cap_type_mode)

    #     S11_with_Cs_array = np.zeros_like(new_f_range, dtype='complex')
    #     S21_with_Cs_array = np.zeros_like(new_f_range, dtype='complex')

    #     S11_with_lengths_array = np.zeros_like(self.freqs_for_fullant, dtype='complex')
    #     S21_with_lengths_array = np.zeros_like(self.freqs_for_fullant, dtype='complex')

    #     # loop over frequencies and load up the S parameters for each frequency for new frequency range cap model 
    #     for i in range(S11_with_Cs_array.shape[0]):
    #         S11, S21 = self.get_full_TWA_network_S11_S21(fullnet=ant_with_Cs, f=new_f_range[i])
    #         S11_with_Cs_array[i] = S11
    #         S21_with_Cs_array[i] = S21

    #     # loop over frequencies and load up the S parameters for each frequency for old model 
    #     for i in range(S11_with_lengths_array.shape[0]):
    #         S11, S21 = self.get_full_TWA_network_S11_S21(fullnet=ant_with_lengths, f=self.freqs_for_fullant[i])
    #         S11_with_lengths_array[i] = S11
    #         S21_with_lengths_array[i] = S21

    #     S11_mag_with_Cs_array = np.zeros_like(new_f_range)
    #     S11_db_with_Cs_array = np.zeros_like(new_f_range)   
    #     S21_mag_with_Cs_array = np.zeros_like(new_f_range)
    #     S21_db_with_Cs_array = np.zeros_like(new_f_range)      

    #     S11_mag_with_lengths_array = np.zeros_like(self.freqs_for_fullant)
    #     S11_db_with_lengths_array = np.zeros_like(self.freqs_for_fullant)
    #     S21_mag_with_lengths_array = np.zeros_like(self.freqs_for_fullant)
    #     S21_db_with_lengths_array = np.zeros_like(self.freqs_for_fullant)

    #     # now, load up arrays of the S parameters for the new range in mag and dB scale. 
    #     for i in range(new_f_range.shape[0]):
    #         S11_mag_with_Cs_array[i] = np.abs(S11_with_Cs_array[i])
    #         S11_db_with_Cs_array[i] = 20*np.log10(S11_mag_with_Cs_array[i])
    #         S21_mag_with_Cs_array[i] = np.abs(S21_with_Cs_array[i])
    #         S21_db_with_Cs_array[i] =  20*np.log10(S21_mag_with_Cs_array[i])

    #     # now, load up arrays of the S parameters for the old range in mag and dB scale. 
    #     for i in range(self.freqs_for_fullant.shape[0]):
    #         S11_mag_with_lengths_array[i] = np.abs(S11_with_lengths_array[i])
    #         S11_db_with_lengths_array[i] = 20*np.log10(S11_mag_with_lengths_array[i])
    #         S21_mag_with_lengths_array[i] = np.abs(S21_with_lengths_array[i])
    #         S21_db_with_lengths_array[i] =  20*np.log10(S21_mag_with_lengths_array[i])

    #     # now, generate the figure 

    #     fig, ax = plt.subplots(2,2,figsize=(10,10))

    #     ax[0,0].plot(new_f_range, S11_mag_with_Cs_array, marker='.', color='blue', label='via C')
    #     ax[0,0].plot(self.freqs_for_fullant, S11_mag_with_lengths_array, marker='.', color='darkblue', label='via length')
    #     ax[0,0].grid()
    #     ax[0,0].set_xlabel('f [MHz]')
    #     ax[0,0].set_ylabel('|S11|')

    #     ax[0,1].plot(new_f_range, S11_db_with_Cs_array, marker='.', color='blue', label='via C')
    #     ax[0,1].plot(self.freqs_for_fullant, S11_db_with_lengths_array, marker='.', color='darkblue', label='via length')
    #     ax[0,1].grid()
    #     ax[0,1].set_xlabel('f [MHz]')
    #     ax[0,1].set_ylabel('20log10(|S11|)')

    #     ax[1,0].plot(new_f_range, S21_mag_with_Cs_array, marker='.', color='red', label='via C')
    #     ax[1,0].plot(self.freqs_for_fullant, S21_mag_with_lengths_array, marker='.', color='darkred', label='via length')
    #     ax[1,0].grid()
    #     ax[1,0].set_xlabel('f [MHz]')
    #     ax[1,0].set_ylabel('|S21|')

    #     ax[1,1].plot(new_f_range, S21_db_with_Cs_array, marker='.', color='red', label='via C')
    #     ax[1,1].plot(self.freqs_for_fullant, S21_db_with_lengths_array, marker='.', color='darkred', label='via length')
    #     ax[1,1].grid()
    #     ax[1,1].set_xlabel('f [MHz]')
    #     ax[1,1].set_ylabel('20log10(|S21|)')