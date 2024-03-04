import numpy as np

class TWA_Design_Toolkit:

    def __init__(self, num_straps, f0, L, C, M, R0, Rt):
        self.w0 = 2*np.pi*f0
        self.num_straps = num_straps
        self.L = L
        self.C = C
        self.M = M
        self.R0 = R0
        self.Rt = Rt
    
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