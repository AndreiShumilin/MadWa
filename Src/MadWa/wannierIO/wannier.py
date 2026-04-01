import numpy as np

from . import wanReads_files as read


__all__ = ['VASP_Wan_correction_Matrix','reconstructW','WanRes']


def VASP_Wan_correction_Matrix(nb, evals, emin, emax):
    r"""
    VASP (and Amn matrix im Wannier) uses different order of Kohn-Sham bands compared to U-matrix in Wannier
    (U-matrix corresponds to the bands starting from the emin value of energy window)

    this procedure calculates the transformation matrix based on array of eigenvalues evals and 
    emin,emax - parameters of the window (should be taken from .win file)
    nb - number of Kohn-Sham bands (from VASP)

    Procedure is expected to be run separately for each k-point
    """
    V = np.zeros((nb, nb), dtype=np.complex128)
    idx = np.where((evals >= emin) & (evals <= emax))[0]
    idx0 = idx[0]
    for i in range(nb - idx0):
        V[i, i + idx0] = 1.0
    for i in range(idx0):
        V[nb - idx0 + i, i] = 1.0
    return V


def reconstructW(Amn, U, kpts, Rvecs):
    r"""
    constructs the transformation matrix relating the wannier functions and the initial projections
    W[ir, iw, ip] = < W_{iw} | P_{ip}(R[ir]) >
    W_{iw} --- wannier function iw
    P_{ip}(R[ir]) --- projection ip in the unit cell displaced by vector R[ir]
    """
    Nrv = len(Rvecs)
    Rdict = {}
    for irv, rv in enumerate(Rvecs):
        Rdict[tuple(rv)] = irv
    Nk = len(kpts)
    Nproj = Amn.shape[2]
    Nw = U.shape[2]

    W = np.zeros((Nrv,Nw,Nproj), dtype=np.complex128)
    for ik in range(Nk):
        k = kpts[ik]
        Wk = U[ik].conj().T @ Amn[ik]
        for irv, rv in enumerate(Rvecs):
            phase = np.exp(-2j * np.pi * np.dot(k, rv))
            W[irv] += Wk * phase
    W /= Nk
    return W
            


class WanRes:
    r"""
    class to store and analize results of wannier90
    """
    def __init__(self, seedname=None, Short=True):
        if seedname is None:
            self.created = False
            self.projections = False
        else:
            infile = seedname + '.win'
            hrfile = seedname + '_hr.dat'
            amnfile = seedname + '.amn'
            Ufile = seedname + '_u.mat'
            UDISfile = seedname + '_u_dis.mat'
            eigfile = seedname + '.eig'
            
            winD = read.readWin(infile)
            self.fullwinD = winD
            self.cell = winD['cell']
            self.atoms = winD['atoms']
            self.Nw = winD['nw']
            self.Nb = winD['nb']
            H, rvecs, deg, Par = read.readHR(hrfile)
            
            self.Hr = H
            self.rvecs = rvecs
            self.Nrv = len(rvecs)
            self.degen = deg

            self.rvNdict = {}
            for irv, rv in enumerate(self.rvecs):
                self.rvNdict[tuple(rv)] = irv
            self.NR0 = self.rvNdict[(0,0,0)]

            if not Short:
                self.Emin = self.fullwinD['eMin']
                self.Emax = self.fullwinD['eMax']
                self.Nk   = self.fullwinD['Nk']
                self.kpts = self.fullwinD['kpts']

                Amn0, pars = read.readAmn(amnfile)
                U0, kar = read.readU(Ufile)
                Udis, kar_dis = read.readU(UDISfile)
                eig = read.readEig(eigfile, self.Nk, self.Nb)

                self.tmp = (Amn0, U0, Udis, eig)
                Amn = np.zeros((self.Nk, self.Nb, self.Nw), dtype=np.complex128)
                for ik in range(self.Nk):
                    V1 = VASP_Wan_correction_Matrix(self.Nb, eig[ik], self.Emin, self.Emax)
                    A1 = V1@Amn0[ik]
                    Amn[ik] = A1
                self.Amn = Amn

                Utotal = np.zeros((self.Nk, self.Nb, self.Nw), dtype=np.complex128)
                for ik in range(self.Nk):
                    Utotal[ik] = Udis[ik]@U0[ik]
                self.Utotal = Utotal

                self.W = reconstructW(self.Amn, self.Utotal, self.kpts, self.rvecs)

    def ShowProj(self, rv=(0,0,0), Lfig=3, vmin=-3, vmax=0):
        plt.figure(figsize=(Lfig,Lfig))
        rvt = tuple(rv)
        if not rvt in self.rvNdict.keys():
            return None
        Nr = self.rvNdict[tuple(rv)]
        W0 = self.W[Nr]
        X1 = np.transpose(np.abs(W0)+1e-20)
        plt.pcolormesh(np.log10(X1), vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.show()
        
                
            

