import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numba as nb
import copy


import MadWa.Tbasic.tbroutines as tbroutines
import MadWa.Utils.utils as ut
import MadWa.Utils.grids2 as gr2

Nkl02D=[60,60,1]
Nkl03D=[30,30,30]
Esmear0 = 0.15

Mklist2D = [30,30,1]
Mklist3D = [10,10,10]
Mul_Smear0 = 4

@nb.njit
def getEgrid(H, rvecs, deg, Kgr_centers, cell):
    r"""
    creates energy-grid based on k-Grid centers and TB-model
    """
    Nk, N3 = Kgr_centers.shape
    Nw = H.shape[1]
    Egr = np.zeros((Nk, Nw), dtype=np.float64)
    for ik in range(Nk):
        kv = Kgr_centers[ik]
        Hk = tbroutines.Hk_njit(H, rvecs, deg, kv, cell)
        evals = np.linalg.eigvalsh(Hk)
        Egr[ik] = evals
    return Egr

def TwoEnergiesZero(vert, H, rvecs, deg, cell, Ef):
    r"""
    function used to detect points in k-space where two energies are close to Ef
    returns the minimal energy interval around Ef, containing two states at at leasy one k-vector
    among verts
    """
    Nv = vert.shape[0]
    Energs = np.zeros(Nv, dtype=np.float64)
    for iv in range(Nv):
        kv = vert[iv]
        Hk = tbroutines.Hk_njit(H, rvecs, deg, kv, cell)
        evals = np.linalg.eigvalsh(Hk)
        evals2 = np.sort( np.abs(evals - Ef)  )
        Energs[iv] = evals2[1]
    return np.min(Energs)

def MatrixElement(H, rvecs, degs, kvec, cell, V, i,j):
    Hk0 = tbroutines.Hk_njit(H, rvecs, degs, kvec, cell)
    Vk = tbroutines.Hk_njit(V, rvecs, degs, kvec, cell)
    eig, evecs = np.linalg.eigh(Hk0)
    psi = evecs[...,i]
    psj = evecs[...,j]
    Mel = np.conjugate(psi)@(Vk@psj)
    return Mel

@nb.njit
def gaussDelta(a,b,smear):
    ex = (a-b)**2 / (2*smear**2)
    norm = np.sqrt(2*np.pi) * smear
    res = np.exp(-ex ) / norm
    return res

####=======================================================================

class damping:
    def __init__(self, H, LdH, rvecs, dim, Ef,
                 muCell, gfac=2,    # magnetic moment of the unit cel [in mu_b] and g-factor
                 NkLinit=None, #### Nkx, Nky,Nkz of the initial Kgr
                 cell=np.eye(3), deg =None,
                 Esmear = 0.1, Eprec=None):
        r"""
            Class for the calculations of magnon-electron damping.
            H, rvecs, deg - Hamiltonian, r-vectors and degenerations composing TB model
            
            LdH - list of possible "directions of modifications of Hamiltonian" - includes both Cartersian coordinates and sublattices
            dim - system dimensionality
            Ef - Fermi energy
            muCell - magnetic moment of the unit cell in Bohr magnetons
            gfac - g-factor
            NkLinit - set of (Nx,Ny,Nz) used to construct the initial k-grid
            cell - unit cell
            Esmear - smearing of the energy delta-functions
            Eprec - precision in energy (should be several times larger than Esmear)
        """
        self.H = H.copy()
        self.Nw = self.H.shape[1]
        self.LdH = copy.deepcopy(LdH)
        self.NdH = len(self.LdH)
        self.dim = dim
        self.rvecs = rvecs.copy()
        self.Nrv = len(self.rvecs)
        self.cell = cell.copy()
        self.muCell = muCell
        self.gfac = gfac
        self.Ef = Ef

        self.Esmear = Esmear
        if Eprec is None:
            self.Eprec = self.Esmear * Mul_Smear0
        else:
            self.Eprec = Eprec
        
        if deg is None:
            self.deg = np.ones(self.Nrv)
        else:
            self.deg = deg
        
        if NkLinit is None:
            if self.dim==2:
                NkLinit = Nkl02D
            else: 
                NkLinit = Nkl03D
        #print(NkL)
        self.Nkx0, self.Nky0, self.Nkz0 = NkLinit
        self.KGr = gr2.Uniform_Krid(self.Nkx0, self.Nky0, self.Nkz0, dim=self.dim, cell=self.cell)
        
    def refineGrid_Eprec_Kmult(self, Eprec=None, MkList=None):
        r"""
        refines K-grid (ones) based on the Energy precision (Eprec) and 
        a single multiplicator set (MkLis)
        """
        if Eprec is None:
            Eprec = self.Eprec

        if MkList is None:
            if self.dim==2:
                MkList = Mklist2D 
            else:
                MkList = Mklist3D 
        
        def Crit1(vert, cen):
            dE2 = TwoEnergiesZero(vert, self.H, self.rvecs, self.deg, self.cell, self.Ef)
            return dE2<Eprec

        self.KGr.refineGrid1(Crit1, *MkList)

    def makeEgrid(self):
        r"""
        makes energy grid
        """
        self.Egr = gr2.getEgrid(self.H, self.rvecs, self.deg, self.KGr.centers, self.cell)

    def findEnPairs(self, evec, Ef=None, Eprec=None):
        r"""
        looks for pairs of energy levels near Ef
        """
        if Ef is None:
            Ef = self.Ef
        if Eprec is None:
            Eprec = self.Eprec
        plist = []
        Ne = len(evec)
        for i in range(Ne):
            for j in range(i+1,Ne):
                #if (i!=j):
                if (np.abs(evec[i]-Ef)<Eprec) and (np.abs(evec[j]-Ef)<Eprec):
                    plist.append((i,j))
        return plist                
                

    def DTensorComponents(self, Ef = None, Esmear=None):
        r"""
        calculates damping as a tensor, based on provided derivatives of directions of magnetizations over
        (sublattice) magnetization directions

        Ef-Fermi energy (None takes value from the object)
        Esmear - width of delta-functions (None takes value from the object)
        """
        const1 = 2*np.pi * self.gfac/self.muCell   ## 2 -> to unclude both band pair b1-b2 and b2-b1
        
        if Ef is None:
            Ef = self.Ef
        if Esmear is None:
            Esmear = self.Esmear
        
        Gten = np.zeros((self.NdH, self.NdH), dtype = np.complex128)

        for ik in range(self.KGr.N):
            kv = self.KGr.centers[ik]
            wei = self.KGr.weights[ik]
            
            eeind = self.findEnPairs(self.Egr[ik])
            for bi,bj in eeind:
                for alp in range(self.NdH):
                    for bet in range(self.NdH):
                        M1 = MatrixElement(self.H, self.rvecs, self.deg, kv, self.cell, self.LdH[alp], bi,bj)
                        M2 = MatrixElement(self.H, self.rvecs, self.deg, kv, self.cell, self.LdH[bet], bj,bi)
                        e1 = self.Egr[ik,bi]
                        e2 = self.Egr[ik,bj]
                        del1 = gaussDelta(e2,Ef,Esmear)*gaussDelta(e1,Ef,Esmear)
                        Gten[alp,bet] += const1*M1*M2*wei*del1
        return np.real(Gten)

    def ananlizeContributions(self, Ef, alp, bet, Esmear=None,
                              ToPlot = False, cmap=pl.cm.Blues, ms=0.1):
        if Esmear is None:
            Esmear = self.Esmear

        Res = 0.0
        xx = []
        yy = []
        vv = []

        const1 = 2*np.pi * self.gfac/self.muCell

        for ik in range(self.KGr.N):
            kv = self.KGr.centers[ik]
            wei = self.KGr.weights[ik]
            eeind = self.findEnPairs(self.Egr[ik])
            v1 = 0
            v1d = 0
            for bi,bj in eeind:
                M1 = MatrixElement(self.H, self.rvecs, self.deg, kv, self.cell, self.LdH[alp], bi,bj)
                M2 = MatrixElement(self.H, self.rvecs, self.deg, kv, self.cell, self.LdH[bet], bj,bi)
                e1 = self.Egr[ik,bi]
                e2 = self.Egr[ik,bj]
                del1 = gaussDelta(e2,Ef,Esmear)*gaussDelta(e1,Ef,Esmear)
                v1d += np.real(M1*M2*del1 * const1 )
                v1 += np.real(M1*M2*wei*del1 * const1)
            Res += v1
            xx.append(kv[0])
            yy.append(kv[1])
            vv.append(v1d)
        xx = np.array(xx)
        yy = np.array(yy)
        vv = np.array(vv)/Res
            
        if ToPlot:
            col = [cmap(v/np.max(vv)) for v in vv]
            plt.scatter(xx,yy,c=col, s=ms)
        
        return xx,yy,vv, Res


    # def ananlizeKpoint(self, kv, Ef, alp=0, bet=0, Esmear=None, logFoo=print):
    #     if Esmear is None:
    #         Esmear = self.Esmear

    #     Hk = tbroutines.Hk_njit(self.H, self.rvecs, self.deg, kv, self.cell)
    #     evals = np.linalg.eigvalsh(Hk)
    #     logFoo('Energies : \n')
    #     sE = ''
    #     for ei in evals:
    #         sE = sE+str(ei)+', '
    #     sE = sE + '\n'
    #     logFoo(sE)
    #     eeind = self.findEnPairs(evals)
    #     logFoo(str(eeind)+'\n')
    #     for bi,bj in eeind:
    #         M1 = MatrixElement(self.H, self.rvecs, self.deg, kv, self.cell, self.LdH[alp], bi,bj)
    #         M2 = MatrixElement(self.H, self.rvecs, self.deg, kv, self.cell, self.LdH[bet], bj,bi)
    #         e1 = self.Egr[ik,bi]
    #         e2 = self.Egr[ik,bj]
    #         del1 = gaussDelta(e2,Ef,Esmear)*gaussDelta(e1,Ef,Esmear)
    #         v1d += np.real(M1*M2*del1 )
    #         s1 = str(bi) + str(bj) + ': \n'
    #         s2 = 'M1, M2 = ' + str(M1) + ', ' + str(M2) + '\n'
    #         s3 = 'del = ' + str(del1) + '\n'
    #         logFoo(s1)
    #         logFoo(s2)
    #         logFoo(s3)
    #         logFoo('------ \n\n')
