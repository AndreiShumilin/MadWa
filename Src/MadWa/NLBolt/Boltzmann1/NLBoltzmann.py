import numpy as np
import numpy.linalg
import scipy as sp
import matplotlib.pyplot as plt
import numba as nb

from ..Math import Gmath
from ..Math import NLBoltMath as math
from ..Utils import TBtools as tools
from ..Utils import Logs

#from Boltzmann1.NumbaBoltzmann import *
from .NumbaBoltzmann import *

K_to_EV = 0.8617333262145177e-4

Const_current_2D = 2434134.8234366034
Const_current_3D = 2434134.8234366034e10




__all__ = ["Boltzmann",]

class Boltzmann:
    def __init__(self, TB, Nk, dim=2, MaxOrd = 3,  der_N =None, der_dk= None,  der_dkGau=None, Name='Blt'):
        
        self.TB = TB
        self.Nk = Nk
        self.dim = dim
        self.MaxOrd = MaxOrd
        self.Name = Name

        self.cell = self.TB.cell
        self.Vcell = tools.cellVolume(self.cell, regime2D=(self.dim==2))

        if dim==2:
            self.KGr, self.gK = Gmath.KGrid(Nk, Nk, 1, self.TB.cell, regime2D=True)
        else:
            self.KGr, self.gK = Gmath.KGrid(Nk, Nk, Nk, self.TB.cell, regime2D=False)

        self.Nw = self.TB.Nw
        self.Nderiv = math.igeomS(3,self.MaxOrd+1)
        self.Nx, self.Ny, self.Nz, _ = self.KGr.shape
        self.DerShape = (self.Nx, self.Ny, self.Nz, self.Nw, self.Nderiv)
        
        if der_N is None:
            self.der_N = self.MaxOrd + 2
        else:
            self.der_N = der_N
        
        if der_dk is None:
            dk = self.KGr[1,1,0] - self.KGr[1,0,0]
            adk = np.linalg.norm(dk)
            self.der_dk = adk/2
        else:
            self.der_dk = der_dk
        
        if der_dkGau is None:
            self.der_dkGau =  self.der_dk/3
        else:
            self.der_dkGau = der_dkGau
    
        self.derivatives_ready = False
        
        self.logfile = self.Name + '.log'
        self.Log = Logs.Log(self.logfile)

        self.Log.Twrite2('Non-linear Boltzmann calculations with order = '+str(self.MaxOrd))
        self.Log.write('process name = ' + self.Name)
        self.Log.write('Nw: ' + str(self.Nw)  )
        self.Log.write('K-Grid: ' + str(self.Nx) + 'x'+str(self.Ny)+ 'x' +str(self.Nz) )
        self.Log.write('Maximum order of derivatives: ' + str(self.MaxOrd)  )
        self.Log.cut()

    def EstimateExtremums(self):
        self.EextrU = np.zeros((self.Nw, 2), dtype=np.float64)
        self.EextrD = np.zeros((self.Nw, 2), dtype=np.float64)
        for iw in range(self.Nw):
            eeU = self.eDerivsU[...,iw,0]
            eeD = self.eDerivsD[...,iw,0]
            self.EextrU[iw] = np.min(eeU), np.max(eeU)
            self.EextrD[iw] = np.min(eeD), np.max(eeD)

    def initDerivs(self):
        r'''
        Calculates all the required energy derivatives 
        (usually the slowest part, but has to be done only once)
        '''
        self.rvlist = [rv for rv in self.TB.rvecs]
        self.HlistU = []
        self.HlistD = []
        for rv in self.rvlist:
            Hu, Hd = self.TB.getH(rv)
            self.HlistU.append(Hu.astype(np.complex128))
            self.HlistD.append(Hd.astype(np.complex128))

        DershapeArr = np.array(self.DerShape, dtype=np.int32)
        self.eDerivsU, self.eDerivsD = NumbaInitDerivs(DershapeArr, self.rvlist, self.HlistU, self.HlistD, self.KGr, 
                                                       self.der_dk, self.cell, self.dim, self.der_N, self.MaxOrd, self.der_dkGau)
        self.EstimateExtremums()
        self.derivatives_ready = True
        self.Log.Twrite2('Derivatives calculated')


    def SaveDerivs(self, fname=None):
        r'''
        Saves energy derivatives in a file
        '''
        if not self.derivatives_ready:
            return None
        if fname is None:
            fname = self.Name + '_derivs.npz'
        np.savez(fname, eDerivsU = self.eDerivsU, eDerivsD = self.eDerivsD)
        self.Log.Twrite2('Derivatives saved to: ' + fname)
        self.Log.cut()

    def LoadDerivs(self, fname=None):
        r'''
        Laods energy derivatives from a file
        '''
        if fname is None:
            fname = self.Name + '_derivs.npz'
        #### TODO: check is file exists
        dat1 = np.load(fname)
        self.rvlist = [rv for rv in self.TB.rvecs]
        self.HlistU = []
        self.HlistD = []
        for rv in self.rvlist:
            Hu, Hd = self.TB.getH(rv)
            self.HlistU.append(Hu.astype(np.complex128))
            self.HlistD.append(Hd.astype(np.complex128))
        self.eDerivsU = dat1['eDerivsU']
        self.eDerivsD = dat1['eDerivsD']
        self.EstimateExtremums()
        self.derivatives_ready = True
        self.Log.Twrite2('Derivatives loaded from: ' + fname)
        self.Log.cut()

########## The set of procedures to get a tipical results to show ##########################################

    def EFcurrent(self, Order, EF1, EF2, Evec, Jvec, Np=100, npz_file=None, txt_file=None, Show=False, fig_file=None, TK=300, tau=100):
        r"""
        Calculates the dependence of the current density on the Fermi energy
        Order - order of the perturbation theory
        EF1, EF2 - minimum and maximum Fermi energy
        Evec - vector of the electric field
        Jvec - direction of the measured current density
        Np - number of "points": Fermi energies
        npz_file: (if provided) saves the results in this file in npz-format
        txt_file: (if provided) saves the results in this file in text format
        Show: if True: makes a plot of the results
        fig_file: (only for Show=True) if provided, saves the figure in the file
        TK:  temperature [K]
        tau: relaxation time [fs]
        """
        eef = np.linspace(EF1, EF2, Np)
        resU = []
        resD = []
        
        jvec1 = Jvec.astype(np.float64)
        for ef in eef:
            # fU, fD = self.getF(Evec, Order, TK, mu=ef, tau=tau)
            # jU0, jD0 = self.getCurrentsF(fU, fD, jvec1)
            # jU = jU0*ccur/self.Vcell
            # jD = jD0*ccur/self.Vcell
            cU, cD = self.getCurrentDensitiesE(Evec, Order, TK, mu=ef, tau=tau, edir=jvec1)
            resU.append(cU)
            resD.append(cD)
        resU = np.array(resU)
        resD = np.array(resD)
        if not npz_file is None:
            np.savez(npz_file, curU = resU, curD=resD, efermi=eef, TK=TK, tau=tau, Efield=Evec, Jvec=Jvec)

        if not txt_file is None:
            A1 = np.array([eef, resU, resD])
            A1 = np.transpose(A1)
            np.savetxt(txt_file, A1)

        if Show:
            plt.figure(figsize=(4,3))
            plt.plot(eef, resU, 'b-', label='up')
            plt.plot(eef, resD, 'r-', label='down')
            plt.xlim(EF1,EF2)
            plt.legend(frameon=False)

            plt.xlabel(r'$E_F$, eV')
            if not fig_file is None:
                plt.savefig(fig_file)
            plt.show()
        return eef, resU, resD


    def Polar_current(self, Order, Ef, Ea=1, tp='par', Nphi=100, theta=np.pi/2, npz_file=None, txt_file=None, Show=False, fig_file=None, TK=300, tau=100):
        r"""
        Calculates the dependence of the current density on the Polar angle phi of the applied electric field
        Order:  order of the perturbation theory
        Ef:  Fermi energy
        Ea:  absolute value of the field [V/m]

        tp = 'par', 'planar', 'planarperp' or 'z'
        type of the calculation (direction of the measured current)
        "par" - along E-field
        "planar" - along E-field projection to xy
        "planarperp" - in xy, perpendicular to the field
        "z" - in z-direction
        
        Nphi- number of
        npz_file: (if provided) saves the results in this file in npz-format
        txt_file: (if provided) saves the results in this file in text format
        Show: if True: makes a plot of the results
        fig_file: (only for Show=True) if provided, saves the figure in the file
        TK:  temperature [K]
        tau: relaxation time [fs]
        """

        types = ['par', 'planar', 'planarperp', 'z',]
        if not tp in types:
            self.Log.Twrite('Warning: incorrect type for Polar_current :' + str(tp))
            return None

        ez = np.array((0,0,1))
        exy = np.array((1,1,0))
        
        lphi = np.linspace(0, 2*np.pi, Nphi)
        
        resU = []
        resD = []
        for phi in lphi:
            Evec = Ea*np.array( (np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)  ) )
            if self.dim == 2:
                Evec[2] = 0.0

            if tp=='par':
                jvec = Evec / np.linalg.norm(Evec)
            elif tp=='planar':
                jvec0 = Evec*exy
                jvec = jvec0 / np.linalg.norm(jvec0)
            elif tp == 'planarperp':
                vec0 = Evec*exy
                vec1 = np.cross(ez, vec1)
                jvec = vec1 / np.linalg.norm(vec1)
            elif tp=='z':
                jvec = ez
            
            #cU, cD = self.getCurrentsF(fU, fD, jvec)      
            cU, cD = self.getCurrentDensitiesE(Evec, Order, TK, mu=Ef, tau=tau, edir=jvec)
            resU.append(cU)
            resD.append(cD)
        resU = np.array(resU)
        resD = np.array(resD)
        
        if not npz_file is None:
            np.savez(npz_file, curU = resU, curD=resD, phi=lphi, TK=TK, tau=tau, E=Ea, typ = tp)

        if not txt_file is None:
            A1 = np.array([lphi, resU, resD])
            A1 = np.transpose(A1)
            np.savetxt(txt_file, A1)

        if Show:
            plt.figure(figsize=(3,3))
            plt.polar(lphi, resU, 'b-', label='up')
            plt.polar(lphi, resD, 'r-', label='down')
            plt.legend(frameon=False)
            if not fig_file is None:
                plt.savefig(fig_file)
            plt.show()
        return lphi, resU, resD

##################### below are technical procedures to be used mainly inside the library ####################################
    

    def getF(self, Efield, Order, TK, mu=0.0, tau=1.0):
        TEV = TK*K_to_EV
        Parti = math.Npartition(Order)
        AParti = math.PartiArray(Parti)
        farShape = (self.Nx, self.Ny, self.Nz, self.Nw)
        farU, farD = NumbaGetF(Efield, Order, TEV, mu, tau, AParti, farShape, self.eDerivsU, self.eDerivsD,
                                                   Extremums=(self.EextrU, self.EextrD))
        return farU, farD

    def getCurrentsF(self, farU, farD, jdir, mu, TK):
        TEV = TK*K_to_EV
        jdir1 = jdir/np.linalg.norm(jdir)
        Const1 = 1.0 * self.gK

        currentU, currentD = NumbagetCurrentsF(jdir, farU, farD, self.DerShape, 
                                               self.eDerivsU, self.eDerivsD, 
                                               mu = mu, TEV=TEV, Extremums=(self.EextrU, self.EextrD))

        currentU *= Const1
        currentD *= Const1
        return currentU, currentD

    def getCurrentDensitiesE(self, Efield, Order, TK, mu=0.0, tau=1.0, edir=None):
        
        if edir is None:
            edir = Efield/np.linalg.norm(Efield)
            
        farU, farD = self.getF(Efield, Order, TK, mu=mu, tau=tau)
        jU0, jD0 = self.getCurrentsF(farU, farD, edir, mu=mu, TK=TK)

        if self.dim==2:
            ccur = Const_current_2D
        else:
            ccur = Const_current_3D
            
        jU = jU0*ccur/self.Vcell
        jD = jD0*ccur/self.Vcell
        
        return jU, jD


        
                        


