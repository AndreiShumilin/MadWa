import numpy as np
import numba as nb

from ..Utils.utils import ex,ey,ez
from ..Utils import utils as ut

from ..Tbasic import tbasic
from ..Tbasic import tbroutines

from . import loperators as loper

from .CMfunctions import *




class combinedModel:
    r"""
    Class controlling the combination of the two "no SOC" TB models and on-atoms spin-orbit coupling
    to get a single TB model "with SOC" dependent on the magnetization direction
    """
    def __init__(self, TBup, TBdown, xiDict={}):
        self.TBu = TBup
        self.TBd = TBdown
        self.Nw = self.TBu.num_wann
        self.Nr = self.TBu.num_rvec
        self.cell = self.TBu.cell
        self.xiDict = xiDict

        self.RMapU =  makeRMap(self.TBu.Irvects)
        self.RMapD =  makeRMap(self.TBd.Irvects)

        self.findVecNumber = self.TBu.findVecNumber

        self.WUDconnect = False
        
    def connect_with_proj(self):
        self.Wud =Wud_from_proj_All(self.TBu.W, self.TBd.W, self.TBu.Irvects, self.TBd.Irvects, self.RMapU, self.RMapD)
        self.Wdu =Wud_from_proj_All(self.TBd.W, self.TBu.W, self.TBd.Irvects, self.TBu.Irvects, self.RMapD, self.RMapU)
        self.WUDconnect = True

    def connect_with_1(self):
        self.Wud = np.zeros((self.Nr, self.Nw, self.Nw), dtype = np.complex128)
        self.Wdu = np.zeros((self.Nr, self.Nw, self.Nw), dtype = np.complex128)
        ir1 = self.TBu.findVecNumber( (0,0,0) )[0]
        self.Wud[ir1] += np.eye(self.Nw)
        self.Wdu[ir1] += np.eye(self.Nw)
        self.WUDconnect = True

    def getDetailedProjections(self, outfile=None):
        if outfile is None:
            self.projDetails = projections_from_TB(self.TBu)
        else:
            self.projDetails = projections_from_out(outfile, cell=self.cell)

    def calculateProjL(self, xiDict):
        r"""
        Note: ProjL already include the strength of 
        """
        LBlocs = loper.makeblocks(self.projDetails)
        self.pLx, self.pLy, self.pLz = loper.L_BlocksToMatr(self.Nw,LBlocs, xiDict) 

    def MakeProjSOHam(self, M):
        """
        Creates SOC Hamiltonian in "projection basis" based on the polarization direction M
        (it is presumed that up- and down- directions for spin are "along M" and "oposite to M")
        """
        mx,my,mz = ut.makebasis(M)
        Lmx = (mx@ex)*self.pLx + (mx@ey)*self.pLy + (mx@ez)*self.pLz
        Lmy = (my@ex)*self.pLx + (my@ey)*self.pLy + (my@ez)*self.pLz
        Lmz = (mz@ex)*self.pLx + (mz@ey)*self.pLy + (mz@ez)*self.pLz
        SOCH = np.kron(Lmx, sx)
        SOCH += np.kron(Lmy, sy)
        SOCH += np.kron(Lmz, sz)
        return SOCH

    def combinedHamiltonian(self, M, SOrvecs = np.array([(0,0,0),], dtype=np.int32)  ):
        r"""
        SOrvecs - set of r-vectors used to calculate SO part of Hamiltonian. Used both for non-diagonal (in r) 
        SO-Hamiltonian elements and as coordinates of unit cells where the orbitals are considered for SO
        can be set to None, in this case full set of rvectors from TB would be used (might be slow)
        """
        Nw2 = 2*self.Nw
        if SOrvecs is None:
            SOrvecs = TRu.rvects
        H = np.zeros((self.Nr, Nw2,Nw2), dtype = np.complex128)
        spup = np.array([(1,0),(0,0)], dtype = np.complex128)
        spdown = np.array([(0,0),(0,1)], dtype = np.complex128)
        for ir in range(self.Nr):
            H[ir] += np.kron(self.TBu.H_ij[ir],spup)
            H[ir] += np.kron(self.TBd.H_ij[ir],spdown)

        pSOC = self.MakeProjSOHam(M)
        for rvH in SOrvecs:
            for rvP in SOrvecs:
                irh,exiH = self.findVecNumber(rvH)
                irp,exiP = self.findVecNumber(rvH)
                rvHP = np.asarray(rvP) - np.asarray(rvH)
                irhp,exiHP = self.findVecNumber(rvHP)
                if exiH and exiP and exiHP:
                    W1u = self.TBu.W[irp]
                    W1d = self.TBd.W[irp]
                    W2u = self.TBu.W[irhp]
                    W2d = self.TBd.W[irhp]
                    W1 = np.kron(W1u, spup) + np.kron(W1d, spdown)
                    W2 = np.kron(W2u, spup) + np.kron(W2d, spdown)
                    Hsoc1 = W1@pSOC@np.transpose(np.conjugate(W2))
                    H[irh] += Hsoc1
        return H

    def generateCombinedTB(self, M, SOrvecs = np.array([(0,0,0),], dtype=np.int32)):
        TBC = tbasic.TBasic()
        rvecs = self.TBu.rvects.copy()
        Hij = self.combinedHamiltonian(M, SOrvecs=SOrvecs)
        TBC.manual(Hij, rvecs, cell=self.cell)
        return TBC