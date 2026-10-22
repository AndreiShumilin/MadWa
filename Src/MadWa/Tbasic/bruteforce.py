import numpy as np
import scipy as sp
import numba as nb

import os


from . import Wann2Proj as w2p
from . import tbasic as tbasic

import MadWa.Utils.utils as ut
import MadWa.vaspIO.OUTCAR as outc
import MadWa.wannierIO as wIO
import MadWa.Math.noisyDerivs as noisy


#### temporary place, don't know where to put this.....



def joinFolder(fol, tilt):
    if tilt is None:
        fol1 = fol
    else:
        fol1 = fol + str(tilt)+'/'
    return fol1

def checkFolder(fol0, tilt=None):
    if tilt is None:
        fol1 = fol0
    else:
        fol1 = fol0 + str(tilt)+'/'
    fwin = fol1+'wannier90.win'
    return os.path.exists(fwin)
    

def getTB(fol0, tilt=None):
    if tilt is None:
        fol1 = fol0
    else:
        fol1 = fol0 + str(tilt)+'/'
    fwin = fol1+'wannier90.win'
    ftb = fol1+'wannier90_tb.dat'
    cenfile = fol1 + 'wannier90_centres.xyz'
    TB = tbasic.TBasic(win_file=fwin, tb_file =ftb)
    TB.readProjections(fol1+"wannier90")
    TB.readCenters(cenfile)
    return TB


def getHPrj(fol0, tilt=None, rvects=[(0,0,0)]):
    if tilt is None:
        fol1 = fol0
    else:
        fol1 = fol0 + str(tilt)+'/'
    seedname = fol1+'wannier90'
    Hprj = wIO.projHam(seedname, rvects)
    return  Hprj



class BForceData:
    def __init__(self, Nsub, Folder0, Tilt0=None, AtSets=None, logFoo=None, logPref = ''):
        self.Nsub = Nsub
        self.Nderiv = 2*self.Nsub
        if AtSets is None:
            self.AtSets = [None for _ in range(self.Nsub)]
        else:
            self.AtSets = AtSets
        self.logFoo = logFoo
        self.logPref = logPref
            
        chk = checkFolder(Folder0, Tilt0)
        if not chk:
            self.log('Now 0-th order Hamiltonian found')
        else:
            self.Folder0 = joinFolder(Folder0, Tilt0)
            self.TB0 = getTB(self.Folder0,  None)
            self.HP0 = getHPrj(fol0=self.Folder0, tilt=None, rvects=self.TB0.Irvects)
            shifts0 = np.zeros(self.Nderiv, dtype=np.float64)
            out0 = self.Folder0 + 'OUTCAR'
            self.ez1 = []
            self.ex1 = []
            self.ey1 = []
            for isub in range(self.Nsub):
                ez11 = outc.MagnetizFromOUTCAR(out0, NNatoms=self.AtSets[isub])
                ex12, ey12,ez12 = ut.makebasis(ez11)
                self.ex1.append(ex12)
                self.ey1.append(ey12)
                self.ez1.append(ez12)                
            self.PHamiltonians = [self.HP0,]
            self.mshifts = [shifts0,]
            self.folderList = [self.Folder0,]
            self.NPH = 0

    def log(self, s):
        if not (self.logFoo is None):
            self.logFoo(selfg.logPref + s + '\n')


    def addInfo(self, Folders, Tilts):
        for fol in Folders:
            for til in Tilts:
                fol1 = joinFolder(fol, til)
                if (checkFolder(fol1)) and (not (fol1 in self.folderList)):
                    mshi1 = []
                    out1 = fol1 + 'OUTCAR'
                    for isub in range(self.Nsub):
                        mdir1 = outc.MagnetizFromOUTCAR(out1, NNatoms=self.AtSets[isub])
                        mshi1.append(mdir1@self.ex1[isub])
                        mshi1.append(mdir1@self.ey1[isub])
                    mshi1 = np.array(mshi1)
                    HP1 =  getHPrj(fol0=fol1, tilt=None, rvects=self.TB0.Irvects)
                    self.PHamiltonians.append(HP1)
                    self.mshifts.append(mshi1)
                    self.folderList.append(fol1)
        self.NPH = len(self.folderList)

    
    def makeDerivs(self, RelVecs = None):
        Nrv = self.HP0.shape[0]
        Nw = self.HP0.shape[1]
        rvecs = self.TB0.Irvects
        DHlist = []
        dma = np.array(self.mshifts)
        for idr in range(self.Nderiv):
            dH1 = np.zeros((Nrv, Nw, Nw), dtype = np.complex128)
            DHlist.append(dH1)
        for irv in range(Nrv):
            rv = rvecs[irv]
            if (RelVecs is None) or (rv in RelVecs):
                irv2, isrv2 = ut.FindRv(rvecs, -rv)
                for iw1 in range(Nw):
                    for iw2 in range(iw1, Nw):
                        valList = [hp[irv,iw1,iw2] for hp in self.PHamiltonians]
                        valList = np.array(valList, dtype=np.complex128)
                        coef1, noise = noisy.NoisyDerivative3(dma, valList, logFoo=self.logFoo)
                        derivs1 = coef1[1:]
                        for idr in range(self.Nderiv):
                             DHlist[idr][irv,iw1,iw2] = derivs1[idr]
                             DHlist[idr][irv2,iw2,iw1] = np.conjugate(derivs1[idr])
        DHlistW = []
        for idr in range(self.Nderiv):
            dHp = DHlist[idr]
            dHw = w2p.PtoWann(dHp, rvecs, self.TB0.W, rvecs)
            DHlistW.append(dHw)
        
        return DHlistW
    