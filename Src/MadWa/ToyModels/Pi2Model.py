import numpy as np
import numba as nb

from ..Tbasic import tbasic 
from ..Tbasic import tbroutines
from ..Utils import utils as ut

r"""
A simplified model of p-states, featuring only px and py
"""

s0 = np.array(((1,0),(0,1)), dtype=np.complex128)
sx = np.array(((0,1),(1,0)), dtype=np.complex128)
sy = np.array(((0,-1j),(1j,0)), dtype=np.complex128)
sz = np.array(((1,0),(0,-1)), dtype=np.complex128)


spup = np.array([(1,0),(0,0)], dtype = np.complex128)
spdown = np.array([(0,0),(0,1)], dtype = np.complex128)


class PiModel2D:
    r"""
    a simple 2D TB-model with two p-states (px and py)
    Used as a test for damping part
    
    ts, tp --- sigma- and pi-bonds respectively
    a - size of the cell
    kap - spin-orbit coupling
    """

    def __init__(self,ts=1, tp=0.0,a=1,kap=0.1, bx=0.3, by=0.3):
        self.ts = ts
        self.tp = tp
        self.a = a
        self.kap = kap
        self.bx = bx
        self.by = by
        self.rvecs = np.array([(0,0,0),(1,0,0),(-1,0,0),(0,1,0),(0,-1,0)])
        self.Nrv = 5
        self.Nw = 2

        self.TBU = tbasic.TBasic()
        self.TBD = tbasic.TBasic()

        self.PLoper = self.kap*sy
        
        HijU  = np.zeros((self.Nrv, self.Nw, self.Nw), dtype=np.complex128)
        HijU[0] += np.diag((self.bx,self.by))
        HijU[1] += np.diag((self.ts,self.tp))
        HijU[2] += np.diag((self.ts,self.tp))
        HijU[3] += np.diag((self.tp,self.ts))
        HijU[4] += np.diag((self.tp,self.ts))
        self.TBU.manual(HijU, self.rvecs)
        self.HijU = HijU

        HijD  = np.zeros((self.Nrv, self.Nw, self.Nw), dtype=np.complex128)
        HijD[0] += np.diag((-self.bx, -self.by))
        HijD[1] += np.diag((self.ts,self.tp))
        HijD[2] += np.diag((self.ts,self.tp))
        HijD[3] += np.diag((self.tp,self.ts))
        HijD[4] += np.diag((self.tp,self.ts))
        self.TBD.manual(HijD, self.rvecs)
        self.HijD = HijD


    def MakeHamiltonian(self, eMz, eMx=None):
        if (eMx is None):
            eMx, eMy, vz = ut.makebasis(eMz)
        else:
            eMy = np.cross(eMx, eMx)
            eMy = eMy/np.linalg.norm(eMy)
            
        sz1 = sx*(eMz@ut.ex) + sy*(eMz@ut.ey) + sz*(eMz@ut.ez)
        Nw2 = 2*self.Nw
        Hij  = np.zeros((self.Nrv, Nw2, Nw2), dtype=np.complex128)
        for i in range(self.Nrv):
            Hij[i] +=  np.kron(self.HijU[i],spup) 
            Hij[i] +=  np.kron(self.HijD[i],spdown) 
        Hij[0] += np.kron(self.PLoper, sz1) 
        degs = np.ones(self.Nrv)
        return Hij, self.rvecs, degs

    def MakeExDerivs(self, eMz, eMx=None):
        if (eMx is None):
            eMx, eMy, vz = ut.makebasis(eMz)
        else:
            eMy = np.cross(eMz, eMx)
            eMy = eMy/np.linalg.norm(eMy)
            
        sx1 = sx*(eMx@ut.ex) + sy*(eMx@ut.ey) + sz*(eMx@ut.ez)
        sy1 = sx*(eMy@ut.ex) + sy*(eMy@ut.ey) + sz*(eMy@ut.ez)
        sz1 = sx*(eMz@ut.ex) + sy*(eMz@ut.ey) + sz*(eMz@ut.ez)
        
        Nw2 = 2*self.Nw

        DHijX  = np.zeros((self.Nrv, Nw2, Nw2), dtype=np.complex128)
        DHijY  = np.zeros((self.Nrv, Nw2, Nw2), dtype=np.complex128)
        for i in range(self.Nrv):
            Hex = (self.HijU[i] - self.HijD[i])/2
            DHijX[i] +=  np.kron(Hex, sx1) 
            DHijY[i] +=  np.kron(Hex, sy1) 
        return DHijX, DHijY
        
        


