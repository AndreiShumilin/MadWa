import numpy as np
import numba as nb

from ..Tbasic import tbasic 
from ..Tbasic import tbroutines

# A toy model of a material composed of 3 p-orbitals with sigma- and pi-bonds ts and tp (respectively)
# lattice is cubic
# SOC can added relating spin to the orbital momentum of p-states


s0 = np.array(((1,0),(0,1)), dtype=np.complex128)
sx = np.array(((0,1),(1,0)), dtype=np.complex128)
sy = np.array(((0,-1j),(1j,0)), dtype=np.complex128)
sz = np.array(((1,0),(0,-1)), dtype=np.complex128)


Lx = np.array([(0,0,0),(0,0,-1j),(0,1j,0)])
Ly = np.array([(0,0,1j),(0,0,0),(-1j,0,0)])
Lz = np.array([(0,-1j,0),(1j,0,0),(0,0,0)])

def PiModel_TB(ts=-1, tp=0.1,a=1):
    r"""
    a TB-object describing the toy model with a cubic lattice and p-orbitals of a single "Fake" atom
    ts, tp --- sigma- and pi-bonds respectively
    a - size of the cell
    no SOC is includes (to include it later)
    """
    TB1 = tbasic.TBasic()
    rvecs = np.array([(0,0,0),(1,0,0),(0,1,0),(0,0,1)])
    Nrv = 4
    Nw = 3
    Hij  = np.zeros((Nrv, Nw, Nw), dtype=np.complex128)
    Hij[1] = np.diag((ts,tp,tp))
    Hij[2] = np.diag((tp,ts,tp))
    Hij[3] = np.diag((tp,tp,ts))
    TB1.manual(Hij, rvecs)

    atoms = [('Fake', np.zeros(3, dtype=np.float64)),]
    proj = [('Fake','p',1),]
    W = np.zeros((Nrv, Nw, Nw), dtype=np.complex128)
    W[0] = np.eye(Nw)
    TB1.manualProjections(atoms, proj, W)
    
    return TB1    

def PiModel_SOHam(k, xi, ts=-1, tp=0.1,a=1):
    HSO = np.kron(Lx,sx) + np.kron(Ly,sy) + np.kron(Lz,sz)
    HSO = HSO * xi
    
    Hx1 = np.diag((ts,tp,tp))*np.cos(k[0]*a)
    Hx = np.kron(Hx1,s0)
    Hy1 = np.diag((tp,ts,tp))*np.cos(k[1]*a)
    Hy = np.kron(Hy1,s0)
    Hz1 = np.diag((tp,tp,ts))*np.cos(k[2]*a)
    Hz = np.kron(Hz1,s0)

    return Hx + Hy + Hz + HSO

def PiModel_spec(kar, xi, ts=-1, tp=0.1,a=1):
    oms = []
    for k1 in kar:
        H1 = PiModel_SOHam(k1,xi,ts=ts, tp=tp, a=a)
        ei1 = np.linalg.eigh(H1)[0]
        oms.append(ei1)
    oms = np.array(oms)
    return oms