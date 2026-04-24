import numpy as np
import scipy as sp
import numba as nb
from numba import int32, float32, float64    # import the types


@nb.jit(nopython=True)
def KGrid(Nkx, Nky, Nkz,  cell, rKXmax=1, rKYmax=1, rKZmax=1, regime2D=True):
    Kgrid1 = np.zeros((Nkx,Nky,Nkz,3), dtype=np.float64)
    a1 = cell[0,...]
    a2 = cell[1,...]
    a3 = cell[2,...]
    b1 = 2*np.pi*np.cross(a2, a3)/ np.dot(a1, np.cross(a2, a3))
    b2 = 2*np.pi*np.cross(a3, a1)/ np.dot(a2, np.cross(a3, a1))
    b3 = 2*np.pi*np.cross(a1, a2)/ np.dot(a3, np.cross(a1, a2))

    if regime2D:
        rKZmax1 = 0.0
    else:
        rKZmax1 = rKZmax
    
    akx = np.linspace(-rKXmax,rKXmax,Nkx )
    aky = np.linspace(-rKYmax,rKYmax,Nky )
    akz = np.linspace(-rKZmax1,rKZmax1,Nkz )
    for ix in range(Nkx):
        for iy in range(Nky):
            for iz in range(Nkz):
                Kgrid1[ix,iy,iz] = akx[ix]*(b1/2) + aky[iy]*(b2/2) + akz[iz]*(b3/2)
    ###gK = (cell[0,0]*rKXmax/np.pi)*(cell[1,1]*rKYmax/np.pi)/(Nkx*Nky)   ####???????  old version  ????????????????

    if regime2D:
        a12 = np.cross(a1,a2)
        Sa = np.sqrt(np.sum(a12*a12))
        b12 = np.cross(b1,b2)
        Sb = np.sqrt(np.sum(b12*b12))
        gK = Sa*Sb/(4*np.pi*np.pi*Nkx*Nky)
        gK *= rKXmax*rKYmax
    else:
        a12 = np.cross(a1,a2)
        Va = np.dot(a12,a3)
        b12 = np.cross(b1,b2)
        Vb = np.dot(b3,b12)
        gK = (Va*Vb)/((2*np.pi)**3 *Nkx *Nky *Nkz)
        gK *= rKXmax*rKYmax*rKZmax
    return Kgrid1, gK