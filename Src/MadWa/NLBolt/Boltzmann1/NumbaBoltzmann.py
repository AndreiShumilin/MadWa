##### some procedures tailored to make more calculations inside numba

import numpy as np
import numpy.linalg
import numba as nb

from ..Math import Gmath
from ..Math import NLBoltMath as math

Ezero = 1e-15
Const_f = 1.519267458894051e-10

__all__ = ['Ezero','Const_f','DisFunction','NumbaGetF','NumbaInitDerivs','NumbagetCurrentsF']

@nb.njit
def CheckBand(iw, mu, T, extr):
    em, eM = extr[iw]
    em -= mu
    eM -= mu
    if em*eM <= 0:
        return True
    elif (np.abs(em/T)>math.boltzmann_maximum) and (np.abs(eM/T)>math.boltzmann_maximum):
        return False
    else:
        return True


#####################################################################################

@nb.njit
def DisFunction(Efield, Ederiv, Nord, Parti, tau=1.0, T=1.0, mu=0.0):
    pref = (-tau)**Nord
    pref = pref*(Const_f)**Nord
    Aalps = math.allAlpha(Nord)
    Res0 = 0
    energy = Ederiv[0]

    crit1 = np.abs((energy-mu)/T)
    if crit1 > math.boltzmann_maximum:
        return 0.0
    
    for alps in Aalps:
        C1 = 1
        for a1 in alps:
            C1 *= Efield[a1]

        T0 = 0
        if np.abs(C1) > Ezero:
            for part in Parti:
                Tpart = 1.0
                Lpart = len(part)
                for set1 in part:
                    alpaSet = [alps[s] for s in set1 if s>=0]
                    derNumber = math.SETtoIND(alpaSet)
                    ederiv1 = Ederiv[derNumber]   #### find ederiv based on alpaset
                    Tpart *= ederiv1
                Tpart *= math.BoltzmanDeriv((energy-mu)/T, Lpart)/(T**Lpart)
                T0 += Tpart
        Res_alps = C1*T0
        Res0 += Res_alps
    return pref*Res0

@nb.njit
def NumbaGetF(Efield, Order, TEV, mu, tau, AParti, farShape, AlleDerivsU, AlleDerivsD, Extremums=None):
    Nx, Ny, Nz, Nw = farShape
    farU = np.zeros((Nx, Ny, Nz, Nw), dtype=np.float64)
    farD = np.zeros((Nx, Ny, Nz, Nw), dtype=np.float64)

    if not Extremums is None:
        extrU, extrD = Extremums
    
    for iw in range(Nw):
        if Extremums is None:
            calcU = True
            calcD = True
        else:
            calcU = CheckBand(iw, mu, TEV, extrU)
            calcD = CheckBand(iw, mu, TEV, extrD)

        if calcU:
            for ix in range(Nx):
                for iy in range(Ny):
                    for iz in range(Nz):
                        EderivU = AlleDerivsU[ix,iy,iz,iw]
                        f1U = DisFunction(Efield, EderivU, Order,  AParti, tau=tau, T=TEV, mu=mu)
                        farU[ix,iy,iz,iw] = f1U
        if calcD:
            for ix in range(Nx):
                for iy in range(Ny):
                    for iz in range(Nz):
                        EderivD = AlleDerivsD[ix,iy,iz,iw]
                        f1D = DisFunction(Efield, EderivD, Order,  AParti, tau=tau, T=TEV, mu=mu)
                        farD[ix,iy,iz,iw] = f1D
    return farU, farD


# @nb.jit(nopython=True)
# def NumbaInitDerivs(DerShape, rvlist, HlistU, HlistD, KGr, der_dk, cell, dim, der_N, MaxOrd, der_dkGau):
#     eDerivsU = np.zeros(DerShape, dtype=np.float64)
#     eDerivsD = np.zeros(DerShape, dtype=np.float64)
#     Nx, Ny, Nz, Nw, Nderiv = DerShape

#     for ix in range(Nx):
#         for iy in range(Ny):
#             for iz in range(Nz):
#                 k1 = KGr[ix,iy,iz]
#                 smallKgr, EgrU, EgrD = math.NumbaGetGrids(HlistU, HlistD, rvlist, k1, der_dk, 
#                                                           Nw, cell, dim=dim, Nk=der_N)
#                 for iw in range(Nw):
#                     eeWU = EgrU[...,iw]
#                     eeWD = EgrD[...,iw]
#                     derivsU = math.getDerivs(smallKgr, eeWU, k1, MaxOrd,  dkGau=der_dkGau)
#                     derivsD = math.getDerivs(smallKgr, eeWD, k1, MaxOrd,  dkGau=der_dkGau)
#                     eDerivsU[ix,iy,iz,iw] = derivsU
#                     eDerivsD[ix,iy,iz,iw] = derivsD
#     return eDerivsU, eDerivsD


@nb.jit(nopython=True, parallel=True)
def NumbaInitDerivs(DerShape, rvlist, HlistU, HlistD, KGr, der_dk, cell, dim, der_N, MaxOrd, der_dkGau):

    Nx, Ny, Nz, Nw, Nderiv = DerShape

    DerShapeT = (Nx,Ny,Nz,Nw,Nderiv)
    
    eDerivsU = np.zeros(DerShapeT, dtype=np.float64)
    eDerivsD = np.zeros(DerShapeT, dtype=np.float64)

    for ix in nb.prange(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
                k1 = KGr[ix,iy,iz]
                smallKgr, EgrU, EgrD = math.NumbaGetGrids(HlistU, HlistD, rvlist, k1, der_dk, 
                                                          Nw, cell, dim=dim, Nk=der_N)
                for iw in range(Nw):
                    eeWU = EgrU[...,iw]
                    eeWD = EgrD[...,iw]
                    derivsU = math.getDerivs(smallKgr, eeWU, k1, MaxOrd,  dkGau=der_dkGau)
                    derivsD = math.getDerivs(smallKgr, eeWD, k1, MaxOrd,  dkGau=der_dkGau)
                    eDerivsU[ix,iy,iz,iw] = derivsU
                    eDerivsD[ix,iy,iz,iw] = derivsD
    return eDerivsU, eDerivsD

@nb.njit
def NumbagetCurrentsF(jdir, farU, farD, DerShape, eDerivsU, eDerivsD, mu, TEV=1, Extremums = None):
    jdir1 = jdir/np.linalg.norm(jdir)
    Nx, Ny, Nz, Nw, Nderiv = DerShape

    if not Extremums is None:
        extrU, extrD = Extremums
    
    currentU = 0.0
    currentD = 0.0
    for iw in range(Nw):
        if Extremums is None:
            calcU = True
            calcD = True
        else:
            calcU = CheckBand(iw, mu, TEV, extrU)
            calcD = CheckBand(iw, mu, TEV, extrD)
        if calcU:
            for ix in range(Nx):
                for iy in range(Ny):
                    for iz in range(Nz):
                        edeU = eDerivsU[ix,iy,iz,iw]
                        vgrU = np.array((edeU[1], edeU[2],edeU[3]))                        
                        currentU += farU[ix,iy,iz,iw] * (vgrU@jdir1)
        if calcD:
            for ix in range(Nx):
                for iy in range(Ny):
                    for iz in range(Nz):
                        edeD = eDerivsD[ix,iy,iz,iw]
                        vgrD = np.array((edeD[1], edeD[2],edeD[3]))
                        currentD += farD[ix,iy,iz,iw] * (vgrD@jdir1)
    return currentU, currentD



#############################################################################################