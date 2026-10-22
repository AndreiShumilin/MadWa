import numpy as np
import numba as nb

from . import utils as ut


r"""
Set of operations with Hamiltonians
the basis of the type (coordinate x spin) is presumed
which is normal for projections but can be different for Wannier
"""


@nb.njit
def convoluteWithSpin(HP, sig, relProjs=None):
    r"""
    convolutes the Hamiltonian HP with the Pauli matrix sig
    can take into account only part of projections relProjs
    """
    Nr = HP.shape[0]
    Np = HP.shape[1]
    Np0 = Np//2
    Hp2 = np.zeros((Nr, Np0, Np0), dtype=np.complex128)
    if not relProjs is None:
        Pm = np.zeros((Np0, Np0), dtype=np.complex128)
        for p in relProjs:
            Pm[p,p] = 1.0

    for irv in range(Nr):
        H1 = np.zeros((Np0, Np0), dtype=np.complex128)
        for i in range(Np0):
            for j in range(Np0):
                Ht1 = HP[irv, 2*i:2*i+2, 2*j:2*j+2]
                H1[i,j] =  np.trace(Ht1@sig)
        if not relProjs is None:
            H1 = Pm@H1@Pm
        Hp2[irv] = H1
    return Hp2


@nb.njit
def ConvertSigma(H, s1, s2):
    r"""
    Takes the part of Hamiltonian ~Pauli matrix s1
    and reproduced it with the Pauli matrix s2
    """
    Nr = H.shape[0]
    Np = H.shape[1]
    H1 = convoluteWithSpin(H, s1)
    Hnew = np.zeros((Nr,Np,Np), dtype=np.complex128)
    for irv in range(Nr):
        H2 = np.kron(H1[irv],s2)
        Hnew[irv] = H2
    return Hnew/2


@nb.njit
def TimeRevH(H,rvecs):
    r"""
    calculates time-reversed TB-Hamiltonian
    H[Nr, Np,Np] matrix (Nr - number of r-vectors, Np - number of projections)
    rvecs - r-vectors of the Hamiltonian

    NOTE: it is implied that the strucrture of eigenstates is kron(projections, s)
          so correct operation in the Wannier basis is not guaranteed
    """
    Nr = H.shape[0]
    Nw = H.shape[1]
    Nw0 = Nw//2
    bsy = np.kron(np.eye(Nw0, dtype=np.complex128), ut.sy)
    Hnew = np.zeros((Nr, Nw,Nw), dtype=np.complex128)
    for irv,rv in enumerate(rvecs):
        irv2, ok = ut.FindRv(rvecs, -rv)
        H1 = H[irv2]
        H2 = bsy@np.transpose(H1)@bsy
        Hnew[irv] = H2
    return Hnew



def TRderivsSelf(HP, rvecs, md=ut.ez):
    r"""
    Calculates the derivatives of the Hamiltonain over directions of the magnetization
    - Ferromagnet is presumed!
    - it is presumed that exchange part of the Hamiltonian is ~sigma_m - Pauli matrix in the magnetization direction

    HP - tb-Hamiltonian 
    rvecs - r-vector structure
    md - direction of the magnetization

    Return list in the relative ccordinates [local ex, local ey]
    """
    Nr = HP.shape[0]
    Nw = HP.shape[1]
    
    ex1, ey1, ez1 = ut.makebasis(md)
    sx1 = ex1[0]*ut.sx + ex1[1]*ut.sy + ex1[2]*ut.sz
    sy1 = ey1[0]*ut.sx + ey1[1]*ut.sy + ey1[2]*ut.sz
    sz1 = ez1[0]*ut.sx + ez1[1]*ut.sy + ez1[2]*ut.sz
    
    HPT = TimeRevH(HP,rvecs)
    Hodd = (HP-HPT)/2
    
    Hd1 = ConvertSigma(HP, sz1, sx1)
    Hd2 = ConvertSigma(HP, sz1, sy1)
   
    return Hd1, Hd2



def TRderivs(HP, rvecs, md=ut.ez):
    r"""
    Calculates the derivatives of the Hamiltonain over directions of the magnetization
    - Ferromagnet is presumed!
    - it is presumed that exchange part of the Hamiltonian is ~sigma_m - Pauli matrix in the magnetization direction

    HP - tb-Hamiltonian 
    rvecs - r-vector structure
    md - direction of the magnetization

    return array in cartesian coordinates
    """
    Nr = HP.shape[0]
    Nw = HP.shape[1]
    
    ex1, ey1, ez1 = ut.makebasis(md)
    sx1 = ex1[0]*ut.sx + ex1[1]*ut.sy + ex1[2]*ut.sz
    sy1 = ey1[0]*ut.sx + ey1[1]*ut.sy + ey1[2]*ut.sz
    sz1 = ez1[0]*ut.sx + ez1[1]*ut.sy + ez1[2]*ut.sz
    
    HPT = TimeRevH(HP,rvecs)
    Hodd = (HP-HPT)/2
    
    Hd1 = ConvertSigma(HP, sz1, sx1)
    Hd2 = ConvertSigma(HP, sz1, sy1)

    ADerivs = np.zeros((3, Nr, Nw, Nw), dtype=np.complex128)
    ADerivs[0] = ex1[0]*Hd1 + ey1[0]*Hd2
    ADerivs[1] = ex1[1]*Hd1 + ey1[1]*Hd2
    ADerivs[2] = ex1[2]*Hd1 + ey1[2]*Hd2
    
    return ADerivs