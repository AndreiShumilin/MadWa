import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import numba as nb

#-----------------------------------------------------#
#---------- Routines for Hr attributes ---------------#
#-----------------------------------------------------#

def cellVolume(cell, Dim2D = False):
    a1 = cell[0,...]
    a2 = cell[1,...]
    a3 = cell[2,...]

    if Dim2D:
        a12 = np.cross(a1,a2)
        cellV = np.sqrt(np.sum(a12*a12))
    else:
        a12 = np.cross(a1,a2)
        cellV = np.dot(a12,a3)
    return np.abs(cellV)

# ------------------------------------------------------------
# H(k)
# ------------------------------------------------------------    
@nb.njit
def Hk_njit(H_ij, rvects, deg, kvec, cell):
    """
    Build k-space Hamiltonian from Wannier90 real-space Hamiltonian.
    Parameters
    ----------
    H_ij: Real-space Hamiltonian matrices (nR, num_wann, num_wann)
    rvects: Lattice vectors R (nR, 3)
    deg: Degeneracy factors from wannier90_hr.dat (nR)
    kvec: k-point np.array(3,)
    num_wann : Number of Wannier functions 
    """
    dim = H_ij.shape[1]
    Hk_tot = np.zeros((dim, dim), dtype=np.complex128)
    nR = len(rvects)
    kvec = kvec.astype(np.float64)
    for i in range(nR):
        #R = rvects[i].astype(np.float64)
        R = rvects[i]
        rvreal = R[0]*cell[0] + R[1]*cell[1]+R[2]*cell[2]
        phase = np.exp(1j * (kvec@rvreal))
        Hk_tot += (H_ij[i]/deg[i]) * phase
    #pass
    return Hk_tot
    
@nb.njit
def fractional_to_cartesian(kfrac, recip):

    return kfrac @ recip
  # # ------------------------------------------------------------
    # # KPATH
    # # ------------------------------------------------------------

@nb.njit
def get_kpath(points, recp, N):
    Points = fractional_to_cartesian(points, recp)
    
    # Calculate total number of points: (num_segments * N) + 1
    num_segments = len(Points) - 1
    total_pts = num_segments * N + 1
    
    # PRE-ALLOCATE arrays instead of using kpoi.append()
    # This is much faster and solves all "Type Determination" errors
    kpoi = np.zeros((total_pts, 3), dtype=np.float64)
    xx = np.zeros(total_pts, dtype=np.float64)
    Xmarks = np.zeros(len(Points), dtype=np.float64)

    # Initialize first point
    kpoi[0] = Points[0]
    xx[0] = 0.0
    Xmarks[0] = 0.0
    
    current_idx = 1
    length = 0.0

    for i in range(num_segments):
        for j in range(N):
            # Calculate interpolation
            # Use float(N) to ensure floating point division
            k = ((N - j - 1) / N) * Points[i] + ((j + 1) / N) * Points[i + 1]
            
            # Now dk is clearly (array - array)
            dk = k - kpoi[current_idx - 1]
            length += np.sqrt(np.sum(dk**2))
            
            kpoi[current_idx] = k
            xx[current_idx] = length
            current_idx += 1
            
        Xmarks[i + 1] = length
        
    return kpoi, xx, Xmarks
# ------------------------------------------------------------------------------
##  BANDS CALCULATION 
# ------------------------------------------------------------------------------
@nb.njit
def bands_w90(kpaths, recip, num_wann, H_ij, rvects, deg, cell, Nst):
    kpoi, xx, Xmarks= kpath_w90(kpaths,recip, Nst)
    Nk = len(kpoi)
    bands = np.zeros((Nk,num_wann), dtype=np.float64)
    for i in range(Nk):
        kpoint = kpoi[i]
        H = Hk_njit(H_ij, rvects, deg, kpoint,cell)
        e, _ = la.eigh(H)
        bands[i] = e
    # pass
    return bands
