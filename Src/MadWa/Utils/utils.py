import numpy as np
import numpy.linalg as la

import numba as nb

__all__ = ['ex','ey','ez','cellVolume','makebasis']


ex = np.array((1.0, 0.0, 0.0), dtype=np.float64)
ey = np.array((0.0, 1.0, 0.0), dtype=np.float64)
ez = np.array((0.0, 0.0, 1.0), dtype=np.float64)


@nb.njit
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

@nb.njit
def makebasis(v, dmin=0.01):
    r"""
    creates a basis where v would be along z
    """
    az = v/np.linalg.norm(v)
    vy = np.cross(az,ex)
    if np.linalg.norm(vy) < dmin:
        vy = np.cross(v,ey)
    ay = vy / np.linalg.norm(vy)
    vx = np.cross(ay,az)
    ax = vx / np.linalg.norm(vx)
    return ax,ay,az



def make_reciprocal_cell(cell):
    a1 = cell[0]
    a2 = cell[1]
    a3 = cell[2]
    V = cellVolume(cell, Dim2D = False)

    b1 = 2*np.pi*(np.cross(a2,a3))/V
    b2 = 2*np.pi*(np.cross(a3,a1))/V
    b3 = 2*np.pi*(np.cross(a1,a2))/V

    #Kcell = np.array([b1,b2,b3], dtype=np.float64)
    Kcell = np.zeros((3,3), dtype=np.float64)
    Kcell[0] = b1
    Kcell[1] = b2
    Kcell[2] = b3

    return Kcell



@nb.njit
def fractional_to_cartesian(kfrac, recip):
    return kfrac @ recip

@nb.njit
def get_kpath(points, recp=None, N=30):
    r"""
    Generates k-path from the keypoints "points"
    if recp is None, "points" are treated as real coordinates
    othervise they are considered as fractional coordinates based on reciprocal cell recp
    N - nuber of points per branch of the k-path
    """
    
    if recp is None:
        Points = points
    else:
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

