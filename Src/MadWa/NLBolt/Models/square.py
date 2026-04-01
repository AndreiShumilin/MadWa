import numpy as np
import numpy.linalg


from ..Utils import tightbinding as tight



def SquareTB(t=1):
    r"""
    A most simple 2D tight-binding model with a single wavefunction
    in a square lattice with side=1 Angst and neighbor hopping term=t
    """
    TB = tight.TBH()
    TB.Nw = 1
    TB.Dim = 2
    TB.cell = np.eye(3)
    TB.rvecs = np.array([ [0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]], dtype=np.int32)

    TB.dH1 = {}
    TB.dH2 = {}
    for rv in TB.rvecs:
        H1 = np.zeros((1,1), dtype=np.float64)
        if not np.linalg.norm(rv) == 0:
            H1[0,0] = t
        TB.dH1[tuple(rv)] = H1.copy()
        TB.dH2[tuple(rv)] = H1.copy()
    
    TB.coords =  np.zeros((TB.Nw,3))
    TB.coords1 = np.zeros((TB.Nw,3))
    TB.coords2 = np.zeros((TB.Nw,3))

    TB.Exist = True
    
    return TB